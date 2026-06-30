import gc
import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import lib.utils.misc as misc
from tqdm import tqdm
'''用于保存结果文件'''
from lib.test.evaluation.environment import env_settings
from liu_tools.liu_train_tools import _save_tracker_output,_check_resultsFile_exist


class TestTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)
    def cycle_dataset_ttt_train(self, loader):
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        # '''fix the normalization layers in the pretrained seqtrackv1 model'''
        # if self.settings.fix_norm:
        #     self.actor.fix_norms()

        self._init_timing()
        print("Current Epoch: ", self.epoch)
        self.state_dict = self.actor.net.module.state_dict()
        for i, data in enumerate(tqdm(loader), 1):
            # if self.move_data_to_gpu:
            #     data = data.to(self.device)
            #self.actor.initialize(data)
            #a 增加一个平均AUC <0.88的限制
            self.actor.initialize(data)
            '''加载权重'''
            for name, param in self.actor.net.module.named_parameters():
                if name in self.state_dict:
                    # Clone the tensor before assigning to avoid the error
                    cloned_param = self.state_dict[name].clone().detach()
                    param.data = cloned_param
                else:
                    print(f"Parameter {name} not found in checkpoint. Initializing with default values.")

            iou_list=[]
            loss_interval = 30

            for idx ,item in enumerate (tqdm(data['search_images'][1:])):
                data['epoch'] = self.epoch
                data['settings'] = self.settings
                #torch.autograd.set_detect_anomaly(True)
                # forward pass
                if not self.use_amp:
                    loss, stats = self.actor(data)
                    loss.backward()
                else:
                    with autocast():
                        loss, stats = self.actor(data)
                        self.scaler.scale(loss).backward()
                #40个自动计算一次梯度
                if (idx+1) % loss_interval == 0:
                    if not self.use_amp:
                        if self.settings.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    else:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    torch.cuda.synchronize()

                iou_list.append(stats['frame_IoU'])
            average_iou = sum(iou_list) / len(iou_list)
            print(str(average_iou) + "average seq IOU-------------------------------------------------------")
            del loss


        # update statistics
        batch_size = 1
        # batch_size = data['template_images'].shape[loader.stack_dim]
        self._update_stats(stats, batch_size, loader)

        # print statistics
        self._print_stats(idx, loader, batch_size)
    def cycle_dataset_ttt_test(self, loader):
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        '''准备环境文件 用于输出结果文件'''
        env = env_settings()
        result_dir = '{}/{}/{}'.format(env.results_path, self.settings.script_name, self.settings.config_name)
        output = {'target_bbox': [], 'time': []}
        tracker = {'name': self.settings.script_name, 'parameter_name': self.settings.config_name,'results_dir': result_dir}
        seq = {'name': 'test', 'dataset': 'test' }

        '''output'''
        # '''fix the normalization layers in the pretrained seqtrackv1 model'''
        # if self.settings.fix_norm:
        #     self.actor.fix_norms()

        self._init_timing()
        print("Current Epoch: ", self.epoch)

        for i, data in enumerate(tqdm(loader), 1):
            # if self.move_data_to_gpu:
            #     data = data.to(self.device)
            #self.actor.initialize(data)
            #a 增加一个平均AUC <0.88的限制

            '''设置输出结果文件'''
            seq['name'] = data['seq_name'][0]
            seq['dataset'] = data['dataset'][0]
            seq['results_dir'] = result_dir
            output['target_bbox'] = data['template_anno'][0].numpy().tolist()

            '''检查输出结果文件是否存在，存在就跳过'''
            if _check_resultsFile_exist(seq, tracker):
                print("test_trainer.py: results file already exists, skipping...")
                continue

            self.actor.initialize(data)

            for idx in range(len(data['search_images']) - 1):
                # data['epoch'] = self.epoch
                # data['settings'] = self.settings

                #torch.autograd.set_detect_anomaly(True)
                # forward pass
                with torch.no_grad():
                    if not self.use_amp:
                        self.actor(data)
                        output['target_bbox'].append(self.actor.state)
                    else:
                        with autocast():
                            self.actor(data)
                            output['target_bbox'].append(self.actor.state)

            '''save results'''
            _save_tracker_output(seq, tracker, output)
            #some error in next line.
            #self._update_stats(seq, batch_size=1, loader=loader)



    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        # '''fix the normalization layers in the pretrained seqtrackv1 model'''
        # if self.settings.fix_norm:
        #     self.actor.fix_norms()

        self._init_timing()
        print("Current Epoch: ", self.epoch)
        # print(loader.training)

        for i, data in enumerate(tqdm(loader), 1):
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            torch.cuda.synchronize()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)


    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset_ttt_test(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            if misc.is_main_process():
                # print(self.settings.log_file)
                with open(self.settings.log_file, 'a') as f:
                    f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
