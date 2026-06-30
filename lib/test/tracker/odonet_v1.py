import time

from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh,box_cxcywh_to_xyxy,box_xyxy_to_xywh
from lib.test.utils.hann import hann2d
from lib.models.odonet_v1 import build_odonet
from lib.test.tracker.utils import Preprocessor,suppress_ambiguous_peaks_by_location
from lib.utils.box_ops import clip_box
import numpy as np
import os
#added by liu for vis_atttn
from lib.test.tracker.tracker_utils import vis_attn_maps


class ODONET(BaseTracker):
    def __init__(self, params, dataset_name):
        super(ODONET, self).__init__(params)
        network = build_odonet(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu',weights_only=False)['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True:  # for window penalty
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.debug = params.debug
        self.frame_id = 0
        # for update
        self.h_state = [None] * self.cfg.MODEL.NECK.N_LAYERS
        if self.debug == 2 or self.debug == 1 :
            save_dir = "vis"
            self.save_dir = os.path.join(save_dir, params.yaml_name)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPT, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPT[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPT.DEFAULT
        print("Update threshold is: ", self.update_threshold)

        if hasattr(self.cfg.TEST.UPH, DATASET_NAME):
            self.update_h_t = self.cfg.TEST.UPH[DATASET_NAME]
        else:
            self.update_h_t = self.cfg.TEST.UPH.DEFAULT
        print("Update hidden state threshold is: ", self.update_h_t)

        if hasattr(self.cfg.TEST.INTER, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.INTER[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.INTER.DEFAULT
        print("Update intervals is: ", self.update_intervals)

        if hasattr(self.cfg.TEST.MB, DATASET_NAME):
            self.memory_bank = self.cfg.TEST.MB[DATASET_NAME]
        else:
            self.memory_bank = self.cfg.TEST.MB.DEFAULT
        print("Update threshold is: ", self.memory_bank)

    def transform_tensor(self,tensor):
        # 复制一份以防修改原始数据
        new_tensor = tensor.clone()
        new_tensor[:, 0:2] = 0.25 + 0.5 * new_tensor[:, 0:2]  # 前两个值
        new_tensor[:, 2:4] = 0.5 * new_tensor[:, 2:4]  # 后两个值
        return new_tensor
    def compute_normalized_offset_xywh(self,box_prev, box_next, H, W):
        """
        Generate normalized offset field (dy, dx) for grid_sample, aligned with your feature map.

        Args:
            box_prev (Tensor): shape (B, 4), [x, y, w, h], top-left normalized (0~1)
            box_next (Tensor): shape (B, 4), same format
            H, W: feature map height and width

        Returns:
            offset: Tensor of shape (B, H, W, 2), dy and dx in [-1, 1] range, others zero.
        """
        B = box_prev.size(0)
        dtype = box_prev.dtype
        device = box_prev.device

        # scale to feature map coordinates
        box_prev = box_prev * torch.tensor([W, H, W, H], dtype=dtype, device=device)
        box_next = box_next * torch.tensor([W, H, W, H], dtype=dtype, device=device)

        # mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=dtype, device=device),
            torch.arange(W, dtype=dtype, device=device),
            indexing='ij'
        )
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)

        # unpack boxes
        x0, y0, w0, h0 = box_prev.unbind(dim=1)
        x1, y1, w1, h1 = box_next.unbind(dim=1)

        x0_min = x0.view(B, 1, 1)
        y0_min = y0.view(B, 1, 1)
        x0_max = (x0 + w0).view(B, 1, 1)
        y0_max = (y0 + h0).view(B, 1, 1)

        # valid region mask
        mask = (grid_x >= x0_min) & (grid_x <= x0_max) & (grid_y >= y0_min) & (grid_y <= y0_max)

        # relative position in prev box
        rel_x = (grid_x - x0_min) / w0.view(B, 1, 1)
        rel_y = (grid_y - y0_min) / h0.view(B, 1, 1)

        # map to next box
        tgt_x = x1.view(B, 1, 1) + rel_x * w1.view(B, 1, 1)
        tgt_y = y1.view(B, 1, 1) + rel_y * h1.view(B, 1, 1)

        # offset in pixels
        dx = tgt_x - grid_x
        dy = tgt_y - grid_y

        # normalize to [-1, 1] for grid_sample
        dx_norm = dx / ((W - 1.0) / 2.0)
        dy_norm = dy / ((H - 1.0) / 2.0)

        # build final offset map (B, H, W, 2)
        offset = torch.zeros((B, H, W, 2), dtype=dtype, device=device)
        offset[..., 0][mask] = dy_norm[mask]  # channel 0: dy
        offset[..., 1][mask] = dx_norm[mask]  # channel 1: dx

        return offset
    def analyze_and_visualize_offset(self,data_real, cell_size=40, arrow_color=(0, 0, 255), thickness=1, tip_length=0.3,visualize=False,save=False):
        B, H, W, _ = data_real.shape

        for b in range(B):
            img = np.ones((H * cell_size, W * cell_size, 3), dtype=np.uint8) * 255

            '''#calc gram'''
            # 3. 提取并重塑该样本
            sample_offset = data_real[b]  # 形状: [H, W, 2]
            sample_offset_transposed = sample_offset.transpose(2, 0, 1)  # 形状: [2, H, W]

            # 4. 展平特征
            C = sample_offset_transposed.shape[0]  # C = 2
            F_flattened = sample_offset_transposed.reshape(C, -1)  # 形状: [2, H*W]

            # 5. 计算 Gram 矩阵
            gram_matrix = F_flattened @ F_flattened.T

            # (可选) 归一化
            num_elements = H * W
            gram_matrix_normalized = gram_matrix / num_elements
            '''end'''

            offset_x = data_real[b, ..., 1]
            offset_y = data_real[b, ..., 0]

            if visualize:
                #打印Gram矩阵和归一化的Gram矩阵
                print("计算出的 Gram 矩阵 (2x2):\n", gram_matrix)
                # print("\n归一化后的 Gram 矩阵:\n", gram_matrix_normalized)

                # 可视化箭头
                for i in range(H):
                    for j in range(W):
                        start_point = (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2)
                        dx_pixel = offset_x[i, j] * ((W - 1.0) / 2.0)
                        dy_pixel = offset_y[i, j] * ((H - 1.0) / 2.0)
                        end_point = (
                            int(start_point[0] + dx_pixel * cell_size),
                            int(start_point[1] + dy_pixel * cell_size)
                        )
                        cv2.arrowedLine(img, start_point, end_point, arrow_color, thickness, tipLength=tip_length)
                # 添加Gram矩阵值的文本显示
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_color = (0, 0, 0)  # 黑色文本
                line_type = 2

                # 格式化Gram矩阵值为字符串
                gram_text = f"Gram Matrix:\n[{gram_matrix[0, 0]:.2f}, {gram_matrix[0, 1]:.2f}]\n[{gram_matrix[1, 0]:.2f}, {gram_matrix[1, 1]:.2f}]"

                # 在图像左上角显示Gram矩阵值
                y0, dy = 30, 25
                for i, line in enumerate(gram_text.split('\n')):
                    y = y0 + i * dy
                    cv2.putText(img, line, (10, y), font, font_scale, font_color, line_type)

                # 显示
                window_name = f"Sample {b + 1}"
                cv2.imshow(window_name, img)
                if save:
                    #save to local
                    os.makedirs(r"{}/gram_matrix/".format(self.save_path),exist_ok=True)
                    with open(r"{}/gram_matrix/gram_matrices.txt".format(self.save_path), "a") as f:
                            g00 = gram_matrix[0, 0]
                            g01 = gram_matrix[0, 1]
                            g10 = gram_matrix[1, 0]
                            g11 = gram_matrix[1 ,1]

                            # 将4个元素写入文件，用逗号隔开
                            line = f"{g00},{g01},{g10},{g11}\n"
                            f.write(line)
                    os.makedirs(r"{}/offset/".format(self.save_path),exist_ok=True)
                    cv2.imwrite(r'{}/offset/{}.jpg'.format(self.save_path,self.frame_id),img)

    def analyze_and_visualize_offset_with_search(
            self,
            data_real,  # [B,H,W,2]，其中[...,0]=offset_y, [...,1]=offset_x
            cell_size=40,
            arrow_color=(0, 0, 255),
            thickness=1,
            tip_length=0.3,
            visualize=False,
            save=False,
            search_img=None,  # 传入对应的搜索区域原图 (H_img,W_img,3)，如 224x224x3，BGR或RGB都行
            draw_points_on_image=True,  # 是否在原图上画点
            draw_start_points=False,  # 是否画起点(网格中心)的点
            start_point_color=(0, 255, 0),  # 绿点
            end_point_color=(0, 0, 255),  # 红点
            point_radius=2,
            alpha=0.75  # 覆盖透明度（点画在叠加层上）
    ):
        import numpy as np, cv2, os

        B, H, W, _ = data_real.shape

        for b in range(B):
            # ====== 计算 Gram（与你原来的相同）======
            sample_offset = data_real[b]  # [H,W,2]
            sample_offset_T = sample_offset.transpose(2, 0, 1)  # [2,H,W]
            C = sample_offset_T.shape[0]
            F_flat = sample_offset_T.reshape(C, -1)  # [2,H*W]
            gram_matrix = F_flat @ F_flat.T
            num_elements = H * W
            gram_matrix_normalized = gram_matrix / num_elements

            # ====== 取 offset 分量（保持你原来的次序）======
            offset_x = data_real[b, ..., 1]  # [H,W]
            offset_y = data_real[b, ..., 0]  # [H,W]

            # ====== 1) 继续在放大网格上画箭头（保持你现有效果）======
            if visualize:
                big_img = np.ones((H * cell_size, W * cell_size, 3), dtype=np.uint8) * 255
                for i in range(H):
                    for j in range(W):
                        start_point = (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2)
                        dx_cell = offset_x[i, j] * ((W - 1.0) / 2.0)  # 以“网格单元”为单位
                        dy_cell = offset_y[i, j] * ((H - 1.0) / 2.0)
                        end_point = (
                            int(start_point[0] + dx_cell * cell_size),
                            int(start_point[1] + dy_cell * cell_size)
                        )
                        cv2.arrowedLine(big_img, start_point, end_point, arrow_color, thickness, tipLength=tip_length)

                # 叠字：Gram
                font = cv2.FONT_HERSHEY_SIMPLEX
                gram_text = f"Gram Matrix:\n[{gram_matrix[0, 0]:.2f}, {gram_matrix[0, 1]:.2f}]\n[{gram_matrix[1, 0]:.2f}, {gram_matrix[1, 1]:.2f}]"
                y0, dy = 30, 25
                for k, line in enumerate(gram_text.split('\n')):
                    y = y0 + k * dy
                    cv2.putText(big_img, line, (10, y), font, 0.6, (0, 0, 0), 2)

                cv2.imshow(f"Sample {b + 1}", big_img)

                if save:
                    os.makedirs(f"{self.save_path}/gram_matrix/", exist_ok=True)
                    with open(f"{self.save_path}/gram_matrix/gram_matrices.txt", "a") as f:
                        g00, g01, g10, g11 = gram_matrix[0, 0], gram_matrix[0, 1], gram_matrix[1, 0], gram_matrix[1, 1]
                        f.write(f"{g00},{g01},{g10},{g11}\n")
                    os.makedirs(f"{self.save_path}/offset/", exist_ok=True)
                    cv2.imwrite(f"{self.save_path}/offset/{self.frame_id}.jpg", big_img)

            # ====== 2) 在原始搜索图上画“点”（这是你要的效果）======
            if draw_points_on_image and (search_img is not None):
                vis = search_img.copy()
                H_img, W_img = vis.shape[:2]
                cell_w = float(W_img) / float(W)
                cell_h = float(H_img) / float(H)

                # 为了更漂亮，用半透明叠加层来画点
                overlay = vis.copy()

                for i in range(H):
                    for j in range(W):
                        # 起点（网格中心）映射到图像像素
                        x0 = (j + 0.5) * cell_w
                        y0 = (i + 0.5) * cell_h

                        # 位移（先是“网格单元”为单位），再换算成像素
                        dx_cell = offset_x[i, j] * ((W - 1.0) / 2.0)
                        dy_cell = offset_y[i, j] * ((H - 1.0) / 2.0)
                        x1 = x0 + dx_cell * cell_w
                        y1 = y0 + dy_cell * cell_h

                        # 可选：画起点
                        if draw_start_points:
                            cx0, cy0 = int(round(x0)), int(round(y0))
                            if 0 <= cx0 < W_img and 0 <= cy0 < H_img:
                                cv2.circle(overlay, (cx0, cy0), point_radius, start_point_color, -1)

                        # 画终点（箭头“指向的位置”）
                        cx1, cy1 = int(round(x1)), int(round(y1))
                        if 0 <= cx1 < W_img and 0 <= cy1 < H_img:
                            cv2.circle(overlay, (cx1, cy1), point_radius, end_point_color, -1)

                # 叠加
                vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

                # 可选保存
                if save:
                    os.makedirs(f"{self.save_path}/offset_points/", exist_ok=True)
                    cv2.imwrite(f"{self.save_path}/offset_points/{self.frame_id}.jpg", vis)

                # 可选显示
                if visualize:
                    cv2.imshow(f"Offset points on image {b + 1}", vis)

    def initialize(self, image, info: dict):
        if self.debug == 2 or self.debug == 1:
            self.save_path = os.path.join(self.save_dir, info['seq_name'])
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                   output_sz=self.params.template_size)
        z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)] * self.num_template
        self.frame_id = 0
        self.memory_template_list = self.template_list.copy()
        self.memory_template_anno_list = self.template_anno_list.copy()

        '''addd by liu for search'''
        # get the initial templates
        x_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.search_factor,
                                                   output_sz=self.params.search_size)
        x_patch_arr = x_patch_arr
        search = self.preprocessor.process(x_patch_arr)
        self.search_list = [search] * 1

        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.search_size, self.params.search_size]),
                                                normalize=True)
        self.search_anno_list = [prev_box_crop.to(search.device).unsqueeze(0)] * 1
        self.memory_search_list = self.search_list.copy()
        self.memory_search_anno_list = self.search_anno_list.copy()

        prev_search_anno_list =  [self.transform_tensor(t) for t in self.template_anno_list]


        self.delta_xywh = self.memory_search_anno_list[0].unsqueeze(1)- prev_search_anno_list[-1].unsqueeze(1)
        self.network.delta_xywh = self.delta_xywh
        '''added by liu for acc'''
        #torch.set_float32_matmul_precision('high')
        '''ended'''
        '''added by liu for offset'''
        self.offset_online = self.compute_normalized_offset_xywh(prev_search_anno_list[-1],self.memory_search_anno_list[0],self.fx_sz,self.fx_sz)
        self.network.neck.interactions[2].deform_attention.offset_online = self.offset_online.to(torch.float32)
        '''ended'''
        '''added by liu for restore history bbox'''
        self.state_list =[info['init_bbox']]
        self.search_img_list = [x_patch_arr]
        self.search_img_factor_list = [resize_factor]

        '''added by liu for hook'''
        # # #
        # def get_module_by_name(model, name):
        #     """
        #     支持类似 'neck.interactions.0.deform_attention.conv_offset' 这种路径的模块访问
        #     """
        #     for attr in name.split('.'):
        #         if attr.isdigit():  # 如果是数字（如 list 或 ModuleList 的 index）
        #             model = model[int(attr)]
        #         else:
        #             model = getattr(model, attr)
        #     return model
        #
        # # 要hook的模块列表
        # target_layers = [
        #     "neck.interactions.2.deform_attention.conv_offset"
        # ]
        # # 存储特征
        # self.feature_maps = {}
        #
        # # 注册hook
        # self.hooks = []
        # for name in target_layers:
        #     layer = get_module_by_name(self.network, name)
        #
        #     def get_hook(name):  # 必须用闭包捕获name
        #         def hook(module, input, output):
        #             self.feature_maps[name] = output.detach()
        #
        #         return hook
        #
        #     self.hooks.append(layer.register_forward_hook(get_hook(name)))


        '''ended'''

        #added by liu for Gram state


    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)

        '''added by liu for 记录搜索区域old state'''
        self.prev_state = self.state
        self.prev_box_crop  = transform_image_to_crop(torch.tensor(self.prev_state),
                                                torch.tensor(self.prev_state),
                                                resize_factor,
                                                torch.Tensor(
                                                    [self.params.search_size, self.params.search_size]),
                                                normalize=True)
        self.search_img_list.append(x_patch_arr)
        self.search_img_factor_list.append(resize_factor)
        '''ended'''

        search = self.preprocessor.process(x_patch_arr)
        search_list = [search]

        '''get the attention weights by hook'''
        # attn_weight = []
        # hooks = []
        # for i in [31]:
        #     hooks.append(self.network.encoder.body.blocks[i].attn.attn_drop.register_forward_hook(
        #         lambda self, input, output: attn_weight.append(output)
        #     ))
        '''ended '''

        # run the encoder
        with torch.no_grad():
            enc_opt = self.network.forward_encoder(self.template_list, search_list, self.template_anno_list,search_anno_list=None,prev_search_anno_list=[self.memory_search_anno_list[-1]])

        # run the time neck
        with torch.no_grad():
            hidden_state = self.h_state.copy()
            encoder_out,out_neck, h = self.network.forward_neck(enc_opt, hidden_state)
        # run the decoder
        with torch.no_grad():
            out_dict = self.network.forward_decoder(feature=out_neck)
        # run the vqvqe head
        with torch.no_grad():
            vqvae_head_input = encoder_out[:, -1:, :]
            vqvae_head_input = self.network(enc_opt=vqvae_head_input, mode='down')
            vqvae_out, vqvae_loss1, vqvae_loss2 = self.network(enc_opt=vqvae_head_input, mode='vqvae_head')
            out_dict['vqvae_head'] = vqvae_out
            out_dict['vqvae_loss1'] = vqvae_loss1
            out_dict['vqvae_loss2'] = vqvae_loss2
        self.delta_xywh = vqvae_out.mean(dim=1, keepdim=True)
        self.network.delta_xywh = self.delta_xywh

        # add hann windows
        pred_score_map = out_dict['score_map']

        '''suppress heatmap mamual'''
        # pred_score_map = suppress_ambiguous_peaks_by_location(pred_score_map,{"peak_thresh_factor":0.3,'distance_threshold':6,'suppression_radius': 3,
        #                'suppression_factor': 0.1})
        '''ended'''


        if self.cfg.TEST.WINDOW == True:  # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['size_map'],
                                                                   out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response,
                                                                   out_dict['offset_map'],
                                                                   return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)

        self.now_box_crop =box_xyxy_to_xywh(box_cxcywh_to_xyxy(pred_boxes.mean(dim=0)))

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.now_box_crop2 = transform_image_to_crop(torch.tensor(self.state),
                                torch.tensor(self.prev_state),
                                resize_factor,
                                torch.Tensor(
                                    [self.params.search_size, self.params.search_size]),
                                normalize=True)
        #self.now_box_crop,self.now_box_crop2,self.prev_box_crop


        # update hiden state
        self.h_state = h
        if conf_score.item() < self.update_h_t:# 每帧低于更新阈值，直接删除hidden state
            # self.h_state = [None] * self.cfg.MODEL.NECK.N_LAYERS
            pass

        # update the template
        if self.num_template > 1:
            # print("conf_score {},threshold {}:".format(conf_score,self.update_threshold))
            if (conf_score > self.update_threshold):# 每帧高于更新阈值，更新模板记忆库，记忆上限500
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                                           output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                self.memory_template_list.append(template)
                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                        torch.tensor(self.state),
                                                        resize_factor,
                                                        torch.Tensor(
                                                            [self.params.template_size, self.params.template_size]),
                                                        normalize=True)
                self.memory_template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
                if len(self.memory_template_list) > self.memory_bank:
                    self.memory_template_list.pop(0)
                    self.memory_template_anno_list.pop(0)

        if (self.frame_id % self.update_intervals == 0):#从记忆库中根据采样间隔，用于实际的推理过程
            assert len(self.memory_template_anno_list) == len(self.memory_template_list)
            len_list = len(self.memory_template_anno_list)
            interval = len_list // self.num_template
            for i in range(1, self.num_template):
                idx = interval * i
                if idx > len_list:
                    idx = len_list
                self.template_list.append(self.memory_template_list[idx])
                self.template_list.pop(1)
                self.template_anno_list.append(self.memory_template_anno_list[idx])
                self.template_anno_list.pop(1)
        assert len(self.template_list) == self.num_template

        """ update the search by liu"""
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)
        prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                torch.tensor(self.state),
                                                resize_factor,
                                                torch.Tensor(
                                                    [self.params.search_size, self.params.search_size]),
                                                normalize=True)

        diff_gt =box_xyxy_to_cxcywh(box_xywh_to_xyxy(self.now_box_crop)).cpu()-box_xyxy_to_cxcywh(box_xywh_to_xyxy(self.prev_box_crop)).cpu()


        if (self.frame_id % 1 == 0):#从记忆库中根据采样间隔，用于实际的推理过程
            self.memory_search_list.append(search)
            self.memory_search_list.pop(0)
            self.memory_search_anno_list.append(prev_box_crop.to(search.device).unsqueeze(0))
            # self.memory_search_anno_list.append(pred_boxes)
            self.memory_search_anno_list.pop(0)

            '''added by liu for update offset'''
            '''type 1,使用预测bbox-begin'''
            # self.offset_online = self.compute_normalized_offset_xywh(self.prev_box_crop.unsqueeze(0).cuda(),
            #                                                          self.now_box_crop.unsqueeze(0), self.fx_sz,
            #                                                          self.fx_sz)
            '''type2 使用delta'''
            self.offset_online = self.compute_normalized_offset_xywh(self.prev_box_crop.unsqueeze(0).cuda(),
                                                                     self.prev_box_crop.unsqueeze(0).cuda()+box_xyxy_to_xywh(box_cxcywh_to_xyxy(self.delta_xywh.squeeze(0))), self.fx_sz,
                                                                     self.fx_sz)
            self.network.neck.interactions[2].deform_attention.offset_online = self.offset_online.to(torch.float32)
            '''ended '''
        '''ended '''

        # for debug
        if self.debug == 2:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_path, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)

        elif self.debug == 1:
            '''see attn map'''
            # vis_attn_maps(attn_weight, 7, 14, 16 ** 2, None, x_patch_arr, 'template', 'search_{}'.format(self.frame_id),
            #               frame_id=self.frame_id,save_path=os.path.join(self.save_path,'attn'))
            '''end attn map'''

            '''show hook'''
            import  einops
            # 打印所有hook的输出 shape
            # for name, fmap in self.feature_maps.items():
            #     print(f"{name}: {einops.rearrange(fmap, 'b p h w -> b h w p').shape}")
            # 一定记得移除 hook
            # for h in self.hooks:
            #     h.remove()

            '''show search area t'''
            x1, y1, w, h = self.now_box_crop.cpu().numpy()  # 差一个resize_factor
            image_t = cv2.cvtColor(self.search_img_list[-1], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_t, (int(x1 * (self.cfg.TEST.SEARCH_SIZE)), int(y1 * (self.cfg.TEST.SEARCH_SIZE))),
                          (int((x1 + w) * (self.cfg.TEST.SEARCH_SIZE)), int((y1 + h) * (self.cfg.TEST.SEARCH_SIZE))),
                          color=(0, 0, 255), thickness=2)
            os.makedirs(r'{}/search_area'.format(self.save_path), exist_ok=True)
            cv2.imwrite(r'{}/search_area/{}.jpg'.format(self.save_path, self.frame_id), image_t)
            cv2.imshow('vis t_search area', image_t)
            '''ended'''

            '''show offset'''
            # data_real = einops.rearrange(self.feature_maps['neck.interactions.2.deform_attention.conv_offset'],'b p h w -> b h w p').cpu()
            data_real = self.h_state[2][2].cpu()

            # self.analyze_and_visualize_offset(data_real.numpy(), cell_size=40, arrow_color=(0, 0, 255), thickness=1, tip_length=0.3,visualize=True,save=True)
            self.analyze_and_visualize_offset_with_search(data_real.numpy(), cell_size=40, arrow_color=(0, 0, 255), thickness=1, tip_length=0.3,visualize=True,save=True,search_img=image_t)

            # data_real  = 0.3*data_real*2+ 0.7*self.offset_online.cpu()
            '''ended '''
            '''show history bbox'''
            x1, y1, w, h = self.state
            self.state_list.append(self.state)
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #
            # # 假设 self.state_list 中每个元素都是 (x1, y1, w, h)
            # for i, state in enumerate(self.state_list):
            #     x1, y1, w, h = state
            #     # 可以选择用不同颜色或渐变色表示轨迹
            #     # 这里示例用蓝色，且越新的轨迹颜色越深（通过alpha值实现）
            #     alpha = (i + 1) / len(self.state_list)  # 计算透明度
            #     color = (255 * (1 - alpha), 0, 255 * alpha)  # 从蓝(0,0,255)渐变到紫(255,0,255)
            #
            #     # 绘制矩形框（或者你可以绘制中心点连线）
            #     cv2.rectangle(image_BGR,
            #                   (int(x1), int(y1)),
            #                   (int(x1 + w), int(y1 + h)),
            #                   color=color,
            #                   thickness=1)
            #     # 如果需要绘制中心点连线
            #     center = (int(x1 + w / 2), int(y1 + h / 2))
            #     if i > 0:  # 从第二个点开始画线
            #         prev_state = self.state_list[i - 1]
            #         prev_center = (int(prev_state[0] + prev_state[2] / 2), int(prev_state[1] + prev_state[3] / 2))
            #         cv2.line(image_BGR, prev_center, center, color=(0, 255, 0), thickness=2)

            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            os.makedirs(os.path.join(self.save_path,'images'), exist_ok=True)
            save_path = os.path.join(self.save_path,'images', "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
            cv2.imshow('vis', image_BGR)

            '''ended'''

            '''show t-1 bbox and t'''
            # # x1, y1, w, h=self.prev_box_crop.numpy() #差一个resize_factor
            # image_t_1 = cv2.cvtColor(self.search_img_list[-2], cv2.COLOR_RGB2BGR)
            # cv2.rectangle(image_t_1, (int(x1*(self.cfg.TEST.SEARCH_SIZE)), int(y1*(self.cfg.TEST.SEARCH_SIZE))), (int((x1+w)*(self.cfg.TEST.SEARCH_SIZE)), int((y1+h)*(self.cfg.TEST.SEARCH_SIZE))), color=(0, 0, 255), thickness=2)
            #
            # # x1, y1, w, h=self.now_box_crop.cpu().numpy() #差一个resize_factor
            # image_t = cv2.cvtColor(self.search_img_list[-1], cv2.COLOR_RGB2BGR)
            # cv2.rectangle(image_t, (int(x1*(self.cfg.TEST.SEARCH_SIZE)), int(y1*(self.cfg.TEST.SEARCH_SIZE))), (int((x1+w)*(self.cfg.TEST.SEARCH_SIZE)), int((y1+h)*(self.cfg.TEST.SEARCH_SIZE))), color=(0, 0, 255), thickness=2)
            # combined = np.hstack((image_t_1, image_t))
            # cv2.imshow('Optical Flow Visualization: t-1 (left) vs t (right)', combined)
            '''added by liu for vis '''
            '''added by liu for vis score map'''
            heatmap = np.array(pred_score_map.view(self.fx_sz, self.fx_sz).cpu())
            # 归一化到 0-255
            heatmap_data = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_data = heatmap_data.astype(np.uint8)

            # 放大到合适尺寸
            heatmap_resized = cv2.resize(heatmap_data, (224, 224), interpolation=cv2.INTER_CUBIC)
            # 应用高斯模糊，参数 (ksize, sigma) 可调
            heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (31, 31), sigmaX=5)

            # 使用伪彩色
            heatmap_color = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)

            # 使用 OpenCV 的窗口展示
            cv2.imshow('Heatmap', heatmap_color)
            #save
            os.makedirs(r'{}/heatmap'.format(self.save_path),exist_ok=True)
            cv2.imwrite(r'{}/heatmap/{}.jpg'.format(self.save_path,self.frame_id),heatmap_color)

            '''ended by liu'''
            # ---- 计算距离和夹角 ----
            import math
            import torch.nn.functional as F

            diff_pred = self.delta_xywh[:, 0].cpu()  # shape: [1, 4]

            # 欧氏距离
            distance = torch.norm(diff_gt[:2] - diff_pred[0][:2]).item()

            # 夹角（角度）
            cos_sim = F.cosine_similarity(diff_gt[:2].unsqueeze(0), diff_pred[0][:2].unsqueeze(0)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = torch.rad2deg(angle_rad).item()

            # ---- 可视化窗口 ----
            # 创建画布
            canvas = np.ones((224 * 2, 224 * 2, 3), dtype=np.uint8) * 255

            # 设置原点坐标和缩放因子
            origin = (112 * 2, 112 * 2)
            scale = 244 * 2 * 10  # 增大缩放因子

            # 绘制坐标轴 (更粗更明显)
            cv2.line(canvas, (origin[0] - 150, origin[1]), (origin[0] + 150, origin[1]), (100, 100, 100), 2)
            cv2.line(canvas, (origin[0], origin[1] - 150), (origin[0], origin[1] + 150), (100, 100, 100), 2)

            # 计算向量终点 (使用相同缩放)
            gt_end = (int(origin[0] + diff_gt[0] * scale), int(origin[1] + diff_gt[1] * scale))
            pred_end = (int(origin[0] + diff_pred[0][0] * scale), int(origin[1] + diff_pred[0][1] * scale))

            # 绘制向量箭头
            cv2.arrowedLine(canvas, origin, gt_end, (0, 0, 255), 4, tipLength=0.3)  # 红色GT向量
            cv2.arrowedLine(canvas, origin, pred_end, (255, 0, 0), 2, tipLength=0.1)  # 蓝色预测向量

            # 添加图例
            cv2.putText(canvas, "GT Vector", (gt_end[0] + 15, gt_end[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(canvas, "Pred Vector", (pred_end[0] + 15, pred_end[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

            # 显示向量信息
            cv2.putText(canvas, f"GT Vector: ({diff_gt[0]:.4f}, {diff_gt[1]:.4f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(canvas, f"Pred Vector: ({diff_pred[0][0]:.4f}, {diff_pred[0][1]:.4f})", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(canvas, f"Euclidean Distance: {distance:.6f}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(canvas, f"Angle: {angle_deg:.2f} deg", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)



            # cv2.namedWindow('vector_diff', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('vis', cv2.WINDOW_NORMAL)

            # cv2.moveWindow('vector_diff', 0, 0)  # (窗口名, x坐标, y坐标)
            # cv2.moveWindow('vis', 400, 0)  # 将vis窗口放在vector_diff窗口右侧
            # 展示窗口
            cv2.imshow('vector_diff', canvas)
            '''ended'''
            # print(conf_score)
            # cv2.waitKey(0)
        return {"target_bbox": self.state,
                "best_score": conf_score,
                "heatmap":np.array(pred_score_map.view(self.fx_sz, self.fx_sz).cpu())
                }

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)



def get_tracker_class():
    return ODONET
