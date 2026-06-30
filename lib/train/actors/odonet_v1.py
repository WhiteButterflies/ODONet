from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.utils.heapmap_utils import generate_heatmap

class ODONET_V1_Actor(BaseActor):
    """ Actor for training the Gohan"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict,other_data_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data,other_data_dict)

        return loss, status

    def transform_tensor(self,tensor):
        # 复制一份以防修改原始数据
        new_tensor = tensor.clone()
        new_tensor[:, 0:2] = 0.25 + 0.5 * new_tensor[:, 0:2]  # 前两个值
        new_tensor[:, 2:4] = 0.5 * new_tensor[:, 2:4]  # 后两个值
        return new_tensor

    def compute_normalized_offset_xywh(self,box_prev, box_next, H, W):
        """
        生成用于grid_sample的归一化偏移场(dy, dx)，修复所有已知问题

        Args:
            box_prev (Tensor): (B, 4) [x, y, w, h]，归一化坐标(0~1)
            box_next (Tensor): (B, 4) 相同格式
            H, W: 特征图尺寸

        Returns:
            offset: (B, H, W, 2) 偏移场，[-1, 1]范围
        """
        B = box_prev.size(0)
        dtype = box_prev.dtype
        device = box_prev.device

        # 转换到特征图坐标（添加1e-6防止除以0）
        scale = torch.tensor([W, H, W, H], dtype=dtype, device=device)
        box_prev = box_prev * scale + torch.tensor([0.5, 0.5, 1e-6, 1e-6], device=device)  # 中心对齐
        box_next = box_next * scale + torch.tensor([0.5, 0.5, 1e-6, 1e-6], device=device)

        # 生成中心对齐的网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=dtype, device=device) + 0.5,  # 像素中心坐标
            torch.arange(W, dtype=dtype, device=device) + 0.5,
            indexing='ij'
        )
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)

        # 解包坐标并确保数值稳定性
        x0, y0 = box_prev[:, 0], box_prev[:, 1]
        w0, h0 = box_prev[:, 2].clamp(min=1e-6), box_prev[:, 3].clamp(min=1e-6)
        x1, y1 = box_next[:, 0], box_next[:, 1]
        w1, h1 = box_next[:, 2].clamp(min=1e-6), box_next[:, 3].clamp(min=1e-6)

        # 计算有效区域mask
        x0_min = x0.view(B, 1, 1)
        y0_min = y0.view(B, 1, 1)
        x0_max = x0_min + w0.view(B, 1, 1)
        y0_max = y0_min + h0.view(B, 1, 1)

        mask = (grid_x >= x0_min) & (grid_x <= x0_max) & \
               (grid_y >= y0_min) & (grid_y <= y0_max)

        # 计算相对位置
        rel_x = (grid_x - x0_min) / w0.view(B, 1, 1)
        rel_y = (grid_y - y0_min) / h0.view(B, 1, 1)

        # 映射到下一帧坐标
        tgt_x = x1.view(B, 1, 1) + rel_x * w1.view(B, 1, 1)
        tgt_y = y1.view(B, 1, 1) + rel_y * h1.view(B, 1, 1)

        # 计算实际偏移量
        dx = tgt_x - grid_x
        dy = tgt_y - grid_y

        # 归一化到[-1, 1]
        dx_norm = dx / ((W - 1.0) / 2.0)
        dy_norm = dy / ((H - 1.0) / 2.0)

        # 构建偏移场
        offset = torch.zeros((B, H, W, 2), dtype=dtype, device=device)
        offset[..., 0] = dy_norm  # channel 0: dy
        offset[..., 1] = dx_norm  # channel 1: dx
        offset[~mask] = 0  # 非有效区域置零

        return offset

    def compute_normalized_offset_xywh_old(self,box_prev, box_next, H, W):
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

    def forward_pass(self, data):
        b = data['search_images'].shape[1]   # n,b,c,h,w
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b,dim=0)  # (n*b, c, h, w)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b,dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b,dim=0)
        search_anno_list = data['search_anno'].view(-1, *data['template_anno'].shape[2:]).split(b,dim=0)
        prev_search_anno_list = data['search_extract_anno'].view(-1, *data['search_extract_anno'].shape[2:]).split(b,dim=0)
        other_data_dict = {"search_anno_list":search_anno_list,
            "prev_search_anno_list":prev_search_anno_list}
        out_list = []
        neck_h_state = [None] * self.cfg.MODEL.NECK.N_LAYERS
        deform_indexes = self.cfg.MODEL.ENCODER.DEFORM_INDEXES
        deform_index = deform_indexes[0]
        for i in range(len(search_list)):
            search_i_list = [search_list[i]]
            search_i_anno_list = [search_anno_list[i]]
            search_i_prev_anno_list = [prev_search_anno_list[i]]
            enc_opt = self.net(template_list=template_list, search_list=search_i_list, template_anno_list=template_anno_list,search_anno_list=search_i_anno_list,prev_search_anno_list=search_i_prev_anno_list, mode='encoder') # forward the encoder
            encoder_out,neck_out,neck_h_state = self.net(enc_opt=enc_opt,neck_h_state=neck_h_state,search_anno_list=search_i_anno_list,mode="neck")
            outputs = self.net(feature=neck_out, mode="decoder")

            '''added by liu for storage offset'''
            outputs['gt_offset'] =  self.compute_normalized_offset_xywh(search_i_prev_anno_list[0],search_i_anno_list[0],self.net.module.fx_sz,self.net.module.fx_sz)
            outputs['pred_offset'] =neck_h_state[deform_index][0]
            outputs['pred_sampled_x_point'] =neck_h_state[deform_index][1]

            '''ended'''

            '''forward vqvae head for prediction'''
            vqvae_head_input = encoder_out[:, -1:, :]
            vqvae_head_input =self.net(enc_opt=vqvae_head_input, mode='down')
            vqvae_out, vqvae_loss1,vqvae_loss2 = self.net(enc_opt=vqvae_head_input, mode='vqvae_head')
            outputs['vqvae_head'] = vqvae_out
            outputs['vqvae_loss1'] = vqvae_loss1
            outputs['vqvae_loss2'] = vqvae_loss2
            ''''''
            out_dict = outputs
            out_list.append(out_dict)

        return out_list,other_data_dict


    def compute_losses(self, pred_dict, gt_dict,other_data_dict, return_status=True):
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda()  #
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                                 self.cfg.MODEL.ENCODER.STRIDE)  # list of torch.Size([b, H, W])

        # For InfoNCE loss. Skip the objective entirely when disabled so
        # odonet_wyq_v1 can run as a pure MixStyle ablation.
        simclr_weight = float(self.loss_weight.get('simclr', 0.0))
        if simclr_weight > 0:
            list_of_sampled_x_point = [pred_dict[i]['pred_sampled_x_point'] for i in range(len(pred_dict))]
            loss_simclr = self.objective['simclr'](list_of_sampled_x_point) * simclr_weight
            total_loss += loss_simclr
        else:
            loss_simclr = total_loss.new_tensor(0.0)

        for i in range(len(pred_dict)):

            diff_boxes_search2template = box_xyxy_to_cxcywh(box_xywh_to_xyxy(other_data_dict['search_anno_list'][i])) -  box_xyxy_to_cxcywh(box_xywh_to_xyxy(other_data_dict['prev_search_anno_list'][i]))

            codebook_vqvae_loss1 = pred_dict[i]['vqvae_loss1'][0] + pred_dict[i]['vqvae_loss1'][1] + \
                                   pred_dict[i]['vqvae_loss1'][2]
            # vqvae_loss2 = pred_dict[i]['vqvae_loss2']

            # gt gaussian map
            gt_bbox = gt_dict['search_anno'][-len(pred_dict):][i]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = gt_gaussian_maps_list[-len(pred_dict):][i].unsqueeze(1)  # torch.Size([b, 1, H, W])

            # Get boxes
            pred_boxes = pred_dict[i]['pred_boxes']  # torch.Size([b, 1, 4])
            pred_boxes_vq = pred_dict[i]['vqvae_head'].view(-1,4)  # torch.Size([b, 1, 4])

            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                               max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            pred_boxes_vq_vec = pred_boxes_vq.view(-1,4)
            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            l1_loss_vq = self.objective['l1'](pred_boxes_vq_vec, diff_boxes_search2template)

            # compute location loss
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            #compute offset loss
            offset_l1_loss = self.objective['offset_l1'](pred_dict[i]['pred_offset'],pred_dict[i]['gt_offset'])
            offset_cosine_loss = self.objective['offset_cosine'](pred_dict[i]['pred_offset'],pred_dict[i]['gt_offset'])

            # weighted sum
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
                'focal'] * location_loss +self.loss_weight['pos_l1']* l1_loss_vq + codebook_vqvae_loss1+ self.loss_weight['offset_l1']*offset_l1_loss +self.loss_weight['offset_cosine']*offset_cosine_loss
            total_loss += loss

            if return_status:
                # status for log
                mean_iou = iou.detach().mean()
                status = {f"{i}frame_Loss/total": loss.item(),
                          f"{i}frame_Loss/giou": giou_loss.item(),
                          f"{i}frame_Loss/l1": l1_loss.item(),
                          f"{i}frame_Loss/location": location_loss.item(),
                          f"{i}frame_IoU": mean_iou.item(),
                          f"{i}frame_Loss/offset_l1": offset_l1_loss.item(),
                          f"{i}frame_Loss/offset_cosine": offset_cosine_loss.item(),
                          f"{i}frame_Loss/VQ": l1_loss_vq.item(),
                          f"{i}frame_Loss/InfoNCE": loss_simclr.item(),
                          # f"{i}frame_Loss/CodebookLoss": codebook_vqvae_loss1.item(),
                          # f"{i}frame_Loss/CodebookUsage": pred_dict[i]['vqvae_loss1'][3].item(),
                          }
                total_status.update(status)

        if return_status:
            return total_loss, total_status
        else:
            return total_loss
