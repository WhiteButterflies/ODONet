"""
MCITrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_xywh_to_xyxy,box_cxcywh_to_xyxy
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from .neck import build_neck
from collections import OrderedDict

import time

class ODONET(nn.Module):
    """ This is the base class for MCITrack """
    def __init__(self, encoder, decoder, neck,cfg,
                 num_frames=1, num_template=1, decoder_type="CENTER",other_model_dict = None):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type
        self.neck = neck

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder
        self.other_model_dict = other_model_dict


        self.num_frames = num_frames
        self.num_template = num_template
        self.freeze_en = cfg.TRAIN.FREEZE_ENCODER
        self.interaction_indexes = cfg.MODEL.ENCODER.INTERACTION_INDEXES

    def forward(self, template_list=None, search_list=None, template_anno_list=None,enc_opt=None,neck_h_state=None, feature=None,search_anno_list=None,prev_search_anno_list = None,mode="encoder"):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list,search_anno_list,prev_search_anno_list)
        elif mode=="down":
            return self.other_model_dict["down"](enc_opt)
        elif mode == "vqvae_head":
            if isinstance( self.other_model_dict["vqvae_head"],nn.Identity):
                return self.other_model_dict["vqvae_head"](enc_opt),[torch.tensor(0).cuda(),torch.tensor(0).cuda(),torch.tensor(0).cuda(),torch.tensor(0).cuda()],None
            else:
                return self.other_model_dict["vqvae_head"](enc_opt)
        elif mode == "neck":
            return self.forward_neck(enc_opt,neck_h_state,search_anno_list)
        elif mode == "decoder":
            return self.forward_decoder(feature[:,:self.num_patch_x,:])
        else:
            raise ValueError
    def forward_ori(self, template_list=None, search_list=None, template_anno_list=None,enc_opt=None,neck_h_state=None, feature=None, mode="encoder"):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list)
        elif mode == "neck":
            return self.forward_neck(enc_opt,neck_h_state)
        elif mode == "decoder":
            return self.forward_decoder(feature)
        else:
            raise ValueError

    def forward_encoder(self, template_list, search_list, template_anno_list,search_anno_list,prev_search_anno_list):
        # Forward the encoder
        xz = self.encoder(template_list, search_list, template_anno_list)
        #forward for SAM inject
        x = xz[:,:self.num_patch_x]
        z = xz[:,self.num_patch_x:]
        '''等分数据'''
        split_size =self.num_patch_x
        splits = torch.split(x, split_size, dim=1)
        # 将每个等分的数据片段添加到一个列表中
        extracted_x_list = [
            split.transpose(1, 2).reshape(split.shape[0], split.shape[2],
                                          int(math.sqrt(split.shape[1])), int(math.sqrt(split.shape[1])))
            for split in splits]
        '''use last template anno as search inject'''
        batched_x_input = []
        if self.training:
            '''group 1 cxcywh'''
            # delta_xywh = box_xyxy_to_cxcywh(box_xywh_to_xyxy(prev_search_anno_list[0])).unsqueeze(1)
            # delta_xywh = box_xyxy_to_cxcywh(box_xywh_to_xyxy(search_anno_list[0])).unsqueeze(1) -box_xyxy_to_cxcywh(box_xywh_to_xyxy(prev_search_anno_list[0])).unsqueeze(1)
            # delta_xywh = box_xyxy_to_cxcywh(box_xywh_to_xyxy(search_anno_list[0])).unsqueeze(1)
            '''group2 xyxy'''
            delta_xywh = box_xywh_to_xyxy(prev_search_anno_list[0]).unsqueeze(1)
            # delta_xywh = box_xywh_to_xyxy(search_anno_list[0]).unsqueeze(1) -box_xywh_to_xyxy(prev_search_anno_list[0]).unsqueeze(1)
            # delta_xywh = box_xywh_to_xyxy(search_anno_list[0]).unsqueeze(1)
            '''group3 xywh'''
            # delta_xywh =  prev_search_anno_list[0].unsqueeze(1)

        else:
            '''group 1'''
            # delta_xywh = box_xyxy_to_cxcywh(box_xywh_to_xyxy(prev_search_anno_list[-1])).unsqueeze(1)
            # delta_xywh = self.delta_xywh
            # delta_xywh = box_xyxy_to_cxcywh(box_xywh_to_xyxy(prev_search_anno_list[-1])).unsqueeze(1)+ self.delta_xywh
            '''group 2'''
            delta_xywh = box_xywh_to_xyxy(prev_search_anno_list[-1]).unsqueeze(1)
            # delta_xywh = box_cxcywh_to_xyxy(self.d  elta_xywh)
            # delta_xywh = box_xywh_to_xyxy(prev_search_anno_list[-1]).unsqueeze(1)+  box_cxcywh_to_xyxy(self.delta_xywh)
            '''group 3'''
            # delta_xywh = prev_search_anno_list[-1].unsqueeze(1)



        for i in range(int(1)):
            # xywh = template_anno_list[i].unsqueeze(1)
            # xyxy = box_xywh_to_xyxy(xywh)

            single_dict = {'boxes': delta_xywh,
                           "point_coords": delta_xywh[:, :, :2],
                           "point_labels": torch.ones((delta_xywh.shape[0], 1)).cuda(),
                           }
            batched_x_input.append(single_dict)
        x_outputs = []
        x_embeds_list = []
        '''added vy liu for calc time'''
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()
        '''ended'''


        for image_record, curr_embedding in zip(batched_x_input, extracted_x_list):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.other_model_dict['search_prompt_encoder'](
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            # iou_predictions,score_size_offset_head,prompt_tokens_out = self.mask_decoder(
            hs, x_features = self.other_model_dict['search_mask_decoder'](
                image_embeddings=curr_embedding,
                image_pe=self.other_model_dict['search_prompt_encoder'].get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            x_embeds_list.append(hs[:, -1, :].unsqueeze(1))
            x_outputs.append(
                x_features
            )
        x_embeds = torch.cat(x_embeds_list, dim=1)
        x_sam = torch.cat(x_outputs, dim=1)
        x = x + x_sam
        x_z_embed = torch.cat([x,z,x_embeds],dim=1)
        '''added by liu for calc time'''
        # end_event.record()
        # torch.cuda.synchronize()
        # print(f"SAM Injection Latency: {start_event.elapsed_time(end_event)} ms")
        '''ended by liu'''


        return x_z_embed
    def forward_neck(self,enc_out,neck_h_state,search_anno_list=None):
        x = enc_out
        xs = x[:, 0:self.num_patch_x]
        x,xs,h = self.neck(x,xs,neck_h_state,self.encoder.body.blocks,self.interaction_indexes,search_anno_list=search_anno_list)
        x = self.encoder.body.fc_norm(x)
        xs = xs + x[:, 0:self.num_patch_x]
        return x,xs,h

    def forward_decoder(self, feature, gt_score_map=None):
        # feature = feature[0]
        # feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)
        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_odonet(cfg):
    encoder = build_encoder(cfg)
    neck = build_neck(cfg,encoder)
    decoder = build_decoder(cfg, neck)

    '''导入 其他模型'''
    '''TODO:留心VQModel的类型，是否需要修改,测试demo和训练的不一样'''
    # VQ_head_config = {
    #     "codebook_size": 256, "codebook_embed_dim": 4, "commit_loss_beta": 0.25, "entropy_loss_ratio": 0.01,
    #     "tau": 0.07, "num_codebooks": 1,
    #     "codebook_l2_norm": False, "codebook_show_usage": True
    # }
    VQ_head_config = {
        "codebook_size": 256,  # 保持或动态调整
        "codebook_embed_dim": 4,  # 必须保持
        "commit_loss_beta": 0.25,  # 适中
        "entropy_loss_ratio": 0.05,  # 调高以提升codebook利用率
        "tau": 0.07,  # 保持或微调
        "num_codebooks": 1,  # 简单任务保持
        "codebook_l2_norm": False,  # 当前初始化已适配数据分布
        "codebook_show_usage": True  # 必须保持
    }
    # VQ_head_config = {
    #     "codebook_size": 256,  # 每维码本大小，仍适用
    #     "entropy_loss_ratio": 0.05,  # 保持，用于鼓励均匀使用
    #     "tau": 0.07,  # 保持，softmax 温度
    #     "codebook_l2_norm": False,  # 保持，当前分布适配无需归一
    #     "codebook_show_usage": True  # 保持，便于分析利用率
    # }

    from continuous_tokenizer.modelling.tokenizer import SoftVQModel, VectorQuantizer_Head,SoftVectorQuantizerOffset_head
    from segment_anything import sam_model_registry
    model_type_list = ["PromptEncoder", "MaskDecoder"]
    other_model_dict = nn.ModuleDict({
        # "vqvae":SoftVQModel.from_pretrained('/data2/lqh/workspace_pycharm/MCITrack/continuous_tokenizer/SoftVQVAE/softvq-b-49-128-lasot',),
        # "upper_quant": nn.Linear(768, 512),
        'down': nn.Sequential(
            nn.Linear(cfg.MODEL.NECK.D_MODEL, 4),
            nn.Tanh() #这个函数不需要改了
        ),
        "vqvae_head": nn.Identity(),
        # "vqvae_head": VectorQuantizer_Head(VQ_head_config),
        # "vqvae_head": SoftVectorQuantizerOffset_head(VQ_head_config),
        # "template_prompt_encoder":sam_model_registry[model_type_list[0]](prompt_embed_dim=cfg.MODEL.NECK.D_MODEL,
        #                                                          image_embedding_size=cfg.DATA.TEMPLATE.SIZE//cfg.MODEL.ENCODER.STRIDE,
        #                                                          image_size=cfg.DATA.TEMPLATE.SIZE),
        # "template_mask_decoder":sam_model_registry[model_type_list[1]](prompt_embed_dim=cfg.MODEL.NECK.D_MODEL),
        "search_prompt_encoder": sam_model_registry[model_type_list[0]](prompt_embed_dim=cfg.MODEL.NECK.D_MODEL,
                                                                        image_embedding_size=cfg.DATA.SEARCH.SIZE // cfg.MODEL.ENCODER.STRIDE ,
                                                                        image_size=cfg.DATA.SEARCH.SIZE ),
        "search_mask_decoder": sam_model_registry[model_type_list[1]](prompt_embed_dim=cfg.MODEL.NECK.D_MODEL),

    })

    model = ODONET(
        encoder,
        decoder,
        neck,
        cfg,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        other_model_dict=other_model_dict

    )
    return model
