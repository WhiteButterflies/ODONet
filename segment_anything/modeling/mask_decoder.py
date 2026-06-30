# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        # self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        # self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(transformer_dim // 4),
        #     activation(),
        #     nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        #     activation(),
        # )
        #next is added by liu
        # self.output_samescaling = nn.Sequential(
        #     nn.Conv2d(transformer_dim, transformer_dim // 4,kernel_size=3, stride=1,padding=1),
        #     LayerNorm2d(transformer_dim // 4),
        #     activation(),
        #     nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, stride=1,padding=1),
        #     activation(),
        # )

        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for _ in range(self.num_mask_tokens)
        #     ]
        # )

        # self.iou_prediction_head = MLP(
        #     transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        # )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[ torch.Tensor,torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        hs,score_size_offset_head= self.predict_masks(
            image_embeddings=image_embeddings,   #原图转化为的特征 featuremap
            image_pe=image_pe,  #使用高斯噪声以某种细节生成同shape的位置编码
            sparse_prompt_embeddings=sparse_prompt_embeddings,  #point prompt embedding +  bbox prompt embedding，[B,N_point+2*N_box,512]
            dense_prompt_embeddings=dense_prompt_embeddings, #mask embedding
        )

        # Prepare output
        return hs,score_size_offset_head
    '''this method is modifed by liu,not same from origin'''
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        '''ori'''
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        '''modifed'''
        tokens = sparse_prompt_embeddings

        # Expand per-image data in batch direction to be per-mask
        #src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        '''check training'''
        # if self.training is True:
        #     src =image_embeddings

        src = image_embeddings
        src = src + dense_prompt_embeddings  #

        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        # Run the transformer #prompt,image
        hs, src = self.transformer(src, pos_src, tokens)# HS :tokens of query ;; src:key
        return hs,src


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
