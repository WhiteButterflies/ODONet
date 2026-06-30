import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
'''first used in chotrack-v2'''
class TrackingInfoNCELoss_V1(nn.Module):
    def __init__(self, temperature=0.07, output_size=(7, 7), feature_dim=512):
        """
        Args:
            temperature: InfoNCE 的温度系数
            output_size: ROI Align 输出的特征图大小 (H_roi, W_roi)
            feature_dim: 特征通道数
        """
        super().__init__()
        self.temperature = temperature
        self.output_size = output_size
        # 简单的投影头，将 ROI 特征映射到对比空间 (可选)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim * output_size[0] * output_size[1], feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 256)
        )

    def cxcywh_to_xyxy(self, boxes, width, height):
        """
        将归一化 cxcywh 转换为 绝对坐标 x1y1x2y2
        boxes: (N, 4) or (B, N, 4) normalized
        """
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - 0.5 * w) * width
        y1 = (cy - 0.5 * h) * height
        x2 = (cx + 0.5 * w) * width
        y2 = (cy + 0.5 * h) * height
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def extract_roi_features(self, features, boxes, batch_indices):
        """
        执行 ROI Align
        features: (B, C, H, W)
        boxes: (N_total, 4) 绝对坐标 xyxy
        batch_indices: (N_total,) 指示每个 box 属于哪个 batch
        """
        # 构建 ROI Align 需要的 boxes 格式: [batch_idx, x1, y1, x2, y2]
        rois = torch.cat([batch_indices.unsqueeze(1).float(), boxes], dim=1)
        
        # 提取特征 (N_total, C, output_size, output_size)
        roi_feats = roi_align(features, rois, output_size=self.output_size, spatial_scale=1.0)
        
        # 展平特征用于全连接层 (N_total, C*H*W)
        roi_feats = roi_feats.view(roi_feats.size(0), -1)
        
        # 投影到嵌入空间 (N_total, Embed_Dim)
        embeddings = self.projector(roi_feats)
        return F.normalize(embeddings, p=2, dim=1)

    def reshape_features(self, feature_tensor):
        """
        处理截图中的 ViT 格式: (B, 196, 512) -> (B, 512, 14, 14)
        """
        B, L, C = feature_tensor.shape
        H = W = int(L ** 0.5) # 假设是正方形, 196 -> 14x14
        # permute 为 (B, C, L) -> view 为 (B, C, H, W)
        return feature_tensor.permute(0, 2, 1).view(B, C, H, W)

    def forward(self, 
                pos_cxcywh_box_list, 
                neg_cxcywh_box_list, 
                pos_neg_feature_list, 
                use_batch_neg=True, 
                use_hard_neg=True):
        """
        Args:
            pos_cxcywh_box_list: List[Tensor(B, 4)], len=T
            neg_cxcywh_box_list: List[Tensor(B, K, 4)], len=T (K是负样本数)
            pos_neg_feature_list: List[Tensor(B, 196, 512)], len=T
            use_batch_neg: bool, 是否使用同 Batch 其他目标作为负样本
            use_hard_neg: bool, 是否使用 neg_cxcywh 作为负样本
        """
        
        # 1. 准备数据: 提取时刻 0 (Anchor) 和 时刻 1 (Positive Target)
        # 假设我们只计算 Frame 0 -> Frame 1 的对比损失
        feat_map_0 = self.reshape_features(pos_neg_feature_list[0]) # (B, 512, 14, 14)
        feat_map_1 = self.reshape_features(pos_neg_feature_list[1]) 
        
        H_feat, W_feat = feat_map_0.shape[-2:]
        B = feat_map_0.shape[0]

        # --- A. 提取 Anchor (时刻0的目标) ---
        box_0 = pos_cxcywh_box_list[0] # (B, 4)
        box_0_xyxy = self.cxcywh_to_xyxy(box_0, W_feat, H_feat)
        batch_idx_0 = torch.arange(B, device=box_0.device)
        anchor_embed = self.extract_roi_features(feat_map_0, box_0_xyxy, batch_idx_0) # (B, Dim)

        # --- B. 提取 Positive (时刻1的同一目标) ---
        box_1 = pos_cxcywh_box_list[1] # (B, 4)
        box_1_xyxy = self.cxcywh_to_xyxy(box_1, W_feat, H_feat)
        batch_idx_1 = torch.arange(B, device=box_1.device)
        pos_embed = self.extract_roi_features(feat_map_1, box_1_xyxy, batch_idx_1) # (B, Dim)

        # --- C. 计算 Logits (相似度矩阵) ---
        # 1. 正样本相似度 (B, 1): Anchor[i] vs Pos[i]
        # 使用 Einstein Summation 计算点积
        l_pos = torch.einsum('nc,nc->n', [anchor_embed, pos_embed]).unsqueeze(-1) # (B, 1)

        logits_list = [l_pos]

        # 2. Batch 负样本 (同 Batch 的其他目标)
        if use_batch_neg:
            # Anchor[i] vs Pos[j] where i != j
            # 计算所有 Anchor 和 所有 Pos 的相似度矩阵 (B, B)
            sim_matrix = torch.matmul(anchor_embed, pos_embed.T) 
            
            # 创建掩码剔除对角线 (正样本已经在 l_pos 里算过了)
            mask = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
            l_batch_neg = sim_matrix[mask].view(B, -1) # (B, B-1)
            logits_list.append(l_batch_neg)

        # 3. Hard Negatives (显式负样本框)
        if use_hard_neg and len(neg_cxcywh_box_list) > 1:
            # 这里的负样本通常来自时刻 1 的特征图 (或者时刻0，视任务定义而定，通常是在搜索区域内找负样本)
            # 假设我们在时刻 1 的特征图上提取负样本
            neg_boxes = neg_cxcywh_box_list[1] # (B, K, 4)
            K = neg_boxes.shape[1]
            
            # 展平处理 ROI Align
            neg_boxes_flat = neg_boxes.view(-1, 4) # (B*K, 4)
            neg_boxes_xyxy = self.cxcywh_to_xyxy(neg_boxes_flat, W_feat, H_feat)
            
            # 生成对应的 batch index: [0,0,0, 1,1,1, ...]
            neg_batch_idx = torch.arange(B, device=neg_boxes.device).repeat_interleave(K)
            
            neg_embed_flat = self.extract_roi_features(feat_map_1, neg_boxes_xyxy, neg_batch_idx) # (B*K, Dim)
            neg_embed = neg_embed_flat.view(B, K, -1) # (B, K, Dim)
            
            # 计算相似度 Anchor[i] vs Negs[i] -> (B, 1, Dim) * (B, Dim, K) -> (B, 1, K)
            l_hard_neg = torch.bmm(anchor_embed.unsqueeze(1), neg_embed.permute(0, 2, 1)).squeeze(1) # (B, K)
            logits_list.append(l_hard_neg)

        # --- D. 拼接与 Loss 计算 ---
        # Logits 形状: (B, 1 + n_neg)
        logits = torch.cat(logits_list, dim=1)
        
        # 应用温度系数
        logits /= self.temperature
        
        # InfoNCE 的 Label 永远是 0 (因为我们将正样本拼在了第0列)
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

class TrackingInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, output_size=(7, 7), feature_dim=512):
        super().__init__()
        self.temperature = temperature
        self.output_size = output_size
        self.projector = nn.Sequential(
            nn.Linear(feature_dim * output_size[0] * output_size[1], feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 256)
        )

    def cxcywh_to_xyxy(self, boxes, width, height):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - 0.5 * w) * width
        y1 = (cy - 0.5 * h) * height
        x2 = (cx + 0.5 * w) * width
        y2 = (cy + 0.5 * h) * height
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def extract_roi_features(self, features, boxes, batch_indices):
        """
        features: (B, C, H, W)
        boxes: (N, 4) absolute xyxy
        batch_indices: (N,)
        """
        rois = torch.cat([batch_indices.unsqueeze(1).float(), boxes], dim=1)
        roi_feats = roi_align(features, rois, output_size=self.output_size, spatial_scale=1.0)
        roi_feats = roi_feats.view(roi_feats.size(0), -1)
        embeddings = self.projector(roi_feats)
        return F.normalize(embeddings, p=2, dim=1)

    def reshape_features(self, feature_tensor):
        B, L, C = feature_tensor.shape
        H = W = int(L ** 0.5) 
        return feature_tensor.permute(0, 2, 1).view(B, C, H, W)

    def _compute_pair_loss(self, 
                           anchor_feat, anchor_box, 
                           target_feat, target_box, target_neg_boxes, 
                           use_batch_neg, use_hard_neg):
        """
        计算单对 (Frame_i, Frame_j) 之间的 Loss
        Anchor 来自 Frame_i, Positive/Negative 来自 Frame_j
        """
        H_feat, W_feat = anchor_feat.shape[-2:]
        B = anchor_feat.shape[0]

        # --- 1. Anchor (来自时刻 i) ---
        anchor_xyxy = self.cxcywh_to_xyxy(anchor_box, W_feat, H_feat)
        batch_idx = torch.arange(B, device=anchor_box.device)
        anchor_embed = self.extract_roi_features(anchor_feat, anchor_xyxy, batch_idx)

        # --- 2. Positive (来自时刻 j) ---
        pos_xyxy = self.cxcywh_to_xyxy(target_box, W_feat, H_feat)
        pos_embed = self.extract_roi_features(target_feat, pos_xyxy, batch_idx)

        # --- 3. 计算 Logits ---
        # 正样本相似度 (B, 1)
        l_pos = torch.einsum('nc,nc->n', [anchor_embed, pos_embed]).unsqueeze(-1)
        logits_list = [l_pos]

        # Batch 负样本 (同 Batch 其他图片的正样本)
        if use_batch_neg:
            sim_matrix = torch.matmul(anchor_embed, pos_embed.T)
            mask = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
            l_batch_neg = sim_matrix[mask].view(B, -1)
            logits_list.append(l_batch_neg)

        # Hard 负样本 (来自时刻 j 的负样本框)
        if use_hard_neg and target_neg_boxes is not None:
            K = target_neg_boxes.shape[1]
            neg_boxes_flat = target_neg_boxes.view(-1, 4)
            neg_xyxy = self.cxcywh_to_xyxy(neg_boxes_flat, W_feat, H_feat)
            neg_batch_idx = torch.arange(B, device=target_neg_boxes.device).repeat_interleave(K)
            
            neg_embed_flat = self.extract_roi_features(target_feat, neg_xyxy, neg_batch_idx)
            neg_embed = neg_embed_flat.view(B, K, -1)
            
            # (B, 1, Dim) @ (B, Dim, K) -> (B, 1, K) -> (B, K)
            l_hard_neg = torch.bmm(anchor_embed.unsqueeze(1), neg_embed.permute(0, 2, 1)).squeeze(1)
            logits_list.append(l_hard_neg)

        # --- 4. Loss ---
        logits = torch.cat(logits_list, dim=1)
        logits /= self.temperature
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)

    def forward(self, 
                pos_cxcywh_box_list, 
                neg_cxcywh_box_list, 
                pos_neg_feature_list, 
                use_batch_neg=True, 
                use_hard_neg=True,
                mode='all_pairs'):
        """
        Args:
            pos_cxcywh_box_list: List[Tensor(B, 4)], len=T
            neg_cxcywh_box_list: List[Tensor(B, K, 4)], len=T
            pos_neg_feature_list: List[Tensor(B, 196, 512)], len=T
            mode: 'all_pairs' (计算所有i<j组合) 或 'template_only' (只计算 0 vs Others)
        """
        
        T = len(pos_neg_feature_list)
        assert len(pos_cxcywh_box_list) == T
        
        # 1. 预处理所有特征图，避免重复 reshape
        feature_maps = [self.reshape_features(f) for f in pos_neg_feature_list]
        
        total_loss = 0.0
        num_pairs = 0

        # 2. 循环构建正负样本对
        # 我们可以计算所有 i < j 的组合，或者只计算 0 vs All
        start_idx_list = range(T) if mode == 'all_pairs' else [0]
        
        for i in start_idx_list:
            for j in range(i + 1, T):
                
                # 定义: i 是 Anchor (过去), j 是 Target (现在)
                # Hard Negative 通常取自 Target 帧 (j) 的背景
                neg_boxes_j = neg_cxcywh_box_list[j] if len(neg_cxcywh_box_list) > j else None

                loss = self._compute_pair_loss(
                    anchor_feat=feature_maps[i],
                    anchor_box=pos_cxcywh_box_list[i],
                    target_feat=feature_maps[j],
                    target_box=pos_cxcywh_box_list[j],
                    target_neg_boxes=neg_boxes_j,
                    use_batch_neg=use_batch_neg,
                    use_hard_neg=use_hard_neg
                )
                
                total_loss += loss
                num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=pos_neg_feature_list[0].device, requires_grad=True)

# ================= 模拟运行示例 =================
if __name__ == "__main__":
    B = 2
    T = 3 # 3帧序列
    K = 5 # 5个负样本
    
    # 模拟数据
    feat_list = [torch.randn(B, 196, 512) for _ in range(T)]
    pos_box_list = [torch.rand(B, 4) for _ in range(T)]
    neg_box_list = [torch.rand(B, K, 4) for _ in range(T)]
    
    criterion = TrackingInfoNCELoss()
    
    # 模式1: 全局两两对比 (0-1, 0-2, 1-2) -> 监督信号更强
    loss_all = criterion(pos_box_list, neg_box_list, feat_list, mode='all_pairs')
    print(f"All-Pairs Loss (3 pairs): {loss_all.item()}")
    
    # 模式2: 仅 Template 对比 (0-1, 0-2) -> 类似经典 Siamese 训练
    loss_template = criterion(pos_box_list, neg_box_list, feat_list, mode='template_only')
    print(f"Template-Only Loss (2 pairs): {loss_template.item()}")