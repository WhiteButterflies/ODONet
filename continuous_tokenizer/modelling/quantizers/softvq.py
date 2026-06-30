import torch 
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.cluster import KMeans


class SoftVectorQuantizerOffset(nn.Module):
    def __init__(
        self,
        n_e=128,
        entropy_loss_ratio=0.01,
        tau=0.07,
        l2_norm=False,
        show_usage=False,

    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = 1
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.tau = tau
        self.num_codebooks = 4  # 固定：cx, cy, w, h

        self.embedding = nn.Parameter(torch.empty(self.num_codebooks, n_e, self.e_dim))
        for i in range(self.num_codebooks):
            if i in [0, 1]:  # cx, cy
                nn.init.uniform_(self.embedding[i], a=-0.1, b=0.1)
            else:  # w, h
                nn.init.uniform_(self.embedding[i], a=-0.06, b=0.06)

        if self.l2_norm:
            self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)

        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(self.num_codebooks, 65536))

    import numpy as np

    def initialize_with_kmeans(self, data: torch.Tensor):
        """
        使用 KMeans 初始化每个 embedding 的码本中心。
        data: Tensor of shape (N, 4)  # offset 数据：[cx, cy, w, h]
        """
        assert data.shape[-1] == 4, "Expected data to have shape (N, 4)"
        data_np = data.detach().cpu().numpy()

        for i in range(self.num_codebooks):
            dim_data = data_np[:, i:i + 1]  # shape (N, 1)
            kmeans = KMeans(n_clusters=self.n_e, random_state=0, n_init=10)
            kmeans.fit(dim_data)
            centers = kmeans.cluster_centers_  # shape (n_e, 1)
            self.embedding.data[i] = torch.from_numpy(centers).float().to(self.embedding.device)

    def forward(self, z):
        """
        z: shape (B, 4) - [cx, cy, w, h]
        """
        if z.dim() == 3 and z.shape[1] == 1 and z.shape[2] == 4:
            z = z.squeeze(1)  # 支持 (B, 1, 4) 格式

        assert z.shape[-1] == self.num_codebooks, \
            f"Expected input last dim=4, got {z.shape[-1]}"

        batch_size = z.shape[0]
        outs = []
        indices = []
        entropy_losses = []
        probs_list = []
        max_probs_list = []
        cos_similarities = []

        for i in range(self.num_codebooks):
            z_i = z[:, i:i+1]  # (B, 1)
            emb_i = self.embedding[i]  # (n_e, 1)

            if self.l2_norm:
                z_i = F.normalize(z_i, dim=-1)
                emb_i = F.normalize(emb_i, dim=-1)

            logits = torch.matmul(z_i, emb_i.T)  # (B, n_e)
            probs = F.softmax(logits / self.tau, dim=-1)  # (B, n_e)
            z_q_i = torch.matmul(probs, emb_i)  # (B, 1)

            # cosine similarity
            cos_sim = F.cosine_similarity(z_i, z_q_i, dim=-1).mean()
            cos_similarities.append(cos_sim)

            # entropy loss
            entropy_losses.append(compute_entropy_loss(logits))

            # usage
            hard_idx = torch.argmax(probs, dim=-1)
            indices.append(hard_idx)

            if self.show_usage and self.training:
                cur_len = hard_idx.size(0)
                self.codebook_used[i, :-cur_len].copy_(self.codebook_used[i, cur_len:].clone())
                self.codebook_used[i, -cur_len:].copy_(hard_idx)

            outs.append(z_q_i)
            probs_list.append(probs)
            max_probs_list.append(torch.max(probs, dim=-1)[0])

        # 合并输出
        z_q = torch.cat(outs, dim=-1)  # (B, 4)
        hard_indices = torch.stack(indices, dim=-1)  # (B, 4)

        avg_probs = torch.cat(probs_list, dim=1).mean()
        max_probs = torch.cat(max_probs_list, dim=0).mean()
        zq_z_cos = torch.stack(cos_similarities).mean()
        entropy_loss = sum(entropy_losses) * self.entropy_loss_ratio
        codebook_usage = torch.tensor([
            len(torch.unique(self.codebook_used[k])) / self.n_e
            for k in range(self.num_codebooks)
        ]).mean() if self.show_usage else 0

        # 和原版接口保持一致
        return z_q, \
               (0.0, 0.0, entropy_loss, codebook_usage), \
               (None, None, hard_indices, avg_probs, max_probs, z_q.detach(), z.detach(), zq_z_cos)



class SoftVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e,
        e_dim,
        entropy_loss_ratio=0.01,
        tau=0.07,
        num_codebooks=1,
        l2_norm=False,
        show_usage=False,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.n_e = n_e
        self.e_dim = e_dim
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.tau = tau
        
        # Single embedding layer for all codebooks
        '''通用embedding 方式，但受限于Tracking任务，改用'''
        # self.embedding = nn.Parameter(torch.randn(num_codebooks, n_e, e_dim))
        # # self.embedding.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e) #一般方式，但是受限于VQVAE在Tracking的值域，需要下面修改
        # self.embedding.data.uniform_(-0.1,0.1 ) #一般方式，但是受限于VQVAE在Tracking的值域，需要下面修改
        '''ended'''
        '''适合在tracking的embedding方式'''
        # More precise initialization matching observed ranges
        self.embedding = nn.Parameter(torch.empty(num_codebooks, n_e, e_dim))
        #方案1
        # nn.init.uniform_(self.embedding, a=-0.22, b=0.15)  # Exact min/max from sample

        # 方案2：分维度初始化（需确认e_dim顺序）
        for dim in range(e_dim):
            if dim in [0, 1]:  # cx, cy
                nn.init.uniform_(self.embedding[:, :, dim], a=-0.1, b=0.1)
            else:  # w, h
                nn.init.uniform_(self.embedding[:, :, dim], a=-0.06, b=0.06)
        ''''''
        
        if self.l2_norm:
            self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)
        
        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(num_codebooks, 65536))

    def forward(self, z):
        # Handle different input shapes
        if z.dim() == 4:
            z = torch.einsum('b c h w -> b h w c', z).contiguous()
            z = z.view(z.size(0), -1, z.size(-1))
        
        batch_size, seq_length, _ = z.shape
        
        # Ensure sequence length is divisible by number of codebooks
        assert seq_length % self.num_codebooks == 0, \
            f"Sequence length ({seq_length}) must be divisible by number of codebooks ({self.num_codebooks})"
        
        segment_length = seq_length // self.num_codebooks
        z_segments = z.view(batch_size, self.num_codebooks, segment_length, self.e_dim)
        
        # Apply L2 norm if needed
        embedding = F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding
        if self.l2_norm:
            z_segments = F.normalize(z_segments, p=2, dim=-1)
            
        z_flat = z_segments.permute(1, 0, 2, 3).contiguous().view(self.num_codebooks, -1, self.e_dim)
        
        logits = torch.einsum('nbe, nke -> nbk', z_flat, embedding.detach())#可以将z_flat和embedding视作Q与K，计算QK^T

        # Calculate probabilities
        probs = F.softmax(logits / self.tau, dim=-1) #相当于计算softmax(QK^T/tau),即attn的计算
        
        
        # Quantize
        z_q = torch.einsum('nbk, nke -> nbe', probs, embedding)#相当于计算softmax(QK^T/tau)与V的乘积
        
        # Reshape back
        z_q = z_q.view(self.num_codebooks, batch_size, segment_length, self.e_dim).permute(1, 0, 2, 3).contiguous()
        
        
        # Calculate cosine similarity
        with torch.no_grad():
            zq_z_cos = F.cosine_similarity(
                z_segments.view(-1, self.e_dim),
                z_q.view(-1, self.e_dim),
                dim=-1
            ).mean()
        
        # Get indices for usage tracking
        indices = torch.argmax(probs, dim=-1)  # (batch*segment_length, num_codebooks)
        
        # Track codebook usage
        if self.show_usage and self.training:
            for k in range(self.num_codebooks):
                cur_len = indices.size(0)
                self.codebook_used[k, :-cur_len].copy_(self.codebook_used[k, cur_len:].clone())
                self.codebook_used[k, -cur_len:].copy_(indices[:, k])
        
        # Calculate losses if training
        if self.training:
            vq_loss = commit_loss = 0.0            
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(logits.view(-1, self.n_e))
        else:
            vq_loss = commit_loss = entropy_loss = None
        
        # Calculate codebook usage
        codebook_usage = torch.tensor([
            len(torch.unique(self.codebook_used[k])) / self.n_e 
            for k in range(self.num_codebooks)
        ]).mean() if self.show_usage else 0

        z_q = z_q.view(batch_size, -1, self.e_dim)
        
        # Reshape back to match original input shape
        if len(z.shape) == 4:
            z_q = torch.einsum('b h w c -> b c h w', z_q)
        
        # Calculate average probabilities
        avg_probs = torch.mean(torch.mean(probs, dim=-1))
        max_probs = torch.mean(torch.max(probs, dim=-1)[0])
        
        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (
            None,  # perplexity
            None,  # min_encodings
            indices.view(batch_size, self.num_codebooks, segment_length),
            avg_probs,
            max_probs,
            z_q.detach(),
            z.detach(),
            zq_z_cos
        )



def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss
