import torch
import torch.nn.functional as F


def multi_view_contrastive_loss(list_of_X, temperature=0.07):
    N = len(list_of_X)
    B, D = list_of_X[0].shape

    # Flatten all views
    Z = torch.cat(list_of_X, dim=0)  # (N*B, D)
    Z = F.normalize(Z, dim=-1)

    # Similarity matrix (NB, NB)
    sim_matrix = torch.matmul(Z, Z.T) / temperature
    sim_matrix_exp = torch.exp(sim_matrix)

    total_loss = 0.0

    for b in range(B):
        # Positive indices for this sample
        pos_indices = [i * B + b for i in range(N)]
        anchor_indices = pos_indices

        for anchor_idx in anchor_indices:
            pos_mask = torch.zeros(N * B, device=Z.device)
            pos_mask[pos_indices] = 1
            pos_mask[anchor_idx] = 0  # exclude self

            pos_sim = sim_matrix_exp[anchor_idx][pos_mask.bool()].sum()
            neg_sim = sim_matrix_exp[anchor_idx].sum() - sim_matrix_exp[anchor_idx][anchor_idx] - pos_sim

            loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
            total_loss += loss

    return total_loss / (B * N)


def multi_view_contrastive_loss_matrix(list_of_X, temperature=0.07):
    """
    list_of_X: list of (B, D) tensors, N views
    返回：
    scalar loss
    """
    N = len(list_of_X)
    B, D = list_of_X[0].shape

    # Flatten and normalize: (NB, D)
    Z = torch.cat(list_of_X, dim=0)
    Z = F.normalize(Z, dim=-1)

    # Similarity matrix: (NB, NB)
    sim_matrix = torch.matmul(Z, Z.T) / temperature
    sim_matrix_exp = torch.exp(sim_matrix)

    # Build positive mask: same sample across views
    pos_mask = torch.zeros((N * B, N * B), device=Z.device)
    for b in range(B):
        idx = [i * B + b for i in range(N)]
        for i in idx:
            pos_mask[i, idx] = 1
        pos_mask[torch.arange(N * B), torch.arange(N * B)] = 0  # remove self

    # Sum over positives and negatives
    pos_sum = (sim_matrix_exp * pos_mask).sum(dim=1)
    all_sum = sim_matrix_exp.sum(dim=1) - torch.exp(sim_matrix.diag())  # exclude self

    # Compute InfoNCE loss
    loss = -torch.log(pos_sum / (all_sum + 1e-8))
    return loss.mean()


def multi_view_contrastive_loss_matrix_tokenwise(list_of_X, temperature=0.07, pool_type='mean'):
    """
    list_of_X: list of (B, T, C) tensors, N views
    T = H * W
    """
    N = len(list_of_X)
    B, T, C = list_of_X[0].shape

    # Pooling: (B, T, C) -> (B, C)
    if pool_type == 'mean':
        list_of_X = [x.mean(dim=1) for x in list_of_X]  # 均值池化
    elif pool_type == 'max':
        list_of_X = [x.max(dim=1)[0] for x in list_of_X]
    else:
        raise ValueError("Unsupported pooling method.")

    # 直接用前面你已有的函数逻辑
    return multi_view_contrastive_loss_matrix(list_of_X, temperature)
