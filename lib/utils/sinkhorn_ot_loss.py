import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornLoss(nn.Module):
    """
    Sinkhorn OT loss for heatmaps of shape (B, 1, H, W).

    Args:
        epsilon: entropic regularization coefficient
        n_iters: number of Sinkhorn iterations
        reduction: 'mean' | 'sum' | 'none'
        p: ground metric power, usually 1 or 2
        normalize_cost: whether to normalize cost matrix into [0, 1]
    """
    def __init__(
        self,
        epsilon: float = 0.1,
        n_iters: int = 50,
        reduction: str = "mean",
        p: int = 2,
        normalize_cost: bool = True,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.reduction = reduction
        self.p = p
        self.normalize_cost = normalize_cost

        # cache for cost matrix
        self._cached_hw = None
        self._cached_cost = None

    def _build_cost_matrix(self, H: int, W: int, device, dtype):
        """
        Build cost matrix C of shape (H*W, H*W).
        C[i, j] = distance between grid location i and j.
        """
        ys = torch.arange(H, device=device, dtype=dtype)
        xs = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W), (H, W)

        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # (N, 2)
        diff = coords[:, None, :] - coords[None, :, :]                 # (N, N, 2)
        C = torch.norm(diff, p=2, dim=-1)                              # (N, N)

        if self.p == 2:
            C = C ** 2

        if self.normalize_cost:
            C = C / (C.max().clamp_min(1e-12))

        return C

    def _get_cost_matrix(self, H: int, W: int, device, dtype):
        if (
            self._cached_cost is None
            or self._cached_hw != (H, W)
            or self._cached_cost.device != device
            or self._cached_cost.dtype != dtype
        ):
            self._cached_cost = self._build_cost_matrix(H, W, device, dtype)
            self._cached_hw = (H, W)
        return self._cached_cost

    # def _normalize_heatmap(self, x: torch.Tensor):
    #     """
    #     x: (B, 1, H, W)
    #     Convert to probability distributions of shape (B, H*W).
    #     """
    #     B, C, H, W = x.shape
    #     assert C == 1, f"Expected channel=1, got {C}"
    #
    #     # ensure nonnegative
    #     x = F.relu(x)
    #
    #     # flatten
    #     x = x.view(B, -1)
    #
    #     # normalize to sum to 1
    #     x = x / x.sum(dim=1, keepdim=True).clamp_min(1e-12)
    #     return x

    def _normalize_heatmap(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert C == 1

        x = x.view(B, -1)
        x = x.clamp_min(1e-12)
        x = x / x.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred, target: (B, 1, H, W)
        Returns Sinkhorn transport cost.
        """
        assert pred.shape == target.shape, "pred and target must have same shape"
        B, C, H, W = pred.shape
        assert C == 1, "Heatmap must have shape (B,1,H,W)"

        a = self._normalize_heatmap(pred)    # (B, N)
        b = self._normalize_heatmap(target)  # (B, N)
        N = H * W

        Cmat = self._get_cost_matrix(H, W, pred.device, pred.dtype)    # (N, N)

        # kernel matrix K = exp(-C / epsilon)
        K = torch.exp(-Cmat / self.epsilon).clamp_min(1e-12)           # (N, N)

        # Sinkhorn iterations
        u = torch.ones_like(a) / N  # (B, N)
        v = torch.ones_like(b) / N  # (B, N)

        for _ in range(self.n_iters):
            Kv = torch.matmul(v, K.t())                                # (B, N)
            u = a / Kv.clamp_min(1e-12)

            KTu = torch.matmul(u, K)                                   # (B, N)
            v = b / KTu.clamp_min(1e-12)

        # transport plan pi = diag(u) K diag(v)
        # batch form: pi_b = u_b[:,None] * K * v_b[None,:]
        pi = u.unsqueeze(2) * K.unsqueeze(0) * v.unsqueeze(1)          # (B, N, N)

        # OT cost
        cost = (pi * Cmat.unsqueeze(0)).sum(dim=(1, 2))                # (B,)

        if self.reduction == "mean":
            return cost.mean()
        elif self.reduction == "sum":
            return cost.sum()
        elif self.reduction == "none":
            return cost
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")