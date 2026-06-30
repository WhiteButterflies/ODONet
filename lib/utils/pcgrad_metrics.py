import torch
from typing import Dict, Iterable, List, Tuple


def _shared_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _flat_grads(grads: Iterable[torch.Tensor], params: List[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for g, p in zip(grads, params):
        if g is None:
            chunks.append(torch.zeros_like(p).view(-1))
        else:
            chunks.append(g.contiguous().view(-1))
    return torch.cat(chunks)


@torch.no_grad()
def _l2(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(), min=1e-16))


def _flat_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in params])


def collect_task_grads(
    model: torch.nn.Module,
    losses: Dict[str, torch.Tensor],
    retain_graph: bool = True,
) -> Dict[str, torch.Tensor]:
    params = _shared_params(model)
    grads = {}
    for name, loss in losses.items():
        grad_tensors = torch.autograd.grad(
            loss,
            params,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        grads[name] = _flat_grads(grad_tensors, params).detach()
    model.zero_grad(set_to_none=True)
    return grads


def pairwise_cos_phi(grads, eps: float = 1e-12):
    names = list(grads.keys())
    cosij, phiij, conflict = {}, {}, {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            gi = grads[ni].reshape(-1)
            gj = grads[nj].reshape(-1)

            ni_norm = torch.linalg.norm(gi).clamp_min(eps)
            nj_norm = torch.linalg.norm(gj).clamp_min(eps)

            cos = torch.dot(gi, gj) / (ni_norm * nj_norm)
            cos_val = float(cos.item())
            cosij[(ni, nj)] = cos_val
            conflict[(ni, nj)] = cos_val < 0.0

            # PCGrad 定义 2：分母是平方和
            phi = (2.0 * ni_norm * nj_norm) / (ni_norm**2 + nj_norm**2 + eps)
            phiij[(ni, nj)] = float(phi.item())
    return cosij, phiij, conflict


# @torch.no_grad()
def curvature_proxy_virtual(
    model: torch.nn.Module,
    total_loss_closure,
    eta: float = 1e-3,
) -> Tuple[float, float]:
    params = _shared_params(model)

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        loss_before = total_loss_closure()
    if not isinstance(loss_before, torch.Tensor) or not loss_before.requires_grad:
        raise RuntimeError(
            "total_loss_closure must return a differentiable tensor; got one without gradients"
        )
    loss_before.backward()
    grads = _flat_grads((p.grad for p in params), params).detach()

    theta_before = _flat_params(params).clone()
    delta = -eta * grads

    offset = 0
    with torch.no_grad():
        for p in params:
            numel = p.numel()
            p.add_(delta[offset:offset + numel].view_as(p))
            offset += numel

    with torch.no_grad():
        loss_after_value = float(total_loss_closure().item())

    offset = 0
    with torch.no_grad():
        for p in params:
            numel = p.numel()
            p.copy_(theta_before[offset:offset + numel].view_as(p))
            offset += numel

    g_dot_delta = torch.dot(grads, delta).item()
    proxy = 2.0 * (loss_after_value - float(loss_before.item()) - g_dot_delta)
    normed = proxy / (float(delta.pow(2).sum().item()) + 1e-16)
    model.zero_grad(set_to_none=True)
    return proxy, normed


def analyze_triad_on_batch(
    model: torch.nn.Module,
    losses: Dict[str, torch.Tensor],
    total_loss_closure,
    eta: float = 1e-3,
):
    grads = collect_task_grads(model, losses, retain_graph=True)
    cosij, phiij, conflict = pairwise_cos_phi(grads)
    proxy, normed = curvature_proxy_virtual(model, total_loss_closure, eta=eta)
    return {
        "cos": cosij,
        "phi": phiij,
        "conflict": conflict,
        "curvature_proxy": proxy,
        "curvature_proxy_normed": normed,
    }