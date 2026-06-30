import torch
import torch.nn.functional as F

def compute_causal_input_nce_loss(module_obj, causal_input_tokens):
    """Apply NCE on causal_former input tokens so gradients reach backbone features."""
    if causal_input_tokens is None:
        return torch.tensor(0.0, device=next(module_obj.net.parameters()).device)

    series = causal_input_tokens
    if series.dim() == 3:
        series = series.unsqueeze(1)
    if series.dim() != 4 or series.shape[2] < 1:
        return series.sum() * 0.0

    temperature = max(float(getattr(module_obj.cfg.TRAIN, "NCE_TEMPERATURE", 0.07)), 1e-6)
    if series.shape[1] > 1:
        anchor_pos = series[:, :-1, 0, :]
        target_pos = series[:, 1:, 0, :]
        target_neg = series[:, 1:, 1:, :]
    else:
        anchor_pos = series[:, :, 0, :]
        target_pos = series[:, :, 0, :].detach()
        target_neg = series[:, :, 1:, :]

    anchor_pos = F.normalize(anchor_pos, p=2, dim=-1)
    target_pos = F.normalize(target_pos, p=2, dim=-1)

    pos_logits = (anchor_pos * target_pos).sum(dim=-1, keepdim=True)
    logits_list = [pos_logits]

    if target_neg.shape[2] > 0:
        hard_neg = F.normalize(target_neg, p=2, dim=-1)
        hard_neg_logits = torch.einsum("btc,btkc->btk", anchor_pos, hard_neg)
        logits_list.append(hard_neg_logits)

    batch_size = anchor_pos.shape[0]
    if batch_size > 1:
        batch_logits = torch.einsum("btc,dtc->btd", anchor_pos, target_pos)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=batch_logits.device)
        batch_logits = batch_logits.permute(1, 0, 2)[:, mask].view(
            anchor_pos.shape[1], batch_size, batch_size - 1
        ).permute(1, 0, 2)
        logits_list.append(batch_logits)

    logits = torch.cat(logits_list, dim=-1) / temperature
    labels = torch.zeros(logits.shape[0] * logits.shape[1], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels)


def compute_causal_regularization_loss(module_obj, reference_tensor):
    causal_reg_weight = float(module_obj.loss_weight.get('causal_reg', 0.0))
    if causal_reg_weight <= 0:
        return reference_tensor.new_tensor(0.0)

    net = module_obj.net.module if hasattr(module_obj.net, "module") else module_obj.net
    causal_former = getattr(net.other_model_dict, "causal_former", None)
    if causal_former is None or not hasattr(causal_former, "regularization"):
        return reference_tensor.new_tensor(0.0)

    return causal_former.regularization().to(device=reference_tensor.device, dtype=reference_tensor.dtype)