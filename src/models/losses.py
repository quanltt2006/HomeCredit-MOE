import torch
import torch.nn.functional as F

LAMBDA_UNI = 0.3
LAMBDA_CONS = 0.1
CONSISTENCY_TEMPERATURE = 2.0

def get_lambda_gate(epoch, max_epochs, initial=0.1, final=0.01):
    if max_epochs <= 1: return initial
    progress = epoch / (max_epochs - 1)
    return initial * (1.0 - progress) + final * progress

def masked_balance_loss(gates, mask, eps=1e-8):
    if gates.shape[1] == 0: return gates.sum() * 0.0
    mask = mask.to(dtype=gates.dtype)
    valid_rows = mask.sum(dim=1) > 0
    if not valid_rows.any(): return gates.sum() * 0.0

    g_valid = gates[valid_rows] * mask[valid_rows]
    m_valid = mask[valid_rows]
    target = m_valid / m_valid.sum(dim=1, keepdim=True).clamp_min(eps)
    n_valid = m_valid.sum(dim=0).float().sum()
    return n_valid * (((g_valid - target) ** 2) * m_valid).sum(dim=1).mean()

def unimodal_auxiliary_loss(unimodal_logits, y, mask, pos_weight):
    losses = []
    for i, logits_m in enumerate(unimodal_logits):
        valid = mask[:, i] > 0
        if valid.any():
            losses.append(F.binary_cross_entropy_with_logits(logits_m[valid], y[valid].float(), pos_weight=pos_weight))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=y.device)

def consistency_loss(fused_logits, unimodal_logits, mask, temperature=CONSISTENCY_TEMPERATURE):
    fused_prob = torch.sigmoid(fused_logits.detach() / temperature)
    losses = []
    for i, logits_m in enumerate(unimodal_logits):
        valid = mask[:, i] > 0
        if valid.any():
            uni_prob = torch.sigmoid(logits_m[valid] / temperature)
            losses.append(F.binary_cross_entropy(uni_prob, fused_prob[valid]))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=fused_logits.device)

def compute_total_loss(logits, gate_weights, unimodal_logits, y, mask, pos_weight, lambda_gate):
    loss_task = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_weight)
    loss_gate = masked_balance_loss(gate_weights, mask)
    loss_uni = unimodal_auxiliary_loss(unimodal_logits, y, mask, pos_weight)
    loss_cons = consistency_loss(logits, unimodal_logits, mask)

    total_loss = loss_task + lambda_gate * loss_gate + LAMBDA_UNI * loss_uni + LAMBDA_CONS * loss_cons

    logs = {
        "loss_total": total_loss.item(),
        "loss_task": loss_task.item(),
        "loss_gate": loss_gate.item(),
        "loss_uni": loss_uni.item(),
        "loss_cons": loss_cons.item(),
    }
    return total_loss, logs