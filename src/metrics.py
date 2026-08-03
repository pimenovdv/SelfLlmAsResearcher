import torch
import torch.nn.functional as F

def logit_difference(logits: torch.Tensor, target_id: int, corrupted_id: int) -> float:
    """
    Вычисляет разницу логитов (Logit Difference) между целевым и ошибочным токеном.

    Args:
        logits (torch.Tensor): Тензор логитов формы [batch, seq_len, vocab_size]
        target_id (int): Идентификатор целевого токена (clean).
        corrupted_id (int): Идентификатор ошибочного токена (corrupted).

    Returns:
        float: Разница логитов (target_logit - corrupted_logit).
    """
    # Берем логиты для последнего токена в последовательности для первого элемента батча
    next_token_logits = logits[0, -1, :]

    target_logit = next_token_logits[target_id].item()
    corrupted_logit = next_token_logits[corrupted_id].item()

    return target_logit - corrupted_logit

def kl_divergence(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor) -> float:
    """
    Вычисляет KL Divergence.
    """
    clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
    corrupted_log_probs = F.log_softmax(corrupted_logits[0, -1, :], dim=-1)
    return F.kl_div(corrupted_log_probs, clean_probs, reduction='sum').item()


def entropy(logits: torch.Tensor) -> float:
    """
    Вычисляет энтропию распределения вероятностей.
    """
    probs = F.softmax(logits[0, -1, :], dim=-1)
    log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
    return -(probs * log_probs).sum().item()


def js_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет Jensen-Shannon Divergence.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    m = 0.5 * (probs_p + probs_q)
    log_m = torch.log(m + 1e-8)

    kl_p = F.kl_div(log_m, probs_p, reduction='sum').item()
    kl_q = F.kl_div(log_m, probs_q, reduction='sum').item()

    return 0.5 * (kl_p + kl_q)


def cross_entropy(logits: torch.Tensor, target_id: int) -> float:
    """
    Вычисляет кросс-энтропию (Cross Entropy) для заданного токена.
    """
    log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
    return -log_probs[target_id].item()


def perplexity(logits: torch.Tensor) -> float:
    """
    Вычисляет перплексию (Perplexity) распределения вероятностей.
    """
    probs = F.softmax(logits[0, -1, :], dim=-1)
    log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
    entropy_val = -(probs * log_probs).sum().item()
    return torch.exp(torch.tensor(entropy_val)).item()


def brier_score(logits: torch.Tensor, target_id: int) -> float:
    """
    Вычисляет Brier Score для заданного токена.
    """
    probs = F.softmax(logits[0, -1, :], dim=-1)
    target_probs = torch.zeros_like(probs)
    target_probs[target_id] = 1.0
    return F.mse_loss(probs, target_probs, reduction='sum').item()
