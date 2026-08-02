import torch

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
    import torch.nn.functional as F
    clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
    corrupted_log_probs = F.log_softmax(corrupted_logits[0, -1, :], dim=-1)
    return F.kl_div(corrupted_log_probs, clean_probs, reduction='sum').item()
