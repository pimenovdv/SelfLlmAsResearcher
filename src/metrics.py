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
