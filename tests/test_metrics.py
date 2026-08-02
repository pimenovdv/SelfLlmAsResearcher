import torch
from src.metrics import logit_difference

def test_logit_difference():
    logits = torch.tensor([[[0.1, 0.2, 0.3], [1.0, 0.5, 0.2]]])
    diff = logit_difference(logits, target_id=0, corrupted_id=1)
    assert diff == 0.5
    diff2 = logit_difference(logits, target_id=1, corrupted_id=2)
    assert round(diff2, 5) == 0.3

def test_kl_divergence():
    from src.metrics import kl_divergence
    import torch
    clean_logits = torch.tensor([[[0.0, 1.0, 0.0]]])
    corrupted_logits = torch.tensor([[[0.0, 0.0, 1.0]]])
    kl = kl_divergence(clean_logits, corrupted_logits)
    assert kl > 0.0
