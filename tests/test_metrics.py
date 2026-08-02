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
    import torch.nn.functional as F

    clean_logits = torch.tensor([[[10.0, 0.0, 0.0]]])
    corrupted_logits = torch.tensor([[[0.0, 10.0, 0.0]]])
    kl = kl_divergence(clean_logits, corrupted_logits)
    assert kl > 0.0

    clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
    corrupted_log_probs = F.log_softmax(corrupted_logits[0, -1, :], dim=-1)
    expected_kl = F.kl_div(corrupted_log_probs, clean_probs, reduction='sum').item()

    assert abs(kl - expected_kl) < 1e-4

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    kl_zero = kl_divergence(same_logits, same_logits)
    assert abs(kl_zero) < 1e-4


def test_entropy():
    from src.metrics import entropy
    import torch
    import math
    logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
    ent = entropy(logits)
    assert abs(ent - math.log(4)) < 1e-4


def test_js_divergence():
    from src.metrics import js_divergence
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    js = js_divergence(logits_p, logits_q)
    assert js > 0.0

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    js_zero = js_divergence(same_logits, same_logits)
    assert abs(js_zero) < 1e-4
