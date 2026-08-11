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


def test_cross_entropy():
    from src.metrics import cross_entropy
    import torch
    import math
    logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
    ce = cross_entropy(logits, 0)
    assert abs(ce - math.log(4)) < 1e-4


def test_perplexity():
    from src.metrics import perplexity
    import torch
    logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
    perp = perplexity(logits)
    assert abs(perp - 4.0) < 1e-4


def test_brier_score():
    from src.metrics import brier_score
    import torch
    logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
    bs = brier_score(logits, 0)
    assert abs(bs - 0.75) < 1e-4

def test_top_k_accuracy():
    from src.metrics import top_k_accuracy
    import torch
    logits = torch.tensor([[[0.1, 0.5, 0.2, 0.9, 0.4]]])
    assert top_k_accuracy(logits, 3, k=2) == 1.0
    assert top_k_accuracy(logits, 1, k=1) == 0.0

def test_mean_reciprocal_rank():
    from src.metrics import mean_reciprocal_rank
    import torch
    logits = torch.tensor([[[0.1, 0.5, 0.2, 0.9, 0.4]]])
    assert mean_reciprocal_rank(logits, 3) == 1.0
    assert mean_reciprocal_rank(logits, 1) == 0.5

def test_exact_match():
    from src.metrics import exact_match
    import torch
    logits = torch.tensor([[[0.1, 0.5, 0.2, 0.9, 0.4]]])
    assert exact_match(logits, 3) == 1.0
    assert exact_match(logits, 1) == 0.0

def test_total_variation_distance():
    from src.metrics import total_variation_distance
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    tvd = total_variation_distance(logits_p, logits_q)
    assert tvd > 0.0

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    tvd_zero = total_variation_distance(same_logits, same_logits)
    assert abs(tvd_zero) < 1e-4

def test_target_probability():
    from src.metrics import target_probability
    import torch

    logits = torch.tensor([[[0.0, 10.0, 0.0]]])
    prob = target_probability(logits, 1)
    assert prob > 0.99
    prob2 = target_probability(logits, 0)
    assert prob2 < 0.01

def test_cosine_similarity():
    from src.metrics import cosine_similarity
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    sim = cosine_similarity(logits_p, logits_q)
    assert sim < 0.01

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    sim_same = cosine_similarity(same_logits, same_logits)
    assert abs(sim_same - 1.0) < 1e-4

def test_chebyshev_distance():
    from src.metrics import chebyshev_distance
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    dist = chebyshev_distance(logits_p, logits_q)
    assert dist > 0.99

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    dist_same = chebyshev_distance(same_logits, same_logits)
    assert abs(dist_same) < 1e-4

def test_euclidean_distance():
    from src.metrics import euclidean_distance
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    dist = euclidean_distance(logits_p, logits_q)
    assert dist > 1.4

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    dist_same = euclidean_distance(same_logits, same_logits)
    assert abs(dist_same) < 1e-4

def test_manhattan_distance():
    from src.metrics import manhattan_distance
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    dist = manhattan_distance(logits_p, logits_q)
    assert dist > 1.9

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    dist_same = manhattan_distance(same_logits, same_logits)
    assert abs(dist_same) < 1e-4

def test_minkowski_distance():
    from src.metrics import minkowski_distance
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    dist = minkowski_distance(logits_p, logits_q, 3.0)
    assert dist > 1.25

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    dist_same = minkowski_distance(same_logits, same_logits, 3.0)
    assert abs(dist_same) < 1e-4

def test_mean_squared_error():
    from src.metrics import mean_squared_error
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    mse = mean_squared_error(logits_p, logits_q)
    assert mse > 0.1

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    mse_same = mean_squared_error(same_logits, same_logits)
    assert abs(mse_same) < 1e-4

def test_mean_absolute_error():
    from src.metrics import mean_absolute_error
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    mae = mean_absolute_error(logits_p, logits_q)
    assert mae > 0.1

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    mae_same = mean_absolute_error(same_logits, same_logits)
    assert abs(mae_same) < 1e-4

def test_pearson_correlation():
    from src.metrics import pearson_correlation
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    corr = pearson_correlation(logits_p, logits_q)
    assert corr < 0.0

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    corr_same = pearson_correlation(same_logits, same_logits)
    assert abs(corr_same - 1.0) < 1e-4

def test_huber_loss():
    from src.metrics import huber_loss
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    loss = huber_loss(logits_p, logits_q)
    assert loss > 0.0

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    loss_same = huber_loss(same_logits, same_logits)
    assert abs(loss_same) < 1e-4

def test_log_cosh_loss():
    from src.metrics import log_cosh_loss
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    loss = log_cosh_loss(logits_p, logits_q)
    assert loss > 0.0

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    loss_same = log_cosh_loss(same_logits, same_logits)
    assert abs(loss_same) < 1e-4

def test_root_mean_squared_error():
    from src.metrics import root_mean_squared_error
    import torch

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])

    loss = root_mean_squared_error(logits_p, logits_q)
    assert loss > 0.0

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    loss_same = root_mean_squared_error(same_logits, same_logits)
    assert abs(loss_same) < 1e-4

def test_r2_score():
    from src.metrics import r2_score
    import torch

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    score_same = r2_score(same_logits, same_logits)
    assert abs(score_same - 1.0) < 1e-4

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])
    score_diff = r2_score(logits_p, logits_q)
    assert score_diff < 1.0

def test_mean_absolute_percentage_error():
    from src.metrics import mean_absolute_percentage_error
    import torch

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    error_same = mean_absolute_percentage_error(same_logits, same_logits)
    assert abs(error_same) < 1e-4

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])
    error_diff = mean_absolute_percentage_error(logits_p, logits_q)
    assert error_diff > 0.0

def test_symmetric_mean_absolute_percentage_error():
    from src.metrics import symmetric_mean_absolute_percentage_error
    import torch

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    error_same = symmetric_mean_absolute_percentage_error(same_logits, same_logits)
    assert abs(error_same) < 1e-4

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])
    error_diff = symmetric_mean_absolute_percentage_error(logits_p, logits_q)
    assert error_diff > 0.0

def test_bhattacharyya_distance():
    from src.metrics import bhattacharyya_distance
    import torch

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    dist_same = bhattacharyya_distance(same_logits, same_logits)
    assert abs(dist_same) < 1e-4

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])
    dist_diff = bhattacharyya_distance(logits_p, logits_q)
    assert dist_diff > 0.0

def test_hellinger_distance():
    from src.metrics import hellinger_distance
    import torch

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    dist_same = hellinger_distance(same_logits, same_logits)
    assert abs(dist_same) < 1e-4

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])
    dist_diff = hellinger_distance(logits_p, logits_q)
    assert dist_diff > 0.0

def test_jaccard_similarity():
    from src.metrics import jaccard_similarity
    import torch

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    sim_same = jaccard_similarity(same_logits, same_logits)
    assert abs(sim_same - 1.0) < 1e-4

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])
    sim_diff = jaccard_similarity(logits_p, logits_q)
    assert sim_diff < 1.0

def test_renyi_divergence():
    from src.metrics import renyi_divergence
    import torch

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    dist_same = renyi_divergence(same_logits, same_logits)
    assert abs(dist_same) < 1e-4

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])
    dist_diff = renyi_divergence(logits_p, logits_q)
    assert dist_diff > 0.0

def test_wasserstein_distance():
    from src.metrics import wasserstein_distance
    import torch

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    dist_same = wasserstein_distance(same_logits, same_logits)
    assert abs(dist_same) < 1e-4

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])
    dist_diff = wasserstein_distance(logits_p, logits_q)
    assert dist_diff > 0.0

def test_chi_square_distance():
    from src.metrics import chi_square_distance
    import torch

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    dist_same = chi_square_distance(same_logits, same_logits)
    assert abs(dist_same) < 1e-4

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])
    dist_diff = chi_square_distance(logits_p, logits_q)
    assert dist_diff > 0.0

def test_canberra_distance():
    from src.metrics import canberra_distance
    import torch

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    dist_same = canberra_distance(same_logits, same_logits)
    assert abs(dist_same) < 1e-4

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])
    dist_diff = canberra_distance(logits_p, logits_q)
    assert dist_diff > 0.0

def test_bray_curtis_distance():
    from src.metrics import bray_curtis_distance
    import torch

    same_logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    dist_same = bray_curtis_distance(same_logits, same_logits)
    assert abs(dist_same) < 1e-4

    logits_p = torch.tensor([[[10.0, 0.0, 0.0]]])
    logits_q = torch.tensor([[[0.0, 10.0, 0.0]]])
    dist_diff = bray_curtis_distance(logits_p, logits_q)
    assert dist_diff > 0.0
