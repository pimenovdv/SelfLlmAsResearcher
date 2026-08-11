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

def top_k_accuracy(logits: torch.Tensor, target_id: int, k: int = 5) -> float:
    """
    Вычисляет Top-K Accuracy.
    """
    next_token_logits = logits[0, -1, :]
    top_k_indices = torch.topk(next_token_logits, k).indices
    return 1.0 if target_id in top_k_indices else 0.0

def mean_reciprocal_rank(logits: torch.Tensor, target_id: int) -> float:
    """
    Вычисляет Mean Reciprocal Rank (MRR).
    """
    next_token_logits = logits[0, -1, :]
    sorted_indices = torch.argsort(next_token_logits, descending=True)
    rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item() + 1
    return 1.0 / rank

def exact_match(logits: torch.Tensor, target_id: int) -> float:
    """
    Вычисляет Exact Match (EM) для заданного токена.
    """
    next_token_logits = logits[0, -1, :]
    predicted_id = torch.argmax(next_token_logits).item()
    return 1.0 if predicted_id == target_id else 0.0

def total_variation_distance(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет Total Variation Distance (TVD) между двумя распределениями.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    return 0.5 * torch.sum(torch.abs(probs_p - probs_q)).item()

def target_probability(logits: torch.Tensor, target_id: int) -> float:
    """
    Вычисляет вероятность целевого токена (Target Probability).
    """
    probs = F.softmax(logits[0, -1, :], dim=-1)
    return probs[target_id].item()

def cosine_similarity(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет косинусное сходство (Cosine Similarity) между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    return F.cosine_similarity(probs_p, probs_q, dim=0).item()

def chebyshev_distance(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет расстояние Чебышёва (Chebyshev Distance) между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    return torch.max(torch.abs(probs_p - probs_q)).item()

def euclidean_distance(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет евклидово расстояние (Euclidean Distance) между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    return torch.sqrt(torch.sum((probs_p - probs_q) ** 2)).item()

def manhattan_distance(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет манхэттенское расстояние (Manhattan Distance) между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    return torch.sum(torch.abs(probs_p - probs_q)).item()

def minkowski_distance(logits_p: torch.Tensor, logits_q: torch.Tensor, p: float = 3.0) -> float:
    """
    Вычисляет расстояние Минковского (Minkowski Distance) между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    return torch.norm(probs_p - probs_q, p=p).item()

def mean_squared_error(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет среднеквадратичную ошибку (MSE) между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    return F.mse_loss(probs_p, probs_q, reduction='mean').item()

def mean_absolute_error(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет среднюю абсолютную ошибку (MAE) между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    return F.l1_loss(probs_p, probs_q, reduction='mean').item()

def pearson_correlation(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет коэффициент корреляции Пирсона между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    mean_p = torch.mean(probs_p)
    mean_q = torch.mean(probs_q)

    p_centered = probs_p - mean_p
    q_centered = probs_q - mean_q

    cov = torch.sum(p_centered * q_centered)
    std_p = torch.sqrt(torch.sum(p_centered ** 2))
    std_q = torch.sqrt(torch.sum(q_centered ** 2))

    if std_p == 0 or std_q == 0:
        return 0.0
    return (cov / (std_p * std_q)).item()

def huber_loss(logits_p: torch.Tensor, logits_q: torch.Tensor, delta: float = 1.0) -> float:
    """
    Вычисляет Huber Loss между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    return F.huber_loss(probs_p, probs_q, reduction='mean', delta=delta).item()

def log_cosh_loss(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет Log-Cosh Loss между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    return torch.mean(torch.log(torch.cosh(probs_p - probs_q))).item()

def root_mean_squared_error(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет корень из среднеквадратичной ошибки (RMSE) между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)
    return torch.sqrt(F.mse_loss(probs_p, probs_q, reduction='mean')).item()

def r2_score(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет коэффициент детерминации R^2 между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    ss_res = torch.sum((probs_p - probs_q) ** 2)
    ss_tot = torch.sum((probs_p - torch.mean(probs_p)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return (1 - ss_res / ss_tot).item()

def mean_absolute_percentage_error(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет среднюю абсолютную процентную ошибку (MAPE) между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    epsilon = 1e-8
    return torch.mean(torch.abs((probs_p - probs_q) / torch.clamp(probs_p, min=epsilon))).item()

def symmetric_mean_absolute_percentage_error(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет симметричную среднюю абсолютную процентную ошибку (SMAPE) между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    epsilon = 1e-8
    numerator = torch.abs(probs_p - probs_q)
    denominator = torch.clamp(torch.abs(probs_p) + torch.abs(probs_q), min=epsilon)
    return torch.mean(2.0 * numerator / denominator).item()

def bhattacharyya_distance(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет расстояние Бхаттачарья между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    bc = torch.sum(torch.sqrt(probs_p * probs_q))
    epsilon = 1e-8
    return -torch.log(torch.clamp(bc, min=epsilon)).item()

def hellinger_distance(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет расстояние Хеллингера между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    return (1.0 / (2.0 ** 0.5)) * torch.sqrt(torch.sum((torch.sqrt(probs_p) - torch.sqrt(probs_q)) ** 2)).item()

def jaccard_similarity(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет коэффициент Жаккара (Jaccard Similarity) между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    intersection = torch.min(probs_p, probs_q).sum()
    union = torch.max(probs_p, probs_q).sum()

    return (intersection / union).item()

def renyi_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor, alpha: float = 2.0) -> float:
    """
    Вычисляет дивергенцию Реньи между двумя распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    epsilon = 1e-8

    if alpha == 1.0:
        return torch.sum(probs_p * torch.log((probs_p + epsilon) / (probs_q + epsilon))).item()

    term = (probs_p ** alpha) * (probs_q ** (1.0 - alpha))
    sum_term = torch.sum(term)

    return (1.0 / (alpha - 1.0)) * torch.log(torch.clamp(sum_term, min=epsilon)).item()

def wasserstein_distance(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет расстояние Вассерштейна-1 (Earth Mover's Distance) между двумя одномерными распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    cdf_p = torch.cumsum(probs_p, dim=-1)
    cdf_q = torch.cumsum(probs_q, dim=-1)

    return torch.sum(torch.abs(cdf_p - cdf_q)).item()

def chi_square_distance(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет Хи-квадрат расстояние (Chi-Square Distance) между двумя одномерными распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    epsilon = 1e-8
    return 0.5 * torch.sum(((probs_p - probs_q) ** 2) / (probs_p + probs_q + epsilon)).item()

def canberra_distance(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет расстояние Канберры (Canberra Distance) между двумя одномерными распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    epsilon = 1e-8
    numerator = torch.abs(probs_p - probs_q)
    denominator = torch.abs(probs_p) + torch.abs(probs_q) + epsilon
    return torch.sum(numerator / denominator).item()

def bray_curtis_distance(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Вычисляет расстояние Брея-Кертиса (Bray-Curtis Distance) между двумя одномерными распределениями вероятностей.
    """
    probs_p = F.softmax(logits_p[0, -1, :], dim=-1)
    probs_q = F.softmax(logits_q[0, -1, :], dim=-1)

    epsilon = 1e-8
    numerator = torch.sum(torch.abs(probs_p - probs_q))
    denominator = torch.sum(torch.abs(probs_p) + torch.abs(probs_q)) + epsilon
    return (numerator / denominator).item()
