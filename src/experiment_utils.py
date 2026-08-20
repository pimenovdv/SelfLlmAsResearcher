import gc
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_seed(seed: int = 42):
    """
    Устанавливает seed для воспроизводимости экспериментов.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_name: str, output_attentions: bool = False):
    """
    Загружает модель и токенизатор по имени.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=output_attentions)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def clear_memory():
    """
    Очищает кэш GPU и вызывает сборщик мусора.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model_memory_footprint(model: torch.nn.Module) -> int:
    """
    Возвращает объем памяти, занимаемый параметрами модели, в байтах.
    """
    mem = 0
    for param in model.parameters():
        mem += param.nelement() * param.element_size()
    return mem

def count_parameters(model: torch.nn.Module) -> dict:
    """
    Возвращает количество параметров модели (всего и обучаемых).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total_params, "trainable": trainable_params}

def get_device() -> torch.device:
    """
    Возвращает доступное устройство (cuda, mps или cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def freeze_model_parameters(model: torch.nn.Module):
    """
    Замораживает все параметры модели.
    """
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model_parameters(model: torch.nn.Module):
    """
    Размораживает все параметры модели.
    """
    for param in model.parameters():
        param.requires_grad = True

def get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    """
    Возвращает модуль по его имени (например, 'transformer.h.0.mlp').
    """
    for name, module in model.named_modules():
        if name == module_name:
            return module
    raise ValueError(f"Module {module_name} not found in model.")

def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Возвращает устройство (device), на котором находятся параметры модели.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def check_model_device_consistency(model: torch.nn.Module) -> bool:
    """
    Проверяет, находятся ли все параметры модели на одном устройстве.
    """
    devices = {param.device for param in model.parameters()}
    return len(devices) <= 1

def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Вычисляет L2 норму градиентов всех параметров модели.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def set_requires_grad(model: torch.nn.Module, requires_grad: bool):
    """
    Устанавливает requires_grad для всех параметров модели.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad

def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """
    Возвращает тип данных (dtype) параметров модели.
    """
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32

def save_model_weights(model: torch.nn.Module, filepath: str):
    """
    Сохраняет веса модели в файл.
    """
    torch.save(model.state_dict(), filepath)

def load_model_weights(model: torch.nn.Module, filepath: str):
    """
    Загружает веса модели из файла.
    """
    model.load_state_dict(torch.load(filepath, map_location=get_model_device(model), weights_only=True))

def get_model_device_map(model: torch.nn.Module) -> dict:
    """
    Возвращает словарь, сопоставляющий имена параметров модели с их устройствами.
    """
    return {name: param.device for name, param in model.named_parameters()}

def has_nan_parameters(model: torch.nn.Module) -> bool:
    """
    Проверяет, содержат ли параметры модели NaN значения.
    """
    for param in model.parameters():
        if torch.isnan(param).any():
            return True
    return False

def has_inf_parameters(model: torch.nn.Module) -> bool:
    """
    Проверяет, содержат ли параметры модели Inf значения.
    """
    for param in model.parameters():
        if torch.isinf(param).any():
            return True
    return False

def replace_module(model: torch.nn.Module, module_name: str, new_module: torch.nn.Module):
    """
    Заменяет модуль в модели по его имени (например, 'transformer.h.0.mlp') на новый модуль.
    """
    parts = module_name.split('.')
    if len(parts) == 1:
        if not hasattr(model, parts[0]):
            raise ValueError(f"Module {module_name} not found in model.")
        setattr(model, parts[0], new_module)
        return

    parent_name = '.'.join(parts[:-1])
    target_name = parts[-1]

    parent_module = get_module_by_name(model, parent_name)

    if not hasattr(parent_module, target_name):
        raise ValueError(f"Module {module_name} not found in model.")
    setattr(parent_module, target_name, new_module)

def get_parameter_by_name(model: torch.nn.Module, parameter_name: str) -> torch.nn.Parameter:
    """
    Возвращает параметр модели по его имени.
    """
    for name, param in model.named_parameters():
        if name == parameter_name:
            return param
    raise ValueError(f"Parameter {parameter_name} not found in model.")

def get_model_sparsity(model: torch.nn.Module) -> float:
    """
    Вычисляет долю нулевых параметров в модели.
    """
    zero_params = 0
    total_params = 0
    for param in model.parameters():
        zero_params += torch.sum(param == 0).item()
        total_params += param.numel()

    if total_params == 0:
        return 0.0
    return zero_params / total_params

def find_modules_by_class(model: torch.nn.Module, module_class: type) -> list:
    """
    Возвращает список имен модулей в модели, которые являются экземплярами указанного класса.
    """
    return [name for name, module in model.named_modules() if isinstance(module, module_class)]

def check_model_weights_equality(model1: torch.nn.Module, model2: torch.nn.Module) -> bool:
    """
    Проверяет, равны ли веса двух моделей.
    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.shape != p2.shape or not torch.allclose(p1, p2):
            return False
    return True

def interpolate_model_weights(model1: torch.nn.Module, model2: torch.nn.Module, alpha: float) -> torch.nn.Module:
    """
    Создает копию model1, веса которой интерполированы между model1 и model2.
    w_new = (1 - alpha) * w1 + alpha * w2
    """
    import copy
    model_interp = copy.deepcopy(model1)
    for p_interp, p1, p2 in zip(model_interp.parameters(), model1.parameters(), model2.parameters()):
        p_interp.data = (1.0 - alpha) * p1.data + alpha * p2.data
    return model_interp


def add_noise_to_weights(model: torch.nn.Module, noise_std: float = 0.01) -> torch.nn.Module:
    """
    Создает копию модели и добавляет гауссовский шум с заданным стандартным отклонением к ее весам.
    """
    import copy
    model_noisy = copy.deepcopy(model)
    with torch.no_grad():
        for param in model_noisy.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)
    return model_noisy

def compute_cosine_similarity_between_models(model1: torch.nn.Module, model2: torch.nn.Module) -> float:
    """
    Вычисляет косинусное сходство между весами двух моделей.
    """
    params1 = [p.flatten() for p in model1.parameters()]
    params2 = [p.flatten() for p in model2.parameters()]

    if not params1 or not params2:
        return 0.0

    vec1 = torch.cat(params1)
    vec2 = torch.cat(params2)

    if vec1.numel() == 0 or vec2.numel() == 0:
        return 0.0

    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()

def compute_l2_distance_between_models(model1: torch.nn.Module, model2: torch.nn.Module) -> float:
    """
    Вычисляет L2 расстояние между весами двух моделей.
    """
    params1 = [p.flatten() for p in model1.parameters()]
    params2 = [p.flatten() for p in model2.parameters()]

    if not params1 or not params2:
        return 0.0

    vec1 = torch.cat(params1)
    vec2 = torch.cat(params2)

    if vec1.numel() == 0 or vec2.numel() == 0:
        return 0.0

    return torch.nn.functional.pairwise_distance(vec1.unsqueeze(0), vec2.unsqueeze(0), p=2).item()

def compute_l1_distance_between_models(model1: torch.nn.Module, model2: torch.nn.Module) -> float:
    """
    Вычисляет L1 расстояние (Манхэттенское расстояние) между весами двух моделей.
    """
    params1 = [p.flatten() for p in model1.parameters()]
    params2 = [p.flatten() for p in model2.parameters()]

    if not params1 or not params2:
        return 0.0

    vec1 = torch.cat(params1)
    vec2 = torch.cat(params2)

    if vec1.numel() == 0 or vec2.numel() == 0:
        return 0.0

    return float(torch.sum(torch.abs(vec1 - vec2)).item())

def compute_linf_distance_between_models(model1: torch.nn.Module, model2: torch.nn.Module) -> float:
    """
    Вычисляет L-infinity (Чебышёвское) расстояние между весами двух моделей.
    """
    params1 = [p.flatten() for p in model1.parameters()]
    params2 = [p.flatten() for p in model2.parameters()]

    if not params1 or not params2:
        return 0.0

    vec1 = torch.cat(params1)
    vec2 = torch.cat(params2)

    if vec1.numel() == 0 or vec2.numel() == 0:
        return 0.0

    return float(torch.max(torch.abs(vec1 - vec2)).item())

def compute_parameter_norm(model: torch.nn.Module, p: float = 2.0) -> float:
    """
    Вычисляет Lp норму всех параметров модели.
    """
    params = [param.flatten() for param in model.parameters()]
    if not params:
        return 0.0
    vec = torch.cat(params)
    return float(vec.norm(p).item())

def prune_model_weights(model: torch.nn.Module, amount: float) -> None:
    """
    Применяет L1 неструктурированный прунинг (l1_unstructured) ко всем Linear слоям модели.
    """
    import torch.nn.utils.prune as prune
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

def get_parameter_statistics(model: torch.nn.Module) -> dict:
    """
    Возвращает статистику параметров модели (mean, std, min, max).
    """
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    vec = torch.cat(params)
    return {
        "mean": float(vec.mean().item()),
        "std": float(vec.std().item()) if vec.numel() > 1 else 0.0,
        "min": float(vec.min().item()),
        "max": float(vec.max().item())
    }

def clip_model_weights(model: torch.nn.Module, min_val: float, max_val: float) -> None:
    """
    Clips all parameters of the model to be within the range [min_val, max_val].
    """
    import torch
    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(min_val, max_val)

def scale_model_weights(model: torch.nn.Module, scale_factor: float) -> None:
    """
    Умножает все параметры модели на scale_factor.
    """
    import torch
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(scale_factor)

def compute_snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """
    Вычисляет Signal-to-Noise Ratio (SNR).
    """
    import torch
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)
    epsilon = 1e-8
    return 10 * torch.log10(signal_power / (noise_power + epsilon)).item()

def compute_psnr(image_true: torch.Tensor, image_test: torch.Tensor, max_val: float) -> float:
    """
    Вычисляет Peak Signal-to-Noise Ratio (PSNR).
    """
    import torch
    mse = torch.mean((image_true - image_test) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((max_val ** 2) / mse).item()

def measure_inference_time(model: torch.nn.Module, input_data: torch.Tensor, num_runs: int = 10) -> float:
    """
    Измеряет среднее время инференса модели на заданных входных данных.
    """
    import time
    model.eval()
    with torch.no_grad():
        # Warmup
        _ = model(input_data)

        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_data)
        end_time = time.time()

    return (end_time - start_time) / num_runs

def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Возвращает размер модели в мегабайтах (MB).
    """
    return get_model_memory_footprint(model) / (1024 * 1024)

def get_trainable_parameters_percentage(model: torch.nn.Module) -> float:
    """
    Возвращает процент обучаемых параметров модели.
    """
    total_params = sum(p.numel() for p in model.parameters())
    if total_params == 0:
        return 0.0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (trainable_params / total_params) * 100.0

def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Создает и возвращает глубокую копию модели.
    """
    import copy
    return copy.deepcopy(model)

def shift_model_weights(model: torch.nn.Module, shift_value: float) -> None:
    """
    Добавляет заданное значение (shift_value) ко всем весам модели.
    """
    import torch
    with torch.no_grad():
        for param in model.parameters():
            param.add_(shift_value)

def randomize_model_weights(model: torch.nn.Module, mean: float = 0.0, std: float = 1.0) -> None:
    """
    Рандомизирует веса модели, используя нормальное распределение с заданными средним и стандартным отклонением.
    """
    import torch
    with torch.no_grad():
        for param in model.parameters():
            torch.nn.init.normal_(param, mean=mean, std=std)
