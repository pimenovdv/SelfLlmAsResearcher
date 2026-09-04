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

def has_nan_gradients(model: torch.nn.Module) -> bool:
    """
    Проверяет, содержат ли градиенты параметров модели NaN значения.
    """
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False

def has_inf_gradients(model: torch.nn.Module) -> bool:
    """
    Проверяет, содержат ли градиенты параметров модели Inf значения.
    """
    for param in model.parameters():
        if param.grad is not None and torch.isinf(param.grad).any():
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

def compute_parameter_mad(model: torch.nn.Module) -> float:
    """
    Вычисляет среднее абсолютное отклонение (MAD) параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    if vec.numel() <= 1:
        return 0.0
    mean_val = vec.mean()
    mad = torch.abs(vec - mean_val).mean()
    return float(mad.item())

def compute_gradient_mad(model: torch.nn.Module) -> float:
    """
    Вычисляет среднее абсолютное отклонение (MAD) градиентов модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    if vec.numel() <= 1:
        return 0.0
    mean_val = vec.mean()
    mad = torch.abs(vec - mean_val).mean()
    return float(mad.item())

def compute_activation_mad(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет среднее абсолютное отклонение (MAD) активаций для заданных слоев при проходе input_data.
    """
    import torch
    stats = {}
    handles = []

    def hook(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            vec = out.detach().flatten()
            if vec.numel() <= 1:
                stats[name] = 0.0
                return
            mean_val = vec.mean()
            mad = torch.abs(vec - mean_val).mean()
            stats[name] = float(mad.item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    with torch.no_grad():
        model(input_data)

    for handle in handles:
        handle.remove()

    return stats

def compute_parameter_iqr(model: torch.nn.Module) -> float:
    """
    Вычисляет межквартильный размах (IQR) параметров модели (75-й процентиль минус 25-й процентиль).
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    if vec.numel() == 0:
        return 0.0
    q = torch.tensor([0.25, 0.75], dtype=vec.dtype, device=vec.device)
    try:
        quantiles = torch.quantile(vec, q).tolist()
        return float(quantiles[1] - quantiles[0])
    except RuntimeError:
        return 0.0

def compute_gradient_iqr(model: torch.nn.Module) -> float:
    """
    Вычисляет межквартильный размах (IQR) градиентов модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    if vec.numel() == 0:
        return 0.0
    q = torch.tensor([0.25, 0.75], dtype=vec.dtype, device=vec.device)
    try:
        quantiles = torch.quantile(vec, q).tolist()
        return float(quantiles[1] - quantiles[0])
    except RuntimeError:
        return 0.0

def compute_activation_iqr(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет межквартильный размах (IQR) активаций для заданных слоев при проходе input_data.
    """
    import torch
    stats = {}
    handles = []

    def hook(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            vec = out.detach().flatten()
            if vec.numel() == 0:
                stats[name] = 0.0
                return
            q = torch.tensor([0.25, 0.75], dtype=vec.dtype, device=vec.device)
            try:
                quantiles = torch.quantile(vec, q).tolist()
                stats[name] = float(quantiles[1] - quantiles[0])
            except RuntimeError:
                stats[name] = 0.0
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    try:
        with torch.no_grad():
            model(input_data)
    finally:
        for handle in handles:
            handle.remove()

    return stats

def compute_parameter_mean(model: torch.nn.Module) -> float:
    """
    Вычисляет среднее значение всех параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    return float(torch.cat(params).mean().item())

def compute_gradient_mean(model: torch.nn.Module) -> float:
    """
    Вычисляет среднее значение всех градиентов модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    return float(torch.cat(grads).mean().item())

def compute_activation_mean(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет среднее значение активаций для заданных слоев модели.
    """
    import torch
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    handles = []
    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook_fn(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)
    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    mean_dict = {}
    for name, act in activations.items():
        if act.numel() > 0:
            mean_dict[name] = float(act.float().mean().item())
        else:
            mean_dict[name] = 0.0

    return mean_dict


def compute_parameter_min(model: torch.nn.Module) -> float:
    """
    Вычисляет минимальное значение всех параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    return float(torch.cat(params).min().item())

def compute_parameter_max(model: torch.nn.Module) -> float:
    """
    Вычисляет максимальное значение всех параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    return float(torch.cat(params).max().item())

def compute_parameter_sum(model: torch.nn.Module) -> float:
    """
    Вычисляет сумму всех параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    return float(torch.cat(params).sum().item())

def compute_parameter_rms(model: torch.nn.Module) -> float:
    """
    Вычисляет среднеквадратичное значение (RMS) всех параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    return float(torch.sqrt((vec ** 2).mean()).item())

def compute_parameter_std(model: torch.nn.Module) -> float:
    """
    Вычисляет стандартное отклонение всех параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    return float(vec.std().item()) if vec.numel() > 1 else 0.0


def compute_parameter_range(model: torch.nn.Module) -> float:
    """
    Вычисляет размах (range = max - min) всех параметров модели.
    """
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    return float((vec.max() - vec.min()).item())


def compute_gradient_range(model: torch.nn.Module) -> float:
    """
    Вычисляет размах (range = max - min) градиентов модели.
    """
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    return float((vec.max() - vec.min()).item())


def compute_gradient_min(model: torch.nn.Module) -> float:
    """
    Вычисляет минимальное значение градиентов параметров модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    return float(torch.cat(grads).min().item())

def compute_gradient_max(model: torch.nn.Module) -> float:
    """
    Вычисляет максимальное значение градиентов параметров модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    return float(torch.cat(grads).max().item())

def compute_gradient_sum(model: torch.nn.Module) -> float:
    """
    Вычисляет сумму градиентов параметров модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    return float(torch.cat(grads).sum().item())

def compute_gradient_rms(model: torch.nn.Module) -> float:
    """
    Вычисляет среднеквадратичное значение (RMS) градиентов параметров модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    return float(torch.sqrt((vec ** 2).mean()).item())

def compute_gradient_std(model: torch.nn.Module) -> float:
    """
    Вычисляет стандартное отклонение всех градиентов модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    return float(vec.std().item()) if vec.numel() > 1 else 0.0


def compute_activation_min(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет минимальное значение активаций для каждого указанного слоя.
    """
    import torch
    stats = {}
    handles = []

    def hook(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                out_tensor = output[0].detach()
            elif isinstance(output, torch.Tensor):
                out_tensor = output.detach()
            else:
                return
            vec = out_tensor.flatten()
            if vec.numel() == 0:
                stats[name] = 0.0
            else:
                stats[name] = float(vec.min().item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return stats

def compute_activation_sum(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет сумму активаций для каждого указанного слоя.
    """
    import torch
    stats = {}
    handles = []

    def hook(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                out_tensor = output[0].detach()
            elif isinstance(output, torch.Tensor):
                out_tensor = output.detach()
            else:
                return
            vec = out_tensor.flatten()
            if vec.numel() == 0:
                stats[name] = 0.0
            else:
                stats[name] = float(vec.sum().item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return stats

def compute_activation_max(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет максимальное значение активаций для каждого указанного слоя.
    """
    import torch
    stats = {}
    handles = []

    def hook(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                out_tensor = output[0].detach()
            elif isinstance(output, torch.Tensor):
                out_tensor = output.detach()
            else:
                return
            vec = out_tensor.flatten()
            if vec.numel() == 0:
                stats[name] = 0.0
            else:
                stats[name] = float(vec.max().item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return stats

def compute_activation_rms(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет среднеквадратичное значение (RMS) активаций для каждого указанного слоя.
    """
    import torch
    stats = {}
    handles = []

    def hook(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            vec = out.detach().flatten()
            stats[name] = float(torch.sqrt((vec ** 2).mean()).item()) if vec.numel() > 0 else 0.0
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    try:
        with torch.no_grad():
            model(input_data)
    finally:
        for handle in handles:
            handle.remove()

    return stats

def compute_activation_std(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет стандартное отклонение активаций для заданных слоев модели.
    """
    import torch
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            elif isinstance(output, torch.Tensor):
                activations[name] = output.detach()
        return hook

    handles = []
    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook_fn(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)
    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    std_dict = {}
    for name, act in activations.items():
        if act.numel() > 1:
            std_dict[name] = float(act.float().std().item())
        else:
            std_dict[name] = 0.0

    return std_dict

def compute_activation_range(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет размах (range = max - min) активаций для заданных слоев при проходе input_data.
    """
    import torch
    range_dict = {}
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0].detach()
            elif isinstance(output, torch.Tensor):
                out = output.detach()
            else:
                return
            range_dict[name] = float((out.max() - out.min()).item())
        return hook

    handles = []
    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(get_hook(name)))

    with torch.no_grad():
        model(input_data)

    for handle in handles:
        handle.remove()

    return range_dict

def compute_activation_norms(model: torch.nn.Module, input_data: torch.Tensor, module_names: list, p: float = 2.0) -> dict:
    """
    Вычисляет Lp норму активаций для списка модулей при прохождении input_data через модель.
    Возвращает словарь {module_name: norm}.
    """
    import torch
    norms = {}

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                norms[name] = torch.norm(output.detach(), p=p).item()
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                norms[name] = torch.norm(output[0].detach(), p=p).item()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name in module_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    with torch.no_grad():
        model(input_data)

    for hook in hooks:
        hook.remove()

    return norms

def compute_activation_sparsity(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str], threshold: float = 1e-7) -> dict[str, float]:
    """
    Вычисляет разреженность активаций для заданных слоев (доля элементов, абсолютное значение которых меньше threshold).
    """
    import torch
    sparsity_dict = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                num_zeros = torch.sum(torch.abs(out) < threshold).item()
                num_elements = out.numel()
                sparsity_dict[name] = float(num_zeros / num_elements) if num_elements > 0 else 0.0
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return sparsity_dict

def compute_activation_statistics(model: torch.nn.Module, input_data: torch.Tensor, module_names: list) -> dict:
    """
    Computes activation statistics (mean, std, min, max) for given layers.
    """
    import torch
    stats = {}
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0].detach()
            elif isinstance(output, torch.Tensor):
                out = output.detach()
            else:
                return
            stats[name] = {
                "mean": float(out.mean().item()),
                "std": float(out.std().item()) if out.numel() > 1 else 0.0,
                "min": float(out.min().item()),
                "max": float(out.max().item()),
            }
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name in module_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    try:
        with torch.no_grad():
            model(input_data)
    finally:
        for hook in hooks:
            hook.remove()

    return stats

def compute_module_parameter_norms(model: torch.nn.Module, p: float = 2.0) -> dict:
    """
    Вычисляет Lp норму параметров каждого модуля (слоя) в модели.
    Возвращает словарь {module_name: norm}.
    """
    import torch
    norms = {}
    for name, module in model.named_modules():
        if name == "": continue
        total_norm = 0.0
        has_params = False
        for param in module.parameters(recurse=False):
            has_params = True
            param_norm = torch.norm(param.detach(), p=p)
            total_norm += param_norm.item() ** p
        if has_params:
            norms[name] = total_norm ** (1.0 / p)
    return norms

def compute_module_gradient_norms(model: torch.nn.Module, p: float = 2.0) -> dict:
    """
    Вычисляет Lp норму градиентов каждого модуля (слоя) в модели.
    Возвращает словарь {module_name: norm}.
    """
    import torch
    norms = {}
    for name, module in model.named_modules():
        if name == "": continue
        total_norm = 0.0
        has_grad = False
        for param in module.parameters(recurse=False):
            if param.grad is not None:
                has_grad = True
                param_norm = torch.norm(param.grad.detach(), p=p)
                total_norm += param_norm.item() ** p
        if has_grad:
            norms[name] = total_norm ** (1.0 / p)
    return norms

def get_module_activations(model: torch.nn.Module, module_name: str, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Возвращает активации (выход) указанного модуля при прохождении input_data через модель.
    """
    import torch
    activation = {}

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activation['out'] = output.detach()
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            activation['out'] = output[0].detach()

    module = None
    for name, mod in model.named_modules():
        if name == module_name:
            module = mod
            break

    if module is None:
        raise ValueError(f"Module {module_name} not found in model.")

    hook = module.register_forward_hook(hook_fn)

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data, **kwargs)

    if training_state:
        model.train()
    hook.remove()
    return activation.get('out', torch.tensor([]))

def get_module_gradients(model: torch.nn.Module, module_name: str, input_data: torch.Tensor, target: torch.Tensor, loss_fn, **kwargs) -> torch.Tensor:
    """
    Возвращает градиенты по выходу указанного модуля при прохождении input_data и вычислении loss.
    """
    import torch
    gradients = {}

    def hook_fn(module, grad_input, grad_output):
        if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
            gradients['out'] = grad_output[0].detach()

    module = None
    for name, mod in model.named_modules():
        if name == module_name:
            module = mod
            break

    if module is None:
        raise ValueError(f"Module {module_name} not found in model.")

    hook = module.register_full_backward_hook(hook_fn)

    training_state = model.training
    model.train()
    model.zero_grad()

    output = model(input_data, **kwargs)
    loss = loss_fn(output, target)
    loss.backward()

    if not training_state:
        model.eval()
    hook.remove()
    return gradients.get('out', torch.tensor([]))

def compute_parameter_entropy(model: torch.nn.Module, bins: int = 256) -> float:
    """
    Вычисляет энтропию параметров модели, оценивая распределение через гистограмму.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    if vec.numel() <= 1:
        return 0.0

    hist = torch.histc(vec, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = -torch.sum(p * torch.log2(p))
    return float(entropy.item())

def compute_gradient_entropy(model: torch.nn.Module, bins: int = 256) -> float:
    """
    Вычисляет энтропию градиентов модели, оценивая распределение через гистограмму.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    if vec.numel() <= 1:
        return 0.0

    hist = torch.histc(vec, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = -torch.sum(p * torch.log2(p))
    return float(entropy.item())

def compute_parameter_coefficient_of_variation(model: torch.nn.Module) -> float:
    """
    Вычисляет коэффициент вариации (coefficient of variation) параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    if vec.numel() <= 1:
        return 0.0
    mean = vec.mean().item()
    if mean == 0:
        return 0.0
    std = vec.std().item()
    return float(std / mean)

def compute_gradient_coefficient_of_variation(model: torch.nn.Module) -> float:
    """
    Вычисляет коэффициент вариации (coefficient of variation) градиентов модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    if vec.numel() <= 1:
        return 0.0
    mean = vec.mean().item()
    if mean == 0:
        return 0.0
    std = vec.std().item()
    return float(std / mean)

def compute_gradient_median(model: torch.nn.Module) -> float:
    """
    Вычисляет медиану градиентов модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    if vec.numel() == 0:
        return 0.0
    return float(torch.median(vec).item())

def compute_gradient_quantiles(model: torch.nn.Module, q: list[float] = None) -> list[float]:
    """
    Вычисляет квантили градиентов модели.
    """
    import torch
    if q is None:
        q = [0.25, 0.5, 0.75]
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return [0.0] * len(q)
    vec = torch.cat(grads)
    if vec.numel() == 0:
        return [0.0] * len(q)
    q_tensor = torch.tensor(q, dtype=vec.dtype, device=vec.device)
    return torch.quantile(vec, q_tensor).tolist()

def compute_gradient_variance(model: torch.nn.Module) -> float:
    """
    Вычисляет дисперсию градиентов модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    if vec.numel() <= 1:
        return 0.0
    return float(vec.var().item())

def compute_gradient_kurtosis(model: torch.nn.Module) -> float:
    """
    Вычисляет эксцесс (kurtosis) градиентов модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    if vec.numel() <= 1:
        return 0.0

    mean = vec.mean()
    std = vec.std()

    if std == 0:
        return 0.0

    kurtosis = torch.mean(((vec - mean) / std) ** 4) - 3.0
    return float(kurtosis.item())

def compute_gradient_skewness(model: torch.nn.Module) -> float:
    """
    Вычисляет асимметрию (skewness) градиентов модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    if vec.numel() <= 1:
        return 0.0

    mean = vec.mean()
    std = vec.std()

    if std == 0:
        return 0.0

    skewness = torch.mean(((vec - mean) / std) ** 3)
    return float(skewness.item())

def compute_parameter_quantiles(model: torch.nn.Module, q: list[float] = None) -> list[float]:
    """
    Вычисляет квантили параметров модели.
    """
    import torch
    if q is None:
        q = [0.25, 0.5, 0.75]
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return [0.0] * len(q)
    vec = torch.cat(params)
    if vec.numel() == 0:
        return [0.0] * len(q)
    q_tensor = torch.tensor(q, dtype=vec.dtype, device=vec.device)
    return torch.quantile(vec, q_tensor).tolist()

def compute_parameter_median(model: torch.nn.Module) -> float:
    """
    Вычисляет медиану всех параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    if vec.numel() == 0:
        return 0.0
    return float(torch.median(vec).item())

def compute_parameter_variance(model: torch.nn.Module) -> float:
    """
    Вычисляет дисперсию всех параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    if vec.numel() <= 1:
        return 0.0
    return float(vec.var().item())

def compute_parameter_kurtosis(model: torch.nn.Module) -> float:
    """
    Вычисляет эксцесс (kurtosis) всех параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    if vec.numel() <= 1:
        return 0.0

    mean = vec.mean()
    std = vec.std()

    if std == 0:
        return 0.0

    kurtosis = torch.mean(((vec - mean) / std) ** 4) - 3.0
    return float(kurtosis.item())

def compute_parameter_skewness(model: torch.nn.Module) -> float:
    """
    Вычисляет асимметрию (skewness) всех параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    if vec.numel() <= 1:
        return 0.0

    mean = vec.mean()
    std = vec.std()

    if std == 0:
        return 0.0

    skewness = torch.mean(((vec - mean) / std) ** 3)
    return float(skewness.item())

def freeze_model_weights(model: torch.nn.Module) -> None:
    """
    Замораживает все веса модели (устанавливает requires_grad = False).
    """
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model_weights(model: torch.nn.Module) -> None:
    """
    Размораживает все веса модели (устанавливает requires_grad = True).
    """
    for param in model.parameters():
        param.requires_grad = True

def check_nan_weights(model: torch.nn.Module) -> bool:
    """
    Проверяет, есть ли NaN значения в весах модели.
    """
    import torch
    for param in model.parameters():
        if torch.isnan(param).any():
            return True
    return False

def check_inf_weights(model: torch.nn.Module) -> bool:
    """
    Проверяет, есть ли Inf (бесконечность) значения в весах модели.
    """
    import torch
    for param in model.parameters():
        if torch.isinf(param).any():
            return True
    return False

def clip_gradients(model: torch.nn.Module, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Обрезает градиенты модели (gradient clipping) по норме.

    Args:
        model: Модель PyTorch.
        max_norm: Максимальная норма градиентов.
        norm_type: Тип используемой нормы (по умолчанию L2).

    Returns:
        Общая норма градиентов до обрезки.
    """
    import torch
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=norm_type))

def zero_gradients(model: torch.nn.Module, set_to_none: bool = False) -> None:
    """
    Обнуляет градиенты всех параметров модели.

    Args:
        model: Модель PyTorch.
        set_to_none: Если True, устанавливает градиенты в None вместо нулей.
    """
    model.zero_grad(set_to_none=set_to_none)

def add_noise_to_gradients(model: torch.nn.Module, noise_std: float = 0.01) -> None:
    """
    Добавляет гауссовский шум с нулевым средним и заданным стандартным отклонением
    к градиентам параметров модели (если они существуют).

    Args:
        model: Модель PyTorch.
        noise_std: Стандартное отклонение шума.
    """
    import torch
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_std
            param.grad.data.add_(noise)

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

def average_model_weights(models: list[torch.nn.Module]) -> torch.nn.Module:
    """
    Усредняет веса списка моделей с одинаковой архитектурой.
    """
    import torch
    from copy import deepcopy
    if not models:
        raise ValueError("The list of models is empty.")

    avg_model = deepcopy(models[0])
    avg_state_dict = avg_model.state_dict()

    for key in avg_state_dict.keys():
        tensors = [model.state_dict()[key] for model in models]
        avg_state_dict[key] = torch.stack(tensors).mean(dim=0)

    avg_model.load_state_dict(avg_state_dict)
    return avg_model

def reset_model_weights(model: torch.nn.Module) -> None:
    """
    Сбрасывает веса модели к значениям по умолчанию, используя методы инициализации каждого модуля.
    """
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()

def copy_model_weights(source_model: torch.nn.Module, target_model: torch.nn.Module) -> None:
    """
    Копирует веса из source_model в target_model.
    """
    target_model.load_state_dict(source_model.state_dict())

def remove_all_hooks(model: torch.nn.Module) -> None:
    """
    Удаляет все хуки (forward, forward_pre, backward, state_dict и т.д.) из всех модулей модели.
    """
    hook_attrs = [
        '_forward_hooks', '_forward_pre_hooks', '_backward_hooks',
        '_full_backward_hooks', '_full_backward_pre_hooks',
        '_state_dict_hooks', '_state_dict_pre_hooks',
        '_load_state_dict_pre_hooks', '_load_state_dict_post_hooks',
        '_backward_pre_hooks'
    ]
    for module in model.modules():
        for attr in hook_attrs:
            if hasattr(module, attr):
                hooks = getattr(module, attr)
                if hooks is not None and hasattr(hooks, 'clear'):
                    hooks.clear()

def set_dropout_prob(model: torch.nn.Module, p: float) -> None:
    """
    Устанавливает вероятность отсева (dropout probability) для всех слоев Dropout в модели.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = p

def compute_parameter_sparsity(model: torch.nn.Module, threshold: float = 1e-7) -> float:
    """
    Вычисляет разреженность параметров модели (доля элементов параметров, абсолютное значение которых меньше threshold).
    """
    import torch
    num_zeros = 0
    num_elements = 0
    for param in model.parameters():
        if param.numel() > 0:
            num_zeros += torch.sum(torch.abs(param.data) < threshold).item()
            num_elements += param.numel()
    if num_elements == 0:
        return 0.0
    return float(num_zeros / num_elements)

def compute_gradient_sparsity(model: torch.nn.Module, threshold: float = 1e-7) -> float:
    """
    Вычисляет разреженность градиентов модели (доля элементов градиентов, абсолютное значение которых меньше threshold).
    """
    import torch
    num_zeros = 0
    num_elements = 0
    for param in model.parameters():
        if param.grad is not None:
            num_zeros += torch.sum(torch.abs(param.grad) < threshold).item()
            num_elements += param.grad.numel()
    if num_elements == 0:
        return 0.0
    return float(num_zeros / num_elements)

def compute_activation_variance(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет дисперсию активаций для заданных слоев.
    """
    import torch
    variance_dict = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out_tensor = out[0].detach()
            elif isinstance(out, torch.Tensor):
                out_tensor = out.detach()
            else:
                return

            vec = out_tensor.flatten()
            if vec.numel() <= 1:
                variance_dict[name] = 0.0
                return

            variance_dict[name] = float(vec.var().item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return variance_dict

def compute_activation_entropy(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str], bins: int = 256) -> dict[str, float]:
    """
    Вычисляет энтропию активаций для заданных слоев, оценивая распределение через гистограмму.
    """
    import torch
    entropy_dict = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out_tensor = out[0].detach()
            elif isinstance(out, torch.Tensor):
                out_tensor = out.detach()
            else:
                return

            vec = out_tensor.flatten()
            if vec.numel() <= 1:
                entropy_dict[name] = 0.0
                return

            hist = torch.histc(vec, bins=bins)
            p = hist / hist.sum()
            p = p[p > 0]
            entropy = -torch.sum(p * torch.log2(p))
            entropy_dict[name] = float(entropy.item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return entropy_dict

def compute_activation_skewness(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет асимметрию (skewness) активаций для заданных слоев.
    """
    import torch
    skewness_dict = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out_tensor = out[0].detach()
            elif isinstance(out, torch.Tensor):
                out_tensor = out.detach()
            else:
                return

            vec = out_tensor.flatten()
            if vec.numel() <= 1:
                skewness_dict[name] = 0.0
                return

            mean = vec.mean()
            std = vec.std()

            if std == 0:
                skewness_dict[name] = 0.0
            else:
                skewness = torch.mean(((vec - mean) / std) ** 3)
                skewness_dict[name] = float(skewness.item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return skewness_dict

def compute_activation_kurtosis(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет эксцесс (kurtosis) активаций для заданных слоев.
    """
    import torch
    kurtosis_dict = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out_tensor = out[0].detach()
            elif isinstance(out, torch.Tensor):
                out_tensor = out.detach()
            else:
                return

            vec = out_tensor.flatten()
            if vec.numel() <= 1:
                kurtosis_dict[name] = 0.0
                return

            mean = vec.mean()
            std = vec.std()

            if std == 0:
                kurtosis_dict[name] = 0.0
            else:
                kurtosis = torch.mean(((vec - mean) / std) ** 4) - 3.0
                kurtosis_dict[name] = float(kurtosis.item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return kurtosis_dict

def compute_activation_median(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет медиану активаций для заданных слоев.
    """
    import torch
    median_dict = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out_tensor = out[0].detach()
            elif isinstance(out, torch.Tensor):
                out_tensor = out.detach()
            else:
                return

            vec = out_tensor.flatten()
            if vec.numel() == 0:
                median_dict[name] = 0.0
                return

            median_val = vec.median()
            median_dict[name] = float(median_val.item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return median_dict

def compute_activation_quantiles(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str], q: list[float] = None) -> dict[str, list[float]]:
    """
    Вычисляет квантили активаций для заданных слоев при проходе input_data.
    """
    import torch
    if q is None:
        q = [0.25, 0.5, 0.75]

    quantiles_dict = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out_tensor = out[0].detach()
            elif isinstance(out, torch.Tensor):
                out_tensor = out.detach()
            else:
                return

            vec = out_tensor.flatten()
            if vec.numel() == 0:
                quantiles_dict[name] = [0.0] * len(q)
                return

            q_tensor = torch.tensor(q, dtype=vec.dtype, device=vec.device)
            try:
                quantiles_val = torch.quantile(vec, q_tensor).tolist()
            except RuntimeError:
                # Fallback if quantile fails
                quantiles_val = [0.0] * len(q)

            quantiles_dict[name] = quantiles_val
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return quantiles_dict

def compute_activation_coefficient_of_variation(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет коэффициент вариации (Coefficient of Variation, std / mean) активаций для заданных слоев при проходе input_data.
    """
    import torch
    cv_dict = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out_tensor = out[0].detach()
            elif isinstance(out, torch.Tensor):
                out_tensor = out.detach()
            else:
                return

            vec = out_tensor.flatten()
            if vec.numel() <= 1:
                cv_dict[name] = 0.0
                return

            mean = vec.mean()
            std = vec.std()

            if mean == 0:
                cv_dict[name] = 0.0
            else:
                cv_dict[name] = float((std / mean).item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return cv_dict

def compute_parameter_mode(model: torch.nn.Module) -> float:
    """
    Вычисляет моду (наиболее частое значение) параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    if vec.numel() == 0:
        return 0.0
    vals, counts = torch.unique(vec, return_counts=True)
    mode_idx = torch.argmax(counts)
    return float(vals[mode_idx].item())


def compute_gradient_mode(model: torch.nn.Module) -> float:
    """
    Вычисляет моду (наиболее частое значение) градиентов параметров модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    if vec.numel() == 0:
        return 0.0
    vals, counts = torch.unique(vec, return_counts=True)
    mode_idx = torch.argmax(counts)
    return float(vals[mode_idx].item())


def compute_activation_mode(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет моду (наиболее частое значение) активаций для заданных слоев модели.
    """
    import torch
    mode_dict = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out_tensor = out[0].detach()
            elif isinstance(out, torch.Tensor):
                out_tensor = out.detach()
            else:
                return

            vec = out_tensor.flatten()
            if vec.numel() == 0:
                mode_dict[name] = 0.0
                return

            vals, counts = torch.unique(vec, return_counts=True)
            mode_idx = torch.argmax(counts)
            mode_dict[name] = float(vals[mode_idx].item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return mode_dict


def compute_parameter_energy(model: torch.nn.Module) -> float:
    """
    Вычисляет энергию (сумму квадратов значений) параметров модели.
    """
    import torch
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    return float(torch.sum(vec ** 2).item())


def compute_gradient_energy(model: torch.nn.Module) -> float:
    """
    Вычисляет энергию (сумму квадратов значений) градиентов параметров модели.
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    return float(torch.sum(vec ** 2).item())


def compute_activation_energy(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет энергию (сумму квадратов значений) активаций для заданных слоев модели.
    """
    import torch
    energy_dict = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out_tensor = out[0].detach()
            elif isinstance(out, torch.Tensor):
                out_tensor = out.detach()
            else:
                return

            vec = out_tensor.flatten()
            if vec.numel() == 0:
                energy_dict[name] = 0.0
                return

            energy_dict[name] = float(torch.sum(vec ** 2).item())
        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(module.register_forward_hook(hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for handle in handles:
        handle.remove()

    return energy_dict


def get_gradient_statistics(model: torch.nn.Module) -> dict:
    """
    Возвращает статистику градиентов модели (mean, std, min, max).
    """
    import torch
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    vec = torch.cat(grads)
    return {
        "mean": float(vec.mean().item()),
        "std": float(vec.std().item()) if vec.numel() > 1 else 0.0,
        "min": float(vec.min().item()),
        "max": float(vec.max().item())
    }


def compute_parameter_abs_mean(model: torch.nn.Module) -> float:
    """
    Вычисляет среднее абсолютное значение параметров модели.
    """
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    return float(torch.abs(vec).mean().item())

def compute_gradient_abs_mean(model: torch.nn.Module) -> float:
    """
    Вычисляет среднее абсолютное значение градиентов модели.
    """
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None and p.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    return float(torch.abs(vec).mean().item())

def compute_activation_abs_mean(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет среднее абсолютное значение активаций для заданных слоев.
    """
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                if vec.numel() > 0:
                    activations[name] = float(torch.abs(vec).mean().item())
                else:
                    activations[name] = 0.0
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations


def compute_parameter_vmr(model: torch.nn.Module) -> float:
    """
    Вычисляет Variance-to-Mean Ratio (VMR) всех параметров модели.
    """
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    mean = vec.mean().item()
    if mean == 0.0:
        return 0.0
    var = vec.var().item() if vec.numel() > 1 else 0.0
    return float(var / mean)


def compute_gradient_vmr(model: torch.nn.Module) -> float:
    """
    Вычисляет Variance-to-Mean Ratio (VMR) градиентов всех параметров модели.
    """
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None and p.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    mean = vec.mean().item()
    if mean == 0.0:
        return 0.0
    var = vec.var().item() if vec.numel() > 1 else 0.0
    return float(var / mean)


def compute_activation_vmr(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет Variance-to-Mean Ratio (VMR) активаций для заданных слоев.
    """
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                if vec.numel() > 1:
                    mean = vec.mean().item()
                    if mean != 0.0:
                        var = vec.var().item()
                        activations[name] = float(var / mean)
                    else:
                        activations[name] = 0.0
                else:
                    activations[name] = 0.0
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations


def compute_parameter_sem(model: torch.nn.Module) -> float:
    """
    Вычисляет стандартную ошибку среднего (SEM) всех параметров модели.
    """
    import math
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    n = vec.numel()
    if n <= 1:
        return 0.0
    std = float(vec.float().std().item())
    return std / math.sqrt(n)


def compute_gradient_sem(model: torch.nn.Module) -> float:
    """
    Вычисляет стандартную ошибку среднего (SEM) всех градиентов модели.
    """
    import math
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    n = vec.numel()
    if n <= 1:
        return 0.0
    std = float(vec.float().std().item())
    return std / math.sqrt(n)


def compute_activation_sem(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет стандартную ошибку среднего (SEM) активаций для заданных слоев.
    """
    import math
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                n = vec.numel()
                if n > 1:
                    std = float(vec.float().std().item())
                    activations[name] = std / math.sqrt(n)
                else:
                    activations[name] = 0.0
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations


def compute_parameter_winsorized_mean(model: torch.nn.Module, limits: tuple[float, float] = (0.05, 0.05)) -> float:
    """
    Вычисляет винзоризованное среднее (winsorized mean) параметров модели.
    """
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    n = vec.numel()
    if n == 0:
        return 0.0

    lower_limit, upper_limit = limits
    k_lower = int(n * lower_limit)
    k_upper = int(n * upper_limit)

    if k_lower + k_upper >= n or n <= 2:
        return float(vec.mean().item())

    sorted_vec, _ = torch.sort(vec)
    lower_val = sorted_vec[k_lower].item()
    upper_val = sorted_vec[n - 1 - k_upper].item()

    winsorized_vec = torch.clamp(vec, min=lower_val, max=upper_val)
    return float(winsorized_vec.mean().item())


def compute_gradient_winsorized_mean(model: torch.nn.Module, limits: tuple[float, float] = (0.05, 0.05)) -> float:
    """
    Вычисляет винзоризованное среднее (winsorized mean) градиентов параметров модели.
    """
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    n = vec.numel()
    if n == 0:
        return 0.0

    lower_limit, upper_limit = limits
    k_lower = int(n * lower_limit)
    k_upper = int(n * upper_limit)

    if k_lower + k_upper >= n or n <= 2:
        return float(vec.mean().item())

    sorted_vec, _ = torch.sort(vec)
    lower_val = sorted_vec[k_lower].item()
    upper_val = sorted_vec[n - 1 - k_upper].item()

    winsorized_vec = torch.clamp(vec, min=lower_val, max=upper_val)
    return float(winsorized_vec.mean().item())


def compute_activation_winsorized_mean(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str], limits: tuple[float, float] = (0.05, 0.05)) -> dict[str, float]:
    """
    Вычисляет винзоризованное среднее (winsorized mean) активаций для заданных слоев.
    """
    activations = {}
    hooks = []

    lower_limit, upper_limit = limits

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                n = vec.numel()
                if n == 0:
                    activations[name] = 0.0
                else:
                    k_lower = int(n * lower_limit)
                    k_upper = int(n * upper_limit)

                    if k_lower + k_upper >= n or n <= 2:
                        activations[name] = float(vec.mean().item())
                    else:
                        sorted_vec, _ = torch.sort(vec)
                        lower_val = sorted_vec[k_lower].item()
                        upper_val = sorted_vec[n - 1 - k_upper].item()

                        winsorized_vec = torch.clamp(vec, min=lower_val, max=upper_val)
                        activations[name] = float(winsorized_vec.mean().item())
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations


def compute_parameter_trimmed_mean(model: torch.nn.Module, trim_percent: float = 0.1) -> float:
    """
    Вычисляет усеченное среднее (trimmed mean) параметров модели.
    """
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    n = vec.numel()
    if n == 0:
        return 0.0
    k = int(n * trim_percent)
    if 2 * k >= n or n <= 2:
        return float(vec.mean().item())
    sorted_vec, _ = torch.sort(vec)
    trimmed_vec = sorted_vec[k:n-k]
    return float(trimmed_vec.mean().item())


def compute_gradient_trimmed_mean(model: torch.nn.Module, trim_percent: float = 0.1) -> float:
    """
    Вычисляет усеченное среднее (trimmed mean) градиентов модели.
    """
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    n = vec.numel()
    if n == 0:
        return 0.0
    k = int(n * trim_percent)
    if 2 * k >= n or n <= 2:
        return float(vec.mean().item())
    sorted_vec, _ = torch.sort(vec)
    trimmed_vec = sorted_vec[k:n-k]
    return float(trimmed_vec.mean().item())


def compute_activation_trimmed_mean(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str], trim_percent: float = 0.1) -> dict[str, float]:
    """
    Вычисляет усеченное среднее (trimmed mean) активаций для заданных слоев.
    """
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                n = vec.numel()
                if n == 0:
                    activations[name] = 0.0
                else:
                    k = int(n * trim_percent)
                    if 2 * k >= n or n <= 2:
                        activations[name] = float(vec.mean().item())
                    else:
                        sorted_vec, _ = torch.sort(vec)
                        trimmed_vec = sorted_vec[k:n-k]
                        activations[name] = float(trimmed_vec.mean().item())
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations


def compute_parameter_proportion_zero(model: torch.nn.Module) -> float:
    """
    Вычисляет долю нулевых элементов параметров модели.
    """
    total_elements = 0
    zero_elements = 0
    for param in model.parameters():
        total_elements += param.numel()
        zero_elements += (param == 0).sum().item()
    if total_elements == 0:
        return 0.0
    return float(zero_elements / total_elements)


def compute_gradient_proportion_zero(model: torch.nn.Module) -> float:
    """
    Вычисляет долю нулевых элементов градиентов модели.
    """
    total_elements = 0
    zero_elements = 0
    for param in model.parameters():
        if param.grad is not None:
            total_elements += param.grad.numel()
            zero_elements += (param.grad == 0).sum().item()
    if total_elements == 0:
        return 0.0
    return float(zero_elements / total_elements)


def compute_activation_proportion_zero(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет долю нулевых элементов активаций для заданных слоев.
    """
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                if vec.numel() > 0:
                    activations[name] = float(((vec == 0).sum().item()) / vec.numel())
                else:
                    activations[name] = 0.0
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations


def compute_parameter_outlier_ratio(model: torch.nn.Module, threshold: float = 3.0) -> float:
    """
    Вычисляет долю выбросов (значений, отклоняющихся от среднего более чем на threshold стандартных отклонений) среди параметров модели.
    """
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    if vec.numel() <= 1:
        return 0.0
    mean = vec.mean()
    std = vec.std()
    if torch.isnan(std) or std == 0.0:
        return 0.0
    outliers = torch.sum(torch.abs(vec - mean) > threshold * std)
    return float((outliers / vec.numel()).item())


def compute_gradient_outlier_ratio(model: torch.nn.Module, threshold: float = 3.0) -> float:
    """
    Вычисляет долю выбросов среди градиентов модели.
    """
    grads = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    if vec.numel() <= 1:
        return 0.0
    mean = vec.mean()
    std = vec.std()
    if torch.isnan(std) or std == 0.0:
        return 0.0
    outliers = torch.sum(torch.abs(vec - mean) > threshold * std)
    return float((outliers / vec.numel()).item())


def compute_activation_outlier_ratio(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str], threshold: float = 3.0) -> dict[str, float]:
    """
    Вычисляет долю выбросов среди активаций для заданных слоев.
    """
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                if vec.numel() <= 1:
                    activations[name] = 0.0
                    return
                mean = vec.mean()
                std = vec.std()
                if torch.isnan(std) or std == 0.0:
                    activations[name] = 0.0
                else:
                    outliers = torch.sum(torch.abs(vec - mean) > threshold * std)
                    activations[name] = float((outliers / vec.numel()).item())
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations


def _gini_coefficient(vec: torch.Tensor) -> float:
    """
    Вспомогательная функция для вычисления коэффициента Джини 1D тензора.
    """
    if vec.numel() == 0:
        return 0.0
    vec = torch.abs(vec).flatten().to(torch.float32)
    vec, _ = torch.sort(vec)
    n = vec.numel()
    if n <= 1:
        return 0.0
    index = torch.arange(1, n + 1, dtype=torch.float32, device=vec.device)
    sum_vec = torch.sum(vec)
    if sum_vec == 0:
        return 0.0
    gini = (torch.sum((2 * index - n - 1) * vec)) / (n * sum_vec)
    return float(gini.item())


def compute_parameter_gini(model: torch.nn.Module) -> float:
    """
    Вычисляет коэффициент Джини для всех параметров модели.
    """
    params = [param.flatten() for param in model.parameters()]
    if not params:
        return 0.0
    all_params = torch.cat(params)
    return _gini_coefficient(all_params)


def compute_gradient_gini(model: torch.nn.Module) -> float:
    """
    Вычисляет коэффициент Джини для всех градиентов параметров модели.
    """
    grads = [param.grad.flatten() for param in model.parameters() if param.grad is not None]
    if not grads:
        return 0.0
    all_grads = torch.cat(grads)
    return _gini_coefficient(all_grads)


def compute_activation_gini(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет коэффициент Джини активаций для заданных слоев.
    """
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                activations[name] = _gini_coefficient(vec)
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations

def compute_parameter_geometric_mean(model: torch.nn.Module) -> float:
    """
    Вычисляет среднее геометрическое параметров модели (по абсолютным значениям).
    """
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    vec = torch.abs(vec)
    vec = vec[vec > 0]
    if vec.numel() == 0:
        return 0.0
    return float(torch.exp(torch.mean(torch.log(vec))).item())

def compute_gradient_geometric_mean(model: torch.nn.Module) -> float:
    """
    Вычисляет среднее геометрическое градиентов модели (по абсолютным значениям).
    """
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None and p.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    vec = torch.abs(vec)
    vec = vec[vec > 0]
    if vec.numel() == 0:
        return 0.0
    return float(torch.exp(torch.mean(torch.log(vec))).item())

def compute_activation_geometric_mean(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет среднее геометрическое активаций для заданных слоев (по абсолютным значениям).
    """
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                vec = torch.abs(vec)
                vec = vec[vec > 0]
                if vec.numel() > 0:
                    activations[name] = float(torch.exp(torch.mean(torch.log(vec))).item())
                else:
                    activations[name] = 0.0
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations


def compute_parameter_harmonic_mean(model: torch.nn.Module) -> float:
    """
    Вычисляет среднее гармоническое параметров модели (по абсолютным значениям).
    """
    params = [p.data.flatten() for p in model.parameters() if p.numel() > 0]
    if not params:
        return 0.0
    vec = torch.cat(params)
    vec = torch.abs(vec)
    vec = vec[vec > 0]
    if vec.numel() == 0:
        return 0.0
    return float((vec.numel() / torch.sum(1.0 / vec)).item())

def compute_gradient_harmonic_mean(model: torch.nn.Module) -> float:
    """
    Вычисляет среднее гармоническое градиентов модели (по абсолютным значениям).
    """
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None and p.numel() > 0]
    if not grads:
        return 0.0
    vec = torch.cat(grads)
    vec = torch.abs(vec)
    vec = vec[vec > 0]
    if vec.numel() == 0:
        return 0.0
    return float((vec.numel() / torch.sum(1.0 / vec)).item())

def compute_parameter_proportion_positive(model: torch.nn.Module) -> float:
    """
    Вычисляет долю положительных элементов параметров модели.
    """
    total_elements = 0
    positive_elements = 0
    for param in model.parameters():
        total_elements += param.numel()
        positive_elements += (param > 0).sum().item()
    if total_elements == 0:
        return 0.0
    return float(positive_elements / total_elements)


def compute_gradient_proportion_positive(model: torch.nn.Module) -> float:
    """
    Вычисляет долю положительных элементов градиентов модели.
    """
    total_elements = 0
    positive_elements = 0
    for param in model.parameters():
        if param.grad is not None:
            total_elements += param.grad.numel()
            positive_elements += (param.grad > 0).sum().item()
    if total_elements == 0:
        return 0.0
    return float(positive_elements / total_elements)


def compute_parameter_proportion_negative(model: torch.nn.Module) -> float:
    """
    Вычисляет долю отрицательных элементов параметров модели.
    """
    total_elements = 0
    negative_elements = 0
    for param in model.parameters():
        total_elements += param.numel()
        negative_elements += (param < 0).sum().item()
    if total_elements == 0:
        return 0.0
    return float(negative_elements / total_elements)


def compute_gradient_proportion_negative(model: torch.nn.Module) -> float:
    """
    Вычисляет долю отрицательных элементов градиентов модели.
    """
    total_elements = 0
    negative_elements = 0
    for param in model.parameters():
        if param.grad is not None:
            total_elements += param.grad.numel()
            negative_elements += (param.grad < 0).sum().item()
    if total_elements == 0:
        return 0.0
    return float(negative_elements / total_elements)


def compute_activation_proportion_negative(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет долю отрицательных элементов активаций для заданных слоев.
    """
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                if vec.numel() > 0:
                    activations[name] = float(((vec < 0).sum().item()) / vec.numel())
                else:
                    activations[name] = 0.0
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations


def compute_activation_proportion_positive(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет долю положительных элементов активаций для заданных слоев.
    """
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                if vec.numel() > 0:
                    activations[name] = float(((vec > 0).sum().item()) / vec.numel())
                else:
                    activations[name] = 0.0
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations


def compute_activation_harmonic_mean(model: torch.nn.Module, input_data: torch.Tensor, layer_names: list[str]) -> dict[str, float]:
    """
    Вычисляет среднее гармоническое активаций для заданных слоев (по абсолютным значениям).
    """
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                vec = output.detach().flatten()
                vec = torch.abs(vec)
                vec = vec[vec > 0]
                if vec.numel() > 0:
                    activations[name] = float((vec.numel() / torch.sum(1.0 / vec)).item())
                else:
                    activations[name] = 0.0
            else:
                activations[name] = 0.0
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        model(input_data)

    if training_state:
        model.train()

    for hook in hooks:
        hook.remove()

    return activations
