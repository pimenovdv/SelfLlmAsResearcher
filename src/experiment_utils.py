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
