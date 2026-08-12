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
