import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
