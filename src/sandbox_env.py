import os

class SandboxEnvironment:
    def __init__(self, workspace_dir="agent_workspace"):
        self.workspace_dir = os.path.abspath(workspace_dir)
        os.makedirs(self.workspace_dir, exist_ok=True)

    def resolve_path(self, path: str) -> str:
        """Resolve and validate that the path is within the workspace."""
        abs_path = os.path.abspath(os.path.join(self.workspace_dir, path))
        if not abs_path.startswith(self.workspace_dir):
            raise ValueError(f"Access denied: path {path} is outside the workspace directory.")
        return abs_path

    def setup_templates(self):
        """Setup initial experiment templates in the workspace."""
        templates_dir = os.path.join(self.workspace_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)

        # Example template for Activation Patching
        patching_template = """import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Загружаем модель и токенизатор
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

source_text = "The capital of Italy is"
target_text = "The capital of France is"

source_inputs = tokenizer(source_text, return_tensors="pt")
target_inputs = tokenizer(target_text, return_tensors="pt")

layer_to_patch = model.transformer.h[8].mlp
cached_activation = None

def cache_hook(module, input, output):
    global cached_activation
    cached_activation = output.detach().clone()

handle_cache = layer_to_patch.register_forward_hook(cache_hook)
with torch.no_grad():
    model(**source_inputs)
handle_cache.remove()

def patch_hook(module, input, output):
    patched_output = output.clone()
    patched_output[0, -1, :] = cached_activation[0, -1, :]
    return patched_output

handle_patch = layer_to_patch.register_forward_hook(patch_hook)
with torch.no_grad():
    outputs = model(**target_inputs)
handle_patch.remove()

next_token_logits = outputs.logits[0, -1, :]
predicted_token_id = torch.argmax(next_token_logits).item()
predicted_word = tokenizer.decode(predicted_token_id)

print(f"Target prompt: '{target_text}'")
print(f"Predicted word: '{predicted_word}'")
"""
        with open(os.path.join(templates_dir, "activation_patching.py"), "w") as f:
            f.write(patching_template)

        # Example template for Ablation Studies
        ablation_template = """import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

text = "Mechanistic Interpretability is"
inputs = tokenizer(text, return_tensors="pt")

# Цель: занулить выход определенной головы внимания (Ablation)
layer_idx = 0
head_idx = 4
num_heads = model.config.n_head
head_dim = model.config.n_embd // num_heads

def ablation_hook(module, input, output):
    # output - это кортеж. Нам нужен первый элемент (hidden states)
    hidden_states = output[0]

    # hidden_states имеет форму [batch_size, seq_len, n_embd]
    # Нам нужно занулить конкретную голову
    batch_size, seq_len, n_embd = hidden_states.shape

    # Разделяем эмбеддинг на головы
    hidden_states_reshaped = hidden_states.view(batch_size, seq_len, num_heads, head_dim)

    # Зануляем выбранную голову
    hidden_states_reshaped[:, :, head_idx, :] = 0.0

    # Собираем обратно
    modified_hidden_states = hidden_states_reshaped.view(batch_size, seq_len, n_embd)

    # Возвращаем модифицированный кортеж
    return (modified_hidden_states,) + output[1:]

layer_to_ablate = model.transformer.h[layer_idx].attn
handle = layer_to_ablate.register_forward_hook(ablation_hook)

with torch.no_grad():
    outputs = model(**inputs)

handle.remove()

next_token_logits = outputs.logits[0, -1, :]
predicted_token_id = torch.argmax(next_token_logits).item()
print(f"Predicted word after ablation: '{tokenizer.decode(predicted_token_id)}'")
"""
        with open(os.path.join(templates_dir, "ablation.py"), "w") as f:
            f.write(ablation_template)

        # Example template for Forward Hooks (Activation Extraction)
        forward_hooks_template = """import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

text = "Extracting activations is fun"
inputs = tokenizer(text, return_tensors="pt")

activations = {}

def get_activation(name):
    def hook(module, input, output):
        # Если output это кортеж, берем первый элемент (для многих слоев attention)
        if isinstance(output, tuple):
            activations[name] = output[0].detach().cpu()
        else:
            activations[name] = output.detach().cpu()
    return hook

# Регистрируем хуки на MLP слои первых трех блоков
handles = []
for i in range(3):
    layer = model.transformer.h[i].mlp
    handle = layer.register_forward_hook(get_activation(f'mlp_layer_{i}'))
    handles.append(handle)

with torch.no_grad():
    outputs = model(**inputs)

for handle in handles:
    handle.remove()

# Выводим информацию о сохраненных активациях
for name, act in activations.items():
    print(f"Layer: {name}, Activation shape: {act.shape}, Mean: {act.mean().item():.4f}")
"""
        with open(os.path.join(templates_dir, "forward_hooks.py"), "w") as f:
            f.write(forward_hooks_template)

        # Example template for Metrics Calculation
        metrics_template = """import torch
import torch.nn.functional as F

def logit_difference(logits, target_id_1, target_id_2):
    '''
    Вычисляет разницу логитов между двумя целевыми токенами.
    Полезно для оценки того, насколько модель предпочитает один ответ другому.
    '''
    # Предполагается, что logits имеют форму [..., seq_len, vocab_size]
    # Берем логиты последнего токена
    last_token_logits = logits[0, -1, :]
    return (last_token_logits[target_id_1] - last_token_logits[target_id_2]).item()

def kl_divergence(logits_base, logits_patched):
    '''
    Вычисляет KL-дивергенцию между распределениями оригинальной и модифицированной модели.
    Полезно для оценки общего изменения поведения модели.
    '''
    # Берем логиты последнего токена
    p = F.softmax(logits_base[0, -1, :], dim=-1)
    log_q = F.log_softmax(logits_patched[0, -1, :], dim=-1)

    # KL(P || Q) = sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
    kl = F.kl_div(log_q, p, reduction='batchmean')
    return kl.item()

def loss_degradation(logits_base, logits_patched, target_ids):
    '''
    Вычисляет изменение потерь (Loss) для целевой последовательности.
    '''
    pass # Реализуйте по необходимости

# Пример использования
if __name__ == '__main__':
    # Fake data
    vocab_size = 50257
    logits_base = torch.randn(1, 1, vocab_size)
    logits_patched = logits_base + torch.randn(1, 1, vocab_size) * 0.1

    target_1 = 1234
    target_2 = 5678

    diff = logit_difference(logits_patched, target_1, target_2)
    kl = kl_divergence(logits_base, logits_patched)

    print(f"Logit Diff: {diff:.4f}")
    print(f"KL Divergence: {kl:.4f}")
"""
        with open(os.path.join(templates_dir, "metrics.py"), "w") as f:
            f.write(metrics_template)


if __name__ == "__main__":
    env = SandboxEnvironment()
    env.setup_templates()
    print("Sandbox templates created.")
