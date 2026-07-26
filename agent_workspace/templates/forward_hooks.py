import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
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
