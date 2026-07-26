import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
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
