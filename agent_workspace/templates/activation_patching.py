import torch
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
