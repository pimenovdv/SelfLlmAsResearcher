import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt")

target_word_1 = " Paris"
target_word_2 = " Moscow"
target_id_1 = tokenizer.encode(target_word_1)[0]
target_id_2 = tokenizer.encode(target_word_2)[0]

def logit_difference(logits, target_id_1, target_id_2):
    last_token_logits = logits[0, -1, :]
    return (last_token_logits[target_id_1] - last_token_logits[target_id_2]).item()

with torch.no_grad():
    outputs_clean = model(**inputs)
    clean_diff = logit_difference(outputs_clean.logits, target_id_1, target_id_2)
    print(f"Clean Logit Diff (Paris - Moscow): {clean_diff:.4f}")

heads_to_ablate = [(9, 8), (10, 0)]
num_heads = model.config.n_head
head_dim = model.config.n_embd // num_heads

def get_ablation_pre_hook(head_idx):
    def ablation_pre_hook(module, args):
        # args is a tuple of arguments to c_proj. The first arg is the hidden states before projection.
        hidden_states = args[0]
        batch_size, seq_len, n_embd = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(batch_size, seq_len, num_heads, head_dim)
        # clone to avoid modifying in-place which might cause issues
        hidden_states_reshaped = hidden_states_reshaped.clone()
        hidden_states_reshaped[:, :, head_idx, :] = 0.0
        modified_hidden_states = hidden_states_reshaped.view(batch_size, seq_len, n_embd)
        return (modified_hidden_states,) + args[1:]
    return ablation_pre_hook

handles = []
for layer_idx, head_idx in heads_to_ablate:
    layer_to_ablate = model.transformer.h[layer_idx].attn.c_proj
    handle = layer_to_ablate.register_forward_pre_hook(get_ablation_pre_hook(head_idx))
    handles.append(handle)

with torch.no_grad():
    outputs_ablated = model(**inputs)

for handle in handles:
    handle.remove()

ablated_diff = logit_difference(outputs_ablated.logits, target_id_1, target_id_2)
print(f"Ablated Logit Diff (Paris - Moscow) for L9H8 and L10H0: {ablated_diff:.4f}")
