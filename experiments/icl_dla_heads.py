import os
import sys
import torch
from src.experiment_utils import load_model_and_tokenizer
from einops import rearrange

def main():
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    prompt = "apple -> red, banana -> yellow, grass -> green, sky ->"
    inputs = tokenizer(prompt, return_tensors="pt")

    target_word = " blue"
    target_id = tokenizer.encode(target_word)[0]

    unembed_weight = model.lm_head.weight[target_id]

    activations = {}
    def get_activation(name):
        def hook(module, input): # PRE HOOK DOES NOT HAVE OUTPUT
            # Input to c_proj is the concatenated head outputs
            activations[name] = input[0].detach().cpu()
        return hook

    handles = []
    for layer_idx in range(model.config.n_layer):
        handle = model.transformer.h[layer_idx].attn.c_proj.register_forward_pre_hook(get_activation(f'attn_c_proj_in_l{layer_idx}'))
        handles.append(handle)

    with torch.no_grad():
        outputs = model(**inputs)

    for handle in handles:
        handle.remove()

    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads

    print("\n--- Direct Logit Attribution (DLA) Scores by Head for ' blue' ---")
    for layer_idx in range(model.config.n_layer):
        # [batch, seq_len, n_embd]
        c_proj_in = activations[f'attn_c_proj_in_l{layer_idx}'][0, -1, :]
        # [num_heads, head_dim]
        c_proj_in_heads = rearrange(c_proj_in, '(h d) -> h d', h=num_heads, d=head_dim)

        # We need to project each head's output through c_proj to get its contribution to the residual stream
        c_proj_weight = model.transformer.h[layer_idx].attn.c_proj.weight # [n_embd, n_embd]

        # c_proj_weight can be split by head
        c_proj_weight_heads = rearrange(c_proj_weight, '(h d) out -> h d out', h=num_heads, d=head_dim)

        dla_scores_layer = []
        for head_idx in range(num_heads):
            # Project head output: [head_dim] @ [head_dim, n_embd] -> [n_embd]
            head_out_proj = c_proj_in_heads[head_idx] @ c_proj_weight_heads[head_idx]
            dla_score = torch.dot(head_out_proj, unembed_weight).item()
            dla_scores_layer.append(dla_score)
            if dla_score > 3.0:
                print(f"Layer {layer_idx:2d} Head {head_idx:2d} DLA: {dla_score:8.4f}")

if __name__ == "__main__":
    main()
