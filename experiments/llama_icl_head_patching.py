import os
import sys
import torch
from src.experiment_utils import load_model_and_tokenizer, clear_memory
from einops import rearrange


from src.metrics import logit_difference

def main():
    model_name = "JackFram/llama-160m"
    print(f"Loading {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    clean = "apple -> red, banana -> yellow, grass -> green, sky ->"
    corr = "apple -> dog, banana -> cat, grass -> bird, sky ->"

    inputs_clean = tokenizer(clean, return_tensors="pt")
    inputs_corr = tokenizer(corr, return_tensors="pt")

    blue_id = tokenizer.encode("blue", add_special_tokens=False)[0]
    bird_id = tokenizer.encode("bird", add_special_tokens=False)[0]

    with torch.no_grad():
        clean_logits = model(**inputs_clean).logits
        corr_logits = model(**inputs_corr).logits

    clean_diff = logit_difference(clean_logits, blue_id, bird_id)
    corr_diff = logit_difference(corr_logits, blue_id, bird_id)

    print(f"Clean Logit Diff: {clean_diff:.4f}")
    print(f"Corr Logit Diff: {corr_diff:.4f}")

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    print("\nPatching Clean Attention Heads into Corrupted Run")

    clean_head_activations = {}
    def save_head_hook(head):
        def hook(module, input):
            x = input[0]
            x_reshaped = rearrange(x, 'b s (h d) -> b s h d', h=num_heads, d=head_dim)
            clean_head_activations[head] = x_reshaped[:, :, head, :].clone()
            return input
        return hook

    def patch_head_hook(head):
        def hook(module, input):
            x = input[0]
            x_reshaped = rearrange(x, 'b s (h d) -> b s h d', h=num_heads, d=head_dim)
            x_reshaped[:, :, head, :] = clean_head_activations[head]
            x_restored = rearrange(x_reshaped, 'b s h d -> b s (h d)')
            return (x_restored,)
        return hook

    for layer_idx in range(num_layers):
        clear_memory()
        for head_idx in range(num_heads):
            # LLaMA uses o_proj
            handle1 = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(save_head_hook(head_idx))
            with torch.no_grad():
                model(**inputs_clean)
            handle1.remove()

            handle2 = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(patch_head_hook(head_idx))
            with torch.no_grad():
                patched_logits = model(**inputs_corr).logits
            handle2.remove()

            patched_diff = logit_difference(patched_logits, blue_id, bird_id)
            recovered = (patched_diff - corr_diff) / (clean_diff - corr_diff)
            if recovered > 0.1: # Show heads that recover > 10%
                print(f"L{layer_idx}H{head_idx}: Recovered = {recovered:.2%}, Logit Diff = {patched_diff:.4f}")

if __name__ == "__main__":
    main()
