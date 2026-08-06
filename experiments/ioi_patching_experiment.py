from src.experiment_utils import clear_memory
import sys
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


from src.metrics import logit_difference

def run_experiment():
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # IOI Task setup
    clean_text = "When John and Mary went to the store, John gave a drink to"
    corrupted_text = "When John and Mary went to the store, Mary gave a drink to"

    clean_inputs = tokenizer(clean_text, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt")

    mary_id = tokenizer.encode(" Mary")[0]
    john_id = tokenizer.encode(" John")[0]

    # Run clean and corrupted forward passes to establish baselines
    print("Running baseline forward passes...")
    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupted_outputs = model(**corrupted_inputs)

    clean_logits = clean_outputs.logits
    corrupted_logits = corrupted_outputs.logits

    baseline_clean_diff = logit_difference(clean_logits, mary_id, john_id)
    baseline_corrupted_diff = logit_difference(corrupted_logits, mary_id, john_id)

    print(f"Baseline Clean Logit Diff: {baseline_clean_diff:.4f}")
    print(f"Baseline Corrupted Logit Diff: {baseline_corrupted_diff:.4f}")

    num_layers = model.config.n_layer

    print("\nStarting Activation Patching (Attn & MLP layers)...")

    # To patch, we first need to cache the activations from the corrupted prompt
    # Then we run the clean prompt and patch in the cached corrupted activations

    attn_diffs = []
    mlp_diffs = []

    for layer_idx in range(num_layers):
        clear_memory()
        # 1. Patching Attention
        attn_layer = model.transformer.h[layer_idx].attn
        cached_attn = None

        def cache_attn_hook(module, input, output):
            nonlocal cached_attn
            # output of attn is a tuple: (hidden_states, presents, ...)
            cached_attn = output[0].detach().clone()

        handle_cache_attn = attn_layer.register_forward_hook(cache_attn_hook)
        with torch.no_grad():
            model(**corrupted_inputs)
        handle_cache_attn.remove()

        def patch_attn_hook(module, input, output):
            # output is a tuple (hidden_states, ...)
            patched_hidden = cached_attn
            return (patched_hidden,) + output[1:]

        handle_patch_attn = attn_layer.register_forward_hook(patch_attn_hook)
        with torch.no_grad():
            patched_outputs = model(**clean_inputs)
        handle_patch_attn.remove()

        diff = logit_difference(patched_outputs.logits, mary_id, john_id)
        attn_diffs.append(diff)

        # 2. Patching MLP
        mlp_layer = model.transformer.h[layer_idx].mlp
        cached_mlp = None

        def cache_mlp_hook(module, input, output):
            nonlocal cached_mlp
            cached_mlp = output.detach().clone()

        handle_cache_mlp = mlp_layer.register_forward_hook(cache_mlp_hook)
        with torch.no_grad():
            model(**corrupted_inputs)
        handle_cache_mlp.remove()

        def patch_mlp_hook(module, input, output):
            return cached_mlp

        handle_patch_mlp = mlp_layer.register_forward_hook(patch_mlp_hook)
        with torch.no_grad():
            patched_outputs = model(**clean_inputs)
        handle_patch_mlp.remove()

        diff = logit_difference(patched_outputs.logits, mary_id, john_id)
        mlp_diffs.append(diff)

    print("\nResults (Logit Difference after patching corrupted activation into clean run):")
    print(f"Goal is to see which layer's patch drops the difference closer to the corrupted baseline ({baseline_corrupted_diff:.4f})")

    print("\nLayer\tAttn Patch\tMLP Patch")
    for i in range(num_layers):
        print(f"{i}\t{attn_diffs[i]:.4f}\t\t{mlp_diffs[i]:.4f}")

if __name__ == "__main__":
    run_experiment()
