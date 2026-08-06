import sys
import os
import torch
from src.experiment_utils import load_model_and_tokenizer, clear_memory
import logging


from src.metrics import logit_difference

def run_experiment(model_name="gpt2"):
    print(f"\n--- Running Activation Patching for {model_name} ---")
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Factual Recall Task setup
    clean_text = "The capital of France is"
    corrupted_text = "The capital of Russia is"

    clean_inputs = tokenizer(clean_text, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt")

    target_id_clean = tokenizer.encode(" Paris")[0]
    target_id_corrupted = tokenizer.encode(" Moscow")[0]

    # Run clean and corrupted forward passes to establish baselines
    print("Running baseline forward passes...")
    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupted_outputs = model(**corrupted_inputs)

    clean_logits = clean_outputs.logits
    corrupted_logits = corrupted_outputs.logits

    baseline_clean_diff = logit_difference(clean_logits, target_id_clean, target_id_corrupted)
    baseline_corrupted_diff = logit_difference(corrupted_logits, target_id_clean, target_id_corrupted)

    print(f"Baseline Clean Logit Diff (Paris - Moscow): {baseline_clean_diff:.4f}")
    print(f"Baseline Corrupted Logit Diff (Paris - Moscow): {baseline_corrupted_diff:.4f}")

    num_layers = model.config.n_layer if hasattr(model.config, 'n_layer') else model.config.num_hidden_layers

    print("\nStarting Activation Patching (Attn & MLP layers)...")

    attn_diffs = []
    mlp_diffs = []

    is_llama = "llama" in model_name.lower() or "tinyllama" in model_name.lower()

    for layer_idx in range(num_layers):
        clear_memory()
        # 1. Patching Attention
        if is_llama:
            attn_layer = model.model.layers[layer_idx].self_attn
        else:
            attn_layer = model.transformer.h[layer_idx].attn

        cached_attn = None

        def cache_attn_hook(module, input, output):
            nonlocal cached_attn
            if isinstance(output, tuple):
                cached_attn = output[0].detach().clone()
            else:
                cached_attn = output.detach().clone()

        handle_cache_attn = attn_layer.register_forward_hook(cache_attn_hook)
        with torch.no_grad():
            model(**corrupted_inputs)
        handle_cache_attn.remove()

        def patch_attn_hook(module, input, output):
            patched_hidden = cached_attn
            if isinstance(output, tuple):
                return (patched_hidden,) + output[1:]
            else:
                return patched_hidden

        handle_patch_attn = attn_layer.register_forward_hook(patch_attn_hook)
        with torch.no_grad():
            patched_outputs = model(**clean_inputs)
        handle_patch_attn.remove()

        diff = logit_difference(patched_outputs.logits, target_id_clean, target_id_corrupted)
        attn_diffs.append(diff)

        # 2. Patching MLP
        if is_llama:
            mlp_layer = model.model.layers[layer_idx].mlp
        else:
            mlp_layer = model.transformer.h[layer_idx].mlp

        cached_mlp = None

        def cache_mlp_hook(module, input, output):
            nonlocal cached_mlp
            if isinstance(output, tuple):
                cached_mlp = output[0].detach().clone()
            else:
                cached_mlp = output.detach().clone()

        handle_cache_mlp = mlp_layer.register_forward_hook(cache_mlp_hook)
        with torch.no_grad():
            model(**corrupted_inputs)
        handle_cache_mlp.remove()

        def patch_mlp_hook(module, input, output):
            if isinstance(output, tuple):
                return (cached_mlp,) + output[1:]
            else:
                return cached_mlp

        handle_patch_mlp = mlp_layer.register_forward_hook(patch_mlp_hook)
        with torch.no_grad():
            patched_outputs = model(**clean_inputs)
        handle_patch_mlp.remove()

        diff = logit_difference(patched_outputs.logits, target_id_clean, target_id_corrupted)
        mlp_diffs.append(diff)

    print("\nResults (Logit Difference after patching corrupted activation into clean run):")
    print(f"Goal is to see which layer's patch drops the difference closer to the corrupted baseline ({baseline_corrupted_diff:.4f})")

    print("\nLayer\tAttn Patch\tMLP Patch")
    for i in range(num_layers):
        print(f"{i}\t{attn_diffs[i]:.4f}\t\t{mlp_diffs[i]:.4f}")

if __name__ == "__main__":
    for model_name in ["gpt2", "EleutherAI/gpt-neo-125m"]:
        run_experiment(model_name)
