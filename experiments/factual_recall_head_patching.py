import sys
import os
import torch
from src.experiment_utils import load_model_and_tokenizer

# Ensure templates exist
if not os.path.exists("agent_workspace/templates/metrics.py"):
    from src.sandbox_env import SandboxEnvironment
    env = SandboxEnvironment()
    env.setup_templates()

sys.path.append(os.path.abspath("agent_workspace"))
from templates.metrics import logit_difference

def run_experiment(model_name="gpt2"):
    print(f"\n--- Running Head Level Activation Patching for {model_name} ---")
    model, tokenizer = load_model_and_tokenizer(model_name)

    clean_text = "The capital of France is"
    corrupted_text = "The capital of Russia is"

    clean_inputs = tokenizer(clean_text, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt")

    target_id_clean = tokenizer.encode(" Paris")[0]
    target_id_corrupted = tokenizer.encode(" Moscow")[0]

    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupted_outputs = model(**corrupted_inputs)

    baseline_clean_diff = logit_difference(clean_outputs.logits, target_id_clean, target_id_corrupted)
    baseline_corrupted_diff = logit_difference(corrupted_outputs.logits, target_id_clean, target_id_corrupted)

    print(f"Baseline Clean Logit Diff (Paris - Moscow): {baseline_clean_diff:.4f}")
    print(f"Baseline Corrupted Logit Diff (Paris - Moscow): {baseline_corrupted_diff:.4f}")

    num_layers = model.config.n_layer if hasattr(model.config, 'n_layer') else model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else model.config.n_head
    head_dim = model.config.hidden_size // num_heads if hasattr(model.config, 'hidden_size') else model.config.n_embd // num_heads

    is_llama = "llama" in model_name.lower() or "tinyllama" in model_name.lower()

    if is_llama:
        print("Model uses LLaMA architecture. Exiting for now as we only need to test on GPT-like models for simplicity in this script.")
        return

    print("\nStarting Activation Patching at Attention Head Level...")

    results = {}

    for layer_idx in range(num_layers):
        attn_layer = model.transformer.h[layer_idx].attn
        results[layer_idx] = []

        for head_idx in range(num_heads):
            cached_corrupted_input = None

            def cache_hook(module, args):
                nonlocal cached_corrupted_input
                cached_corrupted_input = args[0].detach().clone()
                return args

            handle_cache = attn_layer.c_proj.register_forward_pre_hook(cache_hook)
            with torch.no_grad():
                model(**corrupted_inputs)
            handle_cache.remove()

            def patch_hook(module, args):
                hidden = args[0].clone()
                batch, seq, n_embd = hidden.shape

                # Reshape to separate heads
                hidden_reshaped = hidden.view(batch, seq, num_heads, head_dim)
                cached_reshaped = cached_corrupted_input.view(batch, seq, num_heads, head_dim)

                # Patch just the specific head
                hidden_reshaped[:, :, head_idx, :] = cached_reshaped[:, :, head_idx, :]

                # Reshape back
                patched_hidden = hidden_reshaped.view(batch, seq, n_embd)
                return (patched_hidden,)

            handle_patch = attn_layer.c_proj.register_forward_pre_hook(patch_hook)
            with torch.no_grad():
                patched_outputs = model(**clean_inputs)
            handle_patch.remove()

            diff = logit_difference(patched_outputs.logits, target_id_clean, target_id_corrupted)
            results[layer_idx].append((head_idx, diff))

            # Print significant drops (patching corrupted into clean lowers the diff towards corrupted baseline)
            drop = baseline_clean_diff - diff
            if drop > 0.5:
                print(f"Layer {layer_idx}, Head {head_idx}: Logit Diff = {diff:.4f} (Drop = {drop:.4f}) -> Potential Factual Recall Head!")

    print("\nDetailed Results (Layer, Head): Logit Difference")
    for layer_idx in range(num_layers):
        print(f"--- Layer {layer_idx} ---")
        for head_idx, diff in results[layer_idx]:
            # only print interesting ones to save space, say diff drops by more than 0.2
            if baseline_clean_diff - diff > 0.2:
                print(f"Head {head_idx:2d}: {diff:.4f}")

if __name__ == "__main__":
    for model_name in ["gpt2"]:
        run_experiment(model_name)
