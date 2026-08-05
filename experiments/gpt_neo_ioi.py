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

def run_experiment():
    model_name = "EleutherAI/gpt-neo-125m"
    print(f"Loading model and tokenizer for {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    clean_text = "When John and Mary went to the store, John gave a drink to"
    corrupted_text = "When John and Mary went to the store, Mary gave a drink to"

    clean_inputs = tokenizer(clean_text, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt")

    # In GPT-Neo 125m, the tokenizer has " Mary" as a token, but let's check correctly
    mary_id = tokenizer.encode(" Mary")[0]
    john_id = tokenizer.encode(" John")[0]

    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupted_outputs = model(**corrupted_inputs)

    baseline_clean_diff = logit_difference(clean_outputs.logits, mary_id, john_id)
    baseline_corrupted_diff = logit_difference(corrupted_outputs.logits, mary_id, john_id)

    print(f"Baseline Clean Logit Diff (Mary - John): {baseline_clean_diff:.4f}")
    print(f"Baseline Corrupted Logit Diff (Mary - John): {baseline_corrupted_diff:.4f}")

    num_heads = model.config.num_heads
    head_dim = model.config.hidden_size // num_heads

    print("\nStarting Activation Patching at Attention Head Level (Layers 9 to 11)...")

    # GPT-Neo 125m has 12 layers (0 to 11). Let's target the last few layers to find Name Mover Heads.
    target_layers = [9, 10, 11]

    results = {}

    for layer_idx in target_layers:
        attn_layer = model.transformer.h[layer_idx].attn.attention
        results[layer_idx] = []

        for head_idx in range(num_heads):
            cached_corrupted_input = None

            def cache_hook(module, args):
                nonlocal cached_corrupted_input
                # For out_proj, input is a tuple of (hidden_states,)
                cached_corrupted_input = args[0].detach().clone()
                return args

            # Hooking into out_proj to patch the output of specific heads BEFORE they are projected
            # Wait, out_proj input is already concatenated across heads.
            # In GPTNeoAttention, the attention output has shape (batch, seq, hidden_size).
            # To patch a head, we intercept the input to `out_proj`.
            handle_cache = attn_layer.out_proj.register_forward_pre_hook(cache_hook)
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

            handle_patch = attn_layer.out_proj.register_forward_pre_hook(patch_hook)
            with torch.no_grad():
                patched_outputs = model(**clean_inputs)
            handle_patch.remove()

            diff = logit_difference(patched_outputs.logits, mary_id, john_id)
            results[layer_idx].append((head_idx, diff))

            # Print significant drops
            drop = baseline_clean_diff - diff
            if drop > 0.5:
                print(f"Layer {layer_idx}, Head {head_idx}: Logit Diff = {diff:.4f} (Drop = {drop:.4f}) -> Potential Name Mover Head!")

    print("\nDetailed Results (Layer, Head): Logit Difference")
    for layer_idx in target_layers:
        print(f"--- Layer {layer_idx} ---")
        for head_idx, diff in results[layer_idx]:
            print(f"Head {head_idx:2d}: {diff:.4f}")

if __name__ == "__main__":
    run_experiment()
