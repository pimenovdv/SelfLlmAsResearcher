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

    # To find S-Inhibition Heads or Previous Token Heads, we want to patch activations
    # at early/middle layers.
    # IOI Task setup
    clean_text = "When John and Mary went to the store, John gave a drink to"
    corrupted_text = "When Mary and John went to the store, Mary gave a drink to"
    # Notice the corrupted text swaps the order of John and Mary, or we can use another corruption:
    # "When John and Mary went to the store, John gave a drink to" vs
    # "When John and Mary went to the store, Mary gave a drink to" -> this changes the S2 subject.

    # S-Inhibition heads inhibit the prediction of the duplicate name (John). They are activated by S2 (John).
    # If we patch S2 from corrupted (Mary) into clean (John), we should see a drop if a head is an S-inhibition head.

    clean_inputs = tokenizer("When John and Mary went to the store, John gave a drink to", return_tensors="pt")
    corrupted_inputs = tokenizer("When John and Mary went to the store, Mary gave a drink to", return_tensors="pt")

    mary_id = tokenizer.encode(" Mary")[0]
    john_id = tokenizer.encode(" John")[0]

    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupted_outputs = model(**corrupted_inputs)

    baseline_clean_diff = logit_difference(clean_outputs.logits, mary_id, john_id)
    baseline_corrupted_diff = logit_difference(corrupted_outputs.logits, mary_id, john_id)

    print(f"Baseline Clean Logit Diff (Mary - John): {baseline_clean_diff:.4f}")
    print(f"Baseline Corrupted Logit Diff (Mary - John): {baseline_corrupted_diff:.4f}")

    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads

    print("\nStarting Activation Patching at Attention Head Level (Layers 0 to 8)...")
    target_layers = list(range(9))
    results = {}

    for layer_idx in target_layers:
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

                hidden_reshaped = hidden.view(batch, seq, num_heads, head_dim)
                cached_reshaped = cached_corrupted_input.view(batch, seq, num_heads, head_dim)

                hidden_reshaped[:, :, head_idx, :] = cached_reshaped[:, :, head_idx, :]
                patched_hidden = hidden_reshaped.view(batch, seq, n_embd)
                return (patched_hidden,)

            handle_patch = attn_layer.c_proj.register_forward_pre_hook(patch_hook)
            with torch.no_grad():
                patched_outputs = model(**clean_inputs)
            handle_patch.remove()

            diff = logit_difference(patched_outputs.logits, mary_id, john_id)
            results[layer_idx].append((head_idx, diff))

            drop = baseline_clean_diff - diff
            if abs(drop) > 0.5:
                print(f"Layer {layer_idx}, Head {head_idx}: Logit Diff = {diff:.4f} (Drop = {drop:.4f}) -> Potential Head")

if __name__ == "__main__":
    run_experiment()
