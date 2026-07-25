import sys
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Ensure templates exist
if not os.path.exists("agent_workspace/templates/metrics.py"):
    from src.sandbox_env import SandboxEnvironment
    env = SandboxEnvironment()
    env.setup_templates()

sys.path.append(os.path.abspath("agent_workspace"))
from templates.metrics import logit_difference

def run_experiment():
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    clean_text = "When John and Mary went to the store, John gave a drink to"
    clean_inputs = tokenizer(clean_text, return_tensors="pt")

    mary_id = tokenizer.encode(" Mary")[0]
    john_id = tokenizer.encode(" John")[0]

    with torch.no_grad():
        clean_outputs = model(**clean_inputs)

    baseline_clean_diff = logit_difference(clean_outputs.logits, mary_id, john_id)
    print(f"Baseline Clean Logit Diff (Mary - John): {baseline_clean_diff:.4f}")

    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads

    # Target heads: (Layer, Head)
    target_heads = [(7, 3), (7, 9), (8, 6), (8, 10)]

    print("\nEvaluating Individual Zero-Ablation...")
    individual_results = {}

    for layer_idx, head_idx in target_heads:
        attn_layer = model.transformer.h[layer_idx].attn

        def single_ablation_hook(module, args):
            hidden = args[0].clone()
            batch, seq, n_embd = hidden.shape

            # Reshape to separate heads
            hidden_reshaped = hidden.view(batch, seq, num_heads, head_dim)

            # Zero out the specific head
            hidden_reshaped[:, :, head_idx, :] = 0.0

            # Reshape back
            ablated_hidden = hidden_reshaped.view(batch, seq, n_embd)
            return (ablated_hidden,)

        handle = attn_layer.c_proj.register_forward_pre_hook(single_ablation_hook)
        with torch.no_grad():
            ablated_outputs = model(**clean_inputs)
        handle.remove()

        diff = logit_difference(ablated_outputs.logits, mary_id, john_id)
        drop = baseline_clean_diff - diff
        individual_results[(layer_idx, head_idx)] = (diff, drop)
        print(f"Ablating L{layer_idx}H{head_idx:2d}: Logit Diff = {diff:.4f} (Drop = {drop:.4f})")

    print("\nEvaluating Cumulative Zero-Ablation...")

    handles = []

    # We need to create a hook for each layer involved in cumulative ablation
    layers_to_ablate = {}
    for layer_idx, head_idx in target_heads:
        if layer_idx not in layers_to_ablate:
            layers_to_ablate[layer_idx] = []
        layers_to_ablate[layer_idx].append(head_idx)

    for layer_idx, heads in layers_to_ablate.items():
        attn_layer = model.transformer.h[layer_idx].attn

        # We need a closure that captures the specific 'heads' list for this layer
        def get_multi_ablation_hook(heads_to_ablate):
            def multi_ablation_hook(module, args):
                hidden = args[0].clone()
                batch, seq, n_embd = hidden.shape

                # Reshape to separate heads
                hidden_reshaped = hidden.view(batch, seq, num_heads, head_dim)

                # Zero out the specific heads
                for h_idx in heads_to_ablate:
                    hidden_reshaped[:, :, h_idx, :] = 0.0

                # Reshape back
                ablated_hidden = hidden_reshaped.view(batch, seq, n_embd)
                return (ablated_hidden,)
            return multi_ablation_hook

        hook_func = get_multi_ablation_hook(heads)
        handle = attn_layer.c_proj.register_forward_pre_hook(hook_func)
        handles.append(handle)

    with torch.no_grad():
        cumulative_outputs = model(**clean_inputs)

    for handle in handles:
        handle.remove()

    cumulative_diff = logit_difference(cumulative_outputs.logits, mary_id, john_id)
    cumulative_drop = baseline_clean_diff - cumulative_diff

    print(f"Cumulative Ablation of {target_heads}:")
    print(f"Logit Diff = {cumulative_diff:.4f} (Drop = {cumulative_drop:.4f})")

    print("\nSummary:")
    print(f"Baseline: {baseline_clean_diff:.4f}")
    sum_individual_drops = sum([drop for diff, drop in individual_results.values()])
    print(f"Sum of Individual Drops: {sum_individual_drops:.4f}")
    print(f"Cumulative Drop: {cumulative_drop:.4f}")

if __name__ == "__main__":
    run_experiment()
