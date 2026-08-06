import sys
import os
import torch
from src.experiment_utils import load_model_and_tokenizer, clear_memory


from src.metrics import logit_difference

def run_experiment():
    model_name = "EleutherAI/gpt-neo-125m"
    print(f"Loading model and tokenizer for {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    clean_text = "When John and Mary went to the store, John gave a drink to"
    corrupted_text = "When John and Mary went to the store, Mary gave a drink to"

    clean_inputs = tokenizer(clean_text, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt")

    # Pos 1: S1 (John)
    # Pos 3: IO (Mary)
    # Pos 9: S2 (John)
    # Pos 13: END (to)

    mary_id = tokenizer.encode(" Mary")[0]
    john_id = tokenizer.encode(" John")[0]

    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupted_outputs = model(**corrupted_inputs)

    baseline_clean_diff = logit_difference(clean_outputs.logits, mary_id, john_id)
    baseline_corrupted_diff = logit_difference(corrupted_outputs.logits, mary_id, john_id)

    print(f"Baseline Clean Logit Diff: {baseline_clean_diff:.4f}")
    print(f"Baseline Corrupted Logit Diff: {baseline_corrupted_diff:.4f}")

    num_layers = model.config.num_layers
    seq_len = clean_inputs.input_ids.shape[1]

    print("\nStarting Positional MLP Activation Patching...")

    positions_to_patch = [1, 3, 9, 13]
    position_names = {1: "S1 (John)", 3: "IO (Mary)", 9: "S2 (John)", 13: "END (to)"}

    print("\nLayer\tPosition\tLogit Diff")

    for layer_idx in range(num_layers):
        clear_memory()
        mlp_layer = model.transformer.h[layer_idx].mlp

        for pos in positions_to_patch:
            cached_mlp = None

            def cache_mlp_hook(module, input, output):
                nonlocal cached_mlp
                cached_mlp = output.detach().clone()

            handle_cache = mlp_layer.register_forward_hook(cache_mlp_hook)
            with torch.no_grad():
                model(**corrupted_inputs)
            handle_cache.remove()

            def patch_mlp_hook(module, input, output):
                patched_output = output.clone()
                patched_output[:, pos, :] = cached_mlp[:, pos, :]
                return patched_output

            handle_patch = mlp_layer.register_forward_hook(patch_mlp_hook)
            with torch.no_grad():
                patched_outputs = model(**clean_inputs)
            handle_patch.remove()

            diff = logit_difference(patched_outputs.logits, mary_id, john_id)
            print(f"{layer_idx}\t{position_names[pos]}\t{diff:.4f}")

if __name__ == "__main__":
    run_experiment()
