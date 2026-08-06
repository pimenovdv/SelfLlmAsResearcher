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

    # Token positions (0-indexed)
    # 0: When, 1: John, 2: and, 3: Mary, 4: went, 5: to, 6: the, 7: store, 8:,, 9: John, 10: gave, 11: a, 12: drink, 13: to
    # Pos 1: Subject 1 (S1)
    # Pos 3: Indirect Object (IO)
    # Pos 9: Subject 2 (S2)
    # Pos 13: END token ("to")

    mary_id = tokenizer.encode(" Mary")[0]
    john_id = tokenizer.encode(" John")[0]

    # Baseline forward passes
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
    seq_len = clean_inputs.input_ids.shape[1]

    print("\nStarting Positional MLP Activation Patching...")

    positions_to_patch = [1, 3, 9, 13] # S1, IO, S2, END
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

            # Cache corrupted
            handle_cache = mlp_layer.register_forward_hook(cache_mlp_hook)
            with torch.no_grad():
                model(**corrupted_inputs)
            handle_cache.remove()

            def patch_mlp_hook(module, input, output):
                # output shape: [batch, seq_len, hidden_size]
                # We only patch the specific position
                patched_output = output.clone()
                patched_output[:, pos, :] = cached_mlp[:, pos, :]
                return patched_output

            # Patch clean run with corrupted cache at specific position
            handle_patch = mlp_layer.register_forward_hook(patch_mlp_hook)
            with torch.no_grad():
                patched_outputs = model(**clean_inputs)
            handle_patch.remove()

            diff = logit_difference(patched_outputs.logits, mary_id, john_id)
            print(f"{layer_idx}\t{position_names[pos]}\t{diff:.4f}")

if __name__ == "__main__":
    run_experiment()
