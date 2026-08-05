import sys
import os
import torch
from src.experiment_utils import load_model_and_tokenizer

if not os.path.exists("agent_workspace/templates/metrics.py"):
    from src.sandbox_env import SandboxEnvironment
    env = SandboxEnvironment()
    env.setup_templates()

sys.path.append(os.path.abspath("agent_workspace"))
from templates.metrics import logit_difference

def run_experiment():
    model_name = "JackFram/llama-160m"
    print(f"Loading model and tokenizer for {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    clean_text = "When John and Mary went to the store, John gave a drink to"
    corrupted_text = "When John and Mary went to the store, Mary gave a drink to"

    clean_inputs = tokenizer(clean_text, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt")

    # To find correct positions, we tokenize
    tokens = tokenizer.convert_ids_to_tokens(clean_inputs.input_ids[0])
    print(f"Tokens: {list(enumerate(tokens))}")

    # Expected positions:
    # 2: John (S1)
    # 4: Mary (IO)
    # 10: John (S2)
    # 14: to (END) - the last token index is len(tokens)-1 = 14

    mary_id = tokenizer.encode("Mary", add_special_tokens=False)[0]
    john_id = tokenizer.encode("John", add_special_tokens=False)[0]

    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupted_outputs = model(**corrupted_inputs)

    baseline_clean_diff = logit_difference(clean_outputs.logits, mary_id, john_id)
    baseline_corrupted_diff = logit_difference(corrupted_outputs.logits, mary_id, john_id)

    print(f"Baseline Clean Logit Diff: {baseline_clean_diff:.4f}")
    print(f"Baseline Corrupted Logit Diff: {baseline_corrupted_diff:.4f}")

    num_layers = model.config.num_hidden_layers
    seq_len = clean_inputs.input_ids.shape[1]

    print("\nStarting Positional MLP Activation Patching...")

    # We will patch MLP output

    positions_to_patch = [2, 4, 10, 14]
    position_names = {2: "S1 (John)", 4: "IO (Mary)", 10: "S2 (John)", 14: "END (to)"}

    print("\nLayer\tPosition\tLogit Diff")

    for layer_idx in range(num_layers):
        mlp_layer = model.model.layers[layer_idx].mlp

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
            print(f"{layer_idx}\t{position_names.get(pos, pos)}\t{diff:.4f}")

if __name__ == "__main__":
    run_experiment()
