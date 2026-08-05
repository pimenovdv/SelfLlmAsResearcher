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

def run_experiment(model_name="gpt2"):
    print(f"\n--- Running Positional MLP Patching for {model_name} ---")
    model, tokenizer = load_model_and_tokenizer(model_name)

    clean_text = "The capital of France is"
    corrupted_text = "The capital of Russia is"

    clean_inputs = tokenizer(clean_text, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt")

    tokens = tokenizer.convert_ids_to_tokens(clean_inputs.input_ids[0])
    print(f"Clean tokens: {tokens}")

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
    is_llama = "llama" in model_name.lower() or "tinyllama" in model_name.lower()

    positions_to_patch = [3, 4]
    position_names = {3: "Subject (France)", 4: "END (is)"}

    print("\nLayer\tPosition\tLogit Diff")

    for layer_idx in range(num_layers):
        if is_llama:
            mlp_layer = model.model.layers[layer_idx].mlp
        else:
            mlp_layer = model.transformer.h[layer_idx].mlp

        for pos in positions_to_patch:
            cached_mlp = None

            def cache_mlp_hook(module, input, output):
                nonlocal cached_mlp
                if isinstance(output, tuple):
                    cached_mlp = output[0].detach().clone()
                else:
                    cached_mlp = output.detach().clone()

            handle_cache = mlp_layer.register_forward_hook(cache_mlp_hook)
            with torch.no_grad():
                model(**corrupted_inputs)
            handle_cache.remove()

            def patch_mlp_hook(module, input, output):
                if isinstance(output, tuple):
                    patched_output = output[0].clone()
                    patched_output[:, pos, :] = cached_mlp[:, pos, :]
                    return (patched_output,) + output[1:]
                else:
                    patched_output = output.clone()
                    patched_output[:, pos, :] = cached_mlp[:, pos, :]
                    return patched_output

            handle_patch = mlp_layer.register_forward_hook(patch_mlp_hook)
            with torch.no_grad():
                patched_outputs = model(**clean_inputs)
            handle_patch.remove()

            diff = logit_difference(patched_outputs.logits, target_id_clean, target_id_corrupted)
            print(f"{layer_idx}\t{position_names[pos]}\t{diff:.4f}")

if __name__ == "__main__":
    for model_name in ["gpt2", "EleutherAI/gpt-neo-125m"]:
        run_experiment(model_name)
