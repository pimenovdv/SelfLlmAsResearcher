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
    print(f"\n--- Running Early Layers Subject Patching for {model_name} ---")
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
        print("Llama architecture not tested in this early layers script.")
        return

    pos_to_patch = 3 # Subject token
    print(f"\nPatching at Subject Position ({pos_to_patch}) - Early Layers (0-7)...")

    print("\n--- Early Attention Heads (Subject Pos Patching) ---")
    head_results = {}
    for layer_idx in range(8): # early layers 0-7
        attn_layer = model.transformer.h[layer_idx].attn
        head_results[layer_idx] = []

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

                # Patch just the specific head at specific pos
                hidden_reshaped[:, pos_to_patch, head_idx, :] = cached_reshaped[:, pos_to_patch, head_idx, :]

                patched_hidden = hidden_reshaped.view(batch, seq, n_embd)
                return (patched_hidden,)

            handle_patch = attn_layer.c_proj.register_forward_pre_hook(patch_hook)
            with torch.no_grad():
                patched_outputs = model(**clean_inputs)
            handle_patch.remove()

            diff = logit_difference(patched_outputs.logits, target_id_clean, target_id_corrupted)
            head_results[layer_idx].append((head_idx, diff))

            drop = baseline_clean_diff - diff
            if drop > 0.1:
                print(f"Layer {layer_idx}, Head {head_idx:2d}: Logit Diff = {diff:.4f} (Drop = {drop:.4f})")

    print("\n--- Early MLPs (Subject Pos Patching) ---")
    mlp_results = []
    for layer_idx in range(8):
        mlp_layer = model.transformer.h[layer_idx].mlp
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
                patched_output[:, pos_to_patch, :] = cached_mlp[:, pos_to_patch, :]
                return (patched_output,) + output[1:]
            else:
                patched_output = output.clone()
                patched_output[:, pos_to_patch, :] = cached_mlp[:, pos_to_patch, :]
                return patched_output

        handle_patch = mlp_layer.register_forward_hook(patch_mlp_hook)
        with torch.no_grad():
            patched_outputs = model(**clean_inputs)
        handle_patch.remove()

        diff = logit_difference(patched_outputs.logits, target_id_clean, target_id_corrupted)
        mlp_results.append((layer_idx, diff))
        drop = baseline_clean_diff - diff
        if drop > 0.1:
            print(f"Layer {layer_idx} MLP: Logit Diff = {diff:.4f} (Drop = {drop:.4f})")

if __name__ == "__main__":
    for model_name in ["gpt2"]:
        run_experiment(model_name)
