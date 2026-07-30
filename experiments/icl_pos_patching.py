import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if not os.path.exists("agent_workspace/templates/metrics.py"):
    from src.sandbox_env import SandboxEnvironment
    env = SandboxEnvironment()
    env.setup_templates()

sys.path.append(os.path.abspath("agent_workspace"))
from templates.metrics import logit_difference

def main():
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    clean = "apple -> red, banana -> yellow, grass -> green, sky ->"
    corr = "apple -> dog, banana -> cat, grass -> bird, sky ->"

    inputs_clean = tokenizer(clean, return_tensors="pt")
    inputs_corr = tokenizer(corr, return_tensors="pt")

    blue_id = tokenizer.encode(" blue")[0]
    bird_id = tokenizer.encode(" bird")[0]

    with torch.no_grad():
        clean_logits = model(**inputs_clean).logits
        corr_logits = model(**inputs_corr).logits

    clean_diff = logit_difference(clean_logits, blue_id, bird_id)
    corr_diff = logit_difference(corr_logits, blue_id, bird_id)

    print(f"Clean Logit Diff: {clean_diff:.4f}")
    print(f"Corr Logit Diff: {corr_diff:.4f}")

    num_layers = model.config.n_layer
    seq_len = inputs_clean.input_ids.shape[1]

    tokens = [tokenizer.decode([t]) for t in inputs_clean.input_ids[0]]
    print("Tokens:", tokens)

    print("\nPatching Clean Residual Stream into Corrupted Run by Position and Layer")

    clean_activations = {}
    def save_hook(name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            clean_activations[name] = out.clone()
            return output
        return hook

    def patch_hook(name, pos):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            out[:, pos, :] = clean_activations[name][:, pos, :]
            return (out,) if isinstance(output, tuple) else out
        return hook

    for layer_idx in range(num_layers):
        for pos in range(seq_len):
            name = f"L{layer_idx}"

            handle1 = model.transformer.h[layer_idx].register_forward_hook(save_hook(name))
            with torch.no_grad():
                model(**inputs_clean)
            handle1.remove()

            handle2 = model.transformer.h[layer_idx].register_forward_hook(patch_hook(name, pos))
            with torch.no_grad():
                patched_logits = model(**inputs_corr).logits
            handle2.remove()

            patched_diff = logit_difference(patched_logits, blue_id, bird_id)
            recovered = (patched_diff - corr_diff) / (clean_diff - corr_diff)
            if recovered > 0.1: # threshold
                print(f"L{layer_idx} Pos {pos} ('{tokens[pos]}'): Recovered = {recovered:.2%}, Logit Diff = {patched_diff:.4f}")

if __name__ == "__main__":
    main()
