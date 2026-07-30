import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure templates exist
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

    print(f"Clean Logit Diff (blue - bird): {clean_diff:.4f}")
    print(f"Corr Logit Diff (blue - bird): {corr_diff:.4f}")

    num_layers = model.config.n_layer

    print("\nStarting Activation Patching (Attn & MLP layers)...")
    print("Goal is to see which layer's patch recovers the prediction of ' blue' when run on corrupted input")
    print("\nLayer\tAttn Patch\tMLP Patch")

    for layer_idx in range(num_layers):
        attn_layer = model.transformer.h[layer_idx].attn
        mlp_layer = model.transformer.h[layer_idx].mlp

        # Attn Patch
        clean_attn_act = None
        def save_attn_hook(module, input, output):
            nonlocal clean_attn_act
            if isinstance(output, tuple):
                clean_attn_act = output[0].clone()
            else:
                clean_attn_act = output.clone()
            return output

        def patch_attn_hook(module, input, output):
            if isinstance(output, tuple):
                return (clean_attn_act,) + output[1:]
            else:
                return clean_attn_act

        handle1 = attn_layer.register_forward_hook(save_attn_hook)
        with torch.no_grad():
            model(**inputs_clean)
        handle1.remove()

        handle2 = attn_layer.register_forward_hook(patch_attn_hook)
        with torch.no_grad():
            patched_attn_logits = model(**inputs_corr).logits
        handle2.remove()

        attn_diff = logit_difference(patched_attn_logits, blue_id, bird_id)

        # MLP Patch
        clean_mlp_act = None
        def save_mlp_hook(module, input, output):
            nonlocal clean_mlp_act
            if isinstance(output, tuple):
                clean_mlp_act = output[0].clone()
            else:
                clean_mlp_act = output.clone()
            return output

        def patch_mlp_hook(module, input, output):
            if isinstance(output, tuple):
                return (clean_mlp_act,) + output[1:]
            else:
                return clean_mlp_act

        handle3 = mlp_layer.register_forward_hook(save_mlp_hook)
        with torch.no_grad():
            model(**inputs_clean)
        handle3.remove()

        handle4 = mlp_layer.register_forward_hook(patch_mlp_hook)
        with torch.no_grad():
            patched_mlp_logits = model(**inputs_corr).logits
        handle4.remove()

        mlp_diff = logit_difference(patched_mlp_logits, blue_id, bird_id)

        print(f"{layer_idx}\t{attn_diff:.4f}\t\t{mlp_diff:.4f}")

if __name__ == "__main__":
    main()
