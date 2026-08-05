import os
import sys
import torch
from src.experiment_utils import load_model_and_tokenizer

def main():
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    prompt = "apple -> red, banana -> yellow, grass -> green, sky ->"
    inputs = tokenizer(prompt, return_tensors="pt")

    target_word = " blue"
    target_id = tokenizer.encode(target_word)[0]

    unembed_weight = model.lm_head.weight[target_id]

    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach().cpu()
            else:
                activations[name] = output.detach().cpu()
        return hook

    # Register hooks on attention outputs of all layers
    handles = []
    for layer_idx in range(model.config.n_layer):
        handle = model.transformer.h[layer_idx].attn.register_forward_hook(get_activation(f'attn_out_l{layer_idx}'))
        handles.append(handle)

    with torch.no_grad():
        outputs = model(**inputs)

    # Clean up hooks
    for handle in handles:
        handle.remove()

    print("\n--- Direct Logit Attribution (DLA) Scores for ' blue' ---")
    for layer_idx in range(model.config.n_layer):
        attn_out = activations[f'attn_out_l{layer_idx}'][0, -1, :]
        dla_score = torch.dot(attn_out, unembed_weight).item()
        print(f"Layer {layer_idx:2d} DLA: {dla_score:8.4f}")

if __name__ == "__main__":
    main()
