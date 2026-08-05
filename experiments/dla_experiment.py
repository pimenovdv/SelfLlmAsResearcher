import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.sandbox_env import SandboxEnvironment
from src.experiment_utils import load_model_and_tokenizer

def main():
    # Setup SandboxEnvironment to ensure templates are generated if they aren't
    env = SandboxEnvironment()
    env.setup_templates()

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer("gpt2")

    prompt = "John and Mary went to the store. John gave a bottle to"
    target_word = " Mary"

    print(f"Prompt: {prompt}")
    print(f"Target word: '{target_word}'")

    inputs = tokenizer(prompt, return_tensors="pt")
    target_id = tokenizer.encode(target_word)[0]

    # Get embeddings for the target token (unembedding weight)
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

    print("\n--- Direct Logit Attribution (DLA) Scores ---")
    # Calculate DLA for each layer
    for layer_idx in range(model.config.n_layer):
        # Activation at the last token [batch, seq_len, n_embd]
        attn_out = activations[f'attn_out_l{layer_idx}'][0, -1, :]

        # Calculate DLA = dot product (attn_out, unembed_weight)
        dla_score = torch.dot(attn_out, unembed_weight).item()
        print(f"Layer {layer_idx:2d} DLA: {dla_score:8.4f}")

if __name__ == "__main__":
    main()
