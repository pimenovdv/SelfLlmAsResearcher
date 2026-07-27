import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# Ensure sandbox templates are setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.sandbox_env import SandboxEnvironment
    env = SandboxEnvironment()
    env.setup_templates()
except ImportError:
    print("Warning: Could not import SandboxEnvironment. Make sure PYTHONPATH is set.")

def get_greater_than_probs(probs, tokenizer, target_year):
    """
    Evaluates the probability of years greater than the target year vs smaller/equal years.
    """
    greater_prob = 0.0
    less_equal_prob = 0.0

    target_decade_year = target_year % 100

    for i in range(100):
        year_str = f"{i:02d}"

        token_id = tokenizer.encode(year_str)
        if len(token_id) == 1:
            token_id = token_id[0]
            prob = probs[token_id].item()

            if i > target_decade_year:
                greater_prob += prob
            else:
                less_equal_prob += prob

    return greater_prob, less_equal_prob

def run_experiment():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    clean_prompt = "The war lasted from the year 1732 to the year 17"
    corrupt_prompt = "The war lasted from the year 1701 to the year 17"
    target_year_clean = 1732

    clean_inputs = tokenizer(clean_prompt, return_tensors="pt")
    corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt")

    # Baseline probabilities
    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupt_outputs = model(**corrupt_inputs)

    clean_probs = F.softmax(clean_outputs.logits[0, -1, :], dim=-1)
    corrupt_probs = F.softmax(corrupt_outputs.logits[0, -1, :], dim=-1)

    clean_greater, clean_le = get_greater_than_probs(clean_probs, tokenizer, target_year_clean)
    corrupt_greater, corrupt_le = get_greater_than_probs(corrupt_probs, tokenizer, target_year_clean)

    print(f"Clean prompt: '{clean_prompt}'")
    print(f"Corrupt prompt: '{corrupt_prompt}'")
    print(f"Target year threshold: > 32")
    print(f"Baseline clean prob > 32: {clean_greater:.4f}")
    print(f"Baseline corrupt prob > 32: {corrupt_greater:.4f}")

    num_layers = model.config.n_layer

    print("\nPatching residual stream (output of block) at the LAST token from clean -> corrupt")
    print("Layer | Prob > 32 (Expect recovery from corrupt prob to clean prob)")
    print("-" * 65)

    for layer_idx in range(num_layers):
        layer_to_patch = model.transformer.h[layer_idx]
        cached_activation = None

        def cache_hook(module, input, output):
            nonlocal cached_activation
            if isinstance(output, tuple):
                cached_activation = output[0].detach().clone()
            else:
                cached_activation = output.detach().clone()

        handle_cache = layer_to_patch.register_forward_hook(cache_hook)
        with torch.no_grad():
            model(**clean_inputs)
        handle_cache.remove()

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                patched_hidden = output[0].clone()
                patched_hidden[:, -1, :] = cached_activation[:, -1, :]
                return (patched_hidden,) + output[1:]
            else:
                patched_hidden = output.clone()
                patched_hidden[:, -1, :] = cached_activation[:, -1, :]
                return patched_hidden

        handle_patch = layer_to_patch.register_forward_hook(patch_hook)
        with torch.no_grad():
            patched_outputs = model(**corrupt_inputs)
        handle_patch.remove()

        patched_probs = F.softmax(patched_outputs.logits[0, -1, :], dim=-1)
        patched_greater, _ = get_greater_than_probs(patched_probs, tokenizer, target_year_clean)

        # Calculate how much of the clean behavior was recovered
        recovery = (patched_greater - corrupt_greater) / (clean_greater - corrupt_greater) if (clean_greater - corrupt_greater) != 0 else 0

        print(f"{layer_idx:5d} | {patched_greater:.4f} | Recovery: {recovery:.2%}")

if __name__ == "__main__":
    run_experiment()
