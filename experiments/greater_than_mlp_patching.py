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
    corrupt_prompt = "The war lasted from the year 1711 to the year 17"
    target_year = 1732

    clean_inputs = tokenizer(clean_prompt, return_tensors="pt")
    corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt")

    # Baseline probabilities
    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupt_outputs = model(**corrupt_inputs)

    clean_probs = F.softmax(clean_outputs.logits[0, -1, :], dim=-1)
    clean_greater, _ = get_greater_than_probs(clean_probs, tokenizer, target_year)

    corrupt_probs = F.softmax(corrupt_outputs.logits[0, -1, :], dim=-1)
    corrupt_greater, _ = get_greater_than_probs(corrupt_probs, tokenizer, target_year)

    diff_max = clean_greater - corrupt_greater
    print(f"Max diff: {diff_max:.4f}\n")

    num_layers = model.config.n_layer

    print("Patching MLP layers one by one...")
    for layer_idx in range(num_layers):
        cached_mlp_activation = None

        def cache_pre_hook(module, args):
            nonlocal cached_mlp_activation
            cached_mlp_activation = args[0].detach().clone()
            return args

        layer_to_patch = model.transformer.h[layer_idx].mlp.c_proj
        cache_handle = layer_to_patch.register_forward_pre_hook(cache_pre_hook)

        with torch.no_grad():
            model(**clean_inputs)

        cache_handle.remove()

        def patch_pre_hook(module, args):
            return (cached_mlp_activation,)

        patch_handle = layer_to_patch.register_forward_pre_hook(patch_pre_hook)

        with torch.no_grad():
            patched_outputs = model(**corrupt_inputs)

        patch_handle.remove()

        patched_probs = F.softmax(patched_outputs.logits[0, -1, :], dim=-1)
        patched_greater, _ = get_greater_than_probs(patched_probs, tokenizer, target_year)

        recovery = (patched_greater - corrupt_greater) / diff_max if diff_max > 0 else 0.0

        if recovery > 0.05 or recovery < -0.05:
            print(f"Layer {layer_idx:02d} | Patched prob > 32: {patched_greater:.4f} | Recovery: {recovery:.2%}")

if __name__ == "__main__":
    run_experiment()
