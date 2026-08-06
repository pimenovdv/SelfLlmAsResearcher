import os
import torch
import torch.nn.functional as F
from src.experiment_utils import load_model_and_tokenizer, clear_memory
import sys

# Ensure sandbox templates are setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.metrics import brier_score, cross_entropy, top_k_accuracy, mean_reciprocal_rank

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
    model, tokenizer = load_model_and_tokenizer("gpt2")

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

    target_token_id = tokenizer.encode(f"{(target_year % 100) + 1:02d}")[0]
    clean_ce = cross_entropy(clean_outputs.logits, target_token_id)
    clean_bs = brier_score(clean_outputs.logits, target_token_id)
    clean_topk = top_k_accuracy(clean_outputs.logits, target_token_id, k=5)
    clean_mrr = mean_reciprocal_rank(clean_outputs.logits, target_token_id)
    corrupt_ce = cross_entropy(corrupt_outputs.logits, target_token_id)
    corrupt_bs = brier_score(corrupt_outputs.logits, target_token_id)
    corrupt_topk = top_k_accuracy(corrupt_outputs.logits, target_token_id, k=5)
    corrupt_mrr = mean_reciprocal_rank(corrupt_outputs.logits, target_token_id)

    print(f"Baseline clean CE: {clean_ce:.4f} | BS: {clean_bs:.4f} | Top5: {clean_topk:.4f} | MRR: {clean_mrr:.4f}")
    print(f"Baseline corrupt CE: {corrupt_ce:.4f} | BS: {corrupt_bs:.4f} | Top5: {corrupt_topk:.4f} | MRR: {corrupt_mrr:.4f}")

    diff_max = clean_greater - corrupt_greater
    print(f"Max diff: {diff_max:.4f}\n")

    num_layers = model.config.n_layer

    print("Patching MLP layers one by one...")
    for layer_idx in range(num_layers):
        clear_memory()
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

        patched_ce = cross_entropy(patched_outputs.logits, target_token_id)
        patched_bs = brier_score(patched_outputs.logits, target_token_id)
        patched_topk = top_k_accuracy(patched_outputs.logits, target_token_id, k=5)
        patched_mrr = mean_reciprocal_rank(patched_outputs.logits, target_token_id)

        recovery = (patched_greater - corrupt_greater) / diff_max if diff_max > 0 else 0.0

        if recovery > 0.05 or recovery < -0.05:
            print(f"Layer {layer_idx:02d} | Patched prob > 32: {patched_greater:.4f} | Recovery: {recovery:.2%} | CE: {patched_ce:.4f} | BS: {patched_bs:.4f} | Top5: {patched_topk:.4f} | MRR: {patched_mrr:.4f}")

if __name__ == "__main__":
    run_experiment()
