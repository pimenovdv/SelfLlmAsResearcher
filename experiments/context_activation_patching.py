import sys
import os
import torch

sys.path.append(os.path.abspath("."))
from src.experiment_utils import load_model_and_tokenizer, clear_memory
from src.metrics import logit_difference, brier_score, top_k_accuracy, mean_reciprocal_rank

def run_experiment():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer("gpt2")

    clean_text = "When John and Mary went to the store, John gave a drink to"
    corrupted_text = "When John and Mary went to the store, Mary gave a drink to"

    context = "Here is a story about some friends. "
    clean_text_ctx = context + clean_text
    corrupted_text_ctx = context + corrupted_text

    clean_inputs = tokenizer(clean_text, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt")

    clean_inputs_ctx = tokenizer(clean_text_ctx, return_tensors="pt")
    corrupted_inputs_ctx = tokenizer(corrupted_text_ctx, return_tensors="pt")

    mary_id = tokenizer.encode(" Mary")[0]
    john_id = tokenizer.encode(" John")[0]

    def get_patching_results(clean_inp, corrupted_inp):
        with torch.no_grad():
            baseline_corrupted_logits = model(**corrupted_inp).logits

        baseline_corrupted_diff = logit_difference(baseline_corrupted_logits, mary_id, john_id)
        baseline_corr_tk = top_k_accuracy(baseline_corrupted_logits, mary_id)
        baseline_corr_mrr = mean_reciprocal_rank(baseline_corrupted_logits, mary_id)

        num_layers = model.config.n_layer
        attn_diffs = []
        attn_tk = []
        attn_mrr = []

        for layer_idx in range(num_layers):
            clear_memory()
            attn_layer = model.transformer.h[layer_idx].attn
            cached_attn = None

            def cache_attn_hook(module, input, output):
                nonlocal cached_attn
                cached_attn = output[0].detach().clone()

            handle_cache_attn = attn_layer.register_forward_hook(cache_attn_hook)
            with torch.no_grad():
                model(**corrupted_inp)
            handle_cache_attn.remove()

            def patch_attn_hook(module, input, output):
                return (cached_attn,) + output[1:]

            handle_patch_attn = attn_layer.register_forward_hook(patch_attn_hook)
            with torch.no_grad():
                patched_outputs = model(**clean_inp)
            handle_patch_attn.remove()

            diff = logit_difference(patched_outputs.logits, mary_id, john_id)
            tk = top_k_accuracy(patched_outputs.logits, mary_id)
            mrr = mean_reciprocal_rank(patched_outputs.logits, mary_id)
            attn_diffs.append(diff)
            attn_tk.append(tk)
            attn_mrr.append(mrr)

        return baseline_corrupted_diff, baseline_corr_tk, baseline_corr_mrr, attn_diffs, attn_tk, attn_mrr

    print("Running patching without context...")
    base_corr_diff, base_corr_tk, base_corr_mrr, base_diffs, base_tk, base_mrr = get_patching_results(clean_inputs, corrupted_inputs)

    print("Running patching with context...")
    ctx_corr_diff, ctx_corr_tk, ctx_corr_mrr, ctx_diffs, ctx_tk, ctx_mrr = get_patching_results(clean_inputs_ctx, corrupted_inputs_ctx)

    print(f"\nBaseline Corrupted Diff (No Context): {base_corr_diff:.4f} (Top-K: {base_corr_tk}, MRR: {base_corr_mrr:.4f})")
    print(f"Baseline Corrupted Diff (With Context): {ctx_corr_diff:.4f} (Top-K: {ctx_corr_tk}, MRR: {ctx_corr_mrr:.4f})")

    print("\nLayer\tNo Ctx Diff\tNo Ctx Top-K\tNo Ctx MRR\tWith Ctx Diff\tWith Ctx Top-K\tWith Ctx MRR")
    for i in range(model.config.n_layer):
        print(f"{i}\t{base_diffs[i]:.4f}\t\t{base_tk[i]}\t\t{base_mrr[i]:.4f}\t\t{ctx_diffs[i]:.4f}\t\t{ctx_tk[i]}\t\t{ctx_mrr[i]:.4f}")

if __name__ == "__main__":
    run_experiment()
