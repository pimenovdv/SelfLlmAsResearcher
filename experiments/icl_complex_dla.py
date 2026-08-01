import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from einops import rearrange
from src.sandbox_env import SandboxEnvironment

def logit_difference(logits, target_id, corrupted_id):
    next_token_logits = logits[0, -1, :]
    target_logit = next_token_logits[target_id].item()
    corrupted_logit = next_token_logits[corrupted_id].item()
    return target_logit - corrupted_logit

def main():
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    prompt = "Q: en - cat, fr - chat. Q: en - dog, fr - chien. Q: en - horse, fr -"
    corr_prompt = "Q: en - dog, fr - chien. Q: en - horse, fr - cheval. Q: en - cat, fr -"

    inputs = tokenizer(prompt, return_tensors="pt")
    corr_inputs = tokenizer(corr_prompt, return_tensors="pt")

    target_word = " cheval"
    corrupted_word = " chat"

    target_id = tokenizer.encode(target_word)[0]
    corrupted_id = tokenizer.encode(corrupted_word)[0]

    unembed_weight_target = model.lm_head.weight[target_id]
    unembed_weight_corr = model.lm_head.weight[corrupted_id]
    unembed_diff = unembed_weight_target - unembed_weight_corr

    with torch.no_grad():
        clean_logits = model(**inputs).logits
        corr_logits = model(**corr_inputs).logits

    clean_diff = logit_difference(clean_logits, target_id, corrupted_id)
    corr_diff = logit_difference(corr_logits, target_id, corrupted_id)

    print(f"Clean Logit Diff (cheval - chat): {clean_diff:.4f}")
    print(f"Corr Logit Diff (cheval - chat): {corr_diff:.4f}")

    activations = {}
    def get_activation(name):
        def hook(module, input):
            activations[name] = input[0].detach().cpu()
        return hook

    handles = []
    for layer_idx in range(model.config.n_layer):
        handle = model.transformer.h[layer_idx].attn.c_proj.register_forward_pre_hook(get_activation(f'attn_c_proj_in_l{layer_idx}'))
        handles.append(handle)

    with torch.no_grad():
        outputs = model(**inputs)

    for handle in handles:
        handle.remove()

    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads

    print(f"\n--- Direct Logit Attribution (DLA) Scores by Head for '{target_word}' - '{corrupted_word}' ---")

    top_head = None
    max_dla = -1e9

    for layer_idx in range(model.config.n_layer):
        c_proj_in = activations[f'attn_c_proj_in_l{layer_idx}'][0, -1, :]
        c_proj_in_heads = rearrange(c_proj_in, '(h d) -> h d', h=num_heads, d=head_dim)
        c_proj_weight = model.transformer.h[layer_idx].attn.c_proj.weight
        c_proj_weight_heads = rearrange(c_proj_weight, '(h d) out -> h d out', h=num_heads, d=head_dim)

        for head_idx in range(num_heads):
            head_out_proj = c_proj_in_heads[head_idx] @ c_proj_weight_heads[head_idx]
            dla_score = torch.dot(head_out_proj, unembed_diff).item()
            if dla_score > 1.0:
                print(f"Layer {layer_idx:2d} Head {head_idx:2d} DLA: {dla_score:8.4f}")
            if dla_score > max_dla:
                max_dla = dla_score
                top_head = (layer_idx, head_idx)

    # Note: L11H0 is predicting the word directly. In ICL, it's often the induction heads (L5-L10)
    # that read from early layers and move the information to the last token.
    # Let's find the top DLA head in the mid-to-late layers (excluding the very last layer).
    top_induction_head = None
    max_ind_dla = -1e9
    for layer_idx in range(4, model.config.n_layer - 1):
        c_proj_in = activations[f'attn_c_proj_in_l{layer_idx}'][0, -1, :]
        c_proj_in_heads = rearrange(c_proj_in, '(h d) -> h d', h=num_heads, d=head_dim)
        c_proj_weight = model.transformer.h[layer_idx].attn.c_proj.weight
        c_proj_weight_heads = rearrange(c_proj_weight, '(h d) out -> h d out', h=num_heads, d=head_dim)
        for head_idx in range(num_heads):
            head_out_proj = c_proj_in_heads[head_idx] @ c_proj_weight_heads[head_idx]
            dla_score = torch.dot(head_out_proj, unembed_diff).item()
            if dla_score > max_ind_dla:
                max_ind_dla = dla_score
                top_induction_head = (layer_idx, head_idx)

    print(f"\n--- Positional Patching on top Mid-Layer DLA head (Layer {top_induction_head[0]}, Head {top_induction_head[1]}) ---")

    target_layer = top_induction_head[0]
    target_head = top_induction_head[1]

    seq_len = inputs.input_ids.shape[1]

    # We patch at the LAST sequence position, because that's where the head writes to the residual stream.
    # BUT we want to see where it attends TO. Actually, positional patching of the head's *output* at position -1
    # will just recover the DLA score.
    # To find WHERE it gets its information, we should patch the residual stream *input* to that head at all positions!

    # Let's do positional patching on the residual stream input to the target head.

    for pos in range(seq_len):
        clean_resid = None
        def save_resid_hook(module, input):
            nonlocal clean_resid
            clean_resid = input[0].clone()

        def patch_resid_hook(module, input):
            patched_input = input[0].clone()
            # Patch the residual stream at a specific position
            patched_input[0, pos, :] = clean_resid[0, pos, :]
            return (patched_input,)

        handle_save = model.transformer.h[target_layer].register_forward_pre_hook(save_resid_hook)
        with torch.no_grad():
            model(**inputs)
        handle_save.remove()

        handle_patch = model.transformer.h[target_layer].register_forward_pre_hook(patch_resid_hook)
        with torch.no_grad():
            patched_logits = model(**corr_inputs).logits
        handle_patch.remove()

        patched_diff = logit_difference(patched_logits, target_id, corrupted_id)

        recovery = (patched_diff - corr_diff) / (clean_diff - corr_diff)

        token_str = tokenizer.decode([inputs.input_ids[0, pos].item()])
        print(f"Pos {pos:2d} ('{token_str}'): Recovery {recovery*100:6.2f}%")

if __name__ == "__main__":
    main()
