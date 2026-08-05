import torch
from src.experiment_utils import load_model_and_tokenizer
import sys, os

if not os.path.exists("agent_workspace/templates/metrics.py"):
    sys.path.append(".")
    from src.sandbox_env import SandboxEnvironment
    env = SandboxEnvironment()
    env.setup_templates()

sys.path.append(os.path.abspath("agent_workspace"))
from templates.metrics import logit_difference

def test_model(model_name):
    model, tokenizer = load_model_and_tokenizer(model_name)

    clean_text = "The capital of France is"
    corrupted_text = "The capital of Russia is"

    clean_inputs = tokenizer(clean_text, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt")

    # ensure single token target matching
    target_id_clean = tokenizer.encode(" Paris", add_special_tokens=False)[-1]
    target_id_corrupted = tokenizer.encode(" Moscow", add_special_tokens=False)[-1]

    with torch.no_grad():
        clean_out = model(**clean_inputs)
        corr_out = model(**corrupted_inputs)

    diff_clean = logit_difference(clean_out.logits, target_id_clean, target_id_corrupted)
    diff_corr = logit_difference(corr_out.logits, target_id_clean, target_id_corrupted)
    print(f"--- {model_name} ---")
    print(f"Clean Diff: {diff_clean:.4f}, Corr Diff: {diff_corr:.4f}")

    num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else model.config.n_layer
    num_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else model.config.n_head
    head_dim = (model.config.hidden_size if hasattr(model.config, 'hidden_size') else model.config.n_embd) // num_heads

    is_llama = "llama" in model_name.lower() or "tinyllama" in model_name.lower()
    is_neo = "neo" in model_name.lower()

    # MLP Patching Layer 0
    if is_llama:
        mlp_layer = model.model.layers[0].mlp
    else:
        mlp_layer = model.transformer.h[0].mlp

    cached_mlp = None
    def cache_mlp(m, i, o):
        nonlocal cached_mlp
        cached_mlp = o[0].detach().clone() if isinstance(o, tuple) else o.detach().clone()
    h = mlp_layer.register_forward_hook(cache_mlp)
    with torch.no_grad(): model(**corrupted_inputs)
    h.remove()

    def patch_mlp(m, i, o):
        return (cached_mlp,) + o[1:] if isinstance(o, tuple) else cached_mlp
    h = mlp_layer.register_forward_hook(patch_mlp)
    with torch.no_grad(): patched_out = model(**clean_inputs)
    h.remove()
    diff = logit_difference(patched_out.logits, target_id_clean, target_id_corrupted)
    print(f"L0 MLP Patch Diff: {diff:.4f} (Drop {diff_clean - diff:.4f})")

    # Head Patching
    results = []
    for l in range(num_layers):
        if is_llama:
            attn_layer = model.model.layers[l].self_attn
            proj_layer = attn_layer.o_proj
        elif is_neo:
            attn_layer = model.transformer.h[l].attn.attention
            proj_layer = attn_layer.out_proj
        else:
            attn_layer = model.transformer.h[l].attn
            proj_layer = attn_layer.c_proj

        for h_idx in range(num_heads):
            cached_head = None
            def cache_head(m, args):
                nonlocal cached_head
                cached_head = args[0].detach().clone()
                return args
            h_hook = proj_layer.register_forward_pre_hook(cache_head)
            with torch.no_grad(): model(**corrupted_inputs)
            h_hook.remove()

            def patch_head(m, args):
                hidden = args[0].clone()
                batch, seq, n_embd = hidden.shape
                hidden_reshaped = hidden.view(batch, seq, num_heads, head_dim)
                cached_reshaped = cached_head.view(batch, seq, num_heads, head_dim)
                hidden_reshaped[:, :, h_idx, :] = cached_reshaped[:, :, h_idx, :]
                return (hidden_reshaped.view(batch, seq, n_embd),)

            p_hook = proj_layer.register_forward_pre_hook(patch_head)
            with torch.no_grad(): p_out = model(**clean_inputs)
            p_hook.remove()

            diff = logit_difference(p_out.logits, target_id_clean, target_id_corrupted)
            drop = diff_clean - diff
            if drop > 0.5:
                results.append((l, h_idx, diff, drop))

    for l, h_idx, diff, drop in sorted(results, key=lambda x: x[3], reverse=True):
        print(f"Head L{l}H{h_idx}: Diff {diff:.4f} (Drop {drop:.4f})")

test_model("gpt2")
test_model("EleutherAI/gpt-neo-125m")
test_model("JackFram/llama-160m")
