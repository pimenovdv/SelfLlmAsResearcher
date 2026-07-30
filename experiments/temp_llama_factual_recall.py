import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, os

if not os.path.exists("agent_workspace/templates/metrics.py"):
    sys.path.append(".")
    from src.sandbox_env import SandboxEnvironment
    env = SandboxEnvironment()
    env.setup_templates()

sys.path.append(os.path.abspath("agent_workspace"))
from templates.metrics import logit_difference

def test_llama(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    clean_text = "The capital of France is"
    corrupted_text = "The capital of Russia is"

    clean_inputs = tokenizer(clean_text, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt")

    target_id_clean = tokenizer.encode(" Paris")[0]
    target_id_corrupted = tokenizer.encode(" Moscow")[0]

    with torch.no_grad():
        clean_out = model(**clean_inputs)
        corr_out = model(**corrupted_inputs)

    diff_clean = logit_difference(clean_out.logits, target_id_clean, target_id_corrupted)
    diff_corr = logit_difference(corr_out.logits, target_id_clean, target_id_corrupted)
    print(f"[{model_name}] Clean Diff: {diff_clean:.4f}, Corr Diff: {diff_corr:.4f}")

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # MLP Patching (Layer 0)
    mlp_layer = model.model.layers[0].mlp
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
    print(f"[{model_name}] L0 MLP Patch Diff: {logit_difference(patched_out.logits, target_id_clean, target_id_corrupted):.4f}")

    # Head Patching
    results = []
    for l in range(num_layers):
        attn_layer = model.model.layers[l].self_attn
        for h_idx in range(num_heads):
            cached_head = None
            def cache_head(m, args):
                nonlocal cached_head
                cached_head = args[0].detach().clone()
                return args
            h_hook = attn_layer.o_proj.register_forward_pre_hook(cache_head)
            with torch.no_grad(): model(**corrupted_inputs)
            h_hook.remove()

            def patch_head(m, args):
                hidden = args[0].clone()
                batch, seq, n_embd = hidden.shape
                hidden_reshaped = hidden.view(batch, seq, num_heads, head_dim)
                cached_reshaped = cached_head.view(batch, seq, num_heads, head_dim)
                hidden_reshaped[:, :, h_idx, :] = cached_reshaped[:, :, h_idx, :]
                return (hidden_reshaped.view(batch, seq, n_embd),)

            p_hook = attn_layer.o_proj.register_forward_pre_hook(patch_head)
            with torch.no_grad(): p_out = model(**clean_inputs)
            p_hook.remove()

            diff = logit_difference(p_out.logits, target_id_clean, target_id_corrupted)
            drop = diff_clean - diff
            if drop > 0.5:
                results.append((l, h_idx, diff, drop))

    for l, h_idx, diff, drop in sorted(results, key=lambda x: x[3], reverse=True):
        print(f"[{model_name}] Head L{l}H{h_idx}: Diff {diff:.4f} (Drop {drop:.4f})")

test_llama("JackFram/llama-160m")
