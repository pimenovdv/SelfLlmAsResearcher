import sys
import os
import torch
from src.experiment_utils import load_model_and_tokenizer

if not os.path.exists("agent_workspace/templates/metrics.py"):
    from src.sandbox_env import SandboxEnvironment
    env = SandboxEnvironment()
    env.setup_templates()

sys.path.append(os.path.abspath("agent_workspace"))
from templates.metrics import logit_difference

def run_experiment():
    model_name = "JackFram/llama-160m"
    print(f"Loading model and tokenizer for {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    clean_text = "When John and Mary went to the store, John gave a drink to"
    clean_inputs = tokenizer(clean_text, return_tensors="pt")

    mary_id = tokenizer.encode("Mary", add_special_tokens=False)[0]
    john_id = tokenizer.encode("John", add_special_tokens=False)[0]

    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
    baseline_clean_diff = logit_difference(clean_outputs.logits, mary_id, john_id)
    print(f"Baseline Clean Logit Diff (Mary - John): {baseline_clean_diff:.4f}")

    # Name Mover Heads to ablate (based on llama_ioi.py)
    heads_to_ablate = [(9, 5), (10, 6)]

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    hooks = []

    def get_ablation_hook(head_idx):
        def ablation_hook(module, args):
            hidden = args[0].clone()
            batch, seq, n_embd = hidden.shape

            # Reshape to separate heads
            hidden_reshaped = hidden.view(batch, seq, num_heads, head_dim)

            # Zero out the specific head (mean ablation with 0)
            hidden_reshaped[:, :, head_idx, :] = 0

            # Reshape back
            patched_hidden = hidden_reshaped.view(batch, seq, n_embd)
            return (patched_hidden,)
        return ablation_hook

    print(f"\nAblating heads: {heads_to_ablate}")

    for layer_idx, head_idx in heads_to_ablate:
        attn_layer = model.model.layers[layer_idx].self_attn
        hook_fn = get_ablation_hook(head_idx)
        handle = attn_layer.o_proj.register_forward_pre_hook(hook_fn)
        hooks.append(handle)

    with torch.no_grad():
        ablated_outputs = model(**clean_inputs)

    for handle in hooks:
        handle.remove()

    ablated_diff = logit_difference(ablated_outputs.logits, mary_id, john_id)
    print(f"Ablated Logit Diff: {ablated_diff:.4f}")
    print(f"Total Drop: {baseline_clean_diff - ablated_diff:.4f}")

if __name__ == "__main__":
    run_experiment()
