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
    target_year_clean = 1732

    clean_inputs = tokenizer(clean_prompt, return_tensors="pt")

    # Baseline probabilities
    with torch.no_grad():
        clean_outputs = model(**clean_inputs)

    clean_probs = F.softmax(clean_outputs.logits[0, -1, :], dim=-1)
    clean_greater, clean_le = get_greater_than_probs(clean_probs, tokenizer, target_year_clean)

    print(f"Clean prompt: '{clean_prompt}'")
    print(f"Target year threshold: > 32")
    print(f"Baseline clean prob > 32: {clean_greater:.4f}\n")

    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    print("Ablating individual attention heads on layers 5-11")
    print("Format: (Layer, Head) | Prob > 32 | Drop in Prob")
    print("-" * 55)

    results = []

    for layer_idx in range(5, 12):
        for head_idx in range(num_heads):

            def get_ablation_hook(head_to_ablate):
                def hook(module, args):
                    hidden_states = args[0]
                    batch_size, seq_len, n_embd = hidden_states.shape
                    head_dim = n_embd // num_heads

                    # Reshape to [batch, seq_len, num_heads, head_dim]
                    hidden_states = hidden_states.view(batch_size, seq_len, num_heads, head_dim).clone()

                    # Zero out the specific head
                    hidden_states[:, :, head_to_ablate, :] = 0.0

                    # Reshape back
                    hidden_states = hidden_states.view(batch_size, seq_len, n_embd)

                    return (hidden_states,)
                return hook

            layer_to_ablate = model.transformer.h[layer_idx].attn.c_proj
            handle = layer_to_ablate.register_forward_pre_hook(get_ablation_hook(head_idx))

            with torch.no_grad():
                outputs = model(**clean_inputs)

            handle.remove()

            probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
            greater_prob, _ = get_greater_than_probs(probs, tokenizer, target_year_clean)
            drop = clean_greater - greater_prob

            results.append({
                'layer': layer_idx,
                'head': head_idx,
                'prob': greater_prob,
                'drop': drop
            })

    # Sort results by drop in probability
    results.sort(key=lambda x: x['drop'], reverse=True)

    print("Top 15 heads with highest drop when ablated:")
    for res in results[:15]:
        print(f"L{res['layer']:02d}H{res['head']:02d} | Prob: {res['prob']:>6.4f} | Drop: {res['drop']:>6.4f}")

if __name__ == "__main__":
    run_experiment()
