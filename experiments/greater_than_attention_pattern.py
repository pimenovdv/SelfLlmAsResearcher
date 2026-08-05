import os
import torch
from src.experiment_utils import load_model_and_tokenizer
import sys

# Ensure sandbox templates are setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.sandbox_env import SandboxEnvironment
    env = SandboxEnvironment()
    env.setup_templates()
except ImportError:
    print("Warning: Could not import SandboxEnvironment. Make sure PYTHONPATH is set.")

def run_experiment():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer("gpt2", output_attentions=True)

    prompt = "The war lasted from the year 1732 to the year 17"
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0]]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    print(f"Outputs has attentions? {hasattr(outputs, 'attentions') and outputs.attentions is not None}")
    print(f"Number of layers with attention: {len(outputs.attentions) if outputs.attentions else 0}")

    layer_idx = 7
    head_idx = 10

    if outputs.attentions and len(outputs.attentions) > layer_idx:
        attentions = outputs.attentions[layer_idx][0, head_idx]

        print(f"Attention pattern for Layer {layer_idx} Head {head_idx}")
        print("Tokens:", tokens)

        print("\nAttention weights to previous tokens from the last token (' 17'):")
        last_token_attentions = attentions[-1]

        for i, token in enumerate(tokens):
            print(f"  {token:15s}: {last_token_attentions[i].item():.4f}")
    else:
        print(f"Error: Could not retrieve attentions.")

if __name__ == "__main__":
    run_experiment()
