import sys
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Make sure agent_workspace/templates exists
if not os.path.exists("agent_workspace/templates/metrics.py"):
    from src.sandbox_env import SandboxEnvironment
    env = SandboxEnvironment()
    env.setup_templates()

sys.path.append(os.path.abspath("agent_workspace"))
from templates.metrics import logit_difference, kl_divergence

def run_experiment():
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # IOI Task
    clean_text = "When John and Mary went to the store, John gave a drink to"
    corrupted_text = "When John and Mary went to the store, Mary gave a drink to"

    clean_inputs = tokenizer(clean_text, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt")

    print("Running forward passes...")
    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupted_outputs = model(**corrupted_inputs)

    clean_logits = clean_outputs.logits
    corrupted_logits = corrupted_outputs.logits

    # Target IDs
    mary_id = tokenizer.encode(" Mary")[0]
    john_id = tokenizer.encode(" John")[0]

    print(f"\nClean prompt: '{clean_text}'")
    print(f"Corrupted prompt: '{corrupted_text}'")
    print(f"Target 1 (Mary) ID: {mary_id}")
    print(f"Target 2 (John) ID: {john_id}")

    # Logit difference on clean prompt (Mary vs John)
    diff_clean = logit_difference(clean_logits, mary_id, john_id)
    print(f"\nLogit Difference (Mary - John) on clean prompt: {diff_clean:.4f}")

    # Logit difference on corrupted prompt (Mary vs John)
    diff_corrupted = logit_difference(corrupted_logits, mary_id, john_id)
    print(f"Logit Difference (Mary - John) on corrupted prompt: {diff_corrupted:.4f}")

    # KL Divergence between clean and corrupted logits
    kl = kl_divergence(clean_logits, corrupted_logits)
    print(f"\nKL Divergence between clean and corrupted model outputs: {kl:.4f}")

if __name__ == "__main__":
    run_experiment()
