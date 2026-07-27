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

def get_greater_than_probs(model, tokenizer, prompt, target_year):
    """
    Evaluates the probability of years greater than the target year vs smaller/equal years.
    Example prompt: "The war lasted from the year 1732 to the year 17"
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]
    probs = F.softmax(next_token_logits, dim=-1)

    # Analyze the probabilities of tokens corresponding to years 00-99
    greater_prob = 0.0
    less_equal_prob = 0.0

    # The target_year is expected to be a 2-digit number, e.g., 32
    target_century = target_year // 100
    target_decade_year = target_year % 100

    valid_tokens = []

    for i in range(100):
        year_str = f"{i:02d}"

        # In GPT-2, tokens often have a leading space, but for years in this context
        # it might just be the numbers. We need to check the token ids.
        # Often numbers are tokenized directly if they follow another number without space,
        # or with a space depending on the prompt structure.
        # Given "year 17" ends without a space, the next token will just be the two digits.
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

    prompts = [
        ("The war lasted from the year 1732 to the year 17", 1732),
        ("The war lasted from the year 1745 to the year 17", 1745),
        ("The war lasted from the year 1920 to the year 19", 1920),
        ("The war lasted from the year 1850 to the year 18", 1850)
    ]

    print("\nEvaluating Greater-Than task...")
    for prompt, year in prompts:
        greater, less_equal = get_greater_than_probs(model, tokenizer, prompt, year)
        print(f"\nPrompt: '{prompt}' (Target > {year % 100})")
        print(f"Prob > {year % 100}: {greater:.4f}")
        print(f"Prob <= {year % 100}: {less_equal:.4f}")
        print(f"Ratio (> / <=): {greater / max(less_equal, 1e-10):.2f}")

if __name__ == "__main__":
    run_experiment()
