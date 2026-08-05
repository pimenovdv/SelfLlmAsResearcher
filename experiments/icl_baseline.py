import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from src.experiment_utils import load_model_and_tokenizer
from src.metrics import brier_score

def main():
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    prompt = "apple -> red, banana -> yellow, grass -> green, sky ->"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    next_token = tokenizer.decode([next_token_id])

    target_id = tokenizer.encode(" blue", add_special_tokens=False)[0]

    brier = brier_score(outputs.logits, target_id)

    print(f"Prompt: '{prompt}'")
    print(f"Predicted next token: '{next_token}'")
    print(f"Brier Score for ' blue': {brier:.4f}")

if __name__ == "__main__":
    main()
