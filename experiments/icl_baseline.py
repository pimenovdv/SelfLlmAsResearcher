import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "apple -> red, banana -> yellow, grass -> green, sky ->"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    next_token = tokenizer.decode([next_token_id])

    print(f"Prompt: '{prompt}'")
    print(f"Predicted next token: '{next_token}'")

if __name__ == "__main__":
    main()
