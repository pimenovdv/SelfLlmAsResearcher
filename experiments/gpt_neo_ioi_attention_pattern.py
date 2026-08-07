import torch
from src.experiment_utils import load_model_and_tokenizer

def main():
    model_name = "EleutherAI/gpt-neo-125m"
    print(f"Loading {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name, output_attentions=True)

    clean_text = "When John and Mary went to the store, John gave a drink to"
    inputs_clean = tokenizer(clean_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs_clean)

    attentions = outputs.attentions
    tokens = [tokenizer.decode([t]) for t in inputs_clean.input_ids[0]]

    print("Tokens:", tokens)

    # Name Mover Heads in GPT-Neo 125m for IOI
    target_heads = [(9, 4), (10, 6), (11, 2)]

    for layer, head in target_heads:
        # GPT-Neo attentions shape: [batch, num_heads, seq_len, seq_len]
        attn = attentions[layer][0, head, -1, :]
        print(f"\nAttention pattern for L{layer}H{head} from the last token ('{tokens[-1]}'):")
        for i, val in enumerate(attn):
            if val > 0.05: # threshold
                print(f"  Token {i} ('{tokens[i]}'): {val:.4f}")

if __name__ == "__main__":
    main()
