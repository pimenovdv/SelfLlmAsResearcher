import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    model.eval()

    clean = "apple -> red, banana -> yellow, grass -> green, sky ->"
    inputs_clean = tokenizer(clean, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs_clean)

    attentions = outputs.attentions

    tokens = [tokenizer.decode([t]) for t in inputs_clean.input_ids[0]]

    print("Tokens:", tokens)

    for layer, head in [(7, 11), (8, 6)]:
        attn = attentions[layer][0, head, -1, :]
        print(f"\nAttention pattern for L{layer}H{head} from the last token ('{tokens[-1]}'):")
        for i, val in enumerate(attn):
            if val > 0.05: # threshold
                print(f"  Token {i} ('{tokens[i]}'): {val:.4f}")

if __name__ == "__main__":
    main()
