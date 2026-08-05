import os
import sys
import torch
from src.experiment_utils import load_model_and_tokenizer

def main():
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name, output_attentions=True)

    prompt = "Q: en - cat, fr - chat. Q: en - dog, fr - chien. Q: en - horse, fr -"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # L8H2 is the top induction head found via DLA
    layer_idx = 8
    head_idx = 2

    # [batch, num_heads, seq_len, seq_len]
    attention_pattern = outputs.attentions[layer_idx][0, head_idx, :, :]

    # We want to see where the last token attends to
    last_token_attention = attention_pattern[-1, :]

    print(f"\n--- Attention Pattern for L{layer_idx}H{head_idx} (Last Token) ---")

    seq_len = inputs.input_ids.shape[1]
    for pos in range(seq_len):
        token_str = tokenizer.decode([inputs.input_ids[0, pos].item()])
        attn_weight = last_token_attention[pos].item()
        print(f"Pos {pos:2d} ('{token_str}'): {attn_weight:.4f}")

if __name__ == "__main__":
    main()
