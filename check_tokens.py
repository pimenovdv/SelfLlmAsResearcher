from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
clean_inputs = tokenizer("When John and Mary went to the store, John gave a drink to", return_tensors="pt")
for i, token_id in enumerate(clean_inputs.input_ids[0]):
    print(f"{i}: '{tokenizer.decode(token_id)}'")
