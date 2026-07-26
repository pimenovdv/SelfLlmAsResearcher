from transformers import AutoTokenizer

model_name = "JackFram/llama-160m"
print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

clean_text = "When John and Mary went to the store, John gave a drink to"
inputs = tokenizer(clean_text)
print(inputs.input_ids)
for i in inputs.input_ids:
    print(f"{i}: {tokenizer.decode([i])}")
