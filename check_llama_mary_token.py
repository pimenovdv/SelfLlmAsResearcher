from transformers import AutoTokenizer

model_name = "JackFram/llama-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)

mary_id = tokenizer.encode("Mary", add_special_tokens=False)[0]
john_id = tokenizer.encode("John", add_special_tokens=False)[0]

print(f"Mary ID: {mary_id}, John ID: {john_id}")

print("Mary token:", tokenizer.decode([mary_id]))
print("John token:", tokenizer.decode([john_id]))
