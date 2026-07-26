import torch
from transformers import AutoModelForCausalLM

model_name = "JackFram/llama-160m"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)
print(model)
