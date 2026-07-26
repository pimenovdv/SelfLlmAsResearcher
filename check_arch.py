from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
print(model)
