import torch
from transformers import AutoModelForCausalLM

model_name = "JackFram/llama-160m"
model = AutoModelForCausalLM.from_pretrained(model_name)

num_heads = model.config.num_attention_heads
hidden_size = model.config.hidden_size
head_dim = hidden_size // num_heads

print(f"Num heads: {num_heads}, Hidden size: {hidden_size}, Head dim: {head_dim}")

# Let's inspect the input to o_proj
def hook_fn(module, args):
    print("o_proj input shape:", args[0].shape)
    return args

handle = model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(hook_fn)
dummy_input = torch.randint(0, 32000, (1, 10))
model(dummy_input)
handle.remove()
