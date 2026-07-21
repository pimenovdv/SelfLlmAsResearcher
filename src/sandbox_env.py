import os

class SandboxEnvironment:
    def __init__(self, workspace_dir="agent_workspace"):
        self.workspace_dir = os.path.abspath(workspace_dir)
        os.makedirs(self.workspace_dir, exist_ok=True)

    def resolve_path(self, path: str) -> str:
        """Resolve and validate that the path is within the workspace."""
        abs_path = os.path.abspath(os.path.join(self.workspace_dir, path))
        if not abs_path.startswith(self.workspace_dir):
            raise ValueError(f"Access denied: path {path} is outside the workspace directory.")
        return abs_path

    def setup_templates(self):
        """Setup initial experiment templates in the workspace."""
        templates_dir = os.path.join(self.workspace_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)

        # Example template for Activation Patching
        patching_template = """import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Загружаем модель и токенизатор
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

source_text = "The capital of Italy is"
target_text = "The capital of France is"

source_inputs = tokenizer(source_text, return_tensors="pt")
target_inputs = tokenizer(target_text, return_tensors="pt")

layer_to_patch = model.transformer.h[8].mlp
cached_activation = None

def cache_hook(module, input, output):
    global cached_activation
    cached_activation = output.detach().clone()

handle_cache = layer_to_patch.register_forward_hook(cache_hook)
with torch.no_grad():
    model(**source_inputs)
handle_cache.remove()

def patch_hook(module, input, output):
    patched_output = output.clone()
    patched_output[0, -1, :] = cached_activation[0, -1, :]
    return patched_output

handle_patch = layer_to_patch.register_forward_hook(patch_hook)
with torch.no_grad():
    outputs = model(**target_inputs)
handle_patch.remove()

next_token_logits = outputs.logits[0, -1, :]
predicted_token_id = torch.argmax(next_token_logits).item()
predicted_word = tokenizer.decode(predicted_token_id)

print(f"Target prompt: '{target_text}'")
print(f"Predicted word: '{predicted_word}'")
"""
        with open(os.path.join(templates_dir, "activation_patching.py"), "w") as f:
            f.write(patching_template)

if __name__ == "__main__":
    env = SandboxEnvironment()
    env.setup_templates()
    print("Sandbox templates created.")
