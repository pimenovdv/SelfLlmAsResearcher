import torch
from src.metrics import entropy
logits = torch.tensor([[[10.0, 0.0, 0.0]]])
ent = entropy(logits)
print(f"Entropy: {ent}")