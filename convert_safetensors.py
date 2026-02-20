from safetensors import safe_open
from safetensors.torch import save_file
import torch

t = safe_open("models/ner_model/model.safetensors", framework="pt")
tensors = {}
for k in t.keys():
    new_k = k.replace(".gamma", ".weight").replace(".beta", ".bias")
    tensors[new_k] = t.get_tensor(k)

save_file(tensors, "models/ner_model/model.safetensors")
print("Converted LayerNorm beta/gamma to bias/weight")
