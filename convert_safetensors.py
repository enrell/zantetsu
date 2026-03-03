from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


model_path = Path("models/ner_model/model.safetensors")
temp_path = model_path.with_suffix(".converted.safetensors")

tensors = {}
with safe_open(model_path, framework="pt") as tensor_file:
    for key in tensor_file.keys():
        converted_key = key.replace(".gamma", ".weight").replace(".beta", ".bias")
        tensors[converted_key] = tensor_file.get_tensor(key)

save_file(tensors, str(temp_path))
temp_path.replace(model_path)

print(f"Converted LayerNorm beta/gamma to bias/weight: {model_path}")
