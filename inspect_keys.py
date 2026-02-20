from safetensors import safe_open
t = safe_open("models/ner_model/model.safetensors", framework="pt")
for k in t.keys():
    print(k)
