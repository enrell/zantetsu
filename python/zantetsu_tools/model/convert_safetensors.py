from __future__ import annotations

from ..common.safetensors_tools import convert_layer_norm_keys


def main() -> None:
    model_path = convert_layer_norm_keys()
    print(f"Converted LayerNorm beta/gamma to bias/weight: {model_path}")


if __name__ == "__main__":
    main()
