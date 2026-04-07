from __future__ import annotations

from ..common.safetensors_tools import iter_keys


def main() -> None:
    for key in iter_keys():
        print(key)


if __name__ == "__main__":
    main()
