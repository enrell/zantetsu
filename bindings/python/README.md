# Zantetsu Python

Fast anime metadata parser for Python. Extracts title, episode, resolution, codecs, and more from anime filenames.

## Installation

```bash
pip install zantetsu
```

## Usage

```python
from zantetsu import HeuristicParser

parser = HeuristicParser()
result = parser.parse("[SubsPlease] Spy x Family - 01 (1080p).mkv")

print(result.title)       # 'Spy x Family'
print(result.episode)     # '1'
print(result.resolution)  # 'FHD1080'
print(result.group)       # 'SubsPlease'
```

## Development

### Prerequisites

- Rust 1.85+
- Python 3.8+
- maturin: `pip install maturin`

### Build

```bash
cd bindings/python
maturin develop  # For local development
maturin build    # Build wheels
```

### Publish to PyPI

```bash
# Build and publish
maturin publish

# Or use twine
maturin build --release
twine upload target/wheels/*
```

## License

MIT