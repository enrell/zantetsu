# Publishing to PyPI

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org
2. **API Token**: Generate an API token at https://pypi.org/manage/account/token/
3. **Tools**: Install maturin and twine
   ```bash
   pip install maturin twine
   ```

## Quick Publish

```bash
cd bindings/python

# Build and publish in one command
maturin publish

# Or build first, then publish
maturin build --release
twine upload target/wheels/*
```

## Multi-Platform Builds (Recommended)

To support Linux, macOS, and Windows, use GitHub Actions or build on each platform:

### Option 1: GitHub Actions (Recommended)

Create `.github/workflows/pypi.yml`:

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12', '3.13']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Rust
      uses: dtolnay/rust-action@stable
    
    - name: Install maturin
      run: pip install maturin
    
    - name: Build wheels
      run: maturin build --release
      working-directory: bindings/python
    
    - name: Upload wheels
      uses: actions/upload-artifact@v3
      with:
        name: wheels
        path: bindings/python/target/wheels/

  publish:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - name: Download wheels
      uses: actions/download-artifact@v3
      with:
        name: wheels
        path: wheels
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

### Option 2: Manual Multi-Platform

```bash
# Linux (manylinux for compatibility)
pip install cibuildwheel
cd bindings/python
cibuildwheel --platform linux

# macOS (on a Mac)
maturin build --release --target universal2-apple-darwin

# Windows (on Windows)
maturin build --release
```

## Local Testing

```bash
cd bindings/python

# Build and install locally
maturin develop

# Test the package
python -c "from zantetsu import HeuristicParser; p = HeuristicParser(); print(p.parse('[SubsPlease] Test - 01 (1080p).mkv'))"

# Run tests
pytest
```

## Version Management

1. Update version in:
   - `Cargo.toml` (both workspace root and python bindings)
   - `pyproject.toml`
   - `python/zantetsu/__init__.py`

2. Create a git tag:
   ```bash
   git tag v0.1.2
   git push origin v0.1.2
   ```

3. The GitHub Action will automatically build and publish

## Troubleshooting

### maturin: command not found
```bash
pip install --user maturin
# Or use pipx
pipx install maturin
```

### Authentication failed
```bash
# Set token as environment variable
export MATURIN_PYPI_TOKEN=pypi-xxxxxxxxxx
maturin publish
```

### Wheels not supported on this platform
Use `cibuildwheel` for maximum compatibility:
```bash
pip install cibuildwheel
cibuildwheel --platform linux bindings/python
```

## Files Structure

```
bindings/python/
├── Cargo.toml           # Rust package config
├── pyproject.toml       # Python package config
├── build.sh            # Build script
├── README.md           # Package documentation
├── src/
│   └── lib.rs          # Rust FFI code
└── python/
    └── zantetsu/
        └── __init__.py # Python package entry
```

## Next Steps

1. Set up the GitHub Action for automatic publishing
2. Create a PyPI account and API token
3. Add the token as a GitHub secret named `PYPI_API_TOKEN`
4. Push a tag to trigger the release