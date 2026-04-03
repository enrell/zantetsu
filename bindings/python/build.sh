#!/bin/bash
set -e

echo "Building Zantetsu Python package for PyPI..."

# Change to the Python bindings directory
cd "$(dirname "$0")"

# Clean previous builds
rm -rf target/wheels dist build

# Build release wheels for current platform
echo "Building release wheels..."
maturin build --release

# Show what was built
echo ""
echo "Built packages:"
ls -la target/wheels/

echo ""
echo "To publish to PyPI, run:"
echo "  maturin publish"
echo ""
echo "Or manually with twine:"
echo "  twine upload target/wheels/*"