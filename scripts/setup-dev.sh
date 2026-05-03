#!/bin/bash
# Developer environment setup for mlx-graphs
# Usage: ./scripts/setup-dev.sh

set -e

echo "🔧 Setting up development environment..."

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create virtual environment and install dependencies
echo "📦 Installing dependencies..."
uv sync --all-extras

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
uv run pre-commit install -t pre-commit -t pre-push

# Create secrets baseline if it doesn't exist
if [ ! -f .secrets.baseline ]; then
    echo "🔐 Creating secrets baseline..."
    uv run detect-secrets scan > .secrets.baseline
fi

# Create test directories if they don't exist
mkdir -p tests/unit tests/integration tests/contract

echo ""
echo "✅ Development environment ready!"
echo ""
echo "Quick commands:"
echo "  uv run pytest tests/unit         - Run unit tests"
echo "  uv run mypy src/                  - Type checking"
echo "  pre-commit run --all-files        - Run all pre-commit checks"
echo ""
