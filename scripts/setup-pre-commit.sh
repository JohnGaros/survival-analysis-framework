#!/bin/bash
# Setup script for pre-commit hooks

set -e  # Exit on error

echo "======================================"
echo "Setting up pre-commit hooks"
echo "======================================"
echo ""

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
else
    echo "✓ pre-commit is already installed"
fi

# Install pre-commit hooks
echo ""
echo "Installing pre-commit hooks to .git/hooks/..."
pre-commit install

# Install commit-msg hook for conventional commits
echo ""
echo "Installing commit-msg hook..."
pre-commit install --hook-type commit-msg

# Update hooks to latest versions
echo ""
echo "Updating pre-commit hooks..."
pre-commit autoupdate

# Run pre-commit on all files (initial check)
echo ""
echo "Running pre-commit on all files (this may take a while)..."
pre-commit run --all-files || {
    echo ""
    echo "⚠️  Some checks failed. This is normal for the first run."
    echo "   Files have been auto-formatted where possible."
    echo "   Review the changes and commit them."
    exit 0
}

echo ""
echo "======================================"
echo "✓ Pre-commit hooks setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Review any auto-formatted files"
echo "  2. Fix any remaining issues reported above"
echo "  3. Commit your changes"
echo ""
echo "Pre-commit will now run automatically on git commit."
echo "To manually run: pre-commit run --all-files"
