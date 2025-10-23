#!/bin/bash
# Auto-format code using black and isort

echo "======================================"
echo "Auto-formatting Code"
echo "======================================"
echo ""

# Format with black
echo "► Running black..."
black src/ tests/
echo "✓ Black formatting complete"
echo ""

# Sort imports with isort
echo "► Running isort..."
isort src/ tests/
echo "✓ Import sorting complete"
echo ""

echo "======================================"
echo "✓ Code formatting complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git diff"
echo "  2. Run checks: ./scripts/run-checks.sh"
echo "  3. Commit if satisfied: git commit -am 'style: format code'"
