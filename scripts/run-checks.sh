#!/bin/bash
# Run all code quality checks manually

set -e  # Exit on error

echo "======================================"
echo "Running Code Quality Checks"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a check
run_check() {
    local name=$1
    local command=$2

    echo -e "${YELLOW}► Running $name...${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✓ $name passed${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ $name failed${NC}"
        echo ""
        return 1
    fi
}

# Track failures
failed=0

# 1. Code formatting with black (check only)
run_check "Black (formatting)" "black --check --diff src/ tests/" || ((failed++))

# 2. Import sorting with isort (check only)
run_check "isort (import sorting)" "isort --check-only --diff src/ tests/" || ((failed++))

# 3. Linting with flake8
run_check "Flake8 (linting)" "flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503,E501 --max-complexity=10" || ((failed++))

# 4. Type checking with mypy
run_check "Mypy (type checking)" "mypy src/ --ignore-missing-imports --no-strict-optional" || ((failed++))

# 5. Pylint checks
run_check "Pylint (code analysis)" "pylint src/ --max-line-length=100 --disable=C0111,R0903,C0103,R0913" || ((failed++))

# 6. Security checks with bandit
run_check "Bandit (security)" "bandit -r src/ -c pyproject.toml" || ((failed++))

# 7. Docstring coverage
run_check "Interrogate (docstring coverage)" "interrogate -vv --fail-under=80 src/" || ((failed++))

# 8. Run tests with coverage
run_check "Pytest (tests + coverage)" "pytest --cov=src/survival_framework --cov-report=term-missing --cov-fail-under=80" || ((failed++))

# Summary
echo "======================================"
if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "======================================"
    exit 0
else
    echo -e "${RED}✗ $failed check(s) failed${NC}"
    echo "======================================"
    echo ""
    echo "To auto-fix formatting issues:"
    echo "  black src/ tests/"
    echo "  isort src/ tests/"
    echo ""
    exit 1
fi
