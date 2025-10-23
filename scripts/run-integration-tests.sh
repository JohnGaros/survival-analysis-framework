#!/bin/bash
# Integration test script - validates end-to-end pipeline execution

set -e  # Exit on error

echo "========================================="
echo "Integration Test Suite"
echo "========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"

    TESTS_RUN=$((TESTS_RUN + 1))
    echo -n "Testing: $test_name ... "

    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Function to check file exists
check_file() {
    local file="$1"
    if [ -f "$file" ]; then
        return 0
    else
        return 1
    fi
}

# Function to check directory exists
check_dir() {
    local dir="$1"
    if [ -d "$dir" ]; then
        return 0
    else
        return 1
    fi
}

echo "Step 1: Clean environment"
echo "-------------------------"
rm -rf artifacts/ models/ data/test_outputs/ integration_test.log
echo "✓ Cleaned artifacts and test outputs"
echo ""

echo "Step 2: Run end-to-end pipeline"
echo "--------------------------------"
echo "Running: PYTHONPATH=src python src/main.py"
echo "(This may take 60-120 seconds...)"
echo ""

START_TIME=$(date +%s)

if PYTHONPATH=src timeout 300 python src/main.py > integration_test.log 2>&1; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "${GREEN}✓ Pipeline completed successfully${NC} (${DURATION}s)"
    PIPELINE_STATUS="PASS"
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "${RED}✗ Pipeline failed${NC} (${DURATION}s)"
    echo ""
    echo "Last 50 lines of output:"
    tail -50 integration_test.log
    PIPELINE_STATUS="FAIL"
    exit 1
fi
echo ""

echo "Step 3: Verify artifacts"
echo "------------------------"

run_test "model_metrics.csv exists" "check_file artifacts/model_metrics.csv"
run_test "model_summary.csv exists" "check_file artifacts/model_summary.csv"
run_test "ph_flags.csv exists" "check_file artifacts/ph_flags.csv"

run_test "cox_ph artifacts exist" "check_dir artifacts/cox_ph"
run_test "coxnet artifacts exist" "check_dir artifacts/coxnet"
run_test "weibull_aft artifacts exist" "check_dir artifacts/weibull_aft"
run_test "gbsa artifacts exist" "check_dir artifacts/gbsa"
run_test "rsf artifacts exist" "check_dir artifacts/rsf"

run_test "models directory exists" "check_dir models"

# Count models
MODEL_COUNT=$(ls models/*.joblib 2>/dev/null | wc -l | tr -d ' ')
if [ "$MODEL_COUNT" -ge 5 ]; then
    echo -e "Testing: Models saved (5+ expected) ... ${GREEN}✓ PASS${NC} (found $MODEL_COUNT)"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "Testing: Models saved (5+ expected) ... ${RED}✗ FAIL${NC} (found $MODEL_COUNT)"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""

echo "Step 4: Validate metrics"
echo "------------------------"

# Run Python validation script
PYTHONPATH=src python -c "
import pandas as pd
import sys

try:
    metrics = pd.read_csv('artifacts/model_metrics.csv')

    # Check models present
    expected_models = {'cox_ph', 'coxnet', 'weibull_aft', 'gbsa', 'rsf'}
    actual_models = set(metrics['model'].unique())

    missing = expected_models - actual_models
    if missing:
        print(f'ERROR: Missing models: {missing}')
        sys.exit(1)

    # Check metrics ranges
    if not metrics['cindex'].between(0.5, 1.0).all():
        print('ERROR: C-index out of range [0.5, 1.0]')
        sys.exit(1)

    if not metrics['ibs'].between(0.0, 0.25).all():
        print('ERROR: IBS out of range [0.0, 0.25]')
        sys.exit(1)

    if metrics['cindex'].isna().any():
        print('ERROR: NaN values in C-index')
        sys.exit(1)

    if metrics['ibs'].isna().any():
        print('ERROR: NaN values in IBS')
        sys.exit(1)

    # Print summary
    print(f\"All 5 models present: {', '.join(sorted(actual_models))}\")
    print(f\"C-index range: [{metrics['cindex'].min():.3f}, {metrics['cindex'].max():.3f}]\")
    print(f\"IBS range: [{metrics['ibs'].min():.3f}, {metrics['ibs'].max():.3f}]\")
    print('All metrics valid')
    sys.exit(0)

except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Metrics validation passed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Metrics validation failed${NC}"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""

echo "Step 5: Check for critical errors"
echo "----------------------------------"

# Check log for critical errors
if grep -q "LAPACK" integration_test.log; then
    echo -e "${RED}✗ LAPACK errors found in log${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
else
    echo -e "${GREEN}✓ No LAPACK errors${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

if grep -q "IndexError" integration_test.log; then
    echo -e "${RED}✗ IndexErrors found in log${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
else
    echo -e "${GREEN}✓ No IndexErrors${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

if grep -q "TypeError.*expected" integration_test.log; then
    echo -e "${RED}✗ TypeErrors found in log${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
else
    echo -e "${GREEN}✓ No TypeErrors${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

echo ""

# Check for warnings (informational only)
echo "Step 6: Check warnings (informational)"
echo "--------------------------------------"

CONVERGENCE_WARNINGS=$(grep -c "ConvergenceWarning" integration_test.log || echo "0")
OVERFLOW_WARNINGS=$(grep -c "overflow" integration_test.log || echo "0")

if [ "$CONVERGENCE_WARNINGS" -gt 0 ]; then
    echo -e "${YELLOW}⚠ $CONVERGENCE_WARNINGS convergence warning(s) found (may be acceptable)${NC}"
fi

if [ "$OVERFLOW_WARNINGS" -gt 0 ]; then
    echo -e "${YELLOW}⚠ $OVERFLOW_WARNINGS overflow warning(s) found (may be acceptable)${NC}"
fi

echo ""

# Final summary
echo "========================================="
echo "Integration Test Summary"
echo "========================================="
echo ""
echo "Pipeline Status: $PIPELINE_STATUS"
echo "Duration: ${DURATION}s"
echo ""
echo "Tests Run: $TESTS_RUN"
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
if [ "$TESTS_FAILED" -gt 0 ]; then
    echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
else
    echo "Tests Failed: $TESTS_FAILED"
fi
echo ""

if [ "$TESTS_FAILED" -eq 0 ]; then
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}  ✓ ALL INTEGRATION TESTS PASSED${NC}"
    echo -e "${GREEN}=========================================${NC}"
    exit 0
else
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}  ✗ INTEGRATION TESTS FAILED${NC}"
    echo -e "${RED}=========================================${NC}"
    echo ""
    echo "Check integration_test.log for details"
    exit 1
fi
