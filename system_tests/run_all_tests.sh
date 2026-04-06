#!/bin/bash
# Master runner for all system tests
# Each test script exits 0 on success (expected outcome) and 1 on unexpected outcome.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASS=0
FAIL=0
RESULTS=()

run_test() {
    local name="$1"
    local script="$2"
    echo ""
    echo "=========================================="
    echo "RUNNING: $name"
    echo "=========================================="
    bash "$script"
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        PASS=$((PASS + 1))
        RESULTS+=("OK   $name")
    else
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL $name")
    fi
}

run_test "test_01_policy_cp_pass"          "$SCRIPT_DIR/test_01_policy_cp_pass.sh"
run_test "test_01_policy_cp_fail"          "$SCRIPT_DIR/test_01_policy_cp_fail.sh"
run_test "test_02_policy_graph_pass"       "$SCRIPT_DIR/test_02_policy_graph_pass.sh"
run_test "test_02_policy_graph_fail"       "$SCRIPT_DIR/test_02_policy_graph_fail.sh"
run_test "test_03_policy_multiply_pass"    "$SCRIPT_DIR/test_03_policy_multiply_pass.sh"
run_test "test_03_policy_multiply_fail"    "$SCRIPT_DIR/test_03_policy_multiply_fail.sh"
run_test "test_04_policy_assert_input_pass" "$SCRIPT_DIR/test_04_policy_assert_input_pass.sh"
run_test "test_04_policy_assert_input_fail" "$SCRIPT_DIR/test_04_policy_assert_input_fail.sh"
run_test "test_05_policy_assert_output_pass" "$SCRIPT_DIR/test_05_policy_assert_output_pass.sh"
run_test "test_05_policy_assert_output_fail" "$SCRIPT_DIR/test_05_policy_assert_output_fail.sh"
run_test "test_06_policy_cp_graph_pass"    "$SCRIPT_DIR/test_06_policy_cp_graph_pass.sh"
run_test "test_06_policy_cp_graph_fail"    "$SCRIPT_DIR/test_06_policy_cp_graph_fail.sh"
run_test "test_07_policy_token_graph_pass" "$SCRIPT_DIR/test_07_policy_token_graph_pass.sh"
run_test "test_07_policy_token_graph_fail" "$SCRIPT_DIR/test_07_policy_token_graph_fail.sh"
run_test "test_08_policy_token_strict_pass" "$SCRIPT_DIR/test_08_policy_token_strict_pass.sh"
run_test "test_08_policy_token_strict_fail" "$SCRIPT_DIR/test_08_policy_token_strict_fail.sh"

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  Passed: $PASS / $((PASS + FAIL))"
echo "=========================================="

if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
