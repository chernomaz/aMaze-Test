#!/bin/bash
# Policy: policy_assert_input.json
# Expectation: FAIL — prompt doesn't contain "humanitarian",
#              assertion "llm input must contain humanitarian" fails
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Tell me about climate change and its effects on the environment"
"$PYTHON" -m amaze.amaze_runner examples/agents/autogen_annotated_agent.py examples/policies/policy_assert_input.json
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: PASS (assertion failure detected as expected)"
    exit 0
else
    echo "RESULT: FAIL (expected assertion failure but test passed unexpectedly)"
    exit 1
fi
