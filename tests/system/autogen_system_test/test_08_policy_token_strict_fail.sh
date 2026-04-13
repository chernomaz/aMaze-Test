#!/bin/bash
# Policy: policy_token_strict.json (max_tokens=800)
# Expectation: FAIL — prompt has no "search", LLM mock doesn't fire,
#              real LLM runs and almost certainly exceeds 800 token budget
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Explain the entire history of the Roman Empire in great detail"
"$PYTHON" -m amaze.amaze_runner examples/agents/autogen_annotated_agent.py examples/policies/policy_token_strict.json
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: PASS (token violation detected as expected)"
    exit 0
else
    echo "RESULT: FAIL (expected token violation but test passed unexpectedly)"
    exit 1
fi
