#!/bin/bash
# Policy: policy_token_graph.json (max_tokens=1500)
# Expectation: FAIL — prompt has no "file", LLM mock doesn't fire, real LLM runs,
#              likely exceeds 1500 token budget -> PolicyViolation: token limit exceeded
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Please write a detailed essay about the history of computing, covering all major milestones from the 1940s to today"
"$PYTHON" -m amaze.amaze_runner examples/agents/one_conversation_agent.py examples/policies/policy_token_graph.json
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: PASS (policy violation detected as expected)"
    exit 0
else
    echo "RESULT: FAIL (expected policy violation but test passed unexpectedly)"
    exit 1
fi
