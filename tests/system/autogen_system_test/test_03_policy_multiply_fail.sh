#!/bin/bash
# Policy: policy_multiply.json (graph: agent->llm->web_search->dummy_email->finish)
# Expectation: FAIL — prompt doesn't contain "email", LLM mock doesn't fire,
#              real LLM takes a different path -> graph violation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Tell me about the history of the Roman Empire"
"$PYTHON" -m amaze.amaze_runner examples/agents/autogen_annotated_agent.py examples/policies/policy_multiply.json
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: PASS (policy violation detected as expected)"
    exit 0
else
    echo "RESULT: FAIL (expected policy violation but test passed unexpectedly)"
    exit 1
fi
