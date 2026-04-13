#!/bin/bash
# Policy: policy_cp_graph.json (control_plane, allowed=[web_search, dummy_email])
# Expectation: FAIL — agent tries to use pdf_search which is NOT in allowed_tools -> PolicyViolation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Search the local PDF documents for information about data management"
"$PYTHON" -m amaze.amaze_runner examples/agents/autogen_annotated_agent.py examples/policies/policy_cp_graph.json
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: PASS (policy violation detected as expected)"
    exit 0
else
    echo "RESULT: FAIL (expected policy violation but test passed unexpectedly)"
    exit 1
fi
