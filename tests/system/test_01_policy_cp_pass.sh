#!/bin/bash
# Policy: policy.json (control_plane, allowed=[pdf_search], max_llm=2, max_tool=1)
# Expectation: PASS — agent uses pdf_search once within limits
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Search the local PDF documents for information about humanitarian data management"
"$PYTHON" -m amaze.amaze_runner examples/agents/one_conversation_agent.py examples/policies/policy.json
echo "RESULT: PASS (policy satisfied as expected)"
