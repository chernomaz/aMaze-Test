#!/bin/bash
# Policy: policy_cp_graph.json (control_plane, allowed=[web_search, dummy_email])
# web_search mock + dummy_email mock set; no LLM mock (real LLM decides tools)
# Expectation: PASS — real LLM calls web_search and dummy_email, both mocked, within limits
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Search the web for the latest company news and then get Bob's email"
"$PYTHON" -m amaze.amaze_runner examples/agents/autogen_annotated_agent.py examples/policies/policy_cp_graph.json
echo "RESULT: PASS (policy satisfied as expected)"
