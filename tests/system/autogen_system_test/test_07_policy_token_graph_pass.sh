#!/bin/bash
# Policy: policy_token_graph.json (graph: agent->llm->web_search->file_read->finish, max_tokens=1500)
# LLM mock: input contains "file" -> web_search; web_search mock + file_read mock set
# Expectation: PASS — all mocks fire, graph completes, token usage stays within 1500
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Search the web for the current price per unit, then read the file at $PROJECT_ROOT/examples/policies/policy.json for additional details"
"$PYTHON" -m amaze.amaze_runner examples/agents/autogen_annotated_agent.py examples/policies/policy_token_graph.json
echo "RESULT: PASS (graph and token limit satisfied as expected)"
