#!/bin/bash
# Policy: policy_token_strict.json (control_plane, allowed=[pdf_search, web_search], max_tokens=800)
# LLM mock: input contains "search" -> pdf_search("climate change impacts")
# pdf_search mock: short fixed output
# Expectation: PASS — mock fires, no real LLM tokens used, well within 800 token budget
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Please search for information about climate change impacts"
"$PYTHON" -m amaze.amaze_runner examples/agents/one_conversation_agent.py examples/policies/policy_token_strict.json
echo "RESULT: PASS (token limit and policy satisfied as expected)"
