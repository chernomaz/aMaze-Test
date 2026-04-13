#!/bin/bash
# Policy: policy_multiply.json (graph: agent->llm->web_search->dummy_email->finish)
# LLM mock: input contains "email" -> return web_search tool call
# web_search mock + dummy_email mock both set
# Expectation: PASS — full graph sequence completes via mocks
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Get the latest company news and then check Alice's email"
"$PYTHON" -m amaze.amaze_runner examples/agents/one_conversation_agent.py examples/policies/policy_multiply.json
echo "RESULT: PASS (graph sequence completed as expected)"
