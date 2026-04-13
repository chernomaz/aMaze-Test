#!/bin/bash
# Policy: policy_graph.json (graph: agent->llm->tool:pdf_search->finish)
# Expectation: FAIL — prompt has no "search", LLM mock doesn't fire,
#              real LLM returns direct answer or calls wrong tool -> graph violation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="What is the capital of France?"
"$PYTHON" -m amaze.amaze_runner examples/agents/one_conversation_agent.py examples/policies/policy_graph.json
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: PASS (policy violation detected as expected)"
    exit 0
else
    echo "RESULT: FAIL (expected policy violation but test passed unexpectedly)"
    exit 1
fi
