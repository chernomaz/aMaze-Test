#!/bin/bash
# Policy: policy_assert_output.json (graph: agent->llm->pdf_search->finish)
# Expectation: FAIL — prompt has no "pdf", LLM mock doesn't fire,
#              real LLM likely returns direct answer -> graph incomplete (never reaches pdf_search)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="What is 2 plus 2?"
"$PYTHON" -m amaze.amaze_runner examples/agents/crewai_annotated_agent.py examples/policies/policy_assert_output.json
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: PASS (policy violation detected as expected)"
    exit 0
else
    echo "RESULT: FAIL (expected policy violation but test passed unexpectedly)"
    exit 1
fi
