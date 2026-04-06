#!/bin/bash
# Policy: policy_assert_output.json (graph: agent->llm->pdf_search->finish)
# Expectation: FAIL — prompt has no "pdf", LLM mock doesn't fire,
#              real LLM likely returns direct answer -> graph incomplete (never reaches pdf_search)
cd /data/cloude/aMazeTest
export AGENT_PROMPT="What is 2 plus 2?"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy_assert_output.json
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: PASS (policy violation detected as expected)"
    exit 0
else
    echo "RESULT: FAIL (expected policy violation but test passed unexpectedly)"
    exit 1
fi
