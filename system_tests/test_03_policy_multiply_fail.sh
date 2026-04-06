#!/bin/bash
# Policy: policy_multiply.json (graph: agent->llm->web_search->dummy_email->finish)
# Expectation: FAIL — prompt doesn't contain "email", LLM mock doesn't fire,
#              real LLM takes a different path -> graph violation
cd /data/cloude/aMazeTest
export AGENT_PROMPT="Tell me about the history of the Roman Empire"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy_multiply.json
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: PASS (policy violation detected as expected)"
    exit 0
else
    echo "RESULT: FAIL (expected policy violation but test passed unexpectedly)"
    exit 1
fi
