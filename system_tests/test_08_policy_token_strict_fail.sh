#!/bin/bash
# Policy: policy_token_strict.json (max_tokens=800)
# Expectation: FAIL — prompt has no "search", LLM mock doesn't fire,
#              real LLM runs and almost certainly exceeds 800 token budget
cd /data/cloude/aMazeTest
export AGENT_PROMPT="Explain the entire history of the Roman Empire in great detail"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy_token_strict.json
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: PASS (token violation detected as expected)"
    exit 0
else
    echo "RESULT: FAIL (expected token violation but test passed unexpectedly)"
    exit 1
fi
