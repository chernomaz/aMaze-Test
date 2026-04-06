#!/bin/bash
# Policy: policy_assert_input.json
# Expectation: FAIL — prompt doesn't contain "humanitarian",
#              assertion "llm input must contain humanitarian" fails
cd /data/cloude/aMazeTest
export AGENT_PROMPT="Tell me about climate change and its effects on the environment"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy_assert_input.json
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "RESULT: PASS (assertion failure detected as expected)"
    exit 0
else
    echo "RESULT: FAIL (expected assertion failure but test passed unexpectedly)"
    exit 1
fi
