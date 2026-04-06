#!/bin/bash
# Policy: policy.json (control_plane, allowed=[pdf_search], max_llm=2, max_tool=1)
# Expectation: PASS — agent uses pdf_search once within limits
set -e
cd /data/cloude/aMazeTest
export AGENT_PROMPT="Search the local PDF documents for information about humanitarian data management"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy.json
echo "RESULT: PASS (policy satisfied as expected)"
