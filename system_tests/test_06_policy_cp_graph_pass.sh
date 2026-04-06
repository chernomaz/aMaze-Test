#!/bin/bash
# Policy: policy_cp_graph.json (control_plane, allowed=[web_search, dummy_email])
# web_search mock + dummy_email mock set; no LLM mock (real LLM decides tools)
# Expectation: PASS — real LLM calls web_search and dummy_email, both mocked, within limits
set -e
cd /data/cloude/aMazeTest
export AGENT_PROMPT="Search the web for the latest company news and then get Bob's email"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy_cp_graph.json
echo "RESULT: PASS (policy satisfied as expected)"
