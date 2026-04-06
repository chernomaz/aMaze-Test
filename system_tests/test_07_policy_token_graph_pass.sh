#!/bin/bash
# Policy: policy_token_graph.json (graph: agent->llm->web_search->file_read->finish, max_tokens=1500)
# LLM mock: input contains "file" -> web_search; web_search mock + file_read mock set
# Expectation: PASS — all mocks fire, graph completes, token usage stays within 1500
set -e
cd /data/cloude/aMazeTest
export AGENT_PROMPT="Search the web for the current price per unit, then read the file at /data/cloude/aMazeTest/policies/policy.json for additional details"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy_token_graph.json
echo "RESULT: PASS (graph and token limit satisfied as expected)"
