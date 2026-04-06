#!/bin/bash
# Policy: policy_multiply.json (graph: agent->llm->web_search->dummy_email->finish)
# LLM mock: input contains "email" -> return web_search tool call
# web_search mock + dummy_email mock both set
# Expectation: PASS — full graph sequence completes via mocks
set -e
cd /data/cloude/aMazeTest
export AGENT_PROMPT="Get the latest company news and then check Alice's email"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy_multiply.json
echo "RESULT: PASS (graph sequence completed as expected)"
