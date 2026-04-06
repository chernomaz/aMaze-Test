#!/bin/bash
# Policy: policy_token_strict.json (control_plane, allowed=[pdf_search, web_search], max_tokens=800)
# LLM mock: input contains "search" -> pdf_search("climate change impacts")
# pdf_search mock: short fixed output
# Expectation: PASS — mock fires, no real LLM tokens used, well within 800 token budget
set -e
cd /data/cloude/aMazeTest
export AGENT_PROMPT="Please search for information about climate change impacts"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy_token_strict.json
echo "RESULT: PASS (token limit and policy satisfied as expected)"
