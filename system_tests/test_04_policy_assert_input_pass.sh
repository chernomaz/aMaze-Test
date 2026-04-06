#!/bin/bash
# Policy: policy_assert_input.json (control_plane, fully mocked)
# Assertions: llm input contains "humanitarian", pdf_search input starts_with "Scenarios"
# LLM mock: contains "humanitarian" -> pdf_search("Scenarios in humanitarian data management")
# pdf_search mock: fixed output
# Expectation: PASS — mocks fire, assertions pass
set -e
cd /data/cloude/aMazeTest
export AGENT_PROMPT="Tell me about humanitarian data management and scenarios for coordination"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy_assert_input.json
echo "RESULT: PASS (assertions and policy satisfied as expected)"
