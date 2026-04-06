#!/bin/bash
# Policy: policy_assert_output.json (graph mode, fully mocked)
# LLM mock: input contains "pdf" -> pdf_search("data governance frameworks")
# pdf_search mock: returns text with "page" and "Source: *.pdf"
# Assertions: output contains "page", output matches regex "Source: .+\.pdf",
#             input equals "data governance frameworks"
# Expectation: PASS — mocks fire, all assertions pass
set -e
cd /data/cloude/aMazeTest
export AGENT_PROMPT="Search the pdf for information about data governance frameworks"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy_assert_output.json
echo "RESULT: PASS (assertions and graph completed as expected)"
