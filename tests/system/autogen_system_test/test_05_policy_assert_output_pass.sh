#!/bin/bash
# Policy: policy_assert_output.json (graph mode, fully mocked)
# LLM mock: input contains "pdf" -> pdf_search("data governance frameworks")
# pdf_search mock: returns text with "page" and "Source: *.pdf"
# Assertions: output contains "page", output matches regex "Source: .+\.pdf",
#             input equals "data governance frameworks"
# Expectation: PASS — mocks fire, all assertions pass
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Search the pdf for information about data governance frameworks"
"$PYTHON" -m amaze.amaze_runner examples/agents/autogen_annotated_agent.py examples/policies/policy_assert_output.json
echo "RESULT: PASS (assertions and graph completed as expected)"
