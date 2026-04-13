#!/bin/bash
# Policy: policy_assert_input.json (control_plane, fully mocked)
# Assertions: llm input contains "humanitarian", pdf_search input starts_with "Scenarios"
# LLM mock: contains "humanitarian" -> pdf_search("Scenarios in humanitarian data management")
# pdf_search mock: fixed output
# Expectation: PASS — mocks fire, assertions pass
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Tell me about humanitarian data management and scenarios for coordination"
"$PYTHON" -m amaze.amaze_runner examples/agents/crewai_annotated_agent.py examples/policies/policy_assert_input.json
echo "RESULT: PASS (assertions and policy satisfied as expected)"
