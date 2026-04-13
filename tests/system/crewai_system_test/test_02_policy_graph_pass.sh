#!/bin/bash
# Policy: policy_graph.json (graph: agent->llm->tool:pdf_search->finish)
# LLM mock: input contains "search" -> return pdf_search tool call
# Expectation: PASS — mock fires, pdf_search called, graph completes
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
export AGENT_PROMPT="Please search for information about humanitarian data management scenarios"
"$PYTHON" -m amaze.amaze_runner examples/agents/crewai_annotated_agent.py examples/policies/policy_graph.json
echo "RESULT: PASS (graph sequence completed as expected)"
