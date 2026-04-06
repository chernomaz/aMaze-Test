#!/bin/bash
# Policy: policy_graph.json (graph: agent->llm->tool:pdf_search->finish)
# LLM mock: input contains "search" -> return pdf_search tool call
# Expectation: PASS — mock fires, pdf_search called, graph completes
set -e
cd /data/cloude/aMazeTest
export AGENT_PROMPT="Please search for information about humanitarian data management scenarios"
/data/venv/bin/python -m amaze.amaze_runner agents/one_conversation_agent.py policies/policy_graph.json
echo "RESULT: PASS (graph sequence completed as expected)"
