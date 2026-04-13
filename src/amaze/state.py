import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from amaze.policy import (
    AssertionOperator,
    ControlPlanePolicy,
    GraphPolicy,
    MockConfig,
)


class PolicyViolation(Exception):
    pass


def _evaluate_assertion(operator: AssertionOperator, expected: Any, actual: Any) -> bool:
    s = str(actual)
    if operator == AssertionOperator.EQUALS:
        return actual == expected
    elif operator == AssertionOperator.CONTAINS:
        return str(expected) in s
    elif operator == AssertionOperator.STARTS_WITH:
        return s.startswith(str(expected))
    elif operator == AssertionOperator.MATCHES_REGEX:
        return bool(re.search(expected, s))
    return False


class RuntimeState:
    def __init__(self, policy, agent_name: str = "agent"):
        self.policy = policy
        self.agent_name = agent_name
        self.trace_id = str(uuid.uuid4())

        # Call counters
        self.llm_calls = 0          # direct LLM calls only
        self.indirect_llm_calls = 0
        self.tool_calls = 0
        self.tool_calls_by_name: dict = {}

        # Token tracking (bug fix: initialize here)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_log: list = []
        self.last_edge_id = None
        # Graph mode state
        if isinstance(policy, GraphPolicy):
            self.current_node: str = policy.nodes[0]
            self._adjacency: dict = policy.adjacency()
        else:
            self.current_node = ""
            self._adjacency = {}

        self.call_sequence: list = ["agent"]  # current turn's call nodes — always starts with "agent"
        self.agent_prompt: str = ""            # captured from the first direct LLM call
        self.final_answer: str = ""            # last non-indirect LLM response with no tool calls
        self.last_turn: dict = {}              # snapshot of completed turn stats (populated after each finish)

        # Per-turn ordered log of every LLM and tool call with inputs and outputs
        self.call_log: list = []

        # Context for indirect LLM hint injection
        self.last_llm_mock = None          # MockConfig used by the last direct LLM call
        self.last_tool_description: str = ""  # description of the last tool invoked

        self.turns: list = []          # per-turn snapshots (both modes)

        self.events: list = []
        self.assertion_failures: list = []
        self.passed = False

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, event_type: str, payload: dict):
        event = {
            "ts": time.time(),
            "trace_id": self.trace_id,
            "type": event_type,
            "payload": payload,
        }
        self.events.append(event)
        print(f"[STATE] {event_type} trace_id={self.trace_id} payload={payload}", flush=True)

    # ------------------------------------------------------------------
    # LLM entry
    # ------------------------------------------------------------------

    def enter_llm(self, model: str = "unknown", is_indirect: bool = False):
        if is_indirect:
            self.indirect_llm_calls += 1
            self.log("llm_call_indirect", {"count": self.indirect_llm_calls, "model": model})
            return

        # Control plane: enforce call limit
        if isinstance(self.policy, ControlPlanePolicy):
            if (self.policy.max_llm_calls is not None
                    and self.llm_calls >= self.policy.max_llm_calls):
                raise PolicyViolation(
                    f"LLM call limit {self.policy.max_llm_calls} exceeded"
                )

        # Graph mode: per-call validation
        if isinstance(self.policy, GraphPolicy):
            self.check_graph_step("llm")

        self.llm_calls += 1
        if isinstance(self.policy, ControlPlanePolicy):
            self.call_sequence.append("llm")
        self.log("llm_call", {"count": self.llm_calls, "model": model})

    # ------------------------------------------------------------------
    # Tool entry
    # ------------------------------------------------------------------

    def enter_tool(self, tool_name: str, input_data):
        if isinstance(self.policy, ControlPlanePolicy):
            if self.policy.allowed_tools and tool_name not in self.policy.allowed_tools:
                raise PolicyViolation(f"Tool '{tool_name}' is not in the allowed list")

            if (self.policy.max_tool_calls is not None
                    and self.tool_calls >= self.policy.max_tool_calls):
                raise PolicyViolation(
                    f"Tool call limit {self.policy.max_tool_calls} exceeded"
                )

            per_limit = self.policy.max_tool_calls_per_tool.get(tool_name)
            if per_limit is not None:
                if self.tool_calls_by_name.get(tool_name, 0) >= per_limit:
                    raise PolicyViolation(
                        f"Tool '{tool_name}' call limit {per_limit} exceeded"
                    )

        # Graph mode: per-call validation
        if isinstance(self.policy, GraphPolicy):
            self.check_graph_step(f"tool:{tool_name}")

        self.tool_calls += 1
        self.tool_calls_by_name[tool_name] = self.tool_calls_by_name.get(tool_name, 0) + 1
        if isinstance(self.policy, ControlPlanePolicy):
            self.call_sequence.append(f"tool:{tool_name}")
        self.log("tool_call", {
            "tool": tool_name,
            "count": self.tool_calls,
            "input": str(input_data)[:200],
        })

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def add_token_usage(self, input_tokens: int = 0, output_tokens: int = 0, model: str = None):
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        total = input_tokens + output_tokens

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total

        for entry in reversed(self.call_log):
            if entry.get("type") == "llm":
                entry["input_tokens"] = input_tokens
                entry["output_tokens"] = output_tokens
                entry["total_tokens"] = total
                entry["model"] = model
                entry["ended_at"] = time.time()
                break

        self.log("token_usage", {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "running_total": self.total_tokens,
        })

        if (self.policy.max_tokens is not None
                and self.total_tokens > self.policy.max_tokens):
            raise PolicyViolation(
                f"Token limit {self.policy.max_tokens} exceeded (used {self.total_tokens})"
            )

    # ------------------------------------------------------------------
    # Graph validation
    # ------------------------------------------------------------------

    def check_graph_step(self, step: str):
        """Validate a graph transition. On bad transition: record violation, reset
        to start so the next agent turn begins cleanly, then raise PolicyViolation."""
        successors = self._adjacency.get(self.current_node, [])
        if step not in successors:
            msg = (
                f"No edge from '{self.current_node}' to '{step}' in graph. "
                f"Allowed next steps: {successors}"
            )
            self.assertion_failures.append(msg)
            self.log("graph_violation", {"message": msg})
            self._reset_for_next_turn()
            raise PolicyViolation(msg)
        self.current_node = step
        self.call_sequence.append(step)

    def advance_finish_if_complete(self):
        """Signal that the current agent turn ended normally (LLM returned a final answer /
        chain_end fired). Snapshots the turn and resets per-turn counters for both modes.
        The final audit write is handled by amaze_runner after passed is determined."""
        if isinstance(self.policy, GraphPolicy):
            if self.current_node == "finish":
                return
            if "finish" not in self._adjacency.get(self.current_node, []):
                return
            self.current_node = "finish"
            self.call_sequence.append("finish")
            self.log("graph_node", {"node": "finish"})
        else:
            # Control plane: guard against double-fire (e.g. patched LLM + chain_end callback)
            if self.llm_calls == 0 and self.tool_calls == 0:
                return
        self._reset_for_next_turn()

    def _reset_for_next_turn(self):
        """Reset per-turn counters and snapshot stats for both graph and control plane modes.
        Idempotent: no-op if there is nothing to reset."""
        if isinstance(self.policy, GraphPolicy):
            start = self.policy.nodes[0]
            if self.current_node == start:
                return
            self.current_node = start
        else:
            # Control plane: no-op if the turn was already reset or never started
            if self.llm_calls == 0 and self.tool_calls == 0:
                return

        # Snapshot completed turn stats before clearing
        turn_snapshot = {
            "turn": len(self.turns) + 1,
            "call_sequence": list(self.call_sequence),
            "llm_calls": self.llm_calls,
            "indirect_llm_calls": self.indirect_llm_calls,
            "tool_calls": self.tool_calls,
            "tool_calls_by_name": dict(self.tool_calls_by_name),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "call_log": list(self.call_log),
        }
        self.last_turn = turn_snapshot
        self.turns.append(turn_snapshot)

        # Reset per-turn counters
        self.llm_calls = 0
        self.indirect_llm_calls = 0
        self.tool_calls = 0
        self.tool_calls_by_name = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_sequence = ["agent"]
        self.call_log = []
        self.last_edge_id = None
        self.last_llm_mock = None
        self.last_tool_description = ""

        if isinstance(self.policy, GraphPolicy):
            self.log("graph_reset", {"reset_to": self.current_node, "turn": len(self.turns)})
        else:
            self.log("turn_reset", {"turn": len(self.turns)})

    def validate_graph_complete(self) -> list:
        """End-of-run check: the run must have explicitly reached 'finish'.
        Acceptable states: current_node is 'finish' (not yet reset) or the start node
        (reset after a successful turn). Any other node means the run ended mid-turn."""
        if not isinstance(self.policy, GraphPolicy):
            return []
        start = self.policy.nodes[0]
        if self.current_node in ("finish", start):
            return []
        return [
            f"Graph sequence incomplete: stopped at '{self.current_node}', "
            f"expected to reach 'finish'.\n"
            f"  Call sequence so far: {self.call_sequence}"
        ]

    # ------------------------------------------------------------------
    # Mock resolution
    # ------------------------------------------------------------------

    def find_mock(self, target: str, input_text: str) -> Optional[MockConfig]:
        """Find first mock whose target matches and match_contains (if set) is in input_text."""
        for mock in self.policy.mocks:
            if mock.target != target:
                continue
            if mock.match_contains is None or mock.match_contains in input_text:
                return mock
        return None

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def run_assertions(self, target: str, check: str, value: Any):
        """Evaluate all matching assertions; failures are recorded (not raised)."""
        for assertion in self.policy.assertions:
            if assertion.target != target:
                continue
            if assertion.check != check:
                continue



            passed = _evaluate_assertion(assertion.operator, assertion.expected, value)
            self.record_assertion(
                description=assertion.description or f"{target}.{check}",
                passed=passed
            )
            if not passed:
                label = assertion.description or f"{target}.{check}"
                msg = (
                    f"Assertion FAILED [{label}]: "
                    f"{check}={repr(str(value))!r} does not satisfy "
                    f"{assertion.operator.value}({repr(assertion.expected)!r})"
                )
                self.assertion_failures.append(msg)
                self.log("assertion_failure", {"message": msg})

    # ------------------------------------------------------------------
    # Call output recording
    # ------------------------------------------------------------------
    def record_llm_output(self, input, output, has_tool_calls, indirect=False, mocked=False):
        edge_id = str(uuid.uuid4())
        now = time.time()
        entry = {
            "id": edge_id,
            "parent_id": self.last_edge_id,
            "type": "llm",
            "indirect": indirect,
            "input": input,
            "output": output,
            "mocked": mocked,
            "has_tool_calls": has_tool_calls,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "status": "ok",
            "started_at": now,
            "ended_at": None,
            "source": "mock" if mocked else "real"
        }
        self.call_log.append(entry)
        self.last_edge_id = edge_id
        if not has_tool_calls and output:
            self.final_answer = output

    def record_tool_output(self, tool_name: str, input_text: str, output: str, mocked: bool = False):
        edge_id = str(uuid.uuid4())
        now = time.time()

        entry = {
            "id": edge_id,
            "parent_id": self.last_edge_id,
            "type": "tool",
            "name": tool_name,
            "input": input_text,
            "output": output,
            "mocked": mocked,
            "status": "ok",
            "started_at": now,
            "ended_at": now,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "source": "mock" if mocked else "real"
        }

        self.call_log.append(entry)
        self.last_edge_id = edge_id


    def record_assertion(self, description, passed):
        self.call_log.append({
            "id": str(uuid.uuid4()),
            "type": "assertion",
            "description": description,
            "passed": passed,
            "status": "ok" if passed else "failed",
            "ts": time.time()
        })

    def write(self, expected_pass=None):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent_name}_audit_{timestamp}.json"
        audit_dir = Path(__file__).resolve().parent.parent.parent / "reports"
        audit_dir.mkdir(parents=True, exist_ok=True)
        path = audit_dir / filename
        print(f"[STATE] writing audit file: {path}", flush=True)
        # Include any uncommitted active turn (write() is called before _reset_for_next_turn)
        all_turns = list(self.turns)
        if self.llm_calls > 0 or self.tool_calls > 0:
            all_turns.append({
                "turn": len(self.turns) + 1,
                "call_sequence": list(self.call_sequence),
                "llm_calls": self.llm_calls,
                "indirect_llm_calls": self.indirect_llm_calls,
                "tool_calls": self.tool_calls,
                "tool_calls_by_name": dict(self.tool_calls_by_name),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "call_log": list(self.call_log),
            })
        self.audit_path = path
        with path.open("w", encoding="utf-8") as f:
            payload = {
                "trace_id": self.trace_id,
                "agent_prompt": self.agent_prompt,
                "final_answer": self.final_answer,
                "passed": self.passed,
                "policy": _serialize_policy(self.policy),
                "turns": all_turns,
                "assertion_failures": self.assertion_failures,
                "events": self.events,
            }
            if expected_pass is not None:
                payload["expected_pass"] = expected_pass
            json.dump(payload, f, indent=2, ensure_ascii=False)


def _serialize_policy(policy) -> dict:
    from amaze.policy import GraphPolicy, ControlPlanePolicy
    if isinstance(policy, GraphPolicy):
        return {
            "mode": "graph",
            "nodes": policy.nodes,
            "edges": policy.edges,
            "ignore_internal_llm": policy.ignore_internal_llm,
            "max_tokens": policy.max_tokens,
            "mocks": [_serialize_mock(m) for m in policy.mocks],
            "assertions": [_serialize_assertion(a) for a in policy.assertions],
        }
    elif isinstance(policy, ControlPlanePolicy):
        return {
            "mode": "control_plane",
            "allowed_tools": sorted(policy.allowed_tools),
            "max_llm_calls": policy.max_llm_calls,
            "max_tool_calls": policy.max_tool_calls,
            "max_tool_calls_per_tool": policy.max_tool_calls_per_tool,
            "max_tokens": policy.max_tokens,
            "mocks": [_serialize_mock(m) for m in policy.mocks],
            "assertions": [_serialize_assertion(a) for a in policy.assertions],
        }
    return {}


def _serialize_mock(mock) -> dict:
    return {k: v for k, v in vars(mock).items() if v is not None}


def _serialize_assertion(assertion) -> dict:
    return {
        "target": assertion.target,
        "check": assertion.check,
        "operator": assertion.operator.value,
        "expected": assertion.expected,
        "description": assertion.description,
    }
