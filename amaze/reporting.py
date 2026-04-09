from pathlib import Path
import json
import html
import os
import webbrowser
from datetime import datetime
from typing import Any


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _pretty_json(value: Any) -> str:
    return _esc(json.dumps(value, indent=2, ensure_ascii=False))


def _fmt_ts(ts):
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except Exception:
        return str(ts)


def _duration_ms(started_at, ended_at):
    if started_at is None or ended_at is None:
        return None
    try:
        return round((float(ended_at) - float(started_at)) * 1000, 1)
    except Exception:
        return None


def _component_lane(edge: dict) -> str:
    t = edge.get("type")
    if t == "llm":
        return "LLM"
    if t == "tool":
        return f"tool:{edge.get('name', 'unknown')}"
    if t == "assertion":
        return "assertion"
    if t == "graph":
        return "graph"
    return t or "unknown"


def _event_severity(event: dict) -> str:
    et = (event or {}).get("type", "")
    if "fail" in et or "violation" in et or "blocked" in et:
        return "failed"
    return "ok"


def _extract_policy(audit: dict) -> dict:
    # Prefer explicit policy serialized by state.py
    if isinstance(audit.get("policy"), dict) and audit["policy"].get("mode"):
        return audit["policy"]

    # Fallback: infer a minimal policy-like block from available data.
    turns = audit.get("turns", [])
    tools = set()
    call_sequence_examples = []
    for turn in turns:
        seq = turn.get("call_sequence") or []
        if seq:
            call_sequence_examples.append(seq)
            for item in seq:
                if isinstance(item, str) and item.startswith("tool:"):
                    tools.add(item.split("tool:", 1)[1])

        for entry in turn.get("call_log", []):
            if entry.get("type") == "tool" and entry.get("name"):
                tools.add(entry["name"])

    return {
        "mode": audit.get("policy_mode", "unknown"),
        "allowed_tools": sorted(tools),
        "call_sequence_examples": call_sequence_examples,
    }


def _format_policy_text(policy: dict) -> str:
    mode = policy.get("mode", "unknown")
    lines = [f"mode: {mode}"]

    if mode == "graph":
        nodes = policy.get("nodes", [])
        edges = policy.get("edges", [])
        lines.append(f"nodes: {' → '.join(nodes)}")
        for src, dst in edges:
            lines.append(f"  edge: {src} → {dst}")
        ignore = policy.get("ignore_internal_llm", True)
        lines.append(f"ignore_internal_llm: {ignore}")
    elif mode == "control_plane":
        tools = policy.get("allowed_tools", [])
        lines.append(f"allowed_tools: {', '.join(tools) if tools else '(any)'}")
        if policy.get("max_llm_calls") is not None:
            lines.append(f"max_llm_calls: {policy['max_llm_calls']}")
        if policy.get("max_tool_calls") is not None:
            lines.append(f"max_tool_calls: {policy['max_tool_calls']}")
        per_tool = policy.get("max_tool_calls_per_tool", {})
        for k, v in per_tool.items():
            lines.append(f"  max_{k}_calls: {v}")
    else:
        # Fallback: show observed sequence
        for seq in policy.get("call_sequence_examples", []):
            lines.append("  " + " → ".join(seq))

    if policy.get("max_tokens") is not None:
        lines.append(f"max_tokens: {policy['max_tokens']}")

    mocks = policy.get("mocks", [])
    if mocks:
        lines.append(f"\nmocks:  ({len(mocks)} configured)")
        for m in mocks:
            target = m.get("target", "")
            match = m.get("match_contains")
            line = f"  - {target}"
            if match:
                line += f"  [match: \"{match}\"]"
            lines.append(line)
            if m.get("return_tool_call"):
                tc = m["return_tool_call"]
                lines.append(f"    → tool_call: {tc.get('tool','?')}({tc.get('args',{})})")
            elif m.get("return_ai_message"):
                msg = str(m["return_ai_message"])
                lines.append(f"    → ai_message: \"{msg[:60]}{'…' if len(msg)>60 else ''}\"")
            elif m.get("output"):
                out = str(m["output"])
                lines.append(f"    → output: \"{out[:60]}{'…' if len(out)>60 else ''}\"")

    assertions = policy.get("assertions", [])
    if assertions:
        lines.append(f"\nassertions:  ({len(assertions)} configured)")
        for a in assertions:
            desc = a.get("description") or f"{a.get('target','')}.{a.get('check','')}"
            lines.append(f"  - [{desc}] {a.get('operator','')}({repr(a.get('expected',''))})")

    return "\n".join(lines)


def _build_edges(turns: list[dict]) -> list[dict]:
    edges = []
    for turn in turns:
        for i, entry in enumerate(turn.get("call_log", []), start=1):
            started_at = entry.get("started_at")
            ended_at = entry.get("ended_at")
            edge = {
                "turn": turn.get("turn"),
                "index": i,
                "id": entry.get("id", ""),
                "parent_id": entry.get("parent_id"),
                "type": entry.get("type"),
                "name": entry.get("name", ""),
                "input": entry.get("input", ""),
                "output": entry.get("output", ""),
                "mocked": entry.get("mocked", False),
                "indirect": entry.get("indirect", False),
                "has_tool_calls": entry.get("has_tool_calls", False),
                "input_tokens": entry.get("input_tokens", 0) or 0,
                "output_tokens": entry.get("output_tokens", 0) or 0,
                "total_tokens": entry.get("total_tokens", 0) or 0,
                "status": entry.get("status", "ok"),
                "started_at": started_at,
                "ended_at": ended_at,
                "started_at_fmt": _fmt_ts(started_at),
                "ended_at_fmt": _fmt_ts(ended_at),
                "duration_ms": _duration_ms(started_at, ended_at),
                "description": entry.get("description", ""),
                "passed": entry.get("passed"),
                "model": entry.get("model", ""),
                "source": entry.get("source", "mock" if entry.get("mocked") else "real"),
            }
            edges.append(edge)
    return edges


def _build_sequence_steps(edges: list[dict]) -> list[dict]:
    """
    Build actual sequence edges using parent_id when available.
    Fallback to previous-step chaining if IDs are missing.
    """
    id_map = {e["id"]: e for e in edges if e.get("id")}
    steps = []

    for pos, edge in enumerate(edges, start=1):
        from_lane = "agent"
        parent_id = edge.get("parent_id")
        if parent_id and parent_id in id_map:
            from_lane = _component_lane(id_map[parent_id])
        elif pos > 1:
            from_lane = _component_lane(edges[pos - 2])

        to_lane = _component_lane(edge)

        label_parts = [edge.get("type", "")]
        if edge.get("type") == "tool" and edge.get("name"):
            label_parts = [f"tool:{edge['name']}"]
        elif edge.get("type") == "llm":
            label_parts = ["llm_indirect" if edge.get("indirect") else "llm"]
        elif edge.get("type") == "assertion":
            label_parts = [f"assert:{edge.get('description') or edge.get('name') or 'assertion'}"]

        steps.append(
            {
                "turn": edge.get("turn"),
                "index": edge.get("index"),
                "edge_id": edge.get("id"),
                "from": from_lane,
                "to": to_lane,
                "label": " ".join(label_parts).strip(),
                "mocked": edge.get("mocked", False),
                "status": edge.get("status", "ok"),
            }
        )

    # Add a finish edge from the last meaningful component.
    if edges:
        last = edges[-1]
        if last.get("type") == "llm" and not last.get("has_tool_calls", False):
            steps.append(
                {
                    "turn": last.get("turn"),
                    "index": (last.get("index") or 0) + 1,
                    "edge_id": "",
                    "from": _component_lane(last),
                    "to": "finish",
                    "label": "final_answer",
                    "mocked": False,
                    "status": "ok",
                }
            )

    return steps


def _build_tool_stats(edges: list[dict]) -> list[dict]:
    by_tool = {}
    for e in edges:
        if e.get("type") != "tool":
            continue
        name = e.get("name") or "unknown"
        row = by_tool.setdefault(
            name,
            {
                "name": name,
                "calls": 0,
                "mocked_calls": 0,
                "failed_calls": 0,
            },
        )
        row["calls"] += 1
        if e.get("mocked"):
            row["mocked_calls"] += 1
        if e.get("status") == "failed":
            row["failed_calls"] += 1
    return sorted(by_tool.values(), key=lambda x: (-x["calls"], x["name"]))


def _build_turn_stats(turns: list[dict]) -> list[dict]:
    rows = []
    for turn in turns:
        rows.append(
            {
                "turn": turn.get("turn"),
                "call_sequence": turn.get("call_sequence", []),
                "llm_calls": turn.get("llm_calls", 0),
                "indirect_llm_calls": turn.get("indirect_llm_calls", 0),
                "tool_calls": turn.get("tool_calls", 0),
                "tool_calls_by_name": turn.get("tool_calls_by_name", {}),
                "total_input_tokens": turn.get("total_input_tokens", 0),
                "total_output_tokens": turn.get("total_output_tokens", 0),
                "total_tokens": turn.get("total_tokens", 0),
            }
        )
    return rows


def _build_event_rows(events: list[dict]) -> list[dict]:
    rows = []
    for i, event in enumerate(events, start=1):
        rows.append(
            {
                "index": i,
                "ts": event.get("ts"),
                "ts_fmt": _fmt_ts(event.get("ts")),
                "type": event.get("type", ""),
                "payload": event.get("payload", {}),
                "severity": _event_severity(event),
            }
        )
    return rows


def build_report_model(audit: dict) -> dict:
    turns = audit.get("turns", [])
    events = audit.get("events", [])
    edges = _build_edges(turns)

    # Fix final_answer: use the last non-indirect LLM output that contains real text.
    # The audit's stored final_answer can be wrong when the first mocked LLM call has
    # has_tool_calls=False (causing state.py to record the mock output as the answer).
    final_answer = audit.get("final_answer", "")
    for turn in reversed(turns):
        for entry in reversed(turn.get("call_log", [])):
            if entry.get("type") == "llm" and not entry.get("indirect") and entry.get("output"):
                final_answer = entry["output"]
                break
        else:
            continue
        break

    # Prepend a synthetic user→agent edge so the Execution Edges table mirrors the diagram
    agent_prompt = audit.get("agent_prompt", "")
    if agent_prompt:
        user_edge = {
            "turn": 1,
            "index": 0,
            "id": "__user_agent__",
            "parent_id": None,
            "type": "user",
            "name": "user",
            "input": agent_prompt,
            "output": "(task accepted)",
            "mocked": False,
            "indirect": False,
            "has_tool_calls": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "status": "ok",
            "started_at": None,
            "ended_at": None,
            "started_at_fmt": "",
            "ended_at_fmt": "",
            "duration_ms": None,
            "description": "",
            "passed": None,
            "model": "",
            "source": "real",
        }
        edges = [user_edge] + edges

    sequence_steps = _build_sequence_steps([e for e in edges if e.get("id") != "__user_agent__"])
    policy = _extract_policy(audit)
    assertion_failures = audit.get("assertion_failures", []) or []

    failed_edges = [e for e in edges if e.get("status") == "failed"]
    tool_stats = _build_tool_stats(edges)
    turn_stats = _build_turn_stats(turns)
    event_rows = _build_event_rows(events)

    llm_edges = [e for e in edges if e.get("type") == "llm"]
    tool_edges = [e for e in edges if e.get("type") == "tool"]

    all_timestamps = []
    for e in edges:
        if e.get("started_at") is not None:
            all_timestamps.append(float(e["started_at"]))
        if e.get("ended_at") is not None:
            all_timestamps.append(float(e["ended_at"]))
    for ev in events:
        if ev.get("ts") is not None:
            try:
                all_timestamps.append(float(ev["ts"]))
            except Exception:
                pass

    run_duration_ms = None
    if all_timestamps:
        run_duration_ms = round((max(all_timestamps) - min(all_timestamps)) * 1000, 1)

    violations = []
    for edge in failed_edges:
        violations.append(
            {
                "kind": edge.get("type", "component"),
                "name": edge.get("name") or edge.get("description") or edge.get("type"),
                "status": edge.get("status"),
                "turn": edge.get("turn"),
                "index": edge.get("index"),
                "details": edge.get("output") or edge.get("input") or "",
            }
        )

    for failure in assertion_failures:
        if isinstance(failure, str):
            violations.append(
                {
                    "kind": "assertion",
                    "name": "assertion_failure",
                    "status": "failed",
                    "turn": "",
                    "index": "",
                    "details": failure,
                }
            )
        else:
            violations.append(
                {
                    "kind": "assertion",
                    "name": failure.get("name") or failure.get("description") or "assertion_failure",
                    "status": "failed",
                    "turn": failure.get("turn", ""),
                    "index": failure.get("index", ""),
                    "details": failure,
                }
            )

    summary = {
        "trace_id": audit.get("trace_id", ""),
        "passed": audit.get("passed", False),
        "agent_prompt": agent_prompt,
        "final_answer": final_answer,
        "turn_count": len(turns),
        "edge_count": len(edges),
        "event_count": len(events),
        "assertion_failures": len(assertion_failures),
        "violations_count": len(violations),
        "run_duration_ms": run_duration_ms,
        "llm_calls": len(llm_edges),
        "indirect_llm_calls": sum(1 for e in llm_edges if e.get("indirect")),
        "tool_calls": len(tool_edges),
        "unique_tools": len({e.get('name') for e in tool_edges if e.get('name')}),
        "assertion_calls": sum(1 for e in edges if e.get("type") == "assertion"),
        "mock_calls": sum(1 for e in edges if e.get("mocked")),
        "failed_components": len(failed_edges),
        "total_input_tokens": sum(e.get("input_tokens", 0) for e in llm_edges),
        "total_output_tokens": sum(e.get("output_tokens", 0) for e in llm_edges),
        "total_tokens": sum(e.get("total_tokens", 0) for e in llm_edges),
    }

    return {
        "summary": summary,
        "policy": policy,
        "edges": edges,
        "sequence_steps": sequence_steps,
        "tool_stats": tool_stats,
        "turn_stats": turn_stats,
        "events": event_rows,
        "violations": violations,
        "assertion_failures": assertion_failures,
        "raw": audit,
    }


_TEMPLATE_PATH = Path(__file__).parent / "report_template.html"


def render_html(report: dict) -> str:
    """Inject report data into the static HTML template.

    The template (report_template.html) contains all CSS and JavaScript.
    Only the JSON data blob changes between runs.
    """
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")

    # Build the report payload for the template (exclude the bulky raw audit)
    payload = {
        "summary": report["summary"],
        "policy": report["policy"],
        "edges": report["edges"],
        "sequence_steps": report["sequence_steps"],
        "violations": report["violations"],
        "tool_stats": report["tool_stats"],
        "assertion_failures": report["assertion_failures"],
    }

    # JSON-safe embedding: unicode-escape <, >, & so JSON.parse works inside a script tag
    report_json = json.dumps(payload, ensure_ascii=False)
    safe_json = report_json.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")

    return template.replace("%%REPORT_JSON%%", safe_json)


# ---------------------------------------------------------------------------
# Legacy: kept only so callers that import individual helpers don't break.
# ---------------------------------------------------------------------------
def _render_html_legacy(report: dict) -> str:
    """Original Python-side renderer (kept for reference, not used)."""
    s = report["summary"]
    policy = report["policy"]
    policy_text = _format_policy_text(policy)

    passed = s["passed"]
    result_label = "PASSED" if passed else "FAILED"
    trace_short = s["trace_id"][:8] if s["trace_id"] else ""

    assertions_total = s["assertion_calls"]
    assertions_failed = s["assertion_failures"]
    assertions_passed = assertions_total - assertions_failed
    assertions_display = f"{assertions_passed} / {assertions_total}" if assertions_total else "—"
    assertions_cls = "fail" if assertions_failed else "ok"
    token_cls = "warn" if s["total_tokens"] > 0 and s["violations_count"] else "ok"
    mock_cls = "warn" if s["mock_calls"] > 0 else "ok"

    # Build edge rows — full input/output, with timestamp and data-id for highlight
    edge_rows = []
    for e in report["edges"]:
        classes = []
        if e["mocked"]:
            classes.append("mock")
        if e["status"] == "failed":
            classes.append("fail")
        if e["type"] == "assertion":
            classes.append("assertion")

        kind = e["type"].upper() if e["type"] else ""
        name = e["name"] or e["description"] or ""
        source = e.get("source", "mocked" if e["mocked"] else "real")
        duration = "" if e["duration_ms"] is None else e["duration_ms"]
        edge_id = e.get("id", "")
        ts_fmt = e.get("started_at_fmt", "")

        edge_rows.append(f"""
            <tr id="edge-{_esc(edge_id)}" class="{' '.join(classes)}">
              <td>{_esc(e['turn'])}</td>
              <td>{_esc(e['index'])}</td>
              <td class="ts-col">{_esc(ts_fmt)}</td>
              <td><span class="badge {e['type'] or ''}">{_esc(kind)}</span></td>
              <td>{_esc(name)}</td>
              <td>{'<span class="badge mock">indirect</span>' if e.get('indirect') else ''}</td>
              <td>{_esc(source)}</td>
              <td>{_esc(e.get('model', ''))}</td>
              <td>{duration}</td>
              <td>{e['input_tokens']}</td>
              <td>{e['output_tokens']}</td>
              <td>{e['total_tokens']}</td>
              <td><span class="badge {'fail' if e['status'] == 'failed' else 'ok-badge'}">{_esc(e['status'])}</span></td>
              <td><div class="code-full">{_esc(e['input'])}</div></td>
              <td><div class="code-full">{_esc(e['output'])}</div></td>
            </tr>""")

    tool_rows = []
    for row in report["tool_stats"]:
        tool_rows.append(f"""
            <tr>
              <td>{_esc(row['name'])}</td>
              <td>{row['calls']}</td>
              <td>{row['mocked_calls']}</td>
              <td>{row['failed_calls']}</td>
            </tr>""")

    violation_rows = []
    for row in report["violations"]:
        violation_rows.append(f"""
            <tr class="fail">
              <td>{_esc(row['kind'])}</td>
              <td>{_esc(row['name'])}</td>
              <td>{_esc(row['turn'])}</td>
              <td>{_esc(row['index'])}</td>
              <td>{_esc(row['status'])}</td>
              <td><div class="code-full">{_esc(row['details'])}</div></td>
            </tr>""")
    if not violation_rows:
        violation_rows.append('<tr><td colspan="6" class="muted">No violations detected</td></tr>')

    # JSON-safe embedding: use unicode escapes so JSON.parse works inside script tag
    report_json = json.dumps(report["raw"], ensure_ascii=False)
    safe_json = report_json.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")

    # Pre-built sequence steps as JSON (avoids rebuilding in JS and carries edge_id for linking)
    steps_json = json.dumps(report["sequence_steps"], ensure_ascii=False)
    safe_steps = steps_json.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>aMaze Audit Report</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121a2b;
      --panel-2: #18233a;
      --text: #e8eefc;
      --muted: #9fb0d1;
      --line: #2b3a59;
      --green: #1fbf75;
      --yellow: #f2b94b;
      --red: #e05252;
      --blue: #5da9ff;
      --violet: #a97cff;
      --cyan: #43d1c6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      background: linear-gradient(180deg, #09101d 0%, #0d1426 100%);
      color: var(--text);
    }}
    .container {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}

    .topbar {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 20px; }}
    .title {{ font-size: 28px; font-weight: 800; letter-spacing: 0.2px; }}
    .subtitle {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .chip {{ border: 1px solid var(--line); background: rgba(255,255,255,0.03); color: var(--muted); padding: 8px 10px; border-radius: 999px; font-size: 12px; }}

    .card {{
      background: linear-gradient(180deg, var(--panel) 0%, #0f1627 100%);
      border: 1px solid var(--line); border-radius: 18px; padding: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.22);
    }}
    .card h3, .section h3 {{
      margin: 0 0 12px 0; font-size: 14px; color: var(--muted);
      font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    }}

    /* Summary 3-col grid */
    .grid-summary {{
      display: grid;
      grid-template-columns: 2fr 1fr 1.4fr;
      gap: 14px;
      margin-bottom: 18px;
      align-items: start;
    }}
    .kvs {{ display: grid; grid-template-columns: 150px 1fr; row-gap: 8px; column-gap: 12px; font-size: 14px; }}
    .kvs .k {{ color: var(--muted); }}

    .prompt-label {{ color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; margin: 14px 0 5px 0; }}
    .codeblock {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 10px; padding: 10px; overflow: auto; max-height: 100px; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.4; color: #d7e2ff; }}

    /* Metrics mini-grid 3×2 */
    .mini-metrics {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 4px; }}
    .mini-metric {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }}
    .mini-metric .mlabel {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; }}
    .mini-metric .mval {{ font-size: 24px; font-weight: 800; margin-top: 6px; }}
    .mval.ok {{ color: var(--green); }}
    .mval.warn {{ color: var(--yellow); }}
    .mval.fail {{ color: var(--red); }}

    .policy-box {{
      background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px;
      padding: 12px; color: #d7e2ff; white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px; line-height: 1.5; max-height: 340px; overflow: auto;
    }}

    /* Sequence diagram */
    .seq-card {{ margin-bottom: 18px; }}
    #diagram {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px; padding: 10px; overflow-x: auto; min-height: 200px; margin-top: 10px; }}
    #diagram svg .seq-step {{ cursor: pointer; }}
    #diagram svg .seq-step:hover rect {{ opacity: 0.12; }}
    .legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 8px; color: var(--muted); font-size: 12px; }}
    .dot {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; }}

    /* Sections */
    .section {{ background: linear-gradient(180deg, var(--panel) 0%, #0f1627 100%); border: 1px solid var(--line); border-radius: 18px; padding: 16px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.22); }}

    /* Tables */
    .scroll-x {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; vertical-align: top; text-align: left; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; background: var(--panel-2); white-space: nowrap; }}
    tr:hover td {{ background: rgba(255,255,255,0.025); }}
    tr.mock {{ background: rgba(169,124,255,0.07); }}
    tr.fail {{ background: rgba(224,82,82,0.09); }}
    tr.assertion {{ background: rgba(67,209,198,0.07); }}
    tr.highlighted td {{ background: rgba(93,169,255,0.18) !important; outline: 2px solid var(--blue); }}
    .muted {{ color: var(--muted); }}
    .ts-col {{ font-size: 11px; color: var(--muted); white-space: nowrap; font-family: ui-monospace, monospace; }}

    /* Badges */
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .badge.llm {{ background: rgba(93,169,255,.16); color: #9fd0ff; }}
    .badge.tool {{ background: rgba(93,169,255,.16); color: #9fd0ff; }}
    .badge.mock {{ background: rgba(169,124,255,.16); color: #cab0ff; }}
    .badge.assertion {{ background: rgba(67,209,198,.18); color: #8ef1e6; }}
    .badge.fail {{ background: rgba(224,82,82,.16); color: #ffb3b3; }}
    .badge.ok-badge {{ background: rgba(31,191,117,.10); color: #6de8ae; }}

    /* Full-content code cell */
    .code-full {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px;
      padding: 6px 8px; font-size: 11px; min-width: 180px; max-width: 400px;
      max-height: 120px; overflow: auto; white-space: pre-wrap;
      word-break: break-word; color: #dce6ff;
    }}

    @media (max-width: 1100px) {{
      .grid-summary {{ grid-template-columns: 1fr 1fr; }}
      .mini-metrics {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">

    <!-- Top bar -->
    <div class="topbar">
      <div>
        <div class="title">aMaze Test Audit Report</div>
        <div class="subtitle">Behavioral test · trace_id: {_esc(s['trace_id'])}</div>
      </div>
      <div class="chips">
        <div class="chip">Turns: {s['turn_count']}</div>
        <div class="chip">Duration: {_esc(s['run_duration_ms'])} ms</div>
        <div class="chip">{'✓ PASSED' if passed else '✗ FAILED'}</div>
      </div>
    </div>

    <!-- Summary grid -->
    <div class="grid-summary">

      <!-- Test Run Summary — includes prompt & final answer -->
      <div class="card">
        <h3>Test Run Summary</h3>
        <div class="kvs">
          <div class="k">Trace ID</div><div>{_esc(trace_short)}…</div>
          <div class="k">Result</div><div><span class="badge {'fail' if not passed else 'ok-badge'}">{result_label}</span></div>
          <div class="k">Turns</div><div>{s['turn_count']}</div>
          <div class="k">Violations</div><div>{s['violations_count']}</div>
          <div class="k">Mock Calls</div><div>{s['mock_calls']}</div>
          <div class="k">Unique Tools</div><div>{s['unique_tools']}</div>
          <div class="k">Indirect LLM</div><div>{s['indirect_llm_calls']}</div>
          <div class="k">Tokens In / Out</div><div>{s['total_input_tokens']} / {s['total_output_tokens']}</div>
        </div>
        <div class="prompt-label">Agent Prompt</div>
        <div class="codeblock"><pre>{_esc(s['agent_prompt']) or '<em style="color:var(--muted)">—</em>'}</pre></div>
        <div class="prompt-label">Final Answer</div>
        <div class="codeblock"><pre>{_esc(s['final_answer']) or '<em style="color:var(--muted)">—</em>'}</pre></div>
      </div>

      <!-- Metrics (6 items, 3-col) -->
      <div class="card">
        <h3>Metrics</h3>
        <div class="mini-metrics">
          <div class="mini-metric">
            <div class="mlabel">Total Tokens</div>
            <div class="mval {token_cls}">{s['total_tokens']}</div>
          </div>
          <div class="mini-metric">
            <div class="mlabel">LLM Calls</div>
            <div class="mval ok">{s['llm_calls']}</div>
          </div>
          <div class="mini-metric">
            <div class="mlabel">Tool Calls</div>
            <div class="mval ok">{s['tool_calls']}</div>
          </div>
          <div class="mini-metric">
            <div class="mlabel">Mock Calls</div>
            <div class="mval {mock_cls}">{s['mock_calls']}</div>
          </div>
          <div class="mini-metric">
            <div class="mlabel">Assertions</div>
            <div class="mval {assertions_cls}">{assertions_display}</div>
          </div>
          <div class="mini-metric">
            <div class="mlabel">Failed</div>
            <div class="mval {'fail' if s['failed_components'] else 'ok'}">{s['failed_components']}</div>
          </div>
        </div>
      </div>

      <!-- Policy -->
      <div class="card">
        <h3>Policy</h3>
        <div class="policy-box">{_esc(policy_text)}</div>
      </div>

    </div>

    <!-- Conversation Sequence Diagram -->
    <div class="card seq-card">
      <h3>Conversation Sequence Diagram <span style="font-size:11px;color:var(--muted);font-weight:400;text-transform:none">(click an arrow to highlight the edge below)</span></h3>
      <div id="diagram"></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--blue);"></i> real</span>
        <span><i class="dot" style="background:var(--violet);"></i> mocked</span>
        <span><i class="dot" style="background:var(--cyan);"></i> assertion</span>
        <span><i class="dot" style="background:var(--red);"></i> failed</span>
      </div>
    </div>

    <!-- Execution Edges -->
    <div class="section" id="edges-section">
      <h3>Execution Edges</h3>
      <div class="scroll-x">
        <table id="edges-table">
          <thead>
            <tr>
              <th>Turn</th><th>#</th><th>Time</th><th>Type</th><th>Name</th>
              <th>Indirect</th><th>Source</th><th>Model</th><th>ms</th>
              <th>In</th><th>Out</th><th>Total</th><th>Status</th>
              <th>Input</th><th>Output</th>
            </tr>
          </thead>
          <tbody id="edges-tbody">{''.join(edge_rows)}</tbody>
        </table>
      </div>
    </div>

    <!-- Violations & Assertion Failures -->
    <div class="section">
      <h3>Violations &amp; Assertion Failures</h3>
      <div class="scroll-x">
        <table>
          <thead><tr><th>Kind</th><th>Name</th><th>Turn</th><th>#</th><th>Status</th><th>Details</th></tr></thead>
          <tbody>{''.join(violation_rows)}</tbody>
        </table>
      </div>
    </div>

    <!-- Tool Statistics -->
    <div class="section">
      <h3>Tool Statistics</h3>
      <div class="scroll-x">
        <table>
          <thead><tr><th>Tool</th><th>Calls</th><th>Mocked</th><th>Failed</th></tr></thead>
          <tbody>{''.join(tool_rows) if tool_rows else '<tr><td colspan="4" class="muted">No tool calls</td></tr>'}</tbody>
        </table>
      </div>
    </div>

  </div>

  <script id="steps-data" type="application/json">{safe_steps}</script>
  <script>
    const steps = JSON.parse(document.getElementById("steps-data").textContent);

    // ── Highlight edge row when clicked ──────────────────────────────
    function highlightEdge(edgeId) {{
      document.querySelectorAll('#edges-tbody tr.highlighted').forEach(function(r) {{
        r.classList.remove('highlighted');
      }});
      if (!edgeId) return;
      const row = document.getElementById('edge-' + edgeId);
      if (row) {{
        row.classList.add('highlighted');
        row.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
      }}
    }}

    // ── Render SVG sequence diagram ───────────────────────────────────
    function renderDiagram(steps) {{
      // Always prepend user → agent
      const allSteps = [{{ from: "user", to: "agent", label: "user_prompt", mocked: false, status: "ok", edge_id: "" }}].concat(steps);

      const lanes = [];
      function addLane(n) {{ if (!lanes.includes(n)) lanes.push(n); }}
      addLane("user"); addLane("agent");
      allSteps.forEach(function(s) {{ addLane(s.from); addLane(s.to); }});

      const diagram = document.getElementById("diagram");
      const padL = 20, padR = 20;
      const laneW = Math.max(130, Math.floor((Math.max(1000, 160 * lanes.length) - padL - padR) / Math.max(1, lanes.length - 1)));
      const totalW = padL + padR + laneW * (lanes.length - 1);
      const rowH = 56;
      const headerH = 44;
      const height = headerH + allSteps.length * rowH + 24;

      function laneX(n) {{ return padL + lanes.indexOf(n) * laneW; }}

      let svg = `<svg width="${{totalW}}" height="${{height}}" viewBox="0 0 ${{totalW}} ${{height}}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="mB" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#5da9ff"/></marker>
          <marker id="mP" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#a97cff"/></marker>
          <marker id="mC" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#43d1c6"/></marker>
          <marker id="mR" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#e05252"/></marker>
        </defs>`;

      // Lane headers + dashed lifelines
      lanes.forEach(function(name) {{
        const x = laneX(name);
        const shortName = name.length > 16 ? name.slice(0, 14) + "…" : name;
        svg += `<rect x="${{x - 52}}" y="4" width="104" height="28" rx="8" fill="#18233a" stroke="#2b3a59" stroke-width="1"/>`;
        svg += `<text x="${{x}}" y="23" fill="#e8eefc" font-size="12" font-weight="700" text-anchor="middle">${{shortName}}</text>`;
        svg += `<line x1="${{x}}" y1="32" x2="${{x}}" y2="${{height - 8}}" stroke="#2b3a59" stroke-dasharray="5 4" stroke-width="1.5"/>`;
      }});

      // Steps / arrows
      allSteps.forEach(function(s, idx) {{
        const y = headerH + idx * rowH + rowH * 0.5;
        const x1 = laneX(s.from), x2 = laneX(s.to);
        let color = "#5da9ff", marker = "url(#mB)";
        if (s.status === "failed") {{ color = "#e05252"; marker = "url(#mR)"; }}
        else if (s.mocked) {{ color = "#a97cff"; marker = "url(#mP)"; }}
        else if ((s.label || "").startsWith("assert")) {{ color = "#43d1c6"; marker = "url(#mC)"; }}

        const edge_id = s.edge_id || "";
        const isSelf = (x1 === x2);
        const clickAttr = edge_id ? ` data-edge-id="${{edge_id}}"` : "";

        svg += `<g class="seq-step"${{clickAttr}}>`;
        if (isSelf) {{
          // Self-loop arc
          svg += `<path d="M ${{x1}} ${{y - 10}} C ${{x1 + 60}} ${{y - 10}}, ${{x1 + 60}} ${{y + 10}}, ${{x1}} ${{y + 10}}" fill="none" stroke="${{color}}" stroke-width="2" marker-end="${{marker}}"/>`;
        }} else {{
          const mx = Math.min(x1,x2) + Math.abs(x2 - x1) * 0.5;
          svg += `<line x1="${{x1}}" y1="${{y}}" x2="${{x2}}" y2="${{y}}" stroke="${{color}}" stroke-width="2" marker-end="${{marker}}"/>`;
        }}
        // Hit area (transparent, wider than arrow)
        svg += `<rect x="${{Math.min(x1,x2) - 4}}" y="${{y - 16}}" width="${{Math.abs(x2 - x1) + 8}}" height="32" fill="transparent"/>`;
        // Label
        const lx = Math.min(x1,x2) + Math.abs(x2 - x1) * 0.1 + (isSelf ? 12 : 4);
        svg += `<text x="${{lx}}" y="${{y - 9}}" fill="${{color}}" font-size="11" font-family="ui-monospace,monospace">${{s.label || ""}}</text>`;
        svg += `</g>`;
      }});

      svg += "</svg>";
      diagram.innerHTML = svg;

      // Attach click handlers after innerHTML is set
      diagram.querySelectorAll('.seq-step[data-edge-id]').forEach(function(g) {{
        g.addEventListener('click', function() {{
          highlightEdge(g.dataset.edgeId);
        }});
      }});
    }}

    renderDiagram(steps);
  </script>
</body>
</html>
"""


def generate_html_report(audit_path: Path) -> Path:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    report = build_report_model(audit)
    html_text = render_html(report)
    out = audit_path.with_suffix(".html")
    out.write_text(html_text, encoding="utf-8")
    return out


def open_report_if_possible(report_path: Path):
    if os.environ.get("AMAZE_OPEN_REPORT", "1") != "1":
        return

    if os.name == "nt" or os.environ.get("DISPLAY"):
        webbrowser.open(report_path.resolve().as_uri())