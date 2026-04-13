"""
amaze/annotations.py — LangChain annotation-based instrumentation decorators.

These decorators are the alternative to monkey-patching for LangChain agents.
Use them when you want explicit, visible instrumentation in your agent code
rather than the automatic monkey-patching that instrumentation.py provides.

Supported framework: LangChain / LangGraph only.

  AutoGen and CrewAI are NOT supported — those frameworks call the LLM
  internally and cannot be intercepted via decorators.  Use the standard
  monkey-patching path (instrumentation.py) for LangChain agents instead,
  or implement a framework-specific adapter for AutoGen / CrewAI.

When to use annotations vs monkey-patching
------------------------------------------
Monkey-patching (instrumentation.py, the default):
  - Zero changes to your agent code.
  - Works with any LangChain / LangGraph agent automatically.
  - Recommended for most cases.

Annotations (this module):
  - You control the LLM call explicitly (manual ReAct loop, not create_react_agent).
  - You want instrumentation to be visible in the source code.
  - The runner auto-detects annotation imports and skips monkey-patching.

Decorators
----------
@amaze_tool(name, description="")
    Wraps any sync or async LangChain tool function.
    Enforces policy limits, runs assertions, applies mocks.

@amaze_llm(model="unknown", token_extractor=None)
    Wraps a sync or async function that explicitly calls a LangChain LLM.
    ``token_extractor``: optional ``callable(result) -> (input_tokens, output_tokens)``

@amaze_agent  /  @amaze_agent()
    Marks agent-turn boundaries on any sync or async function.

All decorators are transparent no-ops when no runtime is active (i.e. when
the agent runs outside of aMazeTest).

Usage example (LangChain manual loop)::

    from amaze.annotations import amaze_tool, amaze_llm, amaze_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o").bind_tools([...])

    @amaze_tool("web_search")
    def web_search(query: str) -> str:
        return tavily.search(query)

    @amaze_llm("gpt-4o")
    def call_llm(messages: list):
        return llm.invoke(messages)      # explicit call — decorator can intercept

    @amaze_agent
    def run_turn(prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        while True:
            response = call_llm(messages)
            if not response.tool_calls:
                return response.content
            # dispatch tool calls ...
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional, Tuple

# ---------------------------------------------------------------------------
# Runtime binding
# ---------------------------------------------------------------------------

_active_runtime = None


def set_runtime(runtime) -> None:
    """Bind a RuntimeState.  Called by amaze_runner when annotation mode is detected."""
    global _active_runtime
    _active_runtime = runtime


def get_runtime():
    """Return the active RuntimeState, or None if not installed."""
    return _active_runtime


# ---------------------------------------------------------------------------
# Input / output helpers
# ---------------------------------------------------------------------------

def _extract_tool_input(fn: Callable, args: tuple, kwargs: dict) -> str:
    """Return a plain-text string from a tool call's arguments.

    Mirrors the logic of ``_args_to_assertion_text`` in instrumentation.py:
    single-key / single-string argument is unwrapped to its bare value;
    everything else becomes str(bound_dict).
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    # Drop 'self' / 'cls' for methods
    actual_args = args
    if params and params[0] in ("self", "cls"):
        params = params[1:]
        actual_args = args[1:]

    bound: dict = {}
    for i, val in enumerate(actual_args):
        if i < len(params):
            bound[params[i]] = val
    bound.update(kwargs)

    if len(bound) == 1:
        val = next(iter(bound.values()))
        if isinstance(val, str):
            return val

    return str(bound) if bound else ""


def _extract_llm_input(args: tuple, kwargs: dict) -> str:
    """Return a plain-text string from an LLM call's arguments.

    Handles:
    - plain str prompt
    - OpenAI-style list of dicts  [{"role": "user", "content": "..."}]
    - LangChain message objects   (have a .content attribute)
    - ``messages=`` or ``prompt=`` kwargs
    - any object with a ``.messages`` attribute
    """
    candidates = list(args)
    if "messages" in kwargs:
        candidates.insert(0, kwargs["messages"])
    if "prompt" in kwargs:
        candidates.insert(0, kwargs["prompt"])

    for candidate in candidates:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, list):
            parts = []
            for m in candidate:
                if isinstance(m, dict):
                    parts.append(str(m.get("content", "")))
                elif hasattr(m, "content"):
                    parts.append(str(m.content))
            if parts:
                return " ".join(parts)
        if hasattr(candidate, "messages"):
            return _extract_llm_input((list(candidate.messages),), {})

    return str(args) if args else ""


def _is_indirect_call(args: tuple, kwargs: dict) -> bool:
    """Return True if the LLM is being called with tool-result messages (indirect call)."""
    msgs = None
    for a in args:
        if isinstance(a, list):
            msgs = a
            break
    if msgs is None and "messages" in kwargs:
        msgs = kwargs["messages"]
    if not msgs:
        return False
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "tool":
            return True
        if hasattr(m, "__class__") and "ToolMessage" in m.__class__.__name__:
            return True
    return False


def _build_llm_mock_response(mock) -> Any:
    """Build a response object from a MockConfig.

    Returns an AIMessage (with tool_calls if applicable) when LangChain is
    available, otherwise a plain string or dict so the caller still works.
    """
    if mock.return_tool_call:
        try:
            from langchain_core.messages import AIMessage
            import uuid
            tc = mock.return_tool_call
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": tc["tool"],
                    "args": tc.get("args", {}),
                    "id": str(uuid.uuid4()),
                    "type": "tool_call",
                }],
            )
        except ImportError:
            # Non-LangChain environment: return a plain dict the caller can inspect
            return {"tool_calls": [mock.return_tool_call]}
    if mock.return_ai_message is not None:
        try:
            from langchain_core.messages import AIMessage
            return AIMessage(content=mock.return_ai_message)
        except ImportError:
            return mock.return_ai_message
    return ""


def _extract_llm_output(result: Any) -> str:
    """Return a plain-text string from whatever an LLM wrapper returns."""
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return str(result.content)
    if hasattr(result, "text"):
        return str(result.text)
    # OpenAI SDK: result.choices[0].message.content
    try:
        return str(result.choices[0].message.content)
    except (AttributeError, IndexError, TypeError):
        pass
    return str(result)


def _default_token_extractor(result: Any) -> Tuple[int, int]:
    """Try common SDK shapes to extract (input_tokens, output_tokens)."""
    # LangChain AIMessage / usage_metadata dict
    usage = getattr(result, "usage_metadata", None)
    if isinstance(usage, dict):
        inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        if inp or out:
            return int(inp), int(out)

    # OpenAI / Anthropic SDK: result.usage object
    usage = getattr(result, "usage", None)
    if usage is not None:
        inp = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
        out = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
        if inp or out:
            return int(inp), int(out)

    return 0, 0


# ---------------------------------------------------------------------------
# @amaze_tool
# ---------------------------------------------------------------------------

def amaze_tool(name: str = None, description: str = ""):
    """Instrument any sync or async function as a named tool.

    Parameters
    ----------
    name:        Tool name used in policy assertions and mocks.
                 Defaults to the function's ``__name__``.
    description: Stored in the audit log for reference.
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        target = f"tool:{tool_name}"

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                runtime = get_runtime()
                if runtime is None:
                    return await fn(*args, **kwargs)

                input_text = _extract_tool_input(fn, args, kwargs)
                runtime.last_tool_description = description
                runtime.run_assertions(target, "input", input_text)
                runtime.enter_tool(tool_name, input_text)

                mock = runtime.find_mock(target, input_text)
                if mock is not None:
                    content = str(mock.output) if mock.output is not None else ""
                    runtime.run_assertions(target, "output", content)
                    runtime.record_tool_output(tool_name, input_text, content, True)
                    return content

                result = await fn(*args, **kwargs)
                output_text = str(result)
                runtime.run_assertions(target, "output", output_text)
                runtime.record_tool_output(tool_name, input_text, output_text, False)
                return result

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            runtime = get_runtime()
            if runtime is None:
                return fn(*args, **kwargs)

            input_text = _extract_tool_input(fn, args, kwargs)
            runtime.last_tool_description = description
            runtime.run_assertions(target, "input", input_text)
            runtime.enter_tool(tool_name, input_text)

            mock = runtime.find_mock(target, input_text)
            if mock is not None:
                content = str(mock.output) if mock.output is not None else ""
                runtime.run_assertions(target, "output", content)
                runtime.record_tool_output(tool_name, input_text, content, True)
                return content

            result = fn(*args, **kwargs)
            output_text = str(result)
            runtime.run_assertions(target, "output", output_text)
            runtime.record_tool_output(tool_name, input_text, output_text, False)
            return result

        return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# @amaze_llm
# ---------------------------------------------------------------------------

def amaze_llm(
    model: str = "unknown",
    token_extractor: Optional[Callable[[Any], Tuple[int, int]]] = None,
):
    """Instrument any sync or async function that calls an LLM.

    Parameters
    ----------
    model:           Model name recorded in the audit log.
    token_extractor: ``callable(result) -> (input_tokens, output_tokens)``.
                     When omitted, common SDK shapes are tried automatically.
    """
    extract_tokens = token_extractor or _default_token_extractor

    def decorator(fn: Callable) -> Callable:

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                runtime = get_runtime()
                if runtime is None:
                    return await fn(*args, **kwargs)

                indirect = _is_indirect_call(args, kwargs)
                input_text = _extract_llm_input(args, kwargs)
                runtime.enter_llm(model=model, is_indirect=indirect)

                if not indirect and not runtime.agent_prompt:
                    runtime.agent_prompt = input_text

                if indirect:
                    # Post-tool call — never mock, always hit real LLM
                    result = await fn(*args, **kwargs)
                    output_text = _extract_llm_output(result)
                    runtime.record_llm_output(input_text, output_text, False, True, False)
                    inp, out = extract_tokens(result)
                    if inp or out:
                        runtime.add_token_usage(input_tokens=inp, output_tokens=out, model=model)
                    runtime.advance_finish_if_complete()
                    return result

                runtime.run_assertions("llm", "input", input_text)
                mock = runtime.find_mock("llm", input_text)
                if mock is not None:
                    result = _build_llm_mock_response(mock)
                    output_text = _extract_llm_output(result)
                    has_tc = bool(getattr(result, "tool_calls", None))
                    runtime.run_assertions("llm", "output", output_text)
                    runtime.record_llm_output(input_text, output_text, False, False, True)
                    if not has_tc:
                        runtime.advance_finish_if_complete()
                    return result

                result = await fn(*args, **kwargs)
                output_text = _extract_llm_output(result)
                runtime.run_assertions("llm", "output", output_text)
                runtime.record_llm_output(input_text, output_text, False, False, False)
                inp, out = extract_tokens(result)
                if inp or out:
                    runtime.add_token_usage(input_tokens=inp, output_tokens=out, model=model)
                runtime.advance_finish_if_complete()
                return result

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            runtime = get_runtime()
            if runtime is None:
                return fn(*args, **kwargs)

            indirect = _is_indirect_call(args, kwargs)
            input_text = _extract_llm_input(args, kwargs)
            runtime.enter_llm(model=model, is_indirect=indirect)

            if not indirect and not runtime.agent_prompt:
                runtime.agent_prompt = input_text

            if indirect:
                result = fn(*args, **kwargs)
                output_text = _extract_llm_output(result)
                runtime.record_llm_output(input_text, output_text, False, True, False)
                inp, out = extract_tokens(result)
                if inp or out:
                    runtime.add_token_usage(input_tokens=inp, output_tokens=out, model=model)
                runtime.advance_finish_if_complete()
                return result

            runtime.run_assertions("llm", "input", input_text)
            mock = runtime.find_mock("llm", input_text)
            if mock is not None:
                result = _build_llm_mock_response(mock)
                output_text = _extract_llm_output(result)
                has_tc = bool(getattr(result, "tool_calls", None))
                runtime.run_assertions("llm", "output", output_text)
                runtime.record_llm_output(input_text, output_text, False, False, True)
                if not has_tc:
                    runtime.advance_finish_if_complete()
                return result

            result = fn(*args, **kwargs)
            output_text = _extract_llm_output(result)
            runtime.run_assertions("llm", "output", output_text)
            runtime.record_llm_output(input_text, output_text, False, False, False)
            inp, out = extract_tokens(result)
            if inp or out:
                runtime.add_token_usage(input_tokens=inp, output_tokens=out, model=model)
            runtime.advance_finish_if_complete()
            return result

        return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# @amaze_agent
# ---------------------------------------------------------------------------

def amaze_agent(_fn=None):
    """Mark agent-turn boundaries on any sync or async function.

    On normal return: calls ``advance_finish_if_complete()`` to snapshot the
    turn and reset counters for the next one.
    On exception: calls ``_reset_for_next_turn()`` to keep state consistent.

    Can be used with or without parentheses::

        @amaze_agent
        def run_turn(task): ...

        @amaze_agent()
        async def run_turn_async(task): ...
    """
    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                runtime = get_runtime()
                if runtime is None:
                    return await fn(*args, **kwargs)
                try:
                    result = await fn(*args, **kwargs)
                    runtime.advance_finish_if_complete()
                    return result
                except Exception:
                    runtime._reset_for_next_turn()
                    raise

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            runtime = get_runtime()
            if runtime is None:
                return fn(*args, **kwargs)
            try:
                result = fn(*args, **kwargs)
                runtime.advance_finish_if_complete()
                return result
            except Exception:
                runtime._reset_for_next_turn()
                raise

        return sync_wrapper

    if _fn is not None:
        return decorator(_fn)
    return decorator
