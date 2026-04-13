"""
LangChain agent with explicit amaze annotations.

Identical functionality to one_conversation_agent.py but instrumented via
@amaze_tool / @amaze_llm / @amaze_agent decorators instead of monkey-patching.
The runner detects these imports and activates annotation mode (no monkey-patching).

The agent implements a manual ReAct loop so that the LLM call is explicit and
can be wrapped with @amaze_llm — enabling LLM mock interception.
"""
import os
import sys
import logging
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable, tracing_context

from amaze.annotations import amaze_tool, amaze_llm, amaze_agent

logger = logging.getLogger(__name__)

# ---------- Vectorstore (optional) ----------

_retriever = None

def _try_init_vectorstore():
    global _retriever
    try:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        DATA_DIR = os.getenv("DATA_DIR", "/home/ubuntu/data/learn/pdf_reader/")
        CHROMA_DIR = os.getenv("CHROMA_DIR", "/home/ubuntu/data/learn/chroma_db/")
        COLLECTION_NAME = "pdf_docs"

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        existing = vectorstore.get(limit=1)
        if not existing["ids"]:
            loader = PyPDFDirectoryLoader(DATA_DIR, recursive=True)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            vectorstore.add_documents(chunks)
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        logger.warning("Vectorstore init skipped: %s", e)
        _retriever = None

_try_init_vectorstore()

# ---------- Tools (wrapped with @amaze_tool) ----------

@amaze_tool("pdf_search", description="Search the local PDF collection and return relevant passages with sources.")
def pdf_search(query: str) -> str:
    """Search the local PDF collection and return relevant passages with sources."""
    if _retriever is None:
        return "PDF search unavailable: vectorstore not initialized."
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant PDF content found."
    parts: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")
        parts.append(f"Source: {src}, page: {page}\nContent: {d.page_content}")
    return "\n\n".join(parts)


_tavily = None
def _get_tavily():
    global _tavily
    if _tavily is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            from tavily import TavilyClient
            _tavily = TavilyClient(api_key=api_key)
    return _tavily


@amaze_tool("web_search", description="Search the web for recent information.")
def web_search(query: str) -> str:
    """Search the web for recent information."""
    client = _get_tavily()
    if client is None:
        return "Web search unavailable: TAVILY_API_KEY not set."
    results = client.search(query=query, max_results=5)
    items = results.get("results", [])
    if not items:
        return "No relevant web results found."
    return "\n\n".join(
        f"Title: {item.get('title')}\nURL: {item.get('url')}\nContent: {item.get('content')}"
        for item in items
    )


EMAILS = [
    {"to": "alice", "subject": "Q1 Report",
     "body": "Hi Alice, please find the Q1 report attached. Let me know if you have any questions."},
    {"to": "bob", "subject": "Meeting Tomorrow",
     "body": "Hi Bob, just a reminder that we have a team meeting tomorrow at 10am. See you there!"},
    {"to": "carol", "subject": "Project Update",
     "body": "Hi Carol, wanted to share a quick update on the project — we are on track for the deadline."},
]


@amaze_tool("dummy_email", description="Return predefined dummy emails for alice, bob, or carol.")
def dummy_email(person: Optional[str] = None) -> str:
    """Return predefined dummy emails. Known recipients: alice, bob, carol."""
    if person:
        match = [e for e in EMAILS if e["to"].lower() == person.strip().lower()]
        if not match:
            return f"No email found for '{person}'."
        e = match[0]
        return f"To: {e['to']}\nSubject: {e['subject']}\n\n{e['body']}"
    parts = []
    for e in EMAILS:
        parts.append(f"To: {e['to']}\nSubject: {e['subject']}\n\n{e['body']}")
    return "\n\n---\n\n".join(parts)


@amaze_tool("file_read", description="Read the contents of a local file by path.")
def file_read(path: str) -> str:
    """Read the contents of a local file and return it as a string."""
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    if not os.path.isfile(path):
        return f"Error: path is not a file: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


# ---------- LLM (wrapped with @amaze_llm for explicit interception) ----------

_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

_TOOLS_SCHEMA = [pdf_search, web_search, dummy_email, file_read]

# The LLM is called via this thin wrapper so @amaze_llm can intercept it.
# bind_tools attaches the tool schemas so the LLM knows what tools are available.
_llm_with_tools = _llm.bind_tools(_TOOLS_SCHEMA)


@amaze_llm("gpt-4.1-mini")
def call_llm(messages: list) -> AIMessage:
    """Call the LLM with the current message history."""
    return _llm_with_tools.invoke(messages)


# ---------- Tool dispatch ----------

_TOOL_MAP = {
    "pdf_search": pdf_search,
    "web_search": web_search,
    "dummy_email": dummy_email,
    "file_read": file_read,
}

_SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Use pdf_search for questions about local PDF documents. "
    "Use web_search for current or external information. "
    "Use dummy_email to retrieve emails for alice, bob, or carol. "
    "Use file_read to read the contents of a local file by path. "
    "Always cite which tool and source you used."
)

# ---------- Agent loop (wrapped with @amaze_agent for turn tracking) ----------

@amaze_agent
def run_turn(prompt: str) -> str:
    """Run a single agent turn: call LLM, dispatch tools, return final answer."""
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    max_iterations = 10
    for _ in range(max_iterations):
        response = call_llm(messages)

        # Normalise: mock may return a plain string instead of AIMessage
        if isinstance(response, str):
            return response

        # Check for tool calls
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return response.content if hasattr(response, "content") else str(response)

        # Append the assistant message (with tool_calls) to history
        messages.append(response)

        # Execute each tool call and collect results
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")
            tool_fn = _TOOL_MAP.get(tool_name)
            if tool_fn is None:
                result = f"Unknown tool: {tool_name}"
            elif len(tool_args) == 1:
                result = tool_fn(next(iter(tool_args.values())))
            else:
                result = tool_fn(**tool_args)
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
            )

    return "Max iterations reached without a final answer."


# ---------- Entry point ----------

def main():
    prompt = os.environ.get("AGENT_PROMPT", "").strip()
    if not prompt and len(sys.argv) > 3:
        prompt = sys.argv[3].strip()
    if not prompt:
        prompt = input("You: ").strip()
    if prompt.lower() in {"exit", "quit"}:
        return

    with tracing_context(enabled=True):
        answer = run_turn(prompt)

    print("\nAssistant:")
    print(answer)
    print()


if __name__ == "__main__":
    main()
