"""
NOTE: AutoGen is not officially supported by aMazeTest.
LLM calls cannot be intercepted — see bug analysis in system_tests/autogen_system_test/.

AutoGen agent with amaze annotations (pyautogen 0.2).

Same 4 tools as one_conversation_agent.py, instrumented via @amaze_tool.

Instrumentation coverage:
  Tool calls    — fully intercepted via @amaze_tool.
  LLM calls     — NOT intercepted: AutoGen calls the OpenAI API internally.
                  LLM mocks, max_llm_calls, and token tracking do not apply.
  Turn boundary — tracked via @amaze_agent on main().
"""
import os
import sys
import logging
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

import autogen

from amaze.annotations import amaze_tool, amaze_agent

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

# ---------- Tools ----------

@amaze_tool("pdf_search", description="Search the local PDF collection and return relevant passages.")
def pdf_search(query: str) -> str:
    """Search the local PDF collection and return relevant passages with sources."""
    if _retriever is None:
        return "PDF search unavailable: vectorstore not initialized."
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant PDF content found."
    parts = []
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
        match = [e for e in EMAILS if e["to"].lower() == str(person).strip().lower()]
        if not match:
            return f"No email found for '{person}'."
        e = match[0]
        return f"To: {e['to']}\nSubject: {e['subject']}\n\n{e['body']}"
    parts = [f"To: {e['to']}\nSubject: {e['subject']}\n\n{e['body']}" for e in EMAILS]
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


# ---------- Agent ----------

_LLM_CONFIG = {
    "config_list": [{
        "model": "gpt-4.1-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }],
    "temperature": 0,
}

_SYSTEM_MESSAGE = (
    "You are a helpful research assistant. "
    "Use pdf_search for questions about local PDF documents. "
    "Use web_search for current or external information. "
    "Use dummy_email to retrieve emails for alice, bob, or carol. "
    "Use file_read to read the contents of a local file by path. "
    "Always cite which tool and source you used. "
    "Reply TERMINATE when the task is complete."
)


@amaze_agent
def main():
    prompt = os.environ.get("AGENT_PROMPT", "").strip()
    if not prompt and len(sys.argv) > 3:
        prompt = sys.argv[3].strip()
    if not prompt:
        prompt = input("You: ").strip()
    if prompt.lower() in {"exit", "quit"}:
        return

    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=_LLM_CONFIG,
        system_message=_SYSTEM_MESSAGE,
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        code_execution_config=False,
        is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content") or ""),
    )

    autogen.register_function(
        pdf_search, caller=assistant, executor=user_proxy,
        name="pdf_search",
        description="Search the local PDF collection and return relevant passages with sources.",
    )
    autogen.register_function(
        web_search, caller=assistant, executor=user_proxy,
        name="web_search",
        description="Search the web for recent information.",
    )
    autogen.register_function(
        dummy_email, caller=assistant, executor=user_proxy,
        name="dummy_email",
        description="Return predefined dummy emails. Known recipients: alice, bob, carol.",
    )
    autogen.register_function(
        file_read, caller=assistant, executor=user_proxy,
        name="file_read",
        description="Read the contents of a local file and return it as a string.",
    )

    user_proxy.initiate_chat(assistant, message=prompt)


if __name__ == "__main__":
    main()
