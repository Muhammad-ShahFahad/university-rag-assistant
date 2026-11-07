"""
Streamlit UI for the Smart University Chatbot.

Your original script is NOT modified. Save it next to this file as `chatbot.py`.
This UI runs everything in it up to `app = graph.compile()` and skips the CLI
parts (terminal programme menu and input() chat loop).

Run with:
    streamlit run streamlit_app.py
"""

import ast
import sys
import types
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Smart University Chatbot",
    page_icon="🎓",
    layout="centered",
)

BACKEND_FILE = Path(__file__).parent / "chatbot.py"

PROGRAMMES = ["BCA", "BBA", "BSCS"]

QUERY_LABELS = {
    "academic": "📘 Academic rules",
    "fee": "💳 Fee structure",
    "general": "💬 General",
}

SUGGESTIONS = {
    "academic": [
        "What is the minimum attendance requirement?",
        "What happens if my CGPA falls below the limit?",
        "How does semester freezing work?",
    ],
    "fee": [
        "What is the tuition fee per semester?",
        "What is the admission fee?",
        "What is the total cost of my programme?",
    ],
}


# ------------------------------------------------------------
# Load the original backend once (PDF indexing is slow)
# ------------------------------------------------------------

@st.cache_resource(show_spinner="Building retrievers from your PDFs. This runs once...")
def load_backend():
    """Executes your original code (up to app = graph.compile()) and returns the graph."""

    if not BACKEND_FILE.is_file():
        raise FileNotFoundError(
            f"Could not find {BACKEND_FILE.name}. "
            "Save your original script next to this file with that name."
        )

    source = BACKEND_FILE.read_text(encoding="utf-8-sig")

    # Keep every top-level statement up to and including `app = graph.compile()`.
    # Everything after it (terminal menu and input() loop) is left out.
    tree = ast.parse(source, filename=str(BACKEND_FILE))

    last_index = None
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "app" for t in node.targets
        ):
            last_index = i
            break

    if last_index is None:
        raise ValueError("Could not find `app = graph.compile()` in chatbot.py")

    backend_tree = ast.Module(body=tree.body[: last_index + 1], type_ignores=[])

    module = types.ModuleType("chatbot_backend")
    module.__file__ = str(BACKEND_FILE)
    sys.modules["chatbot_backend"] = module

    exec(compile(backend_tree, str(BACKEND_FILE), "exec"), module.__dict__)

    return module.app


try:
    graph_app = load_backend()
except Exception as e:
    st.error(f"Could not start the chatbot: {e}")
    st.stop()


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "programme" not in st.session_state:
    st.session_state.programme = PROGRAMMES[0]

if "chat_history" not in st.session_state:
    # LangChain messages passed back into the graph on every turn
    st.session_state.chat_history = []

if "ui_messages" not in st.session_state:
    # What is displayed: {"role", "content", "query_type", "context"}
    st.session_state.ui_messages = []


def reset_chat():
    st.session_state.chat_history = []
    st.session_state.ui_messages = []


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header("🎓 Your profile")

    selected = st.selectbox(
        "Programme",
        PROGRAMMES,
        index=PROGRAMMES.index(st.session_state.programme),
        help="Answers are personalised to your programme. Changing it starts a new chat.",
    )

    if selected != st.session_state.programme:
        st.session_state.programme = selected
        reset_chat()
        st.rerun()

    show_sources = st.toggle(
        "Show retrieved sources",
        value=True,
        help="Display the PDF passages used to write each answer.",
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()
    st.caption(
        "Ask about attendance, GPA, exams, probation, degree requirements, "
        "or fees. Answers about rules and fees come only from the official PDFs."
    )


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("Smart University Chatbot")
st.caption(f"Chatting as a **{st.session_state.programme}** student")


# ------------------------------------------------------------
# Render previous messages
# ------------------------------------------------------------

def render_meta(msg: dict):
    """Shows the detected category and retrieved sources under an answer."""

    label = QUERY_LABELS.get(msg.get("query_type", "general"), "💬 General")
    st.caption(f"Detected topic: {label}")

    context = msg.get("context", "")
    if show_sources and context and context != "NO_RETRIEVAL_NEEDED":
        with st.expander("Sources used"):
            st.text(context)


for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_meta(msg)


# ------------------------------------------------------------
# Empty state with clickable examples
# ------------------------------------------------------------

clicked_suggestion = None

if not st.session_state.ui_messages:
    with st.chat_message("assistant"):
        st.markdown(
            f"Hi! I can answer questions about university rules and fees for "
            f"**{st.session_state.programme}**. Pick an example or type your own question below."
        )

    cols = st.columns(2)
    examples = SUGGESTIONS["academic"][:2] + SUGGESTIONS["fee"][:2]

    for i, text in enumerate(examples):
        if cols[i % 2].button(text, key=f"suggestion_{i}", use_container_width=True):
            clicked_suggestion = text


# ------------------------------------------------------------
# Chat input
# ------------------------------------------------------------

typed_query = st.chat_input("Ask a question about rules or fees...")
user_query = clicked_suggestion or typed_query

if user_query:
    user_query = user_query.strip()

if user_query:
    st.session_state.ui_messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = graph_app.invoke({
                    "programme": st.session_state.programme,
                    "messages": st.session_state.chat_history
                    + [HumanMessage(content=user_query)],
                })

            # Preserve updated messages for the next turn (same as your CLI loop)
            st.session_state.chat_history = result["messages"]

            answer = result["messages"][-1].content

            assistant_msg = {
                "role": "assistant",
                "content": answer,
                "query_type": result.get("query_type", "general"),
                "context": result.get("retrieved_context", ""),
            }

            st.markdown(answer)
            render_meta(assistant_msg)

            st.session_state.ui_messages.append(assistant_msg)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            # Drop the unanswered question so the history stays consistent
            st.session_state.ui_messages.pop()