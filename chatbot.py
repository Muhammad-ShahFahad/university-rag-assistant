
"""
SMART UNIVERSITY CHATBOT
LangGraph + Conditional Workflow + RAG + FAISS + HuggingFace
"""

import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_community.embeddings import FastEmbedEmbeddings

# ============================================================
# 1. ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from your .env file")


# ============================================================
# 2. CHAT MODEL
# ============================================================

model = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq",
    temperature=0
)


# ============================================================
# 3. EMBEDDING MODEL
# ============================================================

# Runs locally. No Gemini API quota required.
embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ============================================================
# 4. BUILD RETRIEVERS
# ============================================================

def build_retriever(pdf_path: str):
    """Loads a PDF, splits it into chunks and creates a FAISS retriever."""

    if not Path(pdf_path).is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError(f"No documents found in {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(f"No text chunks extracted from {pdf_path}")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


# Build both retrievers
print("Building academic retriever...")

academic_retriever = build_retriever(
    r"D:\AGENTIC_AI\Rules_IQRA.pdf"
)

print("Building fee retriever...")

fee_retriever = build_retriever(
    r"D:\AGENTIC_AI\fee_structure.pdf"
)

print("Both retrievers are ready!")


# ============================================================
# 5. CREATE GRAPH STATE
# ============================================================

class State(BaseModel):

    programme: str

    # add_messages preserves previous conversation messages
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list
    )

    query_type: str = "general"

    retrieved_context: str = ""


# ============================================================
# 6. CLASSIFIER NODE
# ============================================================

def classifier_node(state: State) -> dict:
    """Classifies the student's query into academic, fee or general."""

    last_message = state.messages[-1].content

    prompt = f"""
You are a query classification assistant for a university chatbot.

Classify the student's query into exactly ONE category:

1. academic:
Questions about university rules and regulations, including:
attendance, GPA, CGPA, grading, examinations, re-takes,
probation, degree requirements, semester freezing,
withdrawal and disciplinary policies.

2. fee:
Questions about university finances, including:
admission fee, registration fee, tuition fee, student card fee,
total fee, semester cost and programme fee comparisons.

3. general:
Greetings, thanks, small talk, and anything unrelated
to academic regulations or university fees.

Student's programme: {state.programme}

Student's query:
{last_message}

Return ONLY one word:
academic, fee, or general.
"""

    response = model.invoke(prompt)

    category = str(response.content).strip().lower()

    # Normalize the LLM output
    if category.startswith("academic"):
        category = "academic"

    elif category.startswith("fee"):
        category = "fee"

    else:
        category = "general"

    print(f"\n[Classifier] Query type: {category}")

    return {
        "query_type": category
    }


# ============================================================
# 7. ACADEMIC RAG NODE
# ============================================================

def academic_rag_node(state: State) -> dict:
    """Retrieves relevant information from the academic rules PDF."""

    query = state.messages[-1].content

    print("[Academic RAG] Retrieving academic information...")

    docs = academic_retriever.invoke(query)

    context = "\n\n".join(
        [
            f"Page {doc.metadata.get('page', 'Unknown') + 1 if isinstance(doc.metadata.get('page'), int) else 'Unknown'}:\n{doc.page_content}"
            for doc in docs
        ]
    )

    if not context:
        context = "No relevant academic information was retrieved."

    return {
        "retrieved_context": context
    }


# ============================================================
# 8. FEE RAG NODE
# ============================================================

def fee_rag_node(state: State) -> dict:
    """Retrieves relevant information from the fee structure PDF."""

    query = state.messages[-1].content

    print("[Fee RAG] Retrieving fee information...")

    docs = fee_retriever.invoke(query)

    context = "\n\n".join(
        [
            f"Page {doc.metadata.get('page', 'Unknown') + 1 if isinstance(doc.metadata.get('page'), int) else 'Unknown'}:\n{doc.page_content}"
            for doc in docs
        ]
    )

    if not context:
        context = "No relevant fee information was retrieved."

    return {
        "retrieved_context": context
    }


# ============================================================
# 9. GENERAL NODE
# ============================================================

def general_node(state: State) -> dict:
    """Handles general queries without using RAG."""

    print("[General] No retrieval required.")

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


# ============================================================
# 10. RESPONSE NODE
# ============================================================

def response_node(state: State) -> dict:
    """Generates the final answer using retrieved context or general knowledge."""

    query = state.messages[-1].content

    programme = state.programme

    context = state.retrieved_context

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = f"""
You are a friendly and helpful university assistant.

You are talking to a student enrolled in {programme}.

Answer the student's question using your general knowledge.

Instructions:
- Be friendly and professional.
- Keep the response clear and easy to understand.
- Personalize your answer according to the student's programme when relevant.
- Do not invent university-specific rules, fees or policies.
- If the student asks for university-specific information that you
  cannot verify, recommend checking the official university source.

Student's question:
{query}
"""

    else:

        prompt = f"""
You are a friendly university assistant helping a {programme} student.

Answer the student's question using ONLY the retrieved context.

Instructions:
- Do not invent facts or information.
- Do not use outside knowledge to answer university policy or fee questions.
- If the answer is not available in the context, clearly say so.
- Explain information in simple and understandable language.
- Personalize the response according to the student's programme when relevant.
- Mention page numbers when useful.

Retrieved context:

{context}

Student's question:

{query}
"""

    response = model.invoke(prompt)

    answer = str(response.content).strip()

    print("[Response] Answer generated.")

    return {
        "messages": [
            AIMessage(content=answer)
        ]
    }


# ============================================================
# 11. ROUTER FUNCTION
# ============================================================

def route_query(state: State) -> str:
    """Routes the workflow according to the classifier's decision."""

    if state.query_type == "academic":
        return "academic_rag"

    elif state.query_type == "fee":
        return "fee_rag"

    else:
        return "general"


# ============================================================
# 12. BUILD LANGGRAPH WORKFLOW
# ============================================================

graph = StateGraph(State)

# Register nodes
graph.add_node("classifier", classifier_node)

graph.add_node("academic_rag", academic_rag_node)

graph.add_node("fee_rag", fee_rag_node)

graph.add_node("general", general_node)

graph.add_node("response", response_node)


# Define starting point
graph.add_edge(START, "classifier")


# Conditional routing
graph.add_conditional_edges(
    "classifier",
    route_query,
    {
        "academic_rag": "academic_rag",
        "fee_rag": "fee_rag",
        "general": "general"
    }
)


# Connect retrieval paths to response
graph.add_edge("academic_rag", "response")

graph.add_edge("fee_rag", "response")

graph.add_edge("general", "response")


# End workflow
graph.add_edge("response", END)


# Compile graph
app = graph.compile()


# ============================================================
# 13. STUDENT PROGRAMME SELECTION
# ============================================================

print("\n===================================")
print("       SMART UNIVERSITY CHATBOT")
print("===================================\n")

print("Which programme are you enrolled in?\n")

print("1. BCA")
print("2. BBA")
print("3. BSCS")

programme_map = {
    "1": "BCA",
    "2": "BBA",
    "3": "BSCS"
}

while True:

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice in programme_map:
        student_programme = programme_map[choice]
        break

    print("Invalid choice. Please enter 1, 2 or 3.")


print(f"\nGreat! Your programme is {student_programme}.")
print("Your university chatbot is ready!")
print("Type 'exit' to end the conversation.\n")


# ============================================================
# 14. CHAT LOOP
# ============================================================

# Keep conversation history between graph invocations
chat_history = []

while True:

    user_query = input("You: ").strip()

    if user_query.lower() == "exit":
        print("\nChat ended. Goodbye!")
        break

    if not user_query:
        print("Please enter a question.")
        continue

    try:

        result = app.invoke({
            "programme": student_programme,
            "messages": chat_history + [
                HumanMessage(content=user_query)
            ]
        })

        # Preserve updated messages for the next turn
        chat_history = result["messages"]

        print(
            f"\nAssistant: {chat_history[-1].content}\n"
        )

    except Exception as e:

        print(f"\nAn error occurred: {e}\n")
