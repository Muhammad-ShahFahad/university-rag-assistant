# UniAssist

AI-powered university chatbot built with **LangGraph**, **RAG**, **FAISS** and **Groq**. It classifies each student question as academic, fee or general, retrieves answers from official university PDFs, and replies with page references. Personalized by programme (BCA, BBA, BSCS), with an interactive Streamlit chat interface and conversation memory.

## Features

- **Smart query routing**: an LLM classifier sends each question to the right workflow (academic, fee or general).
- **Two RAG pipelines**: separate FAISS retrievers for the academic rules PDF and the fee structure PDF.
- **Grounded answers**: rules and fee answers use only the retrieved context, with page numbers.
- **Programme personalization**: answers are tailored to BCA, BBA or BSCS students.
- **Conversation memory**: previous messages are kept between turns.
- **Streamlit UI**: chat interface with detected topic labels, a sources viewer and clickable example questions.
- **Local embeddings**: runs `BAAI/bge-small-en-v1.5` locally through FastEmbed, so no embedding API quota is needed.

## How It Works

```
Student question
       |
  Classifier  (academic / fee / general)
       |
  +----+-----------+-------------+
  |                |             |
Academic RAG   Fee RAG       General
(Rules PDF)    (Fee PDF)   (no retrieval)
  |                |             |
  +----------------+-------------+
                   |
           Response generation
                   |
               Final answer
```

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) for the conditional workflow
- [LangChain](https://github.com/langchain-ai/langchain) for loaders, splitters and the chat model interface
- [FAISS](https://github.com/facebookresearch/faiss) for the vector store
- [FastEmbed](https://github.com/qdrant/fastembed) for local embeddings
- [Groq](https://groq.com/) for the LLM (`openai/gpt-oss-20b`)
- [Streamlit](https://streamlit.io/) for the web UI

## Project Structure

```
uniassist/
├── chatbot.py          # Core chatbot: RAG, LangGraph workflow, terminal chat
├── streamlit_app.py    # Streamlit UI that reuses chatbot.py
├── data/
│   ├── Rules_IQRA.pdf      # Academic rules and regulations
│   └── fee_structure.pdf   # Fee structure
├── .env                # Your API key (do not commit)
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/uniassist.git
   cd uniassist
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Groq API key**

   Create a `.env` file in the project root:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Add your PDFs**

   Place your academic rules and fee structure PDFs in the project, then update the two paths in `chatbot.py` (`build_retriever(...)` calls) to point to them.

## Usage

### Streamlit web app

```bash
streamlit run streamlit_app.py
```

Choose your programme in the sidebar and start asking questions. The first launch takes a little longer because the PDFs are indexed once.

### Terminal version

```bash
python chatbot.py
```

Pick your programme (1-3) and chat. Type `exit` to quit.

## Example Questions

- "What is the minimum attendance requirement?"
- "What happens if my CGPA falls below the limit?"
- "How does semester freezing work?"
- "What is the tuition fee per semester?"
- "What is the total cost of my programme?"

## requirements.txt

```
langchain
langchain-community
langchain-core
langchain-text-splitters
langchain-groq
langgraph
faiss-cpu
fastembed
pypdf
python-dotenv
pydantic
streamlit
```

## Notes

- Answers about rules and fees come only from the provided PDFs. If the information isn't in them, the bot says so.
- Never commit your `.env` file. Add it to `.gitignore`.
- Replace the PDFs with your own university documents to adapt the project.

## License

This project is licensed under the MIT License.