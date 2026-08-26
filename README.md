[README.md](https://github.com/user-attachments/files/31478823/README.md)
# HR Recruitment Assistant

An AI-powered HR recruitment assistant built with **LangChain**, **Ollama**, and **FAISS**. It answers company policy questions using Retrieval-Augmented Generation (RAG), evaluates candidate eligibility, calculates experience, generates interview questions, and extracts structured candidate details from resumes — all through a simple command-line chat interface.

## Features

- **Company Policy Search (RAG)** — Answers questions about leave policy, notice period, working hours, and job descriptions by retrieving relevant chunks from a local vector database and grounding the LLM's response in that context.
- **Eligibility Checker** — Compares a candidate's listed skills against the required skill set (Python, SQL, Git) and reports whether they're eligible, along with any missing skills.
- **Experience Calculator** — Calculates a candidate's years of experience from their starting year.
- **Interview Question Generator** — Generates 5 interview questions tailored to a given set of skills.
- **Resume Parsing** — Extracts structured candidate details (name, experience, skills) from freeform resume text using structured LLM output.
- **Conversational Agent** — A tool-calling agent decides which tool to invoke based on the user's query.

## Tech Stack

- [LangChain](https://www.langchain.com/) (`langchain-classic`, `langchain-core`) — agent orchestration and prompting
- [Ollama](https://ollama.com/) — local LLM (`qwen2.5:3b`) and embedding model (`nomic-embed-text`)
- [FAISS](https://github.com/facebookresearch/faiss) — local vector store for document retrieval
- [Pydantic](https://docs.pydantic.dev/) — structured output schema for candidate data

## Project Structure

```
.
├── HR_agent1.py           # Main chat application (agent, tools, chat loop)
├── vector_db.py           # Builds the FAISS vector database from company documents
├── documents/              # Source policy/reference documents used for RAG
│   ├── employeehandbook.txt
│   ├── leavepolicy.txt
│   └── pythondevoloperjd.txt
└── HR_vector_db/           # Generated FAISS index (created by vector_db.py)
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running locally
- The following Ollama models pulled locally:
  ```bash
  ollama pull qwen2.5:3b
  ollama pull nomic-embed-text
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. Install the required Python packages:
   ```bash
   pip install langchain langchain-classic langchain-core langchain-community langchain-ollama langchain-text-splitters faiss-cpu pydantic
   ```

3. Make sure Ollama is running in the background:
   ```bash
   ollama serve
   ```

## Usage

### 1. Build the vector database

Place your reference documents (`.txt` files) inside a `documents/` folder in the project root, then run:

```bash
python vector_db.py
```

This loads and chunks the documents, generates embeddings, and saves a FAISS index to a local folder (`HR_vector_db`).

> **Note:** `HR_agent1.py` currently loads the index from a folder named `hr_vector_db` (lowercase), while `vector_db.py` saves it as `HR_vector_db`. Make sure the folder name used in both files matches exactly, since folder names are case-sensitive on Linux/macOS.

### 2. Run the assistant

```bash
python HR_agent1.py
```

You'll see a chat prompt where you can:

- Ask policy questions, e.g.:
  ```
  You : What is the notice period?
  You : How many casual leaves do I get?
  You : What are the required skills for the Python Developer role?
  ```
- Check candidate eligibility or generate interview questions by describing the request in natural language — the agent will pick the right tool.
- Extract structured details from a resume by prefixing your input with `resume:`:
  ```
  You : resume: John Doe has 3 years of experience in Python, SQL, and Git...
  ```
- Type `exit` to quit.

## Tools Overview

| Tool | Description |
|---|---|
| `experience_calculator` | Calculates years of experience from a starting year |
| `eligibility_checker` | Checks if a candidate's skills meet the required set (Python, SQL, Git) |
| `company_policy_search` | Retrieves relevant document chunks and answers policy questions via RAG |
| `interview_questions` | Generates 5 interview questions for a given skill set |

## Sample Documents Included

- **employeehandbook.txt** — Working hours, WFH policy, leave policy, notice period, dress code, code of conduct, and more.
- **leavepolicy.txt** — Detailed leave policy (Casual, Sick, Earned Leave), approval process, and notice period rules.
- **pythondevoloperjd.txt** — Job description for a Python Developer role, including required/preferred skills and responsibilities.

## Future Improvements

- Align vector DB folder naming between `vector_db.py` and `HR_agent1.py`
- Add a `requirements.txt` for easier dependency management
- Add persistent chat memory across sessions
- Build a web UI (e.g., Streamlit or Gradio) instead of the CLI loop
- Add automated tests for tools and the RAG pipeline

## License

Add your license of choice here (e.g., MIT).
