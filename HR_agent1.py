from datetime import datetime
from typing import List
 
from pydantic import BaseModel
 
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
 
from langchain_community.vectorstores import FAISS
 
from langchain_classic.agents import (
    create_tool_calling_agent,
    AgentExecutor
)

from pathlib import Path 
 
# ==================================
# MODEL
# ==================================
 
llm = ChatOllama(
    model="qwen2.5:3b"
)
 
# ==================================e
# MEMORY
# ==================================
 
chat_history = InMemoryChatMessageHistory()
 
# ==================================
# LOAD VECTOR DB
# ==================================
 
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
 
BASE_DIR = Path(__file__).parent

folder = BASE_DIR / 'HR_vector_db'
 
vector_db = FAISS.load_local(
    folder,
    embeddings,
    allow_dangerous_deserialization=True
)
 
retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)
 
# ==================================
# TOOLS
# ==================================
 
@tool
def experience_calculator(
    start_year: int
) -> str:
    """Calculate candidate experience"""
 
    return str(
        datetime.now().year - start_year
    )
 
 
@tool
def eligibility_checker(
    skills: str
) -> str:
    """Check candidate eligibility"""
 
    required = {
        "python",
        "sql",
        "git"
    }
 
    candidate = {
        skill.strip().lower()
        for skill in skills.split(",")
    }
 
    missing = required - candidate
 
    if len(missing) == 0:
        return "Eligible"
 
    return (
        "Not Eligible. Missing: "
        + ", ".join(missing)
    )
 
 
@tool
def company_policy_search(
    question: str
) -> str:
    """Search company documents"""
 
    docs = retriever.invoke(question)
 
    context = "\n".join(
        doc.page_content
        for doc in docs
    )
 
    prompt = f"""
Answer only from the provided context.
 
Context:
{context}
 
Question:
{question}
"""
 
    result = llm.invoke(prompt)
 
    return result.content
 
 
@tool
def interview_questions(
    skills: str
) -> str:
    """Generate interview questions"""
 
    prompt = f"""
Generate 5 interview questions
for:
 
{skills}
"""
 
    return llm.invoke(prompt).content
 
# ==================================
# TOOLS LIST
# ==================================
 
tools = [
    experience_calculator,
    eligibility_checker,
    company_policy_search,
    interview_questions
]
 
# ==================================
# PROMPT
# ==================================
 
prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are an HR Recruitment Assistant.
 
Use tools whenever required.
 
If the user asks:
 
- leave policy
- notice period
- working hours
- job description
- company policy
 
Always use company_policy_search.
"""
),
("human", "{input}"),
("placeholder", "{agent_scratchpad}")
]
)
 
# ==================================
# AGENT
# ==================================
 
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
 
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
 
# ==================================
# STRUCTURED OUTPUT
# ==================================
 
class Candidate(BaseModel):
    name: str
    experience: int
    skills: List[str]
 
structured_llm = llm.with_structured_output(
    Candidate
)
 
# ==================================
# CHAT LOOP
# ==================================
 
print("=" * 60)
print("HR RECRUITMENT ASSISTANT")
print("=" * 60)
 
while True:
 
    user_input = input("\nYou : ")
 
    if user_input.lower() == "exit":
        break
 
    if user_input.startswith("resume:"):
 
        resume = user_input.replace(
            "resume:",
            ""
        )
 
        candidate = structured_llm.invoke(
            f"""
Extract:
 
Name
Experience
Skills
 
Resume:
 
{resume}
"""
        )
 
        print("\nCandidate Details")
        print(candidate)
 
        continue
 
    response = agent_executor.invoke(
        {
            "input": user_input
        }
    )
 
    print(
        "\nAssistant:",
        response["output"]
    )
