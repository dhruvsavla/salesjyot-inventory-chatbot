import os
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware
import re
from dotenv import load_dotenv

# ========== FASTAPI APP ==========
app = FastAPI(title="Warehouse Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== CONFIG ==========
load_dotenv()  # loads .env into os.environ

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONNECTION_STRING = os.getenv("DATABASE_URL")
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# ========== INIT DB + LLM ==========
sql_db = SQLDatabase.from_uri(CONNECTION_STRING)
sql_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
sql_agent = create_sql_agent(
    llm=sql_llm,
    db=sql_db,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=False
)

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="langchain_pg_embedding"
)

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""
You are a warehouse assistant chatbot.
Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}
""")

# ========== OUTPUT SANITIZER ==========
def sanitize_output(text: str) -> str:
    """
    Sanitizes SQL or semantic agent outputs:
    - Removes any email addresses
    - Removes echoed WHERE clauses
    """
    # Remove emails
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", "[REDACTED]", text)
    # Remove any WHERE "user-email" = '...' clauses
    text = re.sub(r'WHERE\s+"?user-email"?\s*=\s*\'[^\']+\'', '', text, flags=re.IGNORECASE)
    return text.strip()

def answer_question(query: str, user_email: str):
    try:
        # Attempt SQL
        filtered_query = f'{query} WHERE "user-email" = \'{user_email}\'' if "where" not in query.lower() else query
        raw_result = sql_agent.run(filtered_query)
        return sanitize_output(raw_result)
    except Exception:
        # Fallback to semantic RAG
        filtered_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": {"user_id": user_email}}
        )
        rag_chain_filtered = (
            {"context": filtered_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        raw_result = rag_chain_filtered.invoke(query)
        return sanitize_output(raw_result)

# ========== API MODELS ==========
class ChatRequest(BaseModel):
    question: str
    user_email: str

class ChatResponse(BaseModel):
    answer: str

# ========== ENDPOINT ==========
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Accepts a question and user_email, returns chatbot answer
    """
    result = answer_question(req.question, req.user_email)
    return ChatResponse(answer=result)
