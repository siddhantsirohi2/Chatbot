import os
import uuid
import json
import hashlib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# --- Configuration ---
USER_DB_FILE = "users.json"
FAISS_INDEX_PATH = "faiss_index_langchain"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RAG Chat API",
    description="An API for a Retrieval-Augmented Generation Chat Application",
    version="1.0.0",
)

# --- Global Objects ---
# In a production environment, consider a more robust way to manage these objects.
try:
    embeddings_model = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    # This dictionary will store active chat sessions in memory.
    # For production, use a more persistent storage like Redis or a database.
    active_sessions = {}
except Exception as e:
    print(f"Error initializing models: {e}")
    # You might want to handle this more gracefully
    embeddings_model = None
    llm = None
    active_sessions = {}


# --- Pydantic Models for Request/Response ---
class UserCredentials(BaseModel):
    username: str
    password: str

class QuestionRequest(BaseModel):
    username: str # Used to identify the session
    question: str
    one_shot_example: str | None = None

class AnswerResponse(BaseModel):
    answer: str
    docs: list

class SummaryResponse(BaseModel):
    summary: str


# --- User Authentication Functions ---
def hash_password(password: str) -> str:
    """Hashes the password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def get_user_db():
    """Loads the user database from the JSON file."""
    if not os.path.exists(USER_DB_FILE):
        return {}
    with open(USER_DB_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_user_db(users: dict):
    """Saves the user database to the JSON file."""
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)


# --- Chat Session Management ---
class ChatSession:
    """Manages a single user chat session."""
    def __init__(self, username: str):
        self.session_id = str(uuid.uuid4())
        self.username = username
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="history")

        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}.")

        self.vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        print(f"Session {self.session_id} started for user {self.username}.")

    def get_rag_chain(self, one_shot_example: str | None = None):
        """Builds and returns the RAG chain."""
        template = """
        You are an intelligent assistant specializing in analyzing computer science papers.
        Answer the following question based only on the provided context and the ongoing chat history.
        If you don't know the answer from the context, just say that you don't know. Do not make up information.
        Keep your answers concise and to the point.
        CONTEXT: {context}
        CHAT HISTORY: {history}
        QUESTION: {question}
        ANSWER:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "history", "question"])

        rag_chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough(),
                "history": lambda x: self.memory.load_memory_variables({})["history"],
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        chain_with_source = RunnableParallel(
            {"answer": rag_chain, "docs": self.retriever}
        )
        return chain_with_source

    def ask(self, question: str, one_shot_example: str | None = None):
        """Asks a question to the RAG chain and stores the conversation."""
        rag_chain = self.get_rag_chain(one_shot_example)
        response = rag_chain.invoke(question)
        self.memory.save_context({"input": question}, {"output": response["answer"]})
        return response

    def get_summary(self):
        """Generates and returns a summary of the conversation."""
        history_variables = self.memory.load_memory_variables({})
        if not history_variables.get("history"):
            return "No conversation history to summarize."

        summary_prompt = PromptTemplate.from_template("Summarize the following conversation concisely:\n{history}\nSummary:")
        summary_chain = summary_prompt | llm | StrOutputParser()
        return summary_chain.invoke(history_variables)

# --- API Endpoints ---

@app.post("/signup")
def signup_endpoint(credentials: UserCredentials):
    """Signs up a new user."""
    users = get_user_db()
    if credentials.username in users:
        raise HTTPException(status_code=400, detail="Username already exists.")

    users[credentials.username] = hash_password(credentials.password)
    save_user_db(users)
    return {"message": "Signup successful."}


@app.post("/login")
def login_endpoint(credentials: UserCredentials):
    """Logs in a user and creates a chat session."""
    users = get_user_db()
    if not (credentials.username in users and users[credentials.username] == hash_password(credentials.password)):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    # Create and store a new session for the user
    if credentials.username not in active_sessions:
        try:
            active_sessions[credentials.username] = ChatSession(credentials.username)
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Login successful. Welcome, {credentials.username}!"}


@app.post("/ask", response_model=AnswerResponse)
def ask_endpoint(request: QuestionRequest):
    """Receives a question and returns the RAG model's answer."""
    if request.username not in active_sessions:
        raise HTTPException(status_code=403, detail="User not logged in or session expired.")

    session = active_sessions[request.username]
    try:
        response = session.ask(request.question, request.one_shot_example)
        # The 'docs' object needs to be serialized into a JSON-friendly format.
        docs_list = []
        for doc in response["docs"]:
            docs_list.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            })
        return {"answer": response["answer"], "docs": docs_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/summary/{username}", response_model=SummaryResponse)
def summary_endpoint(username: str):
    """Returns a summary of the chat history for a user."""
    if username not in active_sessions:
        raise HTTPException(status_code=403, detail="User not logged in or session expired.")

    session = active_sessions[username]
    summary = session.get_summary()
    return {"summary": summary}

# uvicorn app2:app --reload
