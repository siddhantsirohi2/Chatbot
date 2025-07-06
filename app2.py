import os
import uuid
import json
import hashlib
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory

load_dotenv()

USER_DB_FILE = "users.json"
FAISS_INDEX_PATH = "faiss_index_langchain"

embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# --- User Authentication Functions ---
def hash_password(password):
    """Hashes the password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def signup(username, password):
    """Signs up a new user and saves to the user database."""
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "r") as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                users = {}
    else:
        users = {}

    if username in users:
        print("Username already exists.")
        return False

    users[username] = hash_password(password)
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)
    print("Signup successful.")
    return True

def login(username, password):
    """Logs in a user by verifying credentials."""
    if not os.path.exists(USER_DB_FILE):
        print("No users registered. Please sign up first.")
        return False

    with open(USER_DB_FILE, "r") as f:
        try:
            users = json.load(f)
        except json.JSONDecodeError:
            print("User database is corrupted.")
            return False

    if username in users and users[username] == hash_password(password):
        print(f"Login successful. Welcome, {username}!")
        return True

    print("Invalid username or password.")
    return False

class ChatSession:
    """Manages a single user chat session."""
    def __init__(self, username):
        self.session_id = str(uuid.uuid4())
        self.username = username
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="history")

        # Load the vector store
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Please run the embedding script first.")

        self.vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        print(f"Session {self.session_id} started for user {self.username}.")

    def get_rag_chain(self, one_shot_example=None):
        """Builds and returns the RAG chain, optionally with a one-shot example."""
        base_template = """
        You are an intelligent assistant specializing in analyzing computer science papers.
        Answer the following question based only on the provided context and the ongoing chat history.
        If you don't know the answer from the context, just say that you don't know. Do not make up information.
        Keep your answers concise and to the point.

        CONTEXT:
        {context}

        CHAT HISTORY:
        {history}
        """

        if one_shot_example:
            template = base_template + """
            Here is an example of the kind of answer I am looking for:
            "{one_shot_example}"

            QUESTION:
            {question}

            ANSWER:
            """
            prompt = PromptTemplate(template=template, input_variables=["context", "history", "one_shot_example", "question"])
        else:
            template = base_template + """
            QUESTION:
            {question}

            ANSWER:
            """
            prompt = PromptTemplate(template=template, input_variables=["context", "history", "question"])


        inputs = {
            "context": lambda x: self.retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "history": lambda x: self.memory.load_memory_variables({})["history"],
        }
        if one_shot_example:
            inputs["one_shot_example"] = lambda x: one_shot_example

        rag_chain = (
            RunnablePassthrough.assign(**inputs)
            | prompt
            | llm
            | StrOutputParser()
        )

        # We also want to return the retrieved documents for referencing
        chain_with_source = RunnableParallel(
            {"answer": rag_chain, "docs": (lambda x: self.retriever.invoke(x["question"]))}
        )

        return chain_with_source

    def ask(self, question, one_shot_example=None):
        """Asks a question to the RAG chain and stores the conversation."""
        rag_chain = self.get_rag_chain(one_shot_example)

        response = rag_chain.invoke({"question": question})

        self.memory.save_context({"input": question}, {"output": response["answer"]})

        return response

    def get_summary(self):
        """Generates and returns a summary of the conversation."""
        history_variables = self.memory.load_memory_variables({})

        if not history_variables.get("history"):
            return "No conversation history to summarize."

        summary_prompt = PromptTemplate.from_template(
            "Summarize the following conversation concisely:\n{history}\nSummary:"
        )

        summary_chain = (
            summary_prompt
            | llm
            | StrOutputParser()
        )

        return summary_chain.invoke(history_variables)

def main():
    """Main function to run the chat application."""
    print("--- RAG Model Chat Application ---")

    # --- Authentication ---
    while True:
        action = input("Do you want to 'login' or 'signup'? ").lower()
        if action in ["login", "signup"]:
            username = input("Enter username: ")
            password = input("Enter password: ")

            if action == "signup":
                signup(username, password)

            if login(username, password):
                session = ChatSession(username)
                break
        else:
            print("Invalid action. Please choose 'login' or 'signup'.")

    # --- Main Chat Loop ---
    one_shot_example = input("\nOptional: Provide an example of the kind of answer you want (or press Enter to skip): ")
    if not one_shot_example.strip():
        one_shot_example = None

    while True:
        user_question = input("\nAsk a question (or type 'quit' to exit, 'summary' for chat summary): ")

        if user_question.lower() == 'quit':
            print("Exiting chat. Goodbye!")
            break

        if user_question.lower() == 'summary':
            print("\n--- Chat Summary ---")
            print(session.get_summary())
            continue

        response = session.ask(user_question, one_shot_example)

        print("\n--- Generated Response ---")
        print(response["answer"])

        print("\n--- Source Documents ---")
        for i, doc in enumerate(response["docs"]):
            source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
            # In a frontend, this `source_file` could be a clickable link
            print(f"  Reference {i+1}: {source_file}")
            print(f"    Content: {doc.page_content[:150]}...")


if __name__ == "__main__":
    main()