from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os

from dotenv import load_dotenv
load_dotenv()


def load_document(document_path):
    if document_path.endswith(".txt"):
        loader = TextLoader(document_path)
    elif document_path.endswith(".pdf"):
        loader = PyPDFLoader(document_path)
    elif document_path.endswith(".csv"):
        loader = CSVLoader(document_path)
    else:
        raise ValueError("Unsupported file type. Only .txt, .pdf, .csv allowed.")
    return loader.load()

def create_rag_chain(document_path):
    documents = load_document(document_path)
    embedding_doc = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embedding_doc)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    # Define a prompt template to include chat history and sources
    prompt_template = """
    You are a helpful AI medical assistant.

    Conversation history:
    {chat_history}

    Use the following context snippets to answer the question.
    Context:
    {context}

    Question: {question}

    Provide a concise answer and mention the sources of your information (e.g., page number or document name).
    """

    PROMPT = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=prompt_template
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def run_rag_with_memory(question):
        # Get chat history in string form
        chat_history_str = ""
        if memory.buffer:
            # memory.buffer is list of messages, we convert to string for prompt
            chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in memory.buffer])
        
        # Retrieve documents
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([f"[Source: {doc.metadata.get('source', 'unknown')}] {doc.page_content}" for doc in docs])

        # Format prompt
        prompt = PROMPT.format(chat_history=chat_history_str, context=context, question=question)

        # Run LLM
        response = llm.invoke(prompt).content

        # Add user question and AI response to memory
        memory.save_context({"input": question}, {"output": response})

        return response

    return run_rag_with_memory, memory