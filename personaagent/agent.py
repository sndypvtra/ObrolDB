import os
from datetime import datetime
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings  # Impor baru
from langchain_chroma import Chroma

from personaagent.logging import green_border_style, log_panel
from personaagent.tools import call_tool
from personaagent.config import Config

os.makedirs(Config.Path.VECTORS_DIR, exist_ok=True)
# Inisialisasi embeddings dan vector store sebagai variabel global
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = Chroma(
    collection_name="northwind",
    persist_directory=str(Config.Path.VECTORS_DIR),
    embedding_function=embeddings
)


SYSTEM_PROMPT = f"""
You are a master database engineer with exceptional expertise in SQLite query construction and optimization.
Your purpose is to transform natural language requests into precise, efficient SQL queries that deliver exactly what the user needs.

<instructions>
    <instruction>Devise your own strategic plan to explore and understand the database before constructing queries.</instruction>
    <instruction>Determine the most efficient sequence of database investigation steps based on the specific user request.</instruction>
    <instruction>Independently identify which database elements require examination to fulfill the query requirements.</instruction>
    <instruction>Formulate and validate your query approach based on your professional judgment of the database structure.</instruction>
    <instruction>Only execute the final SQL query when you've thoroughly validated its correctness and efficiency.</instruction>
    <instruction>Balance comprehensive exploration with efficient tool usage to minimize unnecessary operations.</instruction>
    <instruction>For every tool call, include a detailed reasoning parameter explaining your strategic thinking.</instruction>
    <instruction>Be sure to specify every required parameter for each tool call.</instruction>
    <instruction>Never include raw SQL queries in the output. Only return formatted tables and insights in Markdown.</instruction>
    <instruction>Ensure all outputs are user-friendly for business analysts and data scientists, avoiding technical SQL syntax.</instruction>
</instructions>

Today is {datetime.now().strftime("%Y-%m-%d")}

Your responses should be formatted as Markdown. Prefer using tables or lists for displaying data where appropriate.
Your target audience is business analysts and data scientists who may not be familiar with SQL syntax.DATABASE_PATH
""".strip()

def create_history() -> List[BaseMessage]:
    return [SystemMessage(content=SYSTEM_PROMPT)]

def ask(
    query: str, history: List[BaseMessage], llm: BaseChatModel, max_iterations: int = 20
) -> str:
    log_panel(title="User Request", content=f"Query: {query}", border_style=green_border_style)

    # Buat retriever untuk mencari dokumen relevan
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Ambil dokumen relevan berdasarkan query
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Gabungkan konteks dokumen dengan pertanyaan pengguna
    contextualized_query = f"Konteks Dokumen:\n{context}\n\nPertanyaan Pengguna: {query}"

    n_iterations = 0
    messages = history.copy()
    messages.append(HumanMessage(content=contextualized_query))

    while n_iterations < max_iterations:
        response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            # Pastikan output akhir adalah Markdown yang terformat
            return response.content
        for tool_call in response.tool_calls:
            response = call_tool(tool_call)
            messages.append(response)
        n_iterations += 1

    raise RuntimeError(
        "Maximum number of iterations reached. Please try again with a different query."
    )