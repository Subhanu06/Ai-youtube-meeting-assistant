from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.vector_store import build_vector_store, load_vector_store, get_retriever
import os


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2
    )

def format_docs(docs):
    print("\n========== RETRIEVED CHUNKS ==========")

    for i, d in enumerate(docs):
        print(f"\n[DOC {i}]")
        print(d.page_content[:300])

    return "\n\n".join([d.page_content for d in docs])


def build_rag_chain(transcript: str):
    print("\n========== BUILDING RAG CHAIN ==========")

    vector_store = build_vector_store(transcript)
    print("\n[DEBUG] Sample question received in chain")
    retriever = get_retriever(vector_store, k=6)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a meeting assistant. Answer ONLY using context.\n"
         "If unsure, still try to infer from context.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    print("\n[DEBUG] Loading existing RAG chain...")

    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=6)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a meeting assistant. Answer ONLY using context.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def ask_question(rag_chain, question: str):

    print("\n========== QUESTION DEBUG ==========")
    print("Question:", question)

    answer = rag_chain.invoke(question)

    print("\n========== FINAL ANSWER ==========")
    print(answer)

    return answer