#Actionableitems , decision , questions 
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os 


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2
    )


def build_chain(system_prompt: str):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    return prompt | llm | StrOutputParser()

def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. Extract action items with:\n"
        "- Task\n- Owner\n- Deadline (or Not specified)\n"
        "Return numbered list or say none found."
    )

    return chain.invoke({"text": transcript})

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "Extract key decisions from meeting transcript. "
        "Return numbered list or 'No key decisions found.'"
    )

    return chain.invoke({"text": transcript})

def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "Extract unresolved questions or follow-ups from meeting transcript. "
        "Return numbered list or 'No open questions found.'"
    )

    return chain.invoke({"text": transcript})