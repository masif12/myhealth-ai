from langchain_groq import ChatGroq
from typing import Dict
import os
from dotenv import load_dotenv
load_dotenv()


def diagnose(state: Dict) -> Dict:
    classification = state.get("classification", "")
    vision = state.get("vision", "")
    symptoms = state.get("symptoms", "")

    llm = ChatGroq(
    model="llama3-70b-8192",  # or mixtral
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
    prompt = (
        f"Patient has these symptoms: {symptoms}\n"
        f"Image analysis says: {vision}\n"
        f"Classified as: {classification}\n\n"
        "What could be the diagnosis?"
    )
    response = llm.invoke(prompt).content
    state["diagnosis"] = response
    return state
