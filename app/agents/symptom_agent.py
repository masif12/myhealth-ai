from langchain_groq import ChatGroq
from typing import Dict
import os
from dotenv import load_dotenv
load_dotenv()



def classify_symptom(state: Dict) -> Dict:
    symptoms = state.get("symptoms", "")
    
    if not symptoms:
        state["classification"] = "No symptoms provided."
        return state

    llm = ChatGroq(
    model="llama3-70b-8192",  # or mixtral
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
    prompt = f"Classify the following symptom into a category (viral, muscular, neurological):\n\n{symptoms}"
    response = llm.invoke(prompt).content
    state["classification"] = response
    return state
