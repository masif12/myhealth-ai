from langchain_groq import ChatGroq
from typing import Dict
import os
from dotenv import load_dotenv
load_dotenv()



def suggest_treatment(state: Dict) -> Dict:
    diagnosis = state.get("diagnosis", "")

    llm = ChatGroq(
    model="llama3-70b-8192",  # or mixtral
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
    prompt = f"Given this diagnosis: {diagnosis}, what should the patient do next?"
    response = llm.invoke(prompt).content
    state["treatment"] = response
    return state
