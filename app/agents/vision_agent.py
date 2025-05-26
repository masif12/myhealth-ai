from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typing import Dict
import os
from dotenv import load_dotenv

load_dotenv()

def vision_analysis(state: Dict) -> Dict:
    image_base64 = state.get("image_base64", "")

    if not image_base64:
        state["vision"] = "No image provided."
        return state

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

    content = [
        {
            "type": "text",
            "text": "What medical condition does this image suggest?"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }
    ]

    # ✅ Wrap the HumanMessage inside a list
    message = HumanMessage(content=content)
    response = llm.invoke([message]).content

    state["vision"] = response
    return state