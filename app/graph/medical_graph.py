from langgraph.graph import StateGraph
from app.agents.vision_agent import vision_analysis
from app.agents.symptom_agent import classify_symptom
from app.agents.diagnosis_agent import diagnose
from app.agents.treatment_agent import suggest_treatment
from typing import TypedDict
from typing import Optional

class GraphState(TypedDict, total=False):
    image_base64: Optional[str]
    symptoms: Optional[str]
    vision: Optional[str]
    classification: Optional[str]
    diagnosis: Optional[str]
    treatment: Optional[str]
# Create the state graph
graph = StateGraph(GraphState)

# Add nodes with unique names
graph.add_node("vision_node", vision_analysis)
graph.add_node("symptom_classifier", classify_symptom)
graph.add_node("diagnosis_node", diagnose)
graph.add_node("treatment_node", suggest_treatment)

# Set entry and finish points
graph.set_entry_point("vision_node")

# Add edges with correct node names
graph.add_edge("vision_node", "symptom_classifier")
graph.add_edge("symptom_classifier", "diagnosis_node")
graph.add_edge("diagnosis_node", "treatment_node")

graph.set_finish_point("treatment_node")

# Compile the graph
app_graph = graph.compile()
