from langgraph.graph import StateGraph
from nodes import inference_node, confidence_check_node, fallback_node

def build_graph():
    builder = StateGraph(dict)  # Accepting any dict

    # Node flow
    builder.add_node("inference", inference_node)
    builder.add_node("confidence_check", confidence_check_node)
    builder.add_node("fallback", fallback_node)

    builder.set_entry_point("inference")

    builder.add_edge("inference", "confidence_check")
    builder.add_conditional_edges("confidence_check", lambda x: "fallback" if x["fallback"] else "__end__")
    builder.add_edge("fallback", "__end__")

    return builder.compile()