from kaggle_toolsets.feature_extractor import extract_features_for_tree
from kaggle_toolsets.sdd import RequirementNode,TreeBacklogState,evaluate_layer_node,decompose_layer_node,route_next_layer
from kaggle_toolsets.feature_extractor import extract_features_for_tree, print_enriched_tree
from kaggle_toolsets.sdd import RequirementNode, TreeBacklogState, evaluate_layer_node, decompose_layer_node, route_next_layer
import json
from typing import List, Dict, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END



llm = ChatOpenAI(base_url="http://localhost:8000/v1",model="Qwen/Qwen3-4B-Instruct-2507",api_key="dummy", temperature=0.1, max_tokens=2048)
workflow = StateGraph(TreeBacklogState)
workflow.add_node("evaluate_layer", evaluate_layer_node)
workflow.add_node("decompose_layer", decompose_layer_node)

workflow.set_entry_point("evaluate_layer")
workflow.add_edge("evaluate_layer", "decompose_layer")
workflow.add_conditional_edges(
    "decompose_layer",
    route_next_layer,
    {
        "loop_to_evaluate": "evaluate_layer",
        "finish_tree": END
    }
)

app = workflow.compile()
if __name__ == "__main__":
    initial_tree = {
        "1": RequirementNode(
            id="1",
            parent_id=None,
            short_title="E-Commerce System",
            content="Build a complete e-commerce system that allows users to view products, add them to their shopping cart, make online payments via credit card, and track order status on their personal page.",
            context_path=[],
            status="PENDING",
            children_ids=[],
            depth=0
        )
    }
    
    
    initial_state = TreeBacklogState(
        tree_store=initial_tree,
        active_node_ids=["1"],
        max_children_n=10,     
        max_tree_depth=10,
        split_score_threshold = 9,
        min_confidence_threshold = 0.5,
        subtask_threshold = 4,
        hard_split_score_threshold = 10,
        hard_subtask_threshold = 6
    )
    
    final_output = app.invoke(initial_state)
    
    def print_tree(tree_store, node_id="1", indent=""):
        node = tree_store.get(node_id)
        if not node:
            return
        
        status_icon = "✅" if node['status'] == 'READY' else "🧩"
        print(f"{indent}{status_icon} [{node['id']}] (Depth {node.get('depth', 0)}) {node['short_title']}")
        
        if node["children_ids"]:
            for child_id in node["children_ids"]:
                print_tree(tree_store, child_id, indent + "  ")

    print("\n📋 KẾT QUẢ backlog (CẤU TRÚC CÂY YÊU CẦU):")
    print_tree(final_output["tree_store"])

    enriched_tree_store = extract_features_for_tree(final_output, max_iterations=2, max_features_per_run=5)

    print("\n📋 KẾT QUẢ features (CẤU TRÚC CÂY YÊU CẦU) :")
    print_tree(enriched_tree_store)
    print_enriched_tree(enriched_tree_store)
