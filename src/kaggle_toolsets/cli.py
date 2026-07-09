from kaggle_toolsets.feature_extractor import extract_features_for_tree, print_enriched_tree,build_feature_extraction_graph
from kaggle_toolsets.sdd import RequirementNode, TreeBacklogState, build_backlog_state_graph
import json
from typing import List, Dict, Optional, TypedDict
import logging
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END



if __name__ == "__main__":
    # Khởi tạo LLM ở đây, tại điểm bắt đầu của ứng dụng
    # --- Cấu hình logging chi tiết ---
    # 1. Lấy root logger và đặt cấp độ thấp nhất (DEBUG) để bắt tất cả các message
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 2. Tạo handler cho console, chỉ hiển thị từ mức INFO trở lên
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 3. Tạo handler cho file, ghi lại tất cả từ mức DEBUG trở lên
    file_handler = logging.FileHandler('app.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 4. Tạo formatter để định dạng log message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 5. Gán formatter cho các handler
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 6. Thêm các handler vào root logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    llm = ChatOpenAI(base_url="http://localhost:8000/v1", model="Qwen/Qwen3-4B-Instruct-2507", api_key="dummy", temperature=0.1, max_tokens=2048)

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
        llm=llm, # Tiêm llm vào trạng thái ban đầu
        split_score_threshold = 9,
        min_confidence_threshold = 0.5,
        subtask_threshold = 4,
        hard_split_score_threshold = 10,
        hard_subtask_threshold = 6
    ) # type: ignore
    backlog_app = build_backlog_state_graph().compile()
    final_output = backlog_app.invoke(initial_state)
    
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

    
    feature_extraction_app = build_feature_extraction_graph().compile()

    enriched_tree_store = extract_features_for_tree(feature_extraction_app, llm, final_output, max_iterations=2, max_features_per_run=5)

    print("\n📋 KẾT QUẢ features (CẤU TRÚC CÂY YÊU CẦU) :")
    print_enriched_tree(enriched_tree_store)
