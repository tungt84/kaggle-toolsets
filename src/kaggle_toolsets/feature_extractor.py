import json
from typing import List, Dict, Any, TypedDict, Optional

# Giả định rằng sdd.py nằm cùng cấp và có thể import trực tiếp
# Nếu không, bạn cần điều chỉnh Python path cho phù hợp.
from .sdd import RequirementNode, TreeBacklogState, llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
# ==========================================
# 1. CẤU TRÚC FEATURE SET
# ==========================================

class Feature(TypedDict):
    """Đại diện cho một đặc trưng (feature) đơn lẻ được trích xuất."""
    feature_name: str
    feature_value: Any
    description: str
    source_prompt: str # Prompt nào đã tạo ra feature này

class EnrichedRequirementNode(RequirementNode):
    """Mở rộng RequirementNode để chứa các đặc trưng được trích xuất."""
    features: List[Feature]
    feature_extraction_status: str # 'PENDING', 'COMPLETED', 'FAILED'
    
class FeatureExtractionState(TypedDict):
    """Trạng thái cho đồ thị trích xuất đặc trưng của một node."""
    node: EnrichedRequirementNode
    iteration_count: int
    max_iterations: int
    should_continue: bool
    max_features_per_run: int

# ==========================================
# 2. CÁC HÀM TRÍCH XUẤT ĐẶC TRƯNG
# ==========================================

def extract_and_decide_node(state: FeatureExtractionState) -> Dict[str, Any]:
    """Node trích xuất các đặc trưng ban đầu và quyết định có tiếp tục không."""
    node = state["node"]
    max_features = state["max_features_per_run"]
    print(f"  [Step 1] Node {node['id']}: Extracting initial features (max: {max_features})...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior Business Analyst. Extract key features from the requirement. "
                   "Return a single JSON object with two keys: 'features' and 'should_continue'. "
                   "'features' is a list of objects, each with 'feature_name', 'feature_value', 'description'. "
                   f"'features' list should not exceed {max_features} items. "
                   "'should_continue' is a boolean indicating if the requirement is complex and needs deeper analysis."),
        ("user", "Requirement: {content}")
    ])
    chain = prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({"content": node["content"]})
        features = result.get("features", [])
        for f in features:
            f["source_prompt"] = "extract_and_decide"
        node["features"].extend(features)
        should_continue = bool(result.get("should_continue", False))
        print(f"    -> Extracted {len(features)} features. Initial decision to continue: {should_continue}")
        return {"node": node, "should_continue": should_continue, "iteration_count": state["iteration_count"] + 1}
    except Exception as e:
        print(f"    -> Error in Step 1: {e}. Stopping.")
        node["feature_extraction_status"] = "FAILED"
        return {"node": node, "should_continue": False}

def extract_deeper_features_node(state: FeatureExtractionState) -> Dict[str, Any]:
    """Node trích xuất các đặc trưng chuyên sâu hơn khi được yêu cầu."""
    node = state["node"]
    print(f"  [Step 3] Node {node['id']}: Extracting deeper features...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical architect. The requirement seems complex. "
                   "Identify potential risks, non-functional requirements, or technical dependencies. "
                   "Return a single JSON object with a 'features' key, which is a list of new findings."),
        ("user", "Requirement: {content}\n\nAlready identified features: {existing_features}")
    ])
    chain = prompt | llm | JsonOutputParser()

    existing_features_str = json.dumps([f["feature_name"] for f in node["features"]], indent=2)
    try:
        result = chain.invoke({"content": node["content"], "existing_features": existing_features_str})
        features = result.get("features", [])
        for f in features:
            f["source_prompt"] = "extract_deeper"
        node["features"].extend(features)
        print(f"    -> Extracted {len(features)} additional features.")
        # Sau khi đào sâu, ta dừng lại ở lần lặp này.
        return {"node": node, "should_continue": False, "iteration_count": state["iteration_count"] + 1}
    except Exception as e:
        print(f"    -> Error in Step 3: {e}. Stopping.")
        node["feature_extraction_status"] = "FAILED"
        return {"node": node, "should_continue": False}

def verify_and_route(state: FeatureExtractionState):
    """Cạnh điều hướng: Kiểm tra và quyết định branche tiếp theo."""
    node = state["node"]
    print(f"  [Step 2] Node {node['id']}: Verifying decision...")

    # Đơn giản hóa: tin tưởng quyết định từ bước 1 cho lần lặp đầu
    # Trong thực tế, có thể gọi LLM lần nữa để xác nhận
    verified_decision = state["should_continue"]
    print(f"    -> Verified decision to continue: {verified_decision}")

    if verified_decision and state["iteration_count"] < state["max_iterations"]:
        print("    -> Routing to deeper extraction.")
        return "extract_deeper"
    else:
        if state["iteration_count"] >= state["max_iterations"]:
            print("    -> Max iterations reached.")
        print("    -> Routing to END.")
        return END

# ==========================================
# 3. ĐỊNH NGHĨA GRAPH
# ==========================================

builder = StateGraph(FeatureExtractionState)

builder.add_node("extract_and_decide", extract_and_decide_node)
builder.add_node("extract_deeper", extract_deeper_features_node)

builder.set_entry_point("extract_and_decide")

builder.add_conditional_edges(
    "extract_and_decide",
    verify_and_route,
    {
        "extract_deeper": "extract_deeper",
        END: END
    }
)
# Sau khi đào sâu, luôn kết thúc chu trình cho node này
builder.add_edge("extract_deeper", END)

# Biên dịch thành một sub-app có thể tái sử dụng
feature_extraction_sub_app = builder.compile()

# ==========================================
# 3. HÀM ĐIỀU PHỐI CHÍNH
# ==========================================

def extract_features_for_tree(
    sdd_output: TreeBacklogState, 
    max_iterations: int = 2,
    max_features_per_run: int = 5
) -> Dict[str, EnrichedRequirementNode]:
    """
    Hàm chính điều phối việc trích xuất đặc trưng cho mỗi node trong cây.

    Args:
        sdd_output: Trạng thái cuối cùng từ việc chạy `sdd.app`.
        max_iterations: Số lần lặp tối đa cho quy trình trích xuất sâu.
        max_features_per_run: Số lượng feature tối đa trích xuất trong một lần gọi LLM.

    Returns:
        Một tree_store mới, nơi mỗi node đã được làm giàu với các đặc trưng.
    """
    original_tree_store = sdd_output.get("tree_store")
    if not original_tree_store:
        raise ValueError("Không tìm thấy 'tree_store' trong kết quả đầu vào.")

    # Chuyển đổi cấu trúc để làm việc
    enriched_tree_store: Dict[str, EnrichedRequirementNode] = {
        node_id: {**node, "features": [], "feature_extraction_status": "PENDING"}
        for node_id, node in original_tree_store.items()
    }

    # Chỉ trích xuất feature cho các node lá (leaf nodes) và đã sẵn sàng (READY)
    target_node_ids = [
        node_id for node_id, node in enriched_tree_store.items()
        if not node.get("children_ids") and node.get("status") == "READY"
    ]

    print(f"\n[START] Bắt đầu trích xuất đặc trưng cho {len(target_node_ids)} node lá.")

    for node_id in target_node_ids:
        node = enriched_tree_store[node_id]
        print(f"\nProcessing Node: {node_id} ('{node['short_title']}')")
        try:
            # Gọi sub-graph cho từng node
            final_state = feature_extraction_sub_app.invoke({
                "node": node,
                "iteration_count": 0,
                "max_iterations": max_iterations,
                "should_continue": True,
                "max_features_per_run": max_features_per_run
            })
            # Cập nhật lại node trong store chính từ trạng thái cuối cùng của graph
            enriched_tree_store[node_id] = final_state["node"]
            node["feature_extraction_status"] = "COMPLETED"
            print(f"  [DONE] Node {node_id}: Completed with {len(node['features'])} features.")

        except Exception as e:
            print(f"  [ERROR] Node {node_id}: Failed during feature extraction. Error: {e}")
            enriched_tree_store[node_id]["feature_extraction_status"] = "FAILED"

    print("\n[FINISH] Hoàn tất quá trình trích xuất đặc trưng cho toàn bộ cây.")
    return enriched_tree_store

# ==========================================
# 4. HÀM TIỆN ÍCH (UTILITY)
# ==========================================

def print_enriched_tree(
    enriched_tree_store: Dict[str, EnrichedRequirementNode],
    node_id: str = "1",
    indent: str = ""
):
    """
    In ra cấu trúc cây đã được làm giàu với các features, tương tự `print_tree` trong `sdd.py`.

    Args:
        enriched_tree_store: Kho chứa các node đã được xử lý bởi `extract_features_for_tree`.
        node_id: ID của node để bắt đầu in (thường là root).
        indent: Chuỗi dùng để thụt đầu dòng, dùng cho đệ quy.
    """
    node = enriched_tree_store.get(node_id)
    if not node:
        return

    status_icon = "✅" if node.get('status') == 'READY' else "🧩"
    extraction_status = node.get("feature_extraction_status", "N/A")
    print(f"{indent}{status_icon} [{node['id']}] {node['short_title']} [Extraction: {extraction_status}]")

    if node.get("features"):
        for feature in node["features"]:
            print(f"{indent}  - Feature: `{feature['feature_name']}` (Nguồn: {feature['source_prompt']})")

    if node.get("children_ids"):
        for child_id in node["children_ids"]:
            print_enriched_tree(enriched_tree_store, child_id, indent + "  ")
