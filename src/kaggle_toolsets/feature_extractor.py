import json
from typing import List, Dict, Any, TypedDict, Optional

# Giả định rằng sdd.py nằm cùng cấp và có thể import trực tiếp
# Nếu không, bạn cần điều chỉnh Python path cho phù hợp. 
from kaggle_toolsets.sdd import RequirementNode, TreeBacklogState
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
    verification_reasoning: Optional[str] # Lý do từ bước xác thực
    llm: object # Đối tượng LLM được truyền vào

# ==========================================
# 2. CÁC HÀM TRÍCH XUẤT ĐẶC TRƯNG
# ==========================================

def _normalize_features(features_raw: Any, source_prompt: str) -> List[Feature]:
    """
    Xác thực và làm sạch danh sách feature thô từ LLM.
    - Đảm bảo đầu vào là một danh sách.
    - Đảm bảo mỗi mục là một dictionary có các key cần thiết.
    - Bỏ qua các mục không hợp lệ.
    """
    if not isinstance(features_raw, list):
        return []

    cleaned_features: List[Feature] = []
    for item in features_raw:
        if not isinstance(item, dict):
            continue # Bỏ qua nếu item không phải là dictionary

        # Linh hoạt chấp nhận 'feature_name' hoặc 'name'
        feature_name = item.get("feature_name") or item.get("name")
        
        if not feature_name or not isinstance(feature_name, str):
            continue # Bỏ qua nếu không có feature_name hoặc nó không phải là chuỗi

        # Chuẩn hóa key: nếu dùng 'name', đổi thành 'feature_name'
        if "name" in item and "feature_name" not in item:
            item["feature_name"] = item.pop("name")

        item["source_prompt"] = source_prompt
        cleaned_features.append(item)  # type: ignore
    return cleaned_features

def extract_features_node(state: FeatureExtractionState) -> Dict[str, Any]:
    """
    Node trích xuất các đặc trưng. Nó có thể hoạt động ở hai chế độ:
    1. Lần đầu: Trích xuất các đặc trưng ban đầu.
    2. Các lần sau: Trích xuất các đặc trưng còn thiếu để hoàn thành danh sách.
    """
    node = state["node"]
    max_features = state["max_features_per_run"]
    llm = state["llm"]
    is_initial_run = state["iteration_count"] == 0
    print(f"  [Step {'1' if is_initial_run else '1.x'}] Node {node['id']}: Extracting {'initial' if is_initial_run else 'remaining'} features (max: {max_features})...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior Business Analyst. Your primary task is to find *new* features for a given requirement that are not already in the provided list.\n\n"
                   "Key Instructions:\n"
                   "- **Analyze the `Already identified features` list carefully.**\n"
                   "- **Your goal is to identify what is MISSING.**\n"
                   "- **ABSOLUTELY DO NOT repeat any features from the `Already identified features` list.**\n\n"
                   "Return a single JSON object with three keys:\n"
                   "1. `estimated_total_features`: An integer estimating the TOTAL number of features needed to fully describe the requirement.\n"
                   "2. `features`: A list of *new* feature objects you discovered. Each object MUST have a `feature_name` key, along with `feature_value` and `description`. This list must NOT exceed {max_features} items.\n"
                   "3. `should_continue`: A boolean. Set this to `true` if your `estimated_total_features` is greater than the number of items in your `features` list, otherwise set it to `false`."),
        ("user", "Requirement: {content}\n\nAlready identified features: {existing_features}")
    ])
    chain = prompt.partial(max_features=max_features) | llm | JsonOutputParser()

    try:
        # Log prompt đầy đủ
        existing_features_str = json.dumps([f["feature_name"] for f in node["features"]], indent=2)
        rendered_prompt = prompt.format_prompt(max_features=max_features, content=node["content"], existing_features=existing_features_str)
        print(f"    -> LLM Prompt (extract_features_node):\n---\n{rendered_prompt.to_string()}\n---")

        result = chain.invoke({"content": node["content"], "existing_features": existing_features_str})
        print(f"    -> LLM Raw Response:\n---\n{json.dumps(result, indent=2, ensure_ascii=False)}\n---")

        valid_features = _normalize_features(result.get("features"), "extract_features")
        node["features"].extend(valid_features)
        
        should_continue = bool(result.get("should_continue", False))
        estimated_total = result.get("estimated_total_features", len(node["features"]))
        
        print(f"    -> Extracted {len(valid_features)} features (Estimated total: {estimated_total}). LLM decision to continue: {should_continue}")
        return {
            "node": node, 
            "should_continue": should_continue, 
            "iteration_count": state["iteration_count"] + 1,
            # Không có lý do ở bước đầu tiên
            "verification_reasoning": None 
        }
    except Exception as e:
        # Ghi log chi tiết hơn khi có lỗi
        print(f"    -> Error in Step 1: {e}. Raw LLM output might be invalid. Stopping.")
        node["feature_extraction_status"] = "FAILED"
        return {"node": node, "should_continue": False}

def verify_and_decide_node(state: FeatureExtractionState) -> Dict[str, Any]:
    """Node xác thực: Gọi LLM để đánh giá và lưu lại lý do."""
    node = state["node"]
    llm = state["llm"]
    # Quyết định từ bước trích xuất trước đó
    extraction_decision = state["should_continue"]
    print(f"  [Step 2] Node {node['id']}: Verifying decision...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Lead Architect. You will review an initial analysis of a requirement.\n"
                   "Based on the requirement and the features already extracted, decide if a deeper technical analysis is truly necessary.\n"
                   "Look for hidden complexities, risks, or non-functional requirements that might have been missed.\n"
                   "Return a single JSON object with two keys:\n"
                   "1. `verified_should_continue`: boolean. `true` if deeper analysis is needed, `false` otherwise.\n"
                   "2. `reasoning`: A brief explanation for your decision."),
        ("user", "Requirement: {content}\n\nInitial Features Extracted:\n{existing_features}")
    ])
    chain = prompt | llm | JsonOutputParser()

    try:
        existing_features_str = json.dumps(node["features"], indent=2)
        # Log prompt đầy đủ
        rendered_prompt = prompt.format_prompt(content=node["content"], existing_features=existing_features_str)
        print(f"    -> LLM Prompt (verify_and_decide):\n---\n{rendered_prompt.to_string()}\n---")

        result = chain.invoke({"content": node["content"], "existing_features": existing_features_str})
        print(f"    -> LLM Raw Response:\n---\n{json.dumps(result, indent=2, ensure_ascii=False)}\n---")
        verified_decision = bool(result.get("verified_should_continue", False))
        reasoning = result.get("reasoning", "No reasoning provided.")
        
        print(f"    -> Architect Review: Need deeper analysis? {verified_decision}. Reason: {reasoning}")
        if extraction_decision != verified_decision:
            print(f"    -> Decision FLIPPED! Extraction step said: {extraction_decision}, Architect review says: {verified_decision}")
        
        # Cập nhật state với quyết định và lý do đã được xác thực
        return {
            "node": node,
            "should_continue": verified_decision,
            "verification_reasoning": reasoning,
            "iteration_count": state["iteration_count"]
        }
    except Exception as e:
        print(f"    -> Error during verification: {e}. Stopping this branch.")
        return {"node": node, "should_continue": False, "verification_reasoning": str(e)}

def route_after_verification(state: FeatureExtractionState):
    """Cạnh điều hướng sau khi xác thực."""
    # Luôn ưu tiên quyết định từ bước trích xuất để hoàn thành danh sách trước
    should_complete_list = state["should_continue"]
    
    if should_complete_list and state["iteration_count"] < state["max_iterations"]:
        print("    -> Feature list is incomplete. Routing back to extraction.")
        return "extract_features"
    else:
        if state["iteration_count"] >= state["max_iterations"]:
            print("    -> Max iterations reached.")
        print("    -> Routing to END.")
        return END

# ==========================================
# 3. HÀM ĐIỀU PHỐI CHÍNH
# ==========================================

def extract_features_for_tree(
    feature_extraction_app,
    llm: object,
    sdd_output: TreeBacklogState, 
    max_iterations: int = 2,
    max_features_per_run: int = 5
) -> Dict[str, EnrichedRequirementNode]:
    """
    Hàm chính điều phối việc trích xuất đặc trưng cho mỗi node trong cây.

    Args:
        llm: Đối tượng LLM để sử dụng cho việc trích xuất.
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
            final_state = feature_extraction_app.invoke({
                "node": node,
                "iteration_count": 0,
                "max_iterations": max_iterations,
                "should_continue": True,
                "max_features_per_run": max_features_per_run,
                "llm": llm,
                "verification_reasoning": None
            }, {"recursion_limit": 10}) # Thêm recursion_limit để cho phép lặp sâu hơn
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

def build_feature_extraction_graph() -> StateGraph:
    """
    Xây dựng đồ thị trạng thái cho quy trình trích xuất đặc trưng.

    Returns:
        Một đối tượng StateGraph đại diện cho quy trình trích xuất.
    """
    builder = StateGraph(FeatureExtractionState)

    builder.add_node("extract_features", extract_features_node)
    builder.add_node("verify_and_decide", verify_and_decide_node)

    builder.set_entry_point("extract_features")

    # Sau khi trích xuất, quyết định đi đâu tiếp theo
    # Hiện tại, chúng ta sẽ dừng lại sau khi hoàn thành danh sách.
    # Trong tương lai, có thể thêm một nhánh "deepen" ở đây.
    builder.add_conditional_edges(
        "extract_features",
        route_after_verification,
        {
            "extract_features": "extract_features", # Vòng lặp để hoàn thành danh sách
            END: END
        }
    )

    # Tạm thời chưa sử dụng node verify_and_decide trong luồng chính
    # builder.add_edge("extract_features", "verify_and_decide")

    return builder