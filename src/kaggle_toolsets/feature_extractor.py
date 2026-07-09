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
    estimated_total_features: int
    verification_completed: bool
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
    
    if is_initial_run:
        # Prompt cho lần chạy đầu tiên, yêu cầu ước tính tổng số features
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior Business Analyst. Your task is to analyze a requirement and identify key features.\n\n"
                       "Return a single JSON object with two keys:\n"
                       "1. `estimated_total_features`: An integer estimating the TOTAL number of features needed to fully describe the requirement.\n"
                       "2. `features`: A list of feature objects. Each object MUST have `feature_name`, `feature_value`, and `description`. This list must NOT exceed {max_features} items."),
            ("user", "Requirement: {content}\n\nAlready identified features: {existing_features}")
        ])
    else:
        # Prompt cho các lần chạy sau, chỉ yêu cầu bổ sung features
        total_needed = state["estimated_total_features"]
        have_now = len(node["features"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior Business Analyst. Your task is to find the remaining features for a requirement.\n\n"
                       f"The target is to have {total_needed} features in total. We currently have {have_now}.\n"
                       "Your goal is to identify what is MISSING from the `Already identified features` list.\n"
                       "ABSOLUTELY DO NOT repeat any features from the list.\n\n"
                       "Return a single JSON object with one key:\n"
                       "1. `features`: A list of *new* feature objects you discovered. Each object MUST have `feature_name`, `feature_value`, and `description`. This list must NOT exceed {max_features} items."),
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
        
        if is_initial_run:
            # Chỉ ước tính ở lần chạy đầu
            estimated_total = int(result.get("estimated_total_features", len(node["features"])))
            # Quyết định tiếp tục dựa trên ước tính ban đầu
            should_continue = len(node["features"]) < estimated_total
        else:
            estimated_total = state["estimated_total_features"] # Giữ nguyên giá trị đã được verify
            should_continue = len(node["features"]) < estimated_total
        
        print(f"    -> Extracted {len(valid_features)} features (Estimated total: {estimated_total}). LLM decision to continue: {should_continue}")
        return {
            "node": node, 
            "should_continue": should_continue, 
            "estimated_total_features": estimated_total if is_initial_run else state["estimated_total_features"],
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
    initial_estimate = state["estimated_total_features"]
    llm = state["llm"]
    # Quyết định từ bước trích xuất trước đó
    extraction_decision = state["should_continue"]
    print(f"  [Step 2] Node {node['id']}: Verifying decision. Initial estimate: {initial_estimate} features.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Lead Architect. You will review an initial analysis of a requirement.\n"
                   "Based on the requirement and the features extracted so far, you must:\n"
                   "1. Verify or adjust the `estimated_total_features` based on the context and extracted features.\n"
                   "2. Provide a brief reasoning for your decision, especially if you change the estimate.\n\n"
                   "Return a single JSON object with two keys:\n"
                   "1. `adjusted_total_features`: An integer. Your final, verified estimate of the total features required.\n"
                   "2. `reasoning`: A brief explanation for your decision."),
        ("user", "Requirement: {content}\n\nInitial Features Extracted ({count}):\n{existing_features}\n\nInitial estimate was {initial_estimate} total features.")
    ])
    chain = prompt | llm | JsonOutputParser()

    try:
        existing_features_str = json.dumps(node["features"], indent=2)
        # Log prompt đầy đủ
        rendered_prompt = prompt.format_prompt(content=node["content"], existing_features=existing_features_str, count=len(node["features"]), initial_estimate=initial_estimate)
        print(f"    -> LLM Prompt (verify_and_decide):\n---\n{rendered_prompt.to_string()}\n---")

        result = chain.invoke({"content": node["content"], "existing_features": existing_features_str})
        print(f"    -> LLM Raw Response:\n---\n{json.dumps(result, indent=2, ensure_ascii=False)}\n---")
        reasoning = result.get("reasoning", "No reasoning provided.")
        
        # Cập nhật lại tổng số feature ước tính từ Architect
        adjusted_total_features = int(result.get("adjusted_total_features", initial_estimate))
        
        # Quyết định tiếp tục được suy ra một cách tất định, không phụ thuộc vào LLM
        have_now = len(node["features"])
        verified_decision = have_now < adjusted_total_features

        print(f"    -> Architect Review: New total estimate: {adjusted_total_features}. Need more features? {verified_decision}. Reason: {reasoning}")
        if initial_estimate != adjusted_total_features:
            print(f"    -> Estimate ADJUSTED! Initial: {initial_estimate}, Architect's: {adjusted_total_features}")
        
        # Cập nhật state với quyết định và lý do đã được xác thực
        return {
            "node": node,
            "should_continue": verified_decision,
            "estimated_total_features": adjusted_total_features,
            "verification_reasoning": reasoning,
            "iteration_count": state["iteration_count"],
            "verification_completed": True # Đánh dấu đã hoàn thành xác thực
        }
    except Exception as e:
        print(f"    -> Error during verification: {e}. Stopping this branch.")
        return {"node": node, "should_continue": False, "verification_reasoning": str(e), "estimated_total_features": initial_estimate}

def route_after_verification(state: FeatureExtractionState):
    """Cạnh điều hướng sau khi xác thực."""
    print("    -> Routing after verification...")
    return route_after_extraction(state)

def route_before_extraction(state: FeatureExtractionState):
    """Quyết định xem có cần chạy node verify hay không."""
    if not state["verification_completed"]:
        print("    -> Verification not completed. Routing to verification node.")
        return "verify_and_decide"
    else:
        print("    -> Verification already completed. Routing to final check.")
        # Thay vì đi đến một node khác, chúng ta sẽ đánh giá điều kiện ngay tại đây
        # và trả về tên của nhánh tiếp theo.
    node = state["node"]
    should_continue_after_verification = state["should_continue"]
    total_needed = state["estimated_total_features"]
    have_now = len(node["features"])
    if should_continue_after_verification and have_now < total_needed and state["iteration_count"] < state["max_iterations"]:
        print(f"    -> Condition met to continue. Have {have_now}/{total_needed} features. Routing back to extraction.")
        return "extract_features"
    else:
        if state["iteration_count"] >= state["max_iterations"]:
            print("    -> Max iterations reached.")
        elif not should_continue_after_verification and have_now < total_needed:
            print("    -> Verification step decided to stop.")
        else:
            print(f"    -> Feature goal reached ({have_now}/{total_needed}).")
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
                "estimated_total_features": 0,
                "verification_completed": False, # Bắt đầu với trạng thái chưa xác thực
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

    # Sau khi extract, quyết định xem có cần verify không
    builder.add_conditional_edges(
        "extract_features",
        route_before_extraction,
        {
            "verify_and_decide": "verify_and_decide",
            # Các nhánh trả về từ route_before_extraction khi đã verify xong
            "extract_features": "extract_features",
            END: END
        }
    )
    # Sau khi verify, quyết định lặp lại hoặc kết thúc
    builder.add_conditional_edges(
        "verify_and_decide",
        route_after_verification,
        {"extract_features": "extract_features", END: END}
    )
    return builder