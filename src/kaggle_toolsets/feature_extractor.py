import json
from typing import List, Dict, Any, TypedDict, Optional

# Giả định rằng sdd.py nằm cùng cấp và có thể import trực tiếp
# Nếu không, bạn cần điều chỉnh Python path cho phù hợp.
from .sdd import RequirementNode, TreeBacklogState, llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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


# ==========================================
# 2. CÁC HÀM TRÍCH XUẤT ĐẶC TRƯNG
# ==========================================

def _step1_extract_and_decide(node: EnrichedRequirementNode) -> tuple[List[Feature], bool]:
    """
    Bước 1: Trích xuất một tập hợp các đặc trưng cơ bản và quyết định có cần tiếp tục không.
    Sử dụng LLM để thực hiện cả hai việc cùng lúc.
    """
    print(f"  [Step 1] Node {node['id']}: Extracting initial features...")
    # --- PHẦN HIỆN THỰC (SKELETON) ---
    # 1. Thiết kế prompt yêu cầu LLM trả về một JSON chứa:
    #    - Một danh sách các features (ví dụ: 'user_story', 'acceptance_criteria', 'dependencies').
    #    - Một trường boolean 'should_continue' để chỉ ra sự cần thiết phải phân tích sâu hơn.
    # 2. Invoke LLM và parse kết quả.
    # ...

    # Dữ liệu giả lập cho khung xương
    mock_features = [
        Feature(feature_name="user_story", feature_value="As a user...", description="User story format", source_prompt="step1_prompt")
    ]
    should_continue = False # Giả sử không cần tiếp tục

    return mock_features, should_continue

def _step2_verify_and_loop(node: EnrichedRequirementNode, initial_decision: bool) -> bool:
    """
    Bước 2: Kiểm tra lại quyết định từ Bước 1.
    Hỏi LLM một câu hỏi xác nhận đơn giản để tăng độ tin cậy.
    """
    print(f"  [Step 2] Node {node['id']}: Verifying decision to continue...")
    # --- PHẦN HIỆN THỰC (SKELETON) ---
    # 1. Thiết kế prompt hỏi LLM: "Dựa trên nội dung sau, có những khía cạnh kỹ thuật phức tạp
    #    hoặc yêu cầu nghiệp vụ chưa rõ ràng nào không? Trả lời 'true' nếu có, 'false' nếu không."
    # 2. Invoke LLM và parse kết quả boolean.
    # ...

    # Dữ liệu giả lập cho khung xương
    verified_decision = initial_decision # Giả sử quyết định ban đầu là đúng

    if initial_decision != verified_decision:
        print(f"    -> Decision flipped! Initial: {initial_decision}, Verified: {verified_decision}")

    return verified_decision

def _step3_extract_deeper_features(node: EnrichedRequirementNode) -> List[Feature]:
    """
    (Tùy chọn) Bước 3: Nếu quy trình quyết định cần tiếp tục, gọi hàm này để trích xuất sâu hơn.
    """
    print(f"  [Step 3] Node {node['id']}: Extracting deeper features...")
    # --- PHẦN HIỆN THỰC (SKELETON) ---
    # 1. Thiết kế một prompt khác, tập trung vào các khía cạnh cụ thể hơn
    #    (ví dụ: 'potential_risks', 'required_apis', 'database_schema_changes').
    # 2. Invoke LLM và parse kết quả.
    # ...

    # Dữ liệu giả lập cho khung xương
    mock_deep_features = [
        Feature(feature_name="potential_risks", feature_value="Payment gateway integration might fail.", description="Potential project risks", source_prompt="step3_prompt")
    ]
    return mock_deep_features

# ==========================================
# 3. HÀM ĐIỀU PHỐI CHÍNH
# ==========================================

def extract_features_for_tree(sdd_output: TreeBacklogState, max_iterations: int = 2) -> Dict[str, EnrichedRequirementNode]:
    """
    Hàm chính điều phối việc trích xuất đặc trưng cho mỗi node trong cây.

    Args:
        sdd_output: Trạng thái cuối cùng từ việc chạy `sdd.app`.
        max_iterations: Số lần lặp tối đa cho quy trình trích xuất sâu.

    Returns:
        Một tree_store mới, nơi mỗi node đã được làm giàu với các đặc trưng.
    """
    original_tree_store = sdd_output.get("tree_store")
    if not tree_store:
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
            should_continue = True
            iteration_count = 0

            while should_continue and iteration_count < max_iterations:
                if iteration_count == 0:
                    # Lần lặp đầu tiên luôn chạy bước 1 & 2
                    features, initial_decision = _step1_extract_and_decide(node)
                    node["features"].extend(features)
                    should_continue = _step2_verify_and_loop(node, initial_decision)
                else:
                    # Các lần lặp sau (nếu có) sẽ trích xuất sâu hơn
                    deep_features = _step3_extract_deeper_features(node)
                    node["features"].extend(deep_features)
                    # Sau khi đào sâu, ta có thể dừng lại
                    should_continue = False

                iteration_count += 1

            node["feature_extraction_status"] = "COMPLETED"
            print(f"  [DONE] Node {node_id}: Completed with {len(node['features'])} features.")

        except Exception as e:
            print(f"  [ERROR] Node {node_id}: Failed during feature extraction. Error: {e}")
            node["feature_extraction_status"] = "FAILED"

    print("\n[FINISH] Hoàn tất quá trình trích xuất đặc trưng cho toàn bộ cây.")
    return enriched_tree_store