import json
from typing import List, Dict, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ==========================================
# 1. CẤU TRÚC STATE 
# ==========================================

class RequirementNode(TypedDict):
    id: str
    parent_id: Optional[str]
    content: str
    short_title: str
    context_path: List[Dict[str, str]] 
    status: str  
    children_ids: List[str]
    depth: int  

class TreeBacklogState(TypedDict):
    tree_store: Dict[str, RequirementNode]
    active_node_ids: List[str]
    max_children_n: int
    max_tree_depth: int 

llm = ChatOpenAI(base_url="http://localhost:8000/v1",model=model_name,api_key="dummy", temperature=0.1, max_tokens=2048)

def get_compact_context(context_path: List[Dict]) -> str:
    if not context_path:
        return "Gốc hệ thống (Root)"
    return "\n".join([f" -> [{node['id']}] {node['short_title']}" for node in context_path])

# ==========================================
# 2. CÁC NODE XỬ LÝ (CHỈNH SỬA SỰ CỐ STATE)
# ==========================================

def evaluate_layer_node(state: TreeBacklogState) -> Dict:
    tree_store = dict(state["tree_store"])
    active_ids = state["active_node_ids"]
    max_depth = state.get("max_tree_depth", 3)

    # Deterministic thresholds (có thể đưa ra config nếu muốn tuning)
    split_score_threshold = 8
    min_confidence_threshold = 0.6
    subtask_threshold = 3

    print(f"\n[INFO] Đang đánh giá Layer gồm các Node: {active_ids}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Bạn là bộ chấm điểm backlog item cho team kỹ thuật.\n"
            "Phải trả về DUY NHẤT 1 JSON object hợp lệ, không markdown, không giải thích.\n"
            "Schema bắt buộc:\n"
            "{\n"
            "  \"scope_breadth\": int,        # 0-4 (0 rất hẹp, 4 rất rộng)\n"
            "  \"dependency_count\": int,     # 0-4 (0 độc lập, 4 phụ thuộc nhiều)\n"
            "  \"ambiguity\": int,            # 0-4 (0 rõ ràng, 4 mơ hồ)\n"
            "  \"testability\": int,          # 0-4 (0 khó test độc lập, 4 dễ test)\n"
            "  \"estimated_subtasks\": int,   # 1-8\n"
            "  \"confidence\": float          # 0.0-1.0\n"
            "}\n"
            "Nguyên tắc chấm: ưu tiên thực dụng để triển khai coding ngay."
        )),
        ("user", (
            "--- BẢN ĐỒ NGỮ CẢNH CHA ---\n{context}\n\n"
            "--- YÊU CẦU CẦN ĐÁNH GIÁ ---\nID: {node_id}\nNội dung: {content}\n\n"
            "Hãy chấm điểm theo schema ở trên."
        ))
    ])

    chain = prompt | llm | JsonOutputParser()

    def _to_int(value, default, low, high):
        try:
            v = int(value)
        except (TypeError, ValueError):
            v = default
        return max(low, min(high, v))

    def _to_float(value, default, low, high):
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = default
        return max(low, min(high, v))

    for node_id in active_ids:
        node = tree_store.get(node_id)
        if not node:
            continue

        if node.get("depth", 0) >= max_depth:
            print(f"  -> 🛑 Node {node_id} đạt giới hạn độ sâu ({node['depth']}). Ép trạng thái READY.")
            node["status"] = "READY"
            continue

        context_str = get_compact_context(node["context_path"])

        try:
            raw = chain.invoke({
                "context": context_str,
                "node_id": node_id,
                "content": node["content"],
            })

            # Validation shape: nếu không phải object thì fail-safe NEED_SPLIT
            if not isinstance(raw, dict):
                node["status"] = "NEED_SPLIT"
                print(f"  -> ⚠️ Node {node_id}: JSON không hợp lệ (không phải object) => NEED_SPLIT")
                continue

            scope_breadth = _to_int(raw.get("scope_breadth"), default=4, low=0, high=4)
            dependency_count = _to_int(raw.get("dependency_count"), default=4, low=0, high=4)
            ambiguity = _to_int(raw.get("ambiguity"), default=4, low=0, high=4)
            testability = _to_int(raw.get("testability"), default=0, low=0, high=4)
            estimated_subtasks = _to_int(raw.get("estimated_subtasks"), default=4, low=1, high=8)
            confidence = _to_float(raw.get("confidence"), default=0.0, low=0.0, high=1.0)

            # Deterministic score:
            # Càng rộng/phụ thuộc/mơ hồ thì càng cần split.
            # testability cao làm giảm nhu cầu split.
            split_score = scope_breadth + dependency_count + ambiguity + (4 - testability)

            should_split = (
                estimated_subtasks >= subtask_threshold
                or split_score >= split_score_threshold
                or confidence < min_confidence_threshold
            )

            node["status"] = "NEED_SPLIT" if should_split else "READY"

            print(
                f"  -> Node {node_id}: status={node['status']} | "
                f"split_score={split_score}, subtasks={estimated_subtasks}, conf={confidence:.2f}, "
                f"s={scope_breadth}, d={dependency_count}, a={ambiguity}, t={testability}"
            )

        except Exception as e:
            # Fail-safe: lỗi parse/invoke thì ưu tiên split thay vì READY
            node["status"] = "NEED_SPLIT"
            print(f"  -> ⚠️ Node {node_id}: lỗi evaluate ({str(e)}) => NEED_SPLIT")

    return {"tree_store": tree_store}


def decompose_layer_node(state: TreeBacklogState) -> Dict:
    """CẬP NHẬT: Tính toán và chuyển active_node_ids sang tầng mới ngay tại ĐÂY"""
    tree_store = dict(state["tree_store"])
    active_ids = state["active_node_ids"]
    max_n = state["max_children_n"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Bạn là một Business Analyst. Phản hồi BẮT BUỘC là một JSON Array dạng: "
            "[{{\"short_title\": \"Tiêu đề con\", \"content\": \"Nội dung chi tiết\"}}, ...]"
        )),
        ("user", (
            "--- BẢN ĐỒ NGỮ CẢNH CHA ---\n{context}\n\n"
            "--- YÊU CẦU CẦN CHIA NHỎ ---\nNội dung: {content}\n\n"
            "Yêu cầu: Hãy tách thành các yêu cầu con trực tiếp (Tối đa {max_n} mục)."
        ))
    ])
    chain = prompt | llm | JsonOutputParser()
    
    next_layer_ids = []
    
    for node_id in active_ids:
        node = tree_store.get(node_id)
        if not node:
            continue
            
        if node["status"] != "NEED_SPLIT":
            continue
            
        context_str = get_compact_context(node["context_path"])
        try:
            children_data = chain.invoke({"context": context_str, "content": node["content"], "max_n": max_n})
            
            if not children_data:
                node["status"] = "READY"
                continue
                
            children_ids = []
            new_context_path = node["context_path"] + [{"id": node["id"], "short_title": node["short_title"]}]
            current_depth = node.get("depth", 0)
            
            for index, child in enumerate(children_data):
                child_id = f"{node_id}.{index + 1}"
                children_ids.append(child_id)
                
                tree_store[child_id] = RequirementNode(
                    id=child_id,
                    parent_id=node_id,
                    short_title=child["short_title"],
                    content=child["content"],
                    context_path=new_context_path,
                    status="PENDING",
                    children_ids=[],
                    depth=current_depth + 1
                )
            
            node["children_ids"] = children_ids
            next_layer_ids.extend(children_ids) # Gom các ID con vào tầng tiếp theo
            print(f"  -> Node {node_id} đã bẻ thành các nút con: {children_ids}")
            
        except Exception as e:
            print(f"⚠️ Lỗi rã nút {node_id}: {str(e)}")
            node["status"] = "READY"
            
    # 🔥 ĐIỂM THAY ĐỔI QUAN TRỌNG: 
    # Cập nhật danh sách active_node_ids mới ngay trong return của Node để LangGraph lưu vào State!
    print(f"[PROCESS] Tầng cũ hoàn tất. Chuyển giao danh sách Node mới cho tầng tiếp theo: {next_layer_ids}")
    return {
        "tree_store": tree_store,
        "active_node_ids": next_layer_ids # Ghi đè thành công danh sách mới vào State
    }

# ==========================================
# 3. ĐIỀU HƯỚNG ĐỒNG (Bây giờ chỉ kiểm tra Có/Không)
# ==========================================

def route_next_layer(state: TreeBacklogState):
    """Cạnh điều hướng bây giờ chỉ đọc State để quyết định Rẽ nhánh, không chỉnh sửa State nữa"""
    active_ids = state["active_node_ids"]
    
    if active_ids and len(active_ids) > 0:
        print(f"[LOOP] Phát hiện có {len(active_ids)} node mới. Tiếp tục vòng lặp quay lại Evaluate.")
        return "loop_to_evaluate"
    
    print("[FINISH] Hàng đợi trống (active_node_ids rỗng). Tất cả các nhánh đã đạt điều kiện dừng.")
    return "finish_tree"

# ==========================================
# 4. KẾT NỐI VÀ CHẠY THỬ NGHIỆM
# ==========================================

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
            short_title="Hệ thống E-Commerce",
            content="Xây dựng một hệ thống thương mại điện tử hoàn chỉnh cho phép người dùng xem sản phẩm, thêm vào giỏ hàng, thực hiện thanh toán trực tuyến qua thẻ tín dụng và theo dõi tình trạng đơn hàng trong trang cá nhân.",
            context_path=[],
            status="PENDING",
            children_ids=[],
            depth=0
        )
    }
    
    initial_state = TreeBacklogState(
        tree_store=initial_tree,
        active_node_ids=["1"],
        max_children_n=3,     
        max_tree_depth=2      # Giới hạn 2 tầng để chạy siêu tốc
    )
    
    final_output = app.invoke(initial_state)
    
    print("\n📋 KẾT QUẢ CUỐI CÙNG (CÁC MỤC BACKLOG ĐỦ MỊN):")
    for nid, node in final_output["tree_store"].items():
        if node["status"] == "READY":
            print(f"- [{nid}] (Depth {node.get('depth')}): {node['short_title']} -> {node['content'][:60]}...")