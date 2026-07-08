import json
from typing import List, Dict, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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
    llm: object # Đối tượng LLM được truyền vào
    max_tree_depth: int 



def get_compact_context(context_path: List[Dict]) -> str:
    if not context_path:
        return "Gốc hệ thống (Root)"
    return "\n".join([f" -> [{node['id']}] {node['short_title']}" for node in context_path])

# ==========================================
# 2. CÁC NODE XỬ LÝ (CHỈNH SỬA SỰ CỐ STATE)
# ==========================================
"""
Đề xuất khung định lượng

Bắt LLM trả JSON scorecard thay vì nhãn nhị phân:
scope_breadth: 0-4 (độ rộng phạm vi)
dependency_count: 0-4 (mức phụ thuộc hệ thống khác)
ambiguity: 0-4 (mức mơ hồ đầu vào/đầu ra)
testability: 0-4 (khả năng viết test độc lập, càng cao càng “READY”)
estimated_subtasks: 1-8
confidence: 0.0-1.0
Tính điểm tách bằng công thức cố định:
split_score=scope_breadth+dependency_count+ambiguity+(4−testability)
Rule quyết định (deterministic):

Nếu depth >= max_tree_depth: READY (giữ như hiện tại)
Nếu estimated_subtasks >= 3: NEED_SPLIT
Hoặc nếu split_score >= 8: NEED_SPLIT
Hoặc nếu confidence < 0.6: NEED_SPLIT
Ngược lại: READY

Không còn phù hợp hoàn toàn nếu bạn để cây lớn 10x10.

Lý do chính:

Bộ ngưỡng hiện tại ở sdd.py:64, sdd.py:65, sdd.py:66 khá “dễ split”.
Điều kiện ở sdd.py:146 là OR, nên chỉ cần một tín hiệu là tách.
Với Qwen 4B, estimated_subtasks thường hay ra 3-5. Ngưỡng 3 làm rất nhiều node bị split liên tục khi max_depth=10 ở sdd.py:295.
Gợi ý ngưỡng mới để bắt đầu tuning:

split_score_threshold: 9 hoặc 10
subtask_threshold: 4 (thậm chí 5 nếu vẫn nổ nhánh)
min_confidence_threshold: 0.5 hoặc 0.45
Bộ khởi điểm mình khuyên dùng ngay:

split_score_threshold = 9
min_confidence_threshold = 0.5
subtask_threshold = 4
Nếu muốn ổn định hơn nữa, dùng ngưỡng theo depth:

Depth 0-2: score>=8, subtasks>=3, conf<0.6
Depth 3-6: score>=9, subtasks>=4, conf<0.5
Depth 7-10: score>=10, subtasks>=5, conf<0.45
"""
def evaluate_layer_node( state: TreeBacklogState) -> Dict:
    tree_store = dict(state["tree_store"])
    active_ids = state["active_node_ids"]
    max_depth = state.get("max_tree_depth", 3)

    # Tuning knobs
    split_score_threshold = state.get("split_score_threshold", 9)
    min_confidence_threshold = state.get("min_confidence_threshold", 0.5)
    subtask_threshold = state.get("subtask_threshold", 4)
    llm = state["llm"]
    hard_split_score_threshold = state.get("hard_split_score_threshold", 10)
    hard_subtask_threshold = state.get("hard_subtask_threshold", 6)

    print(f"\n[INFO] Evaluate layer với {len(active_ids)} node.")

    # Single fallback prompt: used when batch parsing fails
    single_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a backlog item scoring engine for a software engineering team.\n"
            "Return ONLY one valid JSON object, no markdown, no explanations.\n"
            "Schema:\n"
            "{{\n"
            "  \"scope_breadth\": int,\n"
            "  \"dependency_count\": int,\n"
            "  \"ambiguity\": int,\n"
            "  \"testability\": int,\n"
            "  \"estimated_subtasks\": int,\n"
            "  \"confidence\": float\n"
            "}}\n"
        )),
        ("user", (
            "--- CONTEXT ---\n{context}\n\n"
            "--- NODE ---\nID: {node_id}\nContent: {content}\n"
        ))
    ])

    single_chain = single_prompt | llm | JsonOutputParser()

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

    # Loc cac node can evaluate
    candidates = []
    for node_id in active_ids:
        node = tree_store.get(node_id)
        if not node:
            continue
        if node.get("depth", 0) >= max_depth:
            node["status"] = "READY"
            print(f"  -> 🛑 Node {node_id} dat gioi han do sau ({node['depth']}) => READY")
            continue
        candidates.append((node_id, node))

    if not candidates:
        return {"tree_store": tree_store}

    # Chạy từng node một
    for node_id, node in candidates:
        context_str = get_compact_context(node["context_path"])
        try:
            raw = single_chain.invoke({
                "context": context_str,
                "node_id": node_id,
                "content": node["content"],
            })

            if not isinstance(raw, dict):
                raise ValueError("LLM output khong phai object")

            scope_breadth = _to_int(raw.get("scope_breadth"), default=4, low=0, high=4)
            dependency_count = _to_int(raw.get("dependency_count"), default=4, low=0, high=4)
            ambiguity = _to_int(raw.get("ambiguity"), default=4, low=0, high=4)
            testability = _to_int(raw.get("testability"), default=0, low=0, high=4)
            estimated_subtasks = _to_int(raw.get("estimated_subtasks"), default=4, low=1, high=8)
            confidence = _to_float(raw.get("confidence"), default=0.0, low=0.0, high=1.0)

            split_score = scope_breadth + dependency_count + ambiguity + (4 - testability)

            # Soft signals
            signal_subtasks = estimated_subtasks >= subtask_threshold
            signal_score = split_score >= split_score_threshold
            signal_low_conf = confidence < min_confidence_threshold
            signal_count = int(signal_subtasks) + int(signal_score) + int(signal_low_conf)

            # Hard split: đủ mạnh thì tách ngay
            hard_split = (
                split_score >= hard_split_score_threshold
                or (estimated_subtasks >= hard_subtask_threshold and confidence >= min_confidence_threshold)
            )

            # Quyết định cuối
            should_split = hard_split or (signal_count >= 2)
            node["status"] = "NEED_SPLIT" if should_split else "READY"

            print(
                f"  -> Node {node_id}: status={node['status']} | "
                f"split_score={split_score}, subtasks={estimated_subtasks}, conf={confidence:.2f}, "
                f"signals(subtasks={signal_subtasks}, score={signal_score}, low_conf={signal_low_conf}, count={signal_count}), "
                f"hard_split={hard_split}"
            )

        except Exception as e:
            node["status"] = "NEED_SPLIT"
            print(f"  -> ⚠️ Node {node_id}: loi ({str(e)}) => NEED_SPLIT")

    return {"tree_store": tree_store}


def decompose_layer_node( state: TreeBacklogState) -> Dict:
    tree_store = dict(state["tree_store"])
    active_ids = state["active_node_ids"]
    max_n = int(state["max_children_n"])
    llm = state["llm"]

    print(f"\n[INFO] Decompose layer với {len(active_ids)} node.")

    single_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Business Analyst.\n"
            "Return ONLY a valid JSON array, no markdown, no explanations.\n"
            "Each item schema:\n"
            "{{\"short_title\": str, \"content\": str}}\n"
            "Constraints:\n"
            "- Maximum number of items is {max_n}.\n"
            "- Minimum number of children is 2.\n"
        )),
        ("user", (
            "--- PARENT CONTEXT MAP ---\n{context}\n\n"
            "--- REQUIREMENT TO DECOMPOSE ---\nContent: {content}\n\n"
            "Decompose this into direct child requirements."
        ))
    ])
    single_chain = single_prompt | llm | JsonOutputParser()

    next_layer_ids: List[str] = []

    def _normalize_children(children_raw) -> List[Dict[str, str]]:
        if not isinstance(children_raw, list):
            return []
        cleaned: List[Dict[str, str]] = []
        for item in children_raw[:max_n]:
            if not isinstance(item, dict):
                continue
            short_title = str(item.get("short_title", "")).strip()
            content = str(item.get("content", "")).strip()
            if short_title and content:
                cleaned.append({"short_title": short_title, "content": content})
        return cleaned

    def _materialize_children(node_id: str, node: RequirementNode, children: List[Dict[str, str]]) -> None:
        if not children:
            node["status"] = "READY"
            node["children_ids"] = []
            return

        children_ids: List[str] = []
        new_context_path = node["context_path"] + [{"id": node["id"], "short_title": node["short_title"]}]
        current_depth = node.get("depth", 0)

        for index, child in enumerate(children):
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
        next_layer_ids.extend(children_ids)
        print(f"  -> Node {node_id} da be thanh: {children_ids}")

    candidates: List[tuple[str, RequirementNode]] = []
    for node_id in active_ids:
        node = tree_store.get(node_id)
        if not node:
            continue
        if node.get("status") != "NEED_SPLIT":
            continue
        candidates.append((node_id, node))

    if not candidates:
        print("[PROCESS] Khong co node NEED_SPLIT trong layer nay.")
        return {"tree_store": tree_store, "active_node_ids": []}

    for node_id, node in candidates:
        context_str = get_compact_context(node["context_path"])
        try:
            raw_children = single_chain.invoke({
                "context": context_str,
                "content": node["content"],
                "max_n": max_n
            })
            children = _normalize_children(raw_children)
            _materialize_children(node_id, node, children)
        except Exception as e:
            # Nếu có lỗi (parsing, LLM error, etc.), coi như không thể phân rã và chuyển thành READY
            print(f"  -> ⚠️ Node {node_id}: loi ({str(e)}), khong the phan ra => READY")
            node["status"] = "READY"
            node["children_ids"] = []

    print(f"[PROCESS] Chuyen giao layer tiep theo: {next_layer_ids}")
    return {
        "tree_store": tree_store,
        "active_node_ids": next_layer_ids
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
