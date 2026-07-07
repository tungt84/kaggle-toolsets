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
def evaluate_layer_node(state: TreeBacklogState) -> Dict:
    tree_store = dict(state["tree_store"])
    active_ids = state["active_node_ids"]
    max_depth = state.get("max_tree_depth", 3)

    # Tuning knobs
    split_score_threshold = state.get("split_score_threshold", 9)
    min_confidence_threshold = state.get("min_confidence_threshold", 0.5)
    subtask_threshold = state.get("subtask_threshold", 4)
    batch_size = max(1, int(state.get("batch_size", 5)))
    hard_split_score_threshold = state.get("hard_split_score_threshold", 10)
    hard_subtask_threshold = state.get("hard_subtask_threshold", 6)

    print(f"\n[INFO] Evaluate layer với {len(active_ids)} node, batch_size={batch_size}")

    # Batch scoring prompt: input is a JSON string containing a list of nodes
    batch_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a backlog item scoring engine for a software engineering team.\n"
            "Return ONLY a valid JSON array, no markdown, no explanations.\n"
            "Each array item must follow this schema:\n"
            "{{\n"
            "  \"id\": str,\n"
            "  \"scope_breadth\": int,        # 0-4\n"
            "  \"dependency_count\": int,     # 0-4\n"
            "  \"ambiguity\": int,            # 0-4\n"
            "  \"testability\": int,          # 0-4\n"
            "  \"estimated_subtasks\": int,   # 1-8\n"
            "  \"confidence\": float          # 0.0-1.0\n"
            "}}\n"
            "Rules: return exactly one output item per input node and preserve each id."
        )),
        ("user", (
            "--- NODES TO EVALUATE (JSON ARRAY) ---\n{nodes_json}\n\n"
            "Return a JSON array that strictly follows the schema."
        ))
    ])

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

    batch_chain = batch_prompt | llm | JsonOutputParser()
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

    def _apply_score(node_id: str, node: RequirementNode, raw: Dict) -> None:
        if not isinstance(raw, dict):
            node["status"] = "NEED_SPLIT"
            print(f"  -> ⚠️ Node {node_id}: raw khong phai object => NEED_SPLIT")
            return

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
            f"hard_split={hard_split}, "
            f"s={scope_breadth}, d={dependency_count}, a={ambiguity}, t={testability}"
        )

    def _evaluate_single(node_id: str, node: RequirementNode) -> None:
        context_str = get_compact_context(node["context_path"])
        try:
            raw = single_chain.invoke({
                "context": context_str,
                "node_id": node_id,
                "content": node["content"],
            })
            _apply_score(node_id, node, raw)
        except Exception as e:
            node["status"] = "NEED_SPLIT"
            print(f"  -> ⚠️ Node {node_id}: single fallback loi ({str(e)}) => NEED_SPLIT")

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

    # Chay batch
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]

        payload = []
        for node_id, node in batch:
            payload.append({
                "id": node_id,
                "context": get_compact_context(node["context_path"]),
                "content": node["content"],
            })

        try:
            raw_batch = batch_chain.invoke({
                "nodes_json": json.dumps(payload, ensure_ascii=False)
            })

            if not isinstance(raw_batch, list):
                raise ValueError("Batch output khong phai JSON array")

            by_id: Dict[str, Dict] = {}
            for item in raw_batch:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    by_id[item["id"]] = item

            # Node nao batch khong tra ve du thi fallback single
            for node_id, node in batch:
                raw_item = by_id.get(node_id)
                if raw_item is None:
                    print(f"  -> ⚠️ Node {node_id}: thieu ket qua trong batch, fallback single")
                    _evaluate_single(node_id, node)
                    continue
                _apply_score(node_id, node, raw_item)

        except Exception as e:
            print(f"  -> ⚠️ Batch {i//batch_size + 1} loi ({str(e)}), fallback single toan batch")
            for node_id, node in batch:
                _evaluate_single(node_id, node)

    return {"tree_store": tree_store}


def decompose_layer_node(state: TreeBacklogState) -> Dict:
    tree_store = dict(state["tree_store"])
    active_ids = state["active_node_ids"]
    max_n = int(state["max_children_n"])
    batch_size = max(1, int( state.get("batch_size", 5)))

    print(f"\n[INFO] Decompose layer với {len(active_ids)} node, batch_size={batch_size}")

    batch_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Business Analyst.\n"
            "Return ONLY a valid JSON array, no markdown, no explanations.\n"
            "Each output item must correspond to one input node by id, using this schema:\n"
            "{{\n"
            "  \"id\": str,\n"
            "  \"children\": [\n"
            "    {{\"short_title\": str, \"content\": str}}\n"
            "  ]\n"
            "}}\n"
            "Constraints:\n"
            "- You must return one output item for each input id.\n"
            "- children must be direct child requirements.\n"
            "- Minimum number of children is 2.\n"
            "- Maximum number of children is {max_n}.\n"
            "- If decomposition is not possible, return children as [] for that id."
        )),
        ("user", (
            "--- NODES TO DECOMPOSE (JSON ARRAY) ---\n{nodes_json}\n\n"
            "Return a JSON array that strictly follows the schema."
        ))
    ])

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
    batch_chain = batch_prompt | llm | JsonOutputParser()
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

    def _decompose_single(node_id: str, node: RequirementNode) -> None:
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
            print(f"  -> ⚠️ Node {node_id}: single fallback loi ({str(e)})")
            node["status"] = "READY"
            node["children_ids"] = []

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

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        payload = []
        for node_id, node in batch:
            payload.append({
                "id": node_id,
                "context": get_compact_context(node["context_path"]),
                "content": node["content"]
            })

        try:
            raw_batch = batch_chain.invoke({
                "nodes_json": json.dumps(payload, ensure_ascii=False),
                "max_n": max_n
            })

            if not isinstance(raw_batch, list):
                raise ValueError("Batch output khong phai JSON array")

            by_id: Dict[str, Dict] = {}
            for item in raw_batch:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    by_id[item["id"]] = item

            for node_id, node in batch:
                row = by_id.get(node_id)
                if row is None:
                    print(f"  -> ⚠️ Node {node_id}: batch thieu ket qua, fallback single")
                    _decompose_single(node_id, node)
                    continue

                children = _normalize_children(row.get("children"))
                if not children:
                    # Neu batch row loi schema hoac children rong, fallback single de tang ti le tach dung
                    print(f"  -> ⚠️ Node {node_id}: children rong/khong hop le, fallback single")
                    _decompose_single(node_id, node)
                    continue

                _materialize_children(node_id, node, children)

        except Exception as e:
            print(f"  -> ⚠️ Batch {i // batch_size + 1} loi ({str(e)}), fallback single toan batch")
            for node_id, node in batch:
                _decompose_single(node_id, node)

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
        batch_size = 5,
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

    print("\n📋 KẾT QUẢ CUỐI CÙNG (CẤU TRÚC CÂY YÊU CẦU):")
    print_tree(final_output["tree_store"])