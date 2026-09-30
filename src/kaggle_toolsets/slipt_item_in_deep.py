
#Cần build 1 hàm ( function ) để tách các từ một nội dung và tiếng hành phân tách theo chiều sâu.
#hàm function có thêm tham số:
# - số lượng item cần tách(N)
# - độ sâu tối đa (Depth).
# Hàng function này sẽ phải làm tuần tự các bước sau bằng cách dùng LLM Chains của LangChain:
# 1. khởi tạo danh sách items = {yêu cầu ban đầu}
# 2. khởi tạo danh sách kết quả = []
# 3. thực hiện vòng lặp while (items != []):
#    3.1. lấy từng item ra khỏi  danh sách items
#    3.2. LLM sẽ cần tách item này thành N item con (child_items)
#    3.3. thực hiện vòng lặp while (child_items != []):
#       3.3.1. lấy từng item con (child_item) ra khỏi danh sách child_items
#       3.3.2. Kiểm tra lại  item con (child_item) này có cần phân tách nhỏ hơn nữa hay không?
#       3.3.3. Add (child_item) add vào danh sách kết quả
#       3.3.4. Nếu item vẫn còn lớn quá thì thêm item này vào danh sách items để tiếp tục tách
"""Split a requirement into leaf items with LangChain chains."""

from __future__ import annotations

from collections import deque
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


def split_items_in_depth(
	requirement: str,
	llm: Any,
	n: int,
	depth: int,
) -> list[str]:
	"""Return leaf requirements produced by recursively splitting ``requirement``.

	``depth`` is the maximum number of splits from the root. Items at that
	depth are returned as leaves without another LLM call. The LLM must return
	JSON matching the schemas described in the prompts.
	"""
	if not isinstance(requirement, str) or not requirement.strip():
		raise ValueError("requirement must be a non-empty string")
	if not isinstance(n, int) or isinstance(n, bool) or n < 1:
		raise ValueError("n must be a positive integer")
	if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
		raise ValueError("depth must be a non-negative integer")

	split_prompt = ChatPromptTemplate.from_messages([
		(
			"system",
			"You are a requirements decomposition analyst. Return only a valid "
			"JSON object with an `items` array. Split the requirement into at "
			"most {n} direct child requirements. Every item must be a concise "
			"string and must be meaningfully smaller than the parent.",
		),
		("user", "Requirement:\n{item}"),
	])
	split_chain = split_prompt | llm | JsonOutputParser()

	check_prompt = ChatPromptTemplate.from_messages([
		(
			"system",
			"You decide whether a requirement is still too broad to be an "
			"independent implementation item. Return only a valid JSON object "
			"with a boolean `should_split` field.",
		),
		("user", "Requirement:\n{item}"),
	])
	check_chain = check_prompt | llm | JsonOutputParser()

	items: deque[tuple[str, int]] = deque([(requirement.strip(), 0)])
	results: list[str] = []

	while items:
		item, current_depth = items.popleft()
		if current_depth >= depth:
			results.append(item)
			continue

		split_result = split_chain.invoke({"item": item, "n": n})
		child_items = _normalize_items(
			split_result.get("items") if isinstance(split_result, dict) else split_result,
			n,
		)
		if not child_items:
			results.append(item)
			continue

		while child_items:
			child_item = child_items.pop(0)
			check_result = check_chain.invoke({"item": child_item})
			results.append(child_item)
			if check_result.get("should_split") is True:
				items.append((child_item, current_depth + 1))

	return results


def _normalize_items(raw_items: Any, maximum: int) -> list[str]:
	"""Keep only non-empty string children and enforce the requested limit."""
	if not isinstance(raw_items, list):
		return []
	return [
		item.strip()
		for item in raw_items[:maximum]
		if isinstance(item, str) and item.strip()
	]


# Keep a name close to the module's original spelling for callers importing it.
slipt_item_in_deep = split_items_in_depth

