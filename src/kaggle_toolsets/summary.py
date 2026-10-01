import argparse
import json
import logging
import os
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger("RequirementSummarizer")


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    """
    Hàm bóc tách JSON tối ưu cho các mô hình mã nguồn mở như Llama 3.1:
    - Loại bỏ văn bản phụ trước/sau JSON.
    - Xử lý lỗi xuống dòng không hợp lệ trong chuỗi JSON.
    """
    cleaned_text = raw_text.strip()

    # Bóc tách duy nhất khối JSON giữa dấu { và } đầu/cuối cùng
    start_idx = cleaned_text.find('{')
    end_idx = cleaned_text.rfind('}')

    if start_idx != -1 and end_idx != -1:
        cleaned_text = cleaned_text[start_idx:end_idx + 1]

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Fallback: Thay thế các ký tự xuống dòng thô trong string bằng \n
        fixed_text = re.sub(r'(?<!\\)\r?\n', r'\\n', cleaned_text)
        return json.loads(fixed_text)


def summary_requirement(
    file_path: str,
    max_summary_word: int,
    max_loop: int,
    llm: Any
) -> str:
    summary_file_path = f"{file_path}.summary"

    if os.path.exists(summary_file_path):
        logger.info("Tìm thấy file tóm tắt đã tồn tại tại: %s", summary_file_path)
        with open(summary_file_path, "r", encoding="utf-8") as f:
            return f.read()

    logger.info("Chưa có file tóm tắt. Bắt đầu tiến trình tóm tắt tự động...")

    yeu_cau_sua = ""
    noi_dung_tom_tat = ""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File mô tả yêu cầu không tồn tại: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        noi_dung_can_tom_tat = f.read()

    logger.info("Đã đọc thành công nội dung file yêu cầu (%s)", file_path)

    count = 0

    while count < max_loop:
        count += 1
        logger.info("Bắt đầu vòng lặp tóm tắt thứ %d/%d", count, max_loop)

        # Prompt tóm tắt được tối ưu riêng cho Llama 3.1
        system_prompt_summarize = (
            "ROLE: You are an expert Software Requirements Analyst and Technical Writer.\n"
            "TASK: Summarize the provided software requirement document concisely, completely, "
            f"and accurately. The summary MUST NOT exceed {max_summary_word} words.\n\n"
            "CRITICAL OUTPUT FORMAT RULES:\n"
            "1. Output STRICTLY a valid, parseable JSON object.\n"
            "2. DO NOT include any commentary, notes, preambles, or suffixes outside the JSON.\n"
            "3. All string values MUST be single-line or use escaped '\\n' for line breaks. DO NOT use raw unescaped multiline text inside string quotes.\n\n"
            "JSON SCHEMA:\n"
            '{\n  "summary": "Your concise summary text here"\n}'
        )

        user_prompt_summarize = (
            f"Original Document:\n{noi_dung_can_tom_tat}\n\n"
            f"Previous Feedback/Revision Requirements:\n{yeu_cau_sua if yeu_cau_sua else 'None'}"
        )

        logger.debug("[STEP 2.1.4.2 - SUMMARIZE] System Prompt: %s", system_prompt_summarize)
        logger.debug("[STEP 2.1.4.2 - SUMMARIZE] User Prompt: %s", user_prompt_summarize)

        messages_summarize = [
            SystemMessage(content=system_prompt_summarize),
            HumanMessage(content=user_prompt_summarize)
        ]
        
        response_summarize = llm.invoke(messages_summarize)
        raw_output_summarize = response_summarize.content if hasattr(response_summarize, 'content') else str(response_summarize)
        
        logger.debug("[STEP 2.1.4.2 - SUMMARIZE] LLM Output: %s", raw_output_summarize)

        parsed_summarize = parse_json_response(raw_output_summarize)
        noi_dung_tom_tat = parsed_summarize.get("summary", "")

        # Prompt đánh giá được tối ưu riêng cho Llama 3.1
        system_prompt_eval = (
            "ROLE: You are a strict Software Quality Assurance Manager.\n"
            "TASK: Evaluate if the generated summary accurately reflects the original text and follows "
            f"the constraint of max {max_summary_word} words.\n\n"
            "CRITICAL OUTPUT FORMAT RULES:\n"
            "1. Output STRICTLY a valid, parseable JSON object.\n"
            "2. DO NOT include any commentary, notes, preambles, or suffixes outside the JSON.\n"
            "3. Ensure boolean is valid lower-case JSON boolean (true/false).\n\n"
            "JSON SCHEMA:\n"
            '{\n  "needs_revision": boolean,\n  "revision_feedback": "Explanation of what needs fixing if true, otherwise empty string"\n}'
        )

        user_prompt_eval = (
            f"Original Document:\n{noi_dung_can_tom_tat}\n\n"
            f"Generated Summary:\n{noi_dung_tom_tat}"
        )

        logger.debug("[STEP 2.1.4.3 - EVALUATE] System Prompt: %s", system_prompt_eval)
        logger.debug("[STEP 2.1.4.3 - EVALUATE] User Prompt: %s", user_prompt_eval)

        messages_eval = [
            SystemMessage(content=system_prompt_eval),
            HumanMessage(content=user_prompt_eval)
        ]

        response_eval = llm.invoke(messages_eval)
        raw_output_eval = response_eval.content if hasattr(response_eval, 'content') else str(response_eval)

        logger.debug("[STEP 2.1.4.3 - EVALUATE] LLM Output: %s", raw_output_eval)

        parsed_eval = parse_json_response(raw_output_eval)
        needs_revision = parsed_eval.get("needs_revision", False)
        revision_feedback = parsed_eval.get("revision_feedback", "")

        if needs_revision:
            yeu_cau_sua = revision_feedback
            logger.info("Đánh giá bước %d: Cần chỉnh sửa. Lý do: %s", count, yeu_cau_sua)

            if count == max_loop:
                logger.warning("Đã chạm ngưỡng max_loop (%d). Kết thúc quá trình tối ưu.", max_loop)
        else:
            logger.info("Đánh giá bước %d: Tóm tắt đạt yêu cầu. Hoàn tất!", count)
            break

    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(noi_dung_tom_tat)

    logger.info("Đã lưu kết quả tóm tắt vào file: %s", summary_file_path)
    return noi_dung_tom_tat


def main():
    parser = argparse.ArgumentParser(description="Tool tóm tắt file yêu cầu phần mềm sử dụng LangChain và Llama 3.1.")
    parser.add_argument("--file", type=str, required=True, help="Đường dẫn file markdown mô tả yêu cầu")
    parser.add_argument("--max-words", type=int, default=150, help="Số từ tối đa cho bản tóm tắt")
    parser.add_argument("--max-loop", type=int, default=3, help="Số vòng lặp tối đa để cải thiện bản tóm tắt")
    parser.add_argument("--model-name", type=str, default="llama3.1:8b", help="Tên mô hình LLM trên Ollama")
    parser.add_argument("--debug", action="store_true", help="Bật log chi tiết level DEBUG")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Cấu hình ChatOpenAI kết nối tới Ollama + ép kiểu JSON Mode
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="dummy",
        model=args.model_name,
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    try:
        final_summary = summary_requirement(
            file_path=args.file,
            max_summary_word=args.max_words,
            max_loop=args.max_loop,
            llm=llm
        )
        print("\n=== KẾT QUẢ TÓM TẮT CUỐI CÙNG ===")
        print(final_summary)
    except Exception as e:
        logger.error("Lỗi trong quá trình xử lý: %s", str(e), exc_info=True)


if __name__ == "__main__":
    main()