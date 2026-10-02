import argparse
import logging
import os
import re
from typing import Any, Dict, List

from summary import parse_json_response
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("CodeGenerator")


def clean_code_snippet(code_str: str) -> str:
    """
    Hàm làm sạch chuỗi code, loại bỏ các ký tự bọc markdown block (```lang ... ```).
    """
    if not code_str:
        return ""
    code_str = code_str.strip()
    
    # Biểu thức chính quy bóc tách nội dung bên trong ```...```
    pattern = r"^```[a-zA-R0-9_+-]*\n?(.*?)\n?```$"
    match = re.search(pattern, code_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Xử lý trường hợp thẻ ``` không bao bọc toàn bộ chuỗi
    lines = code_str.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
        
    return "\n".join(lines).strip()


def generate_code(
    file_path: str,
    lang: str,
    max_loop: int,
    llm: Any
) -> Dict[str, Any]:
    """
    Hàm sinh code tự động theo quy trình kiểm tra và nhận xét lặp.
    """
    logger.info(f"Khởi chạy generate_code: file='{file_path}', lang='{lang}', max_loop={max_loop}")

    # 2.1. Đọc nội dung file và gán vào biến yeu_cau
    if not os.path.exists(file_path):
        logger.error(f"File không tồn tại: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        yeu_cau = f.read()
    logger.info("Đã đọc xong nội dung file yêu cầu.")

    # 2.2. Gán count = 0, generated_code = "", nhan_xet = []
    count = 0
    generated_code = ""
    nhan_xet: List[Dict[str, Any]] = []

    # Prompt sinh code (Tiếng Anh + ROLE + Định dạng JSON)
    code_gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "ROLE: You are an expert software developer specializing in {lang}."),
        ("user", """Generate source code in {lang} that fully satisfies the given requirements and addresses previous review feedback if present.

REQUIREMENTS:
{requirements}

PREVIOUS REVIEWS / FEEDBACK:
{feedback}

STRICT OUTPUT FORMAT:
You MUST return ONLY a valid JSON object matching this schema:
{{
  "code": "string (the complete implementation in {lang})"
}}
""")
    ])

    # Prompt nhận xét code (Tiếng Anh + ROLE + Định dạng JSON)
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "ROLE: You are a strict Senior Code Reviewer and QA Engineer."),
        ("user", """Evaluate whether the generated code in {lang} accurately satisfies the requirements and previous review feedback.

REQUIREMENTS:
{requirements}

PREVIOUS REVIEWS / FEEDBACK:
{feedback}

GENERATED CODE TO EVALUATE:
{code}

STRICT OUTPUT FORMAT:
You MUST return ONLY a valid JSON object matching this schema:
{{
  "need_fix": boolean (true if code requires modifications, false if fully satisfactory),
  "fix_content": "string or null (detailed instructions on what needs to be fixed if need_fix is true, else null)",
  "reason": "string (explanation of why revision is required or why code passed review)"
}}
""")
    ])

    # 2.3. Vòng lặp trong khi count < max_loop
    while count < max_loop:
        # 2.3.1. count = count + 1
        count += 1
        logger.info(f"Bắt đầu vòng lặp thứ {count}/{max_loop}")

        # 2.3.2. Yêu cầu LLM sinh code
        formatted_gen_prompt = code_gen_prompt.format(
            lang=lang,
            requirements=yeu_cau,
            feedback=nhan_xet if nhan_xet else "None"
        )
        logger.info(f"[LLM PROMPT - CODE GEN]\n{formatted_gen_prompt}")

        gen_response = llm.invoke(formatted_gen_prompt)
        raw_gen_text = gen_response.content if hasattr(gen_response, 'content') else str(gen_response)
        logger.info(f"[LLM OUTPUT - CODE GEN]\n{raw_gen_text}")

        # 2.3.3. Kiểm tra kết quả đầu ra với JSON format và có nội dung code (dùng parse_json_response)
        try:
            parsed_gen = parse_json_response(raw_gen_text)
        except Exception as e:
            logger.warning(f"Lỗi parse JSON ở bước sinh code (Vòng {count}): {e}")
            continue

        if isinstance(parsed_gen, dict) and parsed_gen.get("code"):
            # 2.3.3.1. generated_code = code được sinh ra
            raw_code = parsed_gen["code"]
            generated_code = clean_code_snippet(raw_code)
            logger.info(f"[Code Generated]\n{generated_code}")
            # KIỂM TRA CODE RỖNG: Nếu rỗng thì ép lặp lại vòng tiếp theo, không thực hiện đánh giá
            if not generated_code:
                logger.warning(f"Mã nguồn 'generated_code' rỗng ở vòng {count}. Ép buộc lặp tiếp và bỏ qua bước đánh giá LLM.")
                continue
            logger.info("Đã trích xuất code thành công từ JSON trả về.")

            # 2.3.3.2. Yêu cầu LLM đưa ra nhận xét
            formatted_eval_prompt = eval_prompt.format(
                lang=lang,
                requirements=yeu_cau,
                feedback=nhan_xet if nhan_xet else "None",
                code=generated_code
            )
            logger.info(f"[LLM PROMPT - EVALUATION]\n{formatted_eval_prompt}")

            eval_response = llm.invoke(formatted_eval_prompt)
            raw_eval_text = eval_response.content if hasattr(eval_response, 'content') else str(eval_response)
            logger.info(f"[LLM OUTPUT - EVALUATION]\n{raw_eval_text}")

            try:
                parsed_eval = parse_json_response(raw_eval_text)
            except Exception as e:
                logger.warning(f"Lỗi parse JSON ở bước nhận xét (Vòng {count}): {e}")
                continue

            if isinstance(parsed_eval, dict):
                need_fix = parsed_eval.get("need_fix", False)
                reason = parsed_eval.get("reason", "")
                fix_content = parsed_eval.get("fix_content", "")

                # 2.3.3.2.1. Nếu không cần phải chỉnh sửa
                if not need_fix:
                    logger.info("Code đã phù hợp với yêu cầu, không cần chỉnh sửa thêm.")
                    count = max_loop  # Dừng vòng lặp
                    break

                # Cập nhật thông tin nhận xét từ bước 2.3.3.2 vào nhan_xet
                nhan_xet.append({
                    "iteration": count,
                    "reason": reason,
                    "fix_content": fix_content
                })

                # 2.3.3.2.2. Nếu cần chỉnh sửa nhưng đã count == max_loop
                if count == max_loop:
                    logger.error("Cần chỉnh sửa nhưng đã đạt count == max_loop. Dừng vòng lặp và ghi nhận lỗi.")
                    break
        else:
            logger.warning("Kết quả trả về không chứa thông tin 'code'.")

    return {
        "generated_code": generated_code,
        "nhan_xet": nhan_xet,
        "total_loops": count
    }


def main():
    parser = argparse.ArgumentParser(description="CLI chạy chương trình sinh code tự động bằng LangChain.")
    parser.add_argument("--file", "-f", required=True, help="Đường dẫn tới file markdown yêu cầu (VD: yeu_cau.md)")
    parser.add_argument("--lang", "-l", default="python", help="Ngôn ngữ lập trình mục tiêu")
    parser.add_argument("--max-loop", "-m", type=int, default=3, help="Số vòng lặp tối đa")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Tên LLM model")

    args = parser.parse_args()

    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="dummy",
        model=args.model,
        temperature=0.2,
        timeout=60*5,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    result = generate_code(
        file_path=args.file,
        lang=args.lang,
        max_loop=args.max_loop,
        llm=llm
    )

    print("\n" + "=" * 50)
    print("KẾT QUẢ THỰC THI")
    print("=" * 50)
    print(f"Tổng số vòng lặp: {result['total_loops']}")
    print(f"Lịch sử nhận xét thu thập được ({len(result['nhan_xet'])} lần):")
    for review in result['nhan_xet']:
        print(f" - Vòng {review['iteration']}: Lý do={review['reason']} | Cần fix={review['fix_content']}")
    print("\n--- CODE CUỐI CÙNG ---")
    print(result["generated_code"])


if __name__ == "__main__":
    main()