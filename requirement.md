yêu cầu xây dựng 1 phần mềm bằng python và sử dụng lanchain với các yêu cầu bên dưới
yêu cầu chung
* Có main để chay commandline
* có hàm summary_requirement để dễ dàng sử dụng như thư viện
* log sử dung logging
* các bước call llm cần log đầy đủ các nội dung prompt và output với level debug. các bước còn lại chỉ cần log dể biết chạy đến bước nào và số thông tin cần thiết 
* các nội dung prompt trao đổi llm cần sử dụng **tiếng anh** và bao gồm **ROLE**
* kết quả trả ra từ llm phải là định đạng **JSON** và **định đạng JSON phải định nghĩa từ prompt** để bảo đảm có thể kiểm tra kết quả 1 các chính xác với các điều kiện rẽ nhánh. Ví dụ: Evaluate if the summary accurately reflects the original text and output STRICTLY a JSON object with keys "is_appropriate" (boolean), "reason" (string explaining why or what needs fixing), and "suggested_summary" (string with revised summary if false, else null). Original: [TEXT] Summary: [SUMMARY]

* không dùng **PromptTools**,**Embeddings**
Hàm summary_requirement
1. input 
    1.1. là 1 file md mô tả yêu cầu phần mền (ví dụ: yeu_cau.md)
    1.2. số từ tối đa của phần tóm tắt max_summary_word
    1.3. số vòng lặp tối đa max_loop
    1.4. đối tượng llm được khởi tạo trước đó
2. chương trình sẽ kiểm tra xem có file tóm tắc của file mô tả hay không( tên file yêu cầu + .summary ví dụ:yeu_cau.md.summary)?
    2.1. nếu chưa có file tóm tắt thì sẽ dùng langchain để thực hiện việc tóm tắt:
        2.1.1. gắn yeu_cau_sua = [] (yêu cầu chỉnh sửa) , noi_dung_tom_tat=[](nội dung đã tóm tắt)
        2.1.2. đọc nội dung file yêu cầu gắn noi_dung_can_tom_tat  (nội dung cần tóm tắt)
        2.1.3. gắn count =0
        2.1.4. vòng lặp trong khi count < max_loop
            2.1.4.1 count = count + 1
            2.1.4.2 yêu cầu llm tóm tắt ngắn gọn và đủ ý ( tối đa max_summary_word tử) với noi_dung_can_tom_tat và yeu_cau_sua và gắn vào noi_dung_tom_tat (nội dung đã tóm tắt)
            2.1.4.3 yêu cầu llm kiểm tra kết quả tóm tắt và đưa ra ý kiến có phải chỉnh sửa không? nếu chỉnh sửa thì chỉnh phần nào? và lý do vì sao
            2.1.4.4 nếu kết quả ở bước 2.1.4.3 là cần chỉnh sửa thì
                2.1.5.1 gắn yeu_cau_sua = yêu cầu cần chỉnh sửa có ở 2.1.4.3
                 2.1.5.2 nếu  count == max_loop thì log thông tin chạm ngưỡi max_loop
            2.1.5 nếu không cần chỉnh sửa thì kết thúc, gắn count = max_loop




---

yêu cầu xây dựng 1 phần mềm bằng python và sử dụng lanchain với các yêu cầu bên dưới
yêu cầu chung
* Có main để chay commandline
* có hàm generate_code để dễ dàng sử dụng như thư viện
* log sử dung logging
* các bước call llm cần log đầy đủ các nội dung prompt và output với level info. các bước còn lại chỉ cần log dể biết chạy đến bước nào và số thông tin cần thiết để giúp xác định đang chạy đến đâu.
* các nội dung prompt trao đổi llm cần sử dụng **tiếng anh** và bao gồm **ROLE**
* kết quả từ llm trả về phải dùng  parse_json_response(raw_text: str) -> Dict[str, Any] để đọc (from kaggle_toolsets.summary import parse_json_response)
* kết quả trả ra từ llm phải là định đạng **JSON** và **định đạng JSON phải định nghĩa từ prompt** để bảo đảm có thể kiểm tra kết quả 1 các chính xác với các điều kiện rẽ nhánh. Ví dụ: Evaluate if the summary accurately reflects the original text and output STRICTLY a JSON object with keys "is_appropriate" (boolean), "reason" (string explaining why or what needs fixing), and "suggested_summary" (string with revised summary if false, else null). Original: [TEXT] Summary: [SUMMARY]

Hàm generate_code
1. Input
    1.1. là 1 file md mô tả yêu cầu cần code (ví dụ: yeu_cau.md)
    1.2. tên ngôn ngữ lập trình(lang)
    1.3. số vòng lặp tối đa max_loop
    1.4. đối tượng llm được khởi tạo trước đó
2. các bước xử lý
    2.1. đọc nội dung file và gắn vào biến yeu_cau
    2.2. găn count = 0, generated_code =[], nhan_xet=[]
    2.3 vòng lặp trong khi count < max_loop
        2.3.1. count = count + 1
        2.3.2. yêu cầu llm sinh code với nội dung yêu cầu trong biến yeu_cau, và nội dung nhận xét trong biến nhan_xet và ngôn ngữ được yêu cầu (lang)
        2.3.3. kiểm tra kết quả đầu ra với JSON format và có nội dung code(lưu ý dùng  parse_json_response) thì thực hiện các bước sau:
            2.3.3.1. generated_code = code được sinh ra
            2.3.3.2. yêu cầu llm đưa ra nhận xét, code được sinh ra generated_code đã phù hợp với nội dung yêu cầu trong biến yeu_cau, và nội dung nhận xét trong biến nhan_xet chưa? nội dung trả về  phải bao gồm tối thiểu: có cần chỉnh sửa hay không, nội dung cần chỉnh sửa nếu có, lý do cần chỉnh sửa nếu có (lưu ý dùng  parse_json_response để  parse respone từ llm)
                2.3.3.2.1. nếu không cần phải chỉnh sửa thì count = max_loop , đừng vòng lặp và ghi log thông báo
                2.3.3.2.1. nếu cần chỉnh sửa nhưng đã count == max_loop, thông báo và ghi nhận lỗi