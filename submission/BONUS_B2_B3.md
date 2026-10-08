# B2/B3 — kết quả thí nghiệm GSM8K trên Kaggle

Nguồn: `math_bonus_summary.json` do học viên xuất sau khi chạy notebook Kaggle. Các checksum dataset khớp gói dữ liệu đã tuyển chọn. Đây là thí nghiệm riêng, không thay các số liệu CSKH của lượt core.

## B2: tập dữ liệu và nguồn

Tập con `openai/gsm8k`, revision `740312add88f781978c0658806c59bc2815b9866`, giấy phép MIT. Có 250 mẫu train, 50 validation, 50 eval từ official test split và 15 câu hỏi regression giữ nguyên từ core. Mỗi mẫu chứa lời giải suy luận gốc, đáp án số chuẩn hóa và nguồn split/row. Đã loại trùng câu hỏi chuẩn hóa và mẫu câu chỉ khác số giữa các tập; 5 mẫu được loại sau kiểm tra chất lượng trước huấn luyện.

Đây là tập con được tuyển chọn từ dữ liệu công khai, không phải dữ liệu riêng tự thu thập. Không có bằng chứng đảm bảo GSM8K chưa xuất hiện trong pretraining của Qwen. Do đó không báo cáo số đo như một benchmark tổng quát mới hay điểm GSM8K chính thức đầy đủ.

## B3: phép đối chứng mask

Cùng model `unsloth/Qwen3.5-4B`, model revision `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636`, rank 16, alpha 32, LR 1e-4, seed 42, hai epoch và 32 step cho mỗi mask. `max_length=512` và không cắt mẫu; đánh giá 50 bài toán và 15 câu hỏi regression. Thinking được bật cho bài toán ở cả train lẫn eval; regression dùng thinking=False nhất quán giữa baseline và adapter.

| Run | Accuracy toán | Format | valid_trace_rate | Regression | Latency ms |
|---|---:|---:|---:|---:|---:|
| baseline_a | 0.0200 | 0.0000 | 0.0200 | 0.7911 | 18028.1 |
| baseline_b | 0.0000 | 0.0000 | 0.0200 | 0.7911 | 17954.1 |
| assistant-only | 0.8000 | 0.9800 | 0.9800 | 0.7022 | 7289.2 |
| response-only | 0.2600 | 0.9000 | 0.9000 | 0.8578 | 9378.9 |

`assistant-only` tốt hơn `response-only` 54 điểm phần trăm accuracy và 8 điểm phần trăm valid_trace_rate, nhưng regression thấp hơn. Với dữ liệu và ngân sách này, chưa tái lập được hiện tượng accuracy tăng trong khi trace giảm: giữa hai mask, accuracy và trace thay đổi cùng chiều. Điểm đáng chú ý là đánh đổi giữa học tác vụ toán và duy trì khả năng tổng quát.

Phán quyết theo cổng của script: assistant-only FAILED vì regression giảm khoảng 0.0889 so với base; response-only PASSED vì target tăng 0.2400 và regression tăng khoảng 0.0667. Mốc target được chọn là baseline_a (0.02), mạnh hơn baseline_b (0.00), tránh so adapter với một mốc yếu hơn để làm kết quả đẹp.

## Giới hạn quan trọng cần đọc dự đoán gốc

Cả hai baseline đều có format 0.00 và valid_trace_rate 0.02 với max_new_tokens=512. Các số này gợi ý phần lớn output chưa trả được câu trả lời cuối đúng định dạng, có thể do reasoning không kết thúc trong ngân sách sinh. Chỉ JSON tổng hợp chưa đủ xác nhận nguyên nhân; cần đọc raw_completion trong math_baselines_frozen.json và kiểm tra kết thúc chuỗi. Không diễn giải target 0–2% thành “base model chỉ biết giải 0–2% bài toán”.

PASSED của response-only là kết quả có điều kiện của quy trình sinh/chấm này, không phải chứng minh khả năng toán vượt trội hay khuyến nghị deploy. `valid_trace_rate` chỉ đo block think không rỗng và đã đóng, không đo tính đúng/faithful của từng bước suy luận. Opening tag đã có trong generation prompt được tính vào trace theo cấu trúc thực tế; không tạo thêm body hoặc closing tag.

## File cần giữ

- data/CUSTOM_DATASET.md, dataset_manifest.json, token_stats.json, mask_proof.json.
- math_baselines_frozen.json và math_base_naive.json/math_base_optimized.json: dự đoán mốc.
- assistant-only_train.json, response-only_train.json: cấu hình và số step.
- assistant-only_eval.json, response-only_eval.json: dự đoán đầy đủ và regression.
- Hai adapter và math_bonus_summary.json.

Các giới hạn trên cần được diễn giải trong report; không thay đổi tập eval hoặc làm yếu baseline sau khi xem kết quả.
