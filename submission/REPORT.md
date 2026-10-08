# Lab 21 — Báo cáo đánh giá fine-tuning

**Họ tên:** Trần Nguyễn Tiến Đức
**MSSV:** 2A202602871
**Ngày tổng hợp:** 08/10/2026

> Nộp theo Option B — GitHub + Hugging Face Hub. Số liệu core lấy từ ZIP results của lượt Drive; số liệu toán lấy từ notebook Kaggle đã thực thi. Các giới hạn bằng chứng định tính được nêu rõ ở mục 9.

## 1. Câu hỏi và thiết kế

Tôi kiểm tra xem LoRA có tốt hơn chính base model khi dùng prompt tối ưu hay không. Bài chính là ticket CSKH tiếng Việt sang JSON bốn trường intent, urgency, product, sentiment. Base là unsloth/Qwen3.5-4B, tier T4; 250 mẫu corpus chia 225 train và 25 validation với seed 42. Lượt chính dùng đủ 50 ticket eval và 15 câu hỏi regression. NB1 → NB5 chạy tuần tự, baseline được đo trước train và kết quả được sao lưu lên Drive sau từng notebook.

Run correct dùng text-linear, rank 16, alpha 32, LR 1e-4, fp16, 32,464,896 tham số trainable và 30 step. Rank sweep giữ cùng data, LR, vị trí và số step. Bonus toán là thí nghiệm riêng trên Kaggle; không trộn accuracy toán một đáp án với target CSKH trung bình bốn trường.

## 2. Tính đúng đắn của pipeline

Mask được xây bằng offset ký tự của chat template và tokenize một lần; nhãn pre-tokenized được đưa vào trainer. Như vậy mask đã kiểm chứng không bị thay bằng mask do một flag của thư viện suy ra. Packing tắt để tránh phá căn chỉnh nhãn. Tokenizer/chat template, p95, max_length và phần supervised cần được trích nguyên từ template_check.json, token_stats.json, mask_proof.json của lượt core.

Hai assert đều true: answer_is_supervised và question_is_masked. Có 39/94 token được tính loss, supervised_fraction=0.4149. Template giữ nội dung reasoning trong phép kiểm tra template. Corpus CSKH chỉ có đáp án JSON; không dùng valid_trace_rate=0 ở core để suy ra reasoning collapse.

Độ dài token: mean=93.1, p95=98, max=101, suggested_max_length=256. Tôi giữ max_length=1024 của tier T4 để tái lập công thức cấu hình của lab, dù p95 cho thấy 256 đã đủ với corpus này. Tất cả run core dùng cùng giới hạn; đây là lựa chọn bảo thủ, không phải tối ưu theo p95.

Đoạn supervised được decode từ artifact:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>

```

## 3. Kết quả bài chính — lượt Drive 20261007_221152_dd4c92

| Run | Target | Regression | Format | Latency ms/mẫu |
|---|---:|---:|---:|---:|
| Base + prompt đơn giản | 0.0000 | 0.7911 | 0.0000 | 3278.8 |
| Base + prompt tối ưu | 0.7650 | 0.7911 | 1.0000 | 1038.8 |
| LoRA correct | 0.9700 | 0.6778 | 1.0000 | 1407.4 |

Target của correct tăng 0.205, tức 20.5 điểm phần trăm so với prompt tối ưu. Tuy nhiên regression giảm khoảng 0.1133, vượt ngưỡng 0.020; verdict là FAILED. Format đều 1.000 ở prompt tối ưu và correct, nên việc fine-tune không đem lại lợi ích format so với mốc mạnh này. Correct còn chậm hơn khoảng 35.5% theo latency của lượt đo.

## 4. Phán quyết và hệ quả triển khai

Tôi chưa chọn adapter correct làm model thay thế tổng quát cho base. Lý do không nằm ở loss huấn luyện mà ở việc nó không vượt qua điều kiện bảo toàn khả năng tổng quát. Mặc dù điểm phân loại từng trường tăng từ 0.765 lên 0.970, điểm regression trên cùng bộ probe lại giảm từ 0.7911 xuống 0.6778. Một hệ thống CSKH có thể gặp câu hỏi ngoài phạm vi ticket, vì vậy lợi ích tác vụ hẹp cần được đặt cạnh rủi ro này. Prompt tối ưu vẫn giữ format 1.000 và có latency thấp hơn adapter. Với dữ liệu hiện có, đó là lựa chọn thận trọng hơn. Tôi giữ nguyên bộ eval và ngưỡng hồi quy thay vì nới điều kiện để lấy PASSED. Nếu nghiên cứu thêm, replay dữ liệu tổng quát là giả thuyết cần thử bằng một thí nghiệm mới; báo cáo hiện tại chưa có bằng chứng rằng replay sẽ giải quyết được vấn đề này.

## 5. Đối chứng cấu hình

| Run | Target | Format | Latency ms |
|---|---:|---:|---:|
| correct | 0.9700 | 1.0000 | 1407.4 |
| attn_only | 0.9700 | 1.0000 | 917.4 |
| wrong_lr | 0.0000 | 0.0000 | 5325.7 |
| qlora | 0.9400 | 1.0000 | 1783.7 |

Correct và attn_only hòa trên target. Vì thế không kết luận text-linear thắng attention-only trong thí nghiệm này. Loss thấp hơn cũng không tự động chứng minh chất lượng eval cao hơn; run được xếp hạng bằng điểm target. Wrong_lr giảm LR xuống 1e-5 và là run kém nhất trên target. QLoRA có target thấp hơn correct 3 điểm phần trăm; phân tích đánh đổi VRAM và loss sẽ dùng đúng các dòng runs.csv của lượt Drive, không lấy số thời gian của lượt Colab cũ.

| Run | Placement | r | Trainable | LR | Train loss | Train s | Peak VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 | 32,464,896 | 0.0001 | 0.6272 | 412.2 | 8.78 |
| attn_only | attn-only | 283 | 32,456,704 | 0.0001 | 0.5374 | 268.8 | 8.79 |
| wrong_lr | text-linear | 16 | 32,464,896 | 1e-05 | 1.5702 | 412.7 | 8.78 |
| qlora | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 472.3 | 3.86 |

Ngân sách attn_only là 32,456,704 so với 32,464,896 của correct, chênh khoảng 0.025%, dưới ngưỡng 5%. Cả bốn run có 30 step. Attn_only có train loss thấp hơn correct nhưng hòa target, nên lợi thế loss không tương đương lợi thế eval. LR nhỏ hơn 10 lần ở wrong_lr đi cùng loss cao và target 0; không kết luận LoRA thiếu năng lực khi cấu hình LR đã sai. QLoRA giảm peak VRAM khoảng 56.0% (8.78 → 3.86 GB) nhưng target giảm 0.030 và latency tăng; với cùng T4, LoRA 16-bit là lựa chọn có chất lượng target tốt hơn trong phép đo này.

## 6. Quét rank — B4

| Rank | Tham số trainable | Target | Regression | Peak VRAM GB |
|---|---:|---:|---:|---:|
| 8 | 16,232,448 | 0.8650 | 0.7467 | 8.51 |
| 16 | 32,464,896 | 0.9700 | 0.6778 | 8.78 |
| 64 | 129,859,584 | 1.0000 | 0.0667 | 10.47 |

Rank 8 → 16 tăng target 10.5 điểm phần trăm; 16 → 64 chỉ tăng thêm 3 điểm, trong khi số tham số trainable tăng 4 lần. Rank 64 đạt target 1.000 nhưng regression chỉ 0.0667. Đây là bằng chứng rằng tăng năng lực adapter không tự động tạo ra model phù hợp hơn cho toàn bộ yêu cầu sử dụng. Trong các run quan sát được, ảnh hưởng lên target lớn nhất đến từ LR, tiếp theo là rank; hai vị trí khớp ngân sách tham số không tạo ra chênh lệch target. Không khái quát kết luận này cho mọi dataset hoặc seed.

## 7. Dataset và reasoning-mask — B2/B3

Tôi dùng tập con tuyển chọn từ openai/gsm8k, giấy phép MIT, revision 740312add88f781978c0658806c59bc2815b9866. Có 250 train, 50 validation, 50 eval và 15 regression. Eval lấy từ official test split. Câu hỏi chuẩn hóa và mẫu câu chỉ khác số được khử trùng giữa các tập; không khẳng định đã loại mọi paraphrase hoặc dữ liệu trùng pretraining. Sau kiểm tra mẫu bằng AI assistant, 5 mẫu có vấn đề về cách diễn đạt/lời giải được loại trước huấn luyện. Tôi không trình bày đây là dữ liệu riêng tự thu thập hay điểm benchmark GSM8K đầy đủ.

Hai mask dùng cùng model revision, rank 16, alpha 32, LR 1e-4, seed 42 và 32 step. Thinking được bật ở train và eval tác vụ toán. Mọi mẫu được giữ nguyên toàn bộ trong max_length=512; p95 của tập đã chuẩn bị là 292 và max 368 token.

| Run | Accuracy toán | Format | valid_trace_rate | Regression |
|---|---:|---:|---:|---:|
| baseline_a | 0.0200 | 0.0000 | 0.0200 | 0.7911 |
| baseline_b | 0.0000 | 0.0000 | 0.0200 | 0.7911 |
| assistant-only | 0.8000 | 0.9800 | 0.9800 | 0.7022 |
| response-only | 0.2600 | 0.9000 | 0.9000 | 0.8578 |

Assistant-only tốt hơn response-only trên tác vụ toán và tỷ lệ trace đã đóng, nhưng có regression thấp hơn. Tôi chưa tái lập được hiện tượng accuracy tăng trong khi trace giảm giữa hai mask: hai chỉ số thay đổi cùng chiều. Response-only PASSED theo cổng của script, còn assistant-only FAILED vì regression giảm quá ngưỡng. Tuy nhiên baseline toán chỉ đạt 0–2% và hầu như không hoàn tất output đúng định dạng. Cần đọc dự đoán gốc để xác nhận vai trò của giới hạn 512 token; không diễn giải kết quả này thành “base model chỉ biết giải 0–2% bài”. valid_trace_rate là chỉ số cấu trúc, không chứng minh lập luận bên trong đúng hoặc faithful.

## 8. Merge/hot-swap và Hub — B1/B5

Ở lượt Colab trước, merge được đo trên 50 ticket: trước 0.9700, sau 0.9700, delta 0.0000. Lần hot-swap đầu gặp lỗi offload sau khi giữ tham chiếu model cũ; lần chạy lại riêng phần hot-swap kết thúc returncode=0 sau assert có ít nhất hai adapter. Giữ merge_check.json và log sửa lỗi làm bằng chứng. Không trình bày log failed của lần đầu thành một lần NB6 hoàn tất.

Adapter công khai: https://huggingface.co/duckak12/lab21-qwen35-triage-vi

Link này được tạo từ lượt Colab trước lượt Drive chính. Các số liệu lượt Drive được ghi theo đúng artifact của nó; không khẳng định checkpoint Hub đó chính là checkpoint của lượt Drive mới. Đây là điểm cần minh bạch để tránh trộn provenance giữa các lần chạy.

## 9. Định tính

Bảng dưới dùng dự đoán đầy đủ của hai run rank 8 và rank 64, cùng tập eval đã đóng băng. Hai ca thua là **rank 8 thua rank 64**, không phải bằng chứng correct thua prompt (b). Artifact core chỉ lưu chuỗi FT đã rút gọn, không lưu dự đoán (b) từng mẫu; vì vậy chưa chứng minh đủ hai ca thua so với prompt tối ưu như bảng mẫu của rubric. Tôi nêu giới hạn này và không tạo dự đoán baseline giả.

| # / eval index | Ticket | Nhãn | FT rank 8 | FT rank 64 | Nhận xét |
|---|---|---|---|---|---|
| 1 / 0 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | {"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"} | {"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"} | FT r8 thua r64; sai so với nhãn |
| 2 / 1 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"} | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"} | FT r8 thua r64; sai so với nhãn |
| 3 / 4 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn VN339109. Vỡ khi nhận. Gấp. Shop xem giúp. | {"intent": "san_pham_loi", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "trung_tinh"} | {"intent": "san_pham_loi", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "trung_tinh"} | {"intent": "san_pham_loi", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "trung_tinh"} | Cả hai đúng, không chọn riêng ca thắng |
| 4 / 7 | Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình. | {"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"} | {"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"} | {"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"} | Cả hai đúng, không chọn riêng ca thắng |
| 5 / 8 | Xin chào, mình đặt chuột không dây mã đơn DH139158. Bảo hành bao lâu. Không vội. Mình vẫn tin tưởng shop. | {"intent": "hoi_thong_tin", "urgency": "thap", "product": "chuột không dây", "sentiment": "tich_cuc"} | {"intent": "hoi_thong_tin", "urgency": "thap", "product": "chuột không dây", "sentiment": "tich_cuc"} | {"intent": "hoi_thong_tin", "urgency": "thap", "product": "chuột không dây", "sentiment": "tich_cuc"} | Cả hai đúng, không chọn riêng ca thắng |


## 10. Kết luận và điều tôi học được

Qua lab, tôi thấy việc fine-tune chạy thành công và loss giảm chưa đủ để quyết định sử dụng model. Ở bài CSKH, adapter correct đạt target cao hơn prompt tối ưu, nhưng đánh đổi là điểm regression giảm và latency tăng. Khi quét rank, model rank 64 đạt điểm tác vụ cao nhất nhưng lại giữ khả năng tổng quát kém nhất theo bộ probe của lab. Những kết quả này khiến tôi xem đánh giá nhiều nhóm là một phần của thiết kế thí nghiệm, thay vì một bước trang trí sau huấn luyện. Tôi cũng thấy thay đổi mask có tác động rõ trong thí nghiệm toán: assistant-only học tác vụ tốt hơn, trong khi response-only giữ điểm regression tốt hơn. Tuy nhiên kết quả phụ thuộc dataset, prompt, ngân sách sinh và một seed đã dùng, nên chưa thể đưa ra quy tắc chung rằng một mask luôn tốt hơn. Cuối cùng, tôi mất thời gian vì runtime và chuyển môi trường, cho thấy tính tái lập còn bao gồm việc bảo toàn artifact. Tôi sẽ dùng mốc prompt mạnh, kiểm tra mask và sao lưu dữ liệu trước khi bắt đầu một dự án fine-tuning mới.

Ba bài học từ số liệu và quá trình thực hiện:
1. Tăng target có thể đi cùng giảm regression; giữ ngưỡng kiểm tra có ý nghĩa hơn cố đạt PASSED.
2. Rank lớn hơn không tự động là lựa chọn tốt hơn; cần xem tài nguyên và chất lượng ngoài nhiệm vụ hẹp.
3. Notebook lưu code/output không thay thế việc sao lưu results và adapter; Drive và output ZIP cần được kiểm tra thực tế.

AI assistant hỗ trợ đọc repo, chuẩn bị script, tuyển chọn dữ liệu, kiểm tra lỗi và tổng hợp kết quả. Hướng dẫn ban đầu đã có các điểm phải sửa: giả định biến ROOT còn tồn tại, subprocess không đưa log lên ô notebook, và NB6 giữ model cũ khi nạp base lần hai. Tôi ghi nhận những lỗi đó thay vì coi mã AI tạo ra là bằng chứng tự đủ.

## Bằng chứng cần có trong gói nộp

- Báo cáo đã điền và results core gốc của lượt Drive chính.
- bonus_rank_sweep.json cùng các run rank và dự đoán eval.
- Toàn bộ bonus_gsm8k/data và results, đặc biệt baseline/FT predictions.
- merge_check.json, log hot-swap và link Hub công khai.
- Code và dependency manifest để tái lập.

Gatekeeper được chạy trên artifact core hiện có. Các dự đoán toán đầy đủ và adapter toán chưa được chuyển về repo; notebook thực thi và JSON tổng hợp là bằng chứng được giữ. Không xem PASSED của response-only là chứng minh model toán tốt hơn một baseline không bị giới hạn ngân sách sinh.
