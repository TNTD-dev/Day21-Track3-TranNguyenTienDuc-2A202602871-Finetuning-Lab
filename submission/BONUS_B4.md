# B4 — Quét rank có kiểm soát

Nguồn số liệu: `results/bonus_rank_sweep.json`, do học viên xuất từ Colab. Run rank 16 được tái sử dụng từ lượt core `20261007_221152_dd4c92`.

## Thiết kế thí nghiệm

Base: `unsloth/Qwen3.5-4B`, T4, fp16; vị trí `text-linear`; learning rate `1e-4`; 30 optimizer step; đánh giá đầy đủ 50 ticket. Giữ tỷ lệ alpha/r = 2, nên alpha lần lượt 16, 32, 128. Dataset, split, baseline và code core được ghi checksum trong JSON; tăng số tham số là hệ quả chủ ý của tăng rank.

## Kết quả

| Rank | Tham số trainable | Target | Regression | Format | Peak VRAM (GB) | Train loss |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 16,232,448 | 0.8650 | 0.7467 | 1.0000 | 8.51 | 0.7875 |
| 16 | 32,464,896 | 0.9700 | 0.6778 | 1.0000 | 8.78 | 0.6272 |
| 64 | 129,859,584 | 1.0000 | 0.0667 | 1.0000 | 10.47 | 0.5611 |

Target là độ chính xác trung bình của bốn trường theo scorer của lab, không phải tỷ lệ ticket hoàn toàn đúng. Regression dùng keyword recall trên bộ câu hỏi phổ thông.

## Diễn giải

Tăng rank 8 → 16 cải thiện target 10.5 điểm phần trăm. Tăng tiếp 16 → 64 cải thiện thêm 3 điểm phần trăm, đổi lấy số tham số trainable tăng 4 lần và peak VRAM tăng 1.69 GB (8.78 → 10.47). Tuy vậy, regression giảm từ 0.6778 xuống 0.0667. Vì thế, chỉ chọn model có target cao nhất sẽ bỏ qua suy giảm ở nhóm đánh giá tổng quát.

Baseline prompt tối ưu của lượt core có regression 0.7911. Cả ba rank đều thấp hơn baseline quá ngưỡng 0.02 của lab: rank 8 giảm khoảng 0.0444, rank 16 giảm khoảng 0.1133, rank 64 giảm khoảng 0.7244. Không có rank nào đủ điều kiện PASSED chỉ nhờ đổi rank. Rank 8 giữ điểm regression tốt nhất trong ba run; rank 16 cân bằng target và tài nguyên hơn rank 64, nhưng vẫn chưa đạt cổng hồi quy.

Trong NB4 của cùng lượt core, correct và attn_only cùng target 0.970, trong khi wrong_lr đạt 0.000. Do đó, độ biến thiên target quan sát được khi đổi LR là 0.970; khi quét rank là 0.135; và giữa hai vị trí có ngân sách tham số khớp là 0.000. Trên phép đo này, xếp theo tác động lên target: LR > rank > vị trí. Kết luận về vị trí chỉ áp dụng cho các run và bộ eval này; chưa chứng minh các vị trí tương đương trên mọi nhiệm vụ.

Đây là một run cho mỗi cấu hình; không có ước lượng độ bất định giữa các seed. Latency đo ở những thời điểm khác nhau trên GPU dùng chung nên không dùng các chênh lệch nhỏ để kết luận chắc chắn về tốc độ.

## Bằng chứng cần giữ

- `bonus_rank_sweep.json`: cấu hình, checksum và bảng tổng hợp.
- `bonus_rank_8.json`, `bonus_rank_64.json`: điểm và dự đoán đầy đủ.
- `bonus_rank_runs.csv`: loss, thời gian và VRAM của các run mới.
- Các adapter `bonus_rank_8/` và `bonus_rank_64/` cùng kết quả core rank 16.

Phần phản tư cá nhân trong report chính cần do học viên bổ sung; tài liệu này chỉ tổng hợp bằng chứng và lập luận từ số đo.
