# Reflection — Lab 21

**1. Điều gây ngạc nhiên:** rank 64 đạt target 1.000 nhưng regression chỉ 0.0667. Tôi chọn tập trung vào đánh đổi này thay vì chỉ nhìn điểm tác vụ.

**2. Phần mất thời gian:** chuyển runtime, sao lưu và xử lý notebook không hiện log. Kết quả phải chạy lại khi không còn file JSON/adapter trên VM; notebook có output không thay thế được artifact.

**3. Quan điểm sau lab:** tôi sẽ không dùng loss hoặc target một mình để quyết định deploy. Không có ghi chép về niềm tin trước lab, nên phần này trình bày bài học hiện tại thay vì dựng lại một quan điểm trước đó.

**4. AI assistant:** hỗ trợ đọc repo, tạo script, tìm và tuyển chọn GSM8K, xử lý lỗi và tổng hợp report. Các lỗi đã gặp gồm giả định ROOT còn tồn tại, log subprocess không hiện trong notebook và NB6 giữ tham chiếu model cũ. Tôi cần kiểm tra output thật và lưu bằng chứng, không chỉ tin lời assistant.

**5. Bước đầu cho dự án khách hàng:** xác định schema và tiêu chí chấp nhận, khóa tập eval không trùng train, đo prompt baseline trước training; sau đó kiểm tra mask, chi phí phần cứng và chiến lược sao lưu.
