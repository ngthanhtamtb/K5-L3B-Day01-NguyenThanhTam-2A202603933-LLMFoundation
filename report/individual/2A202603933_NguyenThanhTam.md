# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603933
- **Họ và tên:** Nguyễn Thành Tâm
- **SEED** (in ở ô đầu notebook): 1864

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Bảng kết quả mình chạy được trong [`day01_lab.ipynb`](day01_lab.ipynb) và [`lab1_report.json`](lab1_report.json) với `SEED = 1864` (sau khi bỏ `403` tin trùng lặp còn `5.169` tin: `train = 3.619` tin, `val = 775` tin, `test = 775` tin, và `SUB = 40` tin):

| Phương án đã chạy | Accuracy | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 (ms/tin) | p95 (ms/tin) | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 0.972 | 0.989 | 0.806 | 0.888 | 0.919 | 4.44 | 23,366 | 0.07 | 0.13 | \$0.00 |
| **2. BoW + MLP (deep learning)** | 0.981 | 0.979 | 0.880 | 0.927 | 0.947 | 4.41 | 32,267 | 0.17 | 0.23 | \$0.00 |
| **3a. Encoder (MiniLM) + kNN** | 0.978 | 0.910 | 0.935 | 0.922 | 0.950 | 68.90 | 31 | 54.03 | 89.24 | \$0.00 |
| **3b. Encoder (MiniLM) + LogReg** | 0.987 | 0.980 | 0.926 | 0.952 | 0.976 | 69.54 | 53 | 26.68 | 35.68 | \$0.00 |

## 3. Dự đoán và thực tế

Trong các dự đoán `predict(...)` mình ghi lại trước khi chạy từng phần, có 3 chỗ kết quả thực tế làm mình chú ý nhất:

1. **Dự đoán `dl_curve` (Train mạng MLP 30 epoch):**
   - **Mình đoán:** `"(1)val loss có giảm, (2) MLP sẽ hơn"`
   - **Thực tế:** Điểm F1 của MLP đạt `0.927`, đúng là cao hơn Logistic Regression (`0.888`). Nhưng trên biểu đồ, `val loss` chỉ giảm tới epoch 6 là thấp nhất rồi bắt đầu tăng ngược trở lại, trong khi `train loss` vẫn giảm tiếp.
   - **Vì sao lệch:** Lúc đầu mới tìm hiểu cứ cho máy train càng nhiều vòng thì lỗi trên cả hai tập đều sẽ giảm. Khi chạy xong và tính số tham số => mạng MLP có quá nhiều tham số (`128.129` tham số so với `2.001` của LogReg) trong khi dữ liệu train lại ít (`3.619` tin), nên sau epoch 6 mô hình bắt đầu học thuộc lòng dữ liệu train làm kết quả trên tập val kém đi.

2. **Dự đoán `dl_relu` (Bỏ hàm ReLU trong mạng MLP):**
   - **Mình đoán:** `"hai lớp Linear liên tiếp sẽ tương đương với một lớp Linear duy nhất, F1 có thể thấp hơn MLP có ReLU"`
   - **Thực tế:** Khi bỏ `nn.ReLU()`, kết quả `TEST F1` vẫn ra `0.927`, bằng đúng với mạng MLP lúc có `ReLU`.
   - **Vì sao lệch:** Bài này dữ liệu đầu vào vẫn là Bag-of-Words (chỉ đếm sự xuất hiện của 2.000 từ), bản thân sự xuất hiện của các từ khoá spam riêng lẻ đã gần như đủ để phân loại tuyến tính rồi nên việc có hay không có `ReLU` chưa tạo ra khác biệt lớn trên tập test.

3. **Dự đoán `ml_f1` (Huấn luyện Logistic Regression ở Phần 1):**
   - **Mình đoán:** `"F1 ~0.88, Precision cao hơn Recall vì tập dữ liệu mất cân bằng và ta muốn tránh chặn nhầm tin nhắn Ham"`
   - **Thực tế:** F1 đạt `0.888` (khá sát dự đoán), `Precision` đạt rất cao (`0.989`) nhưng `Recall` chỉ đạt `0.806`.
   - **Vì sao lệch:** `Recall` lại thấp hơn `Precision` nhiều như vậy (bỏ lọt gần 20% số tin spam). Khi xem danh sách 10 câu đoán sai ở ngay bên dưới, thấy có tới 9/10 câu là tin `spam` bị đoán nhầm thành `ham`. Lý do là trong tập dữ liệu số tin `ham` nhiều gấp gần 7 lần tin `spam`, nên mô hình khá "an toàn", chỉ khi gặp câu có nhiều từ khoá spam rõ rệt mới báo là `spam`.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

- Theo `SEED = 1864` của mình (`1864 % 6 = 4`), đề tương ứng là **Đề #5**: *"Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."*
- Trong buổi thực hành mình chưa kịp chạy tới ô code của phần thử thách này, nhưng qua những gì đã học ở Phần 1 và Phần 3 thì mình hiểu ý tưởng chính của đề như sau:
  - Nếu mình viết **2 tin spam né từ khoá** (không dùng các từ như `free`, `win`, `call`, `txt` mà diễn đạt bằng các từ khác mang ý trúng thưởng hoặc giục bấm link), mô hình `BoW + LogReg` sẽ không tìm thấy từ khoá quen thuộc nào trong từ điển nên rất dễ đoán nhầm thành `ham`. Ngược lại, `Embedding + LogReg` hiểu được ý nghĩa tổng thể của câu nên sẽ nhận diện `spam` tốt hơn.
  - Nếu mình viết **2 tin ham có chứa chữ `'free'` và `'win'`** (ví dụ rủ bạn lúc nào rảnh đi ăn hoặc nói chuyện thắng một trận game), `BoW + LogReg` thấy có mặt hai từ khoá mang trọng số dương lớn nên dễ bắt nhầm thành `spam` (giống câu `"Waiting for your call."`), còn `Embedding + LogReg` đặt cả câu vào ngữ cảnh trò chuyện bạn bè nên sẽ đoán đúng là `ham`.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

### Bảng AI Log

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| :--- | :--- | :--- | :--- |
| **Tìm hiểu Repo & Khái niệm** | Nhờ AI phân tích cấu trúc repo, thứ tự đọc file và giải thích dễ hiểu các thuật ngữ trong bài lab (như *Data Leakage, Precision, Recall, F1, Sigmoid, Gradient Descent, ReLU, Overfitting, Embedding, Cosine Similarity*). | AI tóm tắt lộ trình làm bài và giải thích từng khái niệm kèm ví dụ trong bài toán lọc tin nhắn rác. | Đọc đối chiếu với từng phần trong notebook để hiểu bản chất trước khi làm các ô TODO và trả lời câu hỏi giải thích. |
| **Phần 1–3 (Hỗ trợ kiểm tra logic)** | Hỏi AI về công thức cập nhật trọng số trong Logistic Regression, 5 bước trong vòng lặp huấn luyện PyTorch và cách chuẩn hoá độ dài vector để tính Cosine Similarity bằng nhân ma trận. | AI giải thích công thức đạo hàm, ý nghĩa của `opt.zero_grad()` / `loss.backward()` / `opt.step()` và cách chia norm L2. | Tự chạy kiểm tra trên Colab, đạt ✅ toàn bộ 8/8 bài kiểm tra code từ `1.1` đến `3.3` và tự làm 4 bài tính tay trên giấy (`hand_calc.jpg`). |

### Điều mình đã học được hôm nay

Là người mới bắt đầu học về AI, bài thực hành hôm nay giúp mình vỡ ra được 3 điều rất quan trọng mà trước giờ mình chỉ nghe tên chứ chưa hiểu rõ:
1. Hiểu được thông số đánh giá mô hình qua các chỉ số Accuracy, Precision, Recall, F1. Không được chỉ nhìn vào Accuracy: trên tập dữ liệu có tới ~87.4% là tin thường, một mô hình chẳng cần học gì mà cứ đoán tất cả là `ham` cũng đã đạt `87.37%` accuracy, nên bắt buộc phải dùng `Precision`, `Recall` và `F1` mới biết mô hình có thực sự bắt được spam mà không chặn nhầm tin quan trọng hay không.
2. Hiểu được **sự tiến hoá từ Bag-of-Words sang Deep Learning và Embedding**: BoW chỉ đếm mặt chữ rời rạc nên rất nhanh nhưng dễ bị lừa bởi ngữ cảnh; mạng MLP nhiều tham số hơn thì lại rất nhanh bị học vẹt (*overfitting* ngay từ sau epoch 6); còn khi mượn bộ `Encoder MiniLM` đã học sẵn ngôn ngữ từ trước thì dù chỉ gắn một bộ phân loại `LogReg` đơn giản lên trên (`3b`) cũng giúp hiểu được ý nghĩa câu và nâng F1 lên `0.952`.
3. Không phải mô hình nào điểm cao hay mới hơn cũng dùng được ngay mà phải nhìn vào **độ trễ (`p95`)**: ví dụ cùng dùng Embedding nhưng `kNN` (`3a`) mất tới `89.24 ms/tin` vì phải so sánh với toàn bộ tập train, trong khi `LogReg` (`3b`) chỉ mất `35.68 ms/tin` mà lại chính xác hơn.
