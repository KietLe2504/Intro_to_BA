# Intro_to_BA – Fraud Detection

## 1. Giới thiệu dự án

Dự án tập trung vào bài toán **Fraud Detection** (phát hiện giao dịch gian lận) trên dữ liệu giao dịch tài chính PaySim.

Mục tiêu là xây dựng quy trình xử lý dữ liệu, trích xuất đặc trưng (Feature Engineering), xử lý mất cân bằng dữ liệu và thử nghiệm nhiều mô hình Machine Learning nhằm phân biệt giao dịch bình thường và giao dịch gian lận.

- **Dataset:** PaySim
- **Target:** `isFraud`
- **Task:** Binary Classification
- **Main objective:** Phát hiện các giao dịch gian lận với khả năng nhận diện tốt, đặc biệt là hạn chế bỏ sót các giao dịch Fraud.

## 2. Dataset

### Nguồn dữ liệu

Dataset được sử dụng trong dự án:

- Kaggle: https://www.kaggle.com/datasets/almayashassan/ba-dataset
- File: `paysim_final_cleaned.csv` ( Sau khi data cleaning )

Đường dẫn sử dụng trên Kaggle Notebook:

```python
DATA_PATH = "/kaggle/input/datasets/almayashassan/ba-dataset/paysim_final_cleaned.csv"
```

### Mô tả dữ liệu

| Attribute | Description |
|---|---|
| `step` | Thời điểm giao dịch, tính theo đơn vị thời gian mô phỏng |
| `type` | Loại giao dịch |
| `amount` | Số tiền giao dịch |
| `nameOrig` | Tài khoản thực hiện giao dịch |
| `oldbalanceOrg` | Số dư tài khoản nguồn trước giao dịch |
| `newbalanceOrig` | Số dư tài khoản nguồn sau giao dịch |
| `nameDest` | Tài khoản nhận giao dịch |
| `oldbalanceDest` | Số dư tài khoản đích trước giao dịch |
| `newbalanceDest` | Số dư tài khoản đích sau giao dịch |
| `isFraud` | Nhãn xác định giao dịch gian lận |
| `isFlaggedFraud` | Cờ đánh dấu giao dịch đáng ngờ theo hệ thống mô phỏng |

## 3. Quy trình thực hiện

### Step 1. Data Loading & Inspection

Đọc dữ liệu từ file `paysim_final_cleaned.csv`, kiểm tra:

- Kích thước dataset.
- Kiểu dữ liệu của từng cột.
- Giá trị thiếu (Missing Values).
- Dữ liệu trùng lặp (Duplicates).
- Phân phối của biến mục tiêu `isFraud`.

Mục đích là nắm được cấu trúc dữ liệu và đặc điểm của bài toán trước khi tiến hành xử lý.

### Step 2. Data Splitting

Chia dữ liệu thành ba tập:

- **Train:** Dùng để huấn luyện mô hình.
- **Validation:** Dùng để lựa chọn đặc trưng, điều chỉnh mô hình và xác định threshold.
- **Test:** Dùng để đánh giá cuối cùng trên dữ liệu chưa được sử dụng trong quá trình phát triển mô hình.

Quy tắc chia dữ liệu theo `step`:

| Dataset | Condition |
|---|---|
| Train | `step <= 504` |
| Validation | `505 <= step <= 624` |
| Test | `step >= 625` |

Việc chia theo thời gian giúp mô phỏng tình huống thực tế, trong đó mô hình được huấn luyện trên dữ liệu quá khứ và dự đoán các giao dịch xảy ra trong tương lai.

### Step 3. Feature Engineering

Feature Engineering là quá trình tạo ra các đặc trưng mới từ dữ liệu ban đầu nhằm giúp mô hình nhận diện tốt hơn các mẫu giao dịch gian lận.

Các nhóm đặc trưng chính:

- **Transaction Features:** Đặc trưng liên quan đến số tiền và loại giao dịch.
- **Time Features:** Đặc trưng thể hiện thời điểm giao dịch.
- **Behavioral Features:** Đặc trưng mô tả hành vi giao dịch của tài khoản.
- **Risk Features:** Các chỉ số được sử dụng để biểu diễn mức độ rủi ro của giao dịch.

Một số đặc trưng được tạo trong quá trình xử lý:

| Feature | Description |
|---|---|
| `log_amount` | Logarithm của số tiền giao dịch |
| `amount_to_type_median` | Tỷ lệ giữa số tiền giao dịch và median của loại giao dịch tương ứng |
| `is_amount_outlier_3xIQR` | Đánh dấu giao dịch có số tiền nằm ngoài ngưỡng 3xIQR |
| `hour_of_day` | Thời điểm giao dịch trong ngày |
| `type_CASH_IN` | Đặc trưng One-Hot Encoding cho loại giao dịch CASH_IN |
| `type_CASH_OUT` | Đặc trưng One-Hot Encoding cho loại giao dịch CASH_OUT |
| `type_PAYMENT` | Đặc trưng One-Hot Encoding cho loại giao dịch PAYMENT |
| `type_TRANSFER` | Đặc trưng One-Hot Encoding cho loại giao dịch TRANSFER |
| `dest_prior_tx_count` | Số lượng giao dịch trước đó của tài khoản nhận |

Ngoài ra, pipeline còn tạo một số đặc trưng rủi ro bổ sung như `n_risk_flags`, `device_trust_score`, `failed_auth_attempts_24h`, `log_ip_distance` và `account_age_days`.

**Lưu ý:** Các đặc trưng tổng hợp này cần được kiểm tra kỹ về nguồn gốc và cách tạo. Trong phiên bản dữ liệu hiện tại, một số đặc trưng được tạo có phụ thuộc vào `isFraud`. Vì vậy, không nên sử dụng kết quả mô hình có các đặc trưng này để khẳng định hiệu năng thực tế.

### Step 4. Feature Selection & Analysis

Sau khi tạo đặc trưng, tiến hành phân tích và lựa chọn các feature phù hợp.

Các phương pháp được sử dụng:

- **Correlation Analysis:** Kiểm tra mức độ tương quan giữa các đặc trưng.
- **Fraud Rate Analysis:** Phân tích tỷ lệ Fraud theo từng nhóm giá trị.
- **Lift Analysis:** Đánh giá khả năng tập trung các giao dịch Fraud của một đặc trưng.
- **Drift Analysis:** Kiểm tra sự thay đổi phân phối đặc trưng giữa các tập dữ liệu.
- **Ablation Study:** Loại bỏ từng nhóm đặc trưng để đánh giá mức độ đóng góp của chúng vào kết quả mô hình.

Mục tiêu là lựa chọn tập feature có ý nghĩa, hạn chế đặc trưng gây rò rỉ nhãn (Label Leakage) và tránh phụ thuộc quá mức vào dữ liệu tổng hợp.

### Step 5. Feature Scaling

Chuẩn hóa dữ liệu để đưa các đặc trưng về phạm vi giá trị phù hợp với từng thuật toán.

Các scaler được lưu lại để có thể tái sử dụng khi đánh giá hoặc triển khai mô hình.

File được lưu:

- `02_scaler.joblib`

Các tập dữ liệu đã chuẩn hóa:

- `02_train_scaled.parquet`
- `02_valid_scaled.parquet`
- `02_test_scaled.parquet`

### Step 6. Handling Imbalanced Data

Dataset có tỷ lệ giao dịch Fraud rất thấp so với giao dịch bình thường. Đây là hiện tượng **Class Imbalance** (mất cân bằng lớp).

Pipeline sử dụng hai phương pháp chính:

1. **Random Undersampling:** Giảm số lượng mẫu thuộc lớp Majority để cân bằng tương đối với lớp Minority.
2. **SMOTENC:** Tạo thêm mẫu tổng hợp cho lớp Minority, có hỗ trợ xử lý các đặc trưng dạng categorical.

Các file dữ liệu sau khi xử lý:

- `03_train_undersampled.parquet`
- `03_train_under_smote.parquet`

Các phiên bản đã chuẩn hóa cũng được lưu riêng.

**Lưu ý:** Chỉ áp dụng kỹ thuật xử lý mất cân bằng trên tập Train. Không thực hiện undersampling hoặc SMOTE trên Validation và Test để giữ nguyên phân phối dữ liệu đánh giá.

### Step 7. Model Training & Evaluation

Sau khi hoàn thành Feature Engineering và xử lý mất cân bằng, tiến hành huấn luyện và so sánh các mô hình Machine Learning.

Các mô hình được thử nghiệm:

- LightGBM
- XGBoost
- Logistic Regression
- RF
  
Các chỉ số đánh giá:

| Metric | Meaning |
|---|---|
| PR-AUC | Đánh giá khả năng phân biệt hai lớp, đặc biệt hữu ích với dữ liệu mất cân bằng |
| ROC-AUC | Đánh giá khả năng xếp hạng mẫu Positive cao hơn mẫu Negative |
| Precision | Trong các giao dịch được dự đoán là Fraud, có bao nhiêu giao dịch thực sự Fraud |
| Recall | Trong tất cả giao dịch Fraud thực tế, mô hình phát hiện được bao nhiêu |
| F1-score | Trung bình điều hòa giữa Precision và Recall |
| Confusion Matrix | Ma trận thể hiện số lượng dự đoán đúng và sai của từng lớp |

Ngoài ra, mô hình được đánh giá với nhiều giá trị `threshold` khác nhau nhằm lựa chọn ngưỡng phân loại phù hợp với mục tiêu phát hiện Fraud.

### Step 8. Final Test Evaluation

Sau khi lựa chọn mô hình và threshold dựa trên Validation, tiến hành đánh giá cuối cùng trên Test.

Các bước:

1. Chọn mô hình có kết quả tốt trên Validation.
2. Xác định threshold phù hợp từ Validation.
3. Sử dụng mô hình đã chọn để dự đoán trên Test.
4. Tính toán các metric đánh giá.
5. Lưu lại kết quả cuối cùng.

Test
