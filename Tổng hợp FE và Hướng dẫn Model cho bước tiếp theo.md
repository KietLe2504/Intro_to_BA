# Feature Engineering (FE) --- Quy trình và hướng dẫn bàn giao

## 1. Mục tiêu của giai đoạn Feature Engineering

Feature Engineering là giai đoạn chuyển dữ liệu giao dịch ban đầu thành
tập đặc trưng phù hợp cho các mô hình Machine Learning.

Trong bài toán phát hiện giao dịch gian lận (*fraud detection*), FE thực
hiện các công việc chính:

-   Kiểm tra chất lượng và cấu trúc dữ liệu.
-   Phân tích phân phối nhãn, đặc biệt là tình trạng mất cân bằng giữa
    Fraud và Non-Fraud.
-   Tạo và biến đổi đặc trưng từ dữ liệu giao dịch.
-   Kiểm tra mối quan hệ giữa feature và target, đồng thời nhận diện
    nguy cơ leakage.
-   Chia dữ liệu theo thời gian thành Train, Validation và Test.
-   Chọn feature set, xử lý mất cân bằng trên tập Train và xuất dữ liệu
    để bàn giao cho Model Training.

**Đầu ra của FE** là các tập dữ liệu đã được xử lý, có cấu trúc feature
thống nhất và đi kèm metadata mô tả cách tạo dữ liệu cũng như các giới
hạn cần lưu ý.

------------------------------------------------------------------------

## 2. Tổng quan quy trình FE

  -----------------------------------------------------------------------
  Bước                    Nội dung                Ý nghĩa
  ----------------------- ----------------------- -----------------------
  1                       Load dữ liệu và kiểm    Nắm số dòng, số cột,
                          tra cấu trúc            kiểu dữ liệu, missing
                                                  values, duplicates và
                                                  phạm vi thời gian

  2                       Phân tích target và     Hiểu tỷ lệ
                          class imbalance         Fraud/Non-Fraud và phân
                                                  phối giao dịch theo
                                                  loại

  3                       Phân tích đặc trưng ban Kiểm tra biến số, biến
                          đầu                     phân loại, phân phối
                                                  amount và các trường
                                                  balance

  4                       Phân tích synthetic     Kiểm tra nguồn sinh
                          features                feature và mức độ phụ
                                                  thuộc vào target

  5                       Ordinal encoding và     Chuyển các nhóm
                          risk flags              categorical có thứ tự
                                                  thành mã số; tổng hợp
                                                  số lượng risk flags

  6                       Feature engineering và  Tạo feature dẫn xuất,
                          feature selection       xây dựng các nhóm
                                                  feature và chọn bộ
                                                  feature phục vụ thí
                                                  nghiệm

  7                       Temporal                Chia dữ liệu theo thời
                          Train/Valid/Test split  gian để mô phỏng việc
                                                  dự đoán trên giao dịch
                                                  tương lai

  8                       Feature ablation        So sánh đóng góp của
                                                  các nhóm feature, nhận
                                                  diện shortcut hoặc
                                                  feature có ảnh hưởng
                                                  lớn

  9                       Feature drift analysis  Kiểm tra sự thay đổi
                                                  trung bình feature giữa
                                                  Train và Validation

  10                      Resampling và export    Chỉ undersample Train;
                                                  giữ nguyên
                                                  Validation/Test; xuất
                                                  dữ liệu bàn giao
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 3. Dữ liệu và kiểm tra ban đầu

-   Dataset: `data_synthetic.csv`.
-   Tổng số bản ghi: **6,362,620**.
-   Dữ liệu có các thông tin giao dịch như `step`, `type`, `amount`,
    thông tin số dư tài khoản nguồn/đích và nhãn `isFraud`.
-   Các cột phụ trợ không sử dụng trong pipeline được loại khỏi tập
    feature theo cấu hình notebook.
-   Kiểm tra ban đầu ghi nhận không có missing values và duplicate rows
    trong dữ liệu được phân tích.

Target:

-   `isFraud = 1`: giao dịch gian lận.
-   `isFraud = 0`: giao dịch không gian lận.

Dữ liệu có sự mất cân bằng lớn: tỷ lệ Fraud trong Train ban đầu chỉ
khoảng **0.0927%**. Vì vậy, accuracy đơn thuần không phải thước đo phù
hợp để đánh giá mô hình.

------------------------------------------------------------------------

## 4. Synthetic features và cảnh báo quan trọng

Dataset có một số feature categorical được sinh thêm, gồm:

-   `account_age`
-   `device_status`
-   `failed_attempts`
-   `addr_mismatch`
-   `ip_billing_dist`

Các feature này được chuyển thành ordinal features:

-   `account_age_ord`
-   `device_status_ord`
-   `failed_attempts_ord`
-   `addr_mismatch_ord`
-   `ip_billing_dist_ord`

Ngoài ra, `n_risk_flags` được tạo bằng cách tổng hợp số lượng điều kiện
rủi ro từ các ordinal features.

### Giới hạn cần đặc biệt lưu ý

Theo tài liệu mô tả generator, quá trình sinh synthetic features có sử
dụng trực tiếp `isFraud` làm đầu vào. Do đó, các feature này **phụ thuộc
target theo thiết kế**.

Điều này có nghĩa:

-   Chúng có thể mang thông tin liên quan đến nhãn một cách có chủ đích.
-   Kết quả mô hình sử dụng các feature này chỉ phù hợp cho kiểm thử
    pipeline hoặc thí nghiệm có kiểm soát.
-   Không được diễn giải hiệu năng đạt được từ chúng như bằng chứng về
    khả năng phát hiện gian lận trong môi trường thực tế.
-   Feature set B bên dưới không thể được gọi là hoàn toàn leakage-free
    vì vẫn giữ các synthetic features.

Cảnh báo này cần được giữ nguyên khi chuyển dữ liệu sang nhóm Model
Training và khi viết báo cáo kết quả.

------------------------------------------------------------------------

## 5. Feature engineering và feature sets

Các phép biến đổi chính gồm:

-   `log_amount`: biến đổi log cho giá trị giao dịch.
-   `amount_to_type_median`: tỷ lệ/quan hệ giữa amount và median amount
    của loại giao dịch tương ứng.
-   `is_amount_outlier`: đánh dấu giao dịch nằm ngoài ngưỡng outlier
    theo loại giao dịch.
-   `hour_of_day`: trích xuất giờ trong ngày từ `step`.
-   `dest_prior_tx_count`: số giao dịch trước đó của tài khoản đích.
-   Các one-hot features cho loại giao dịch.
-   Các balance-derived features như `errorBalanceOrig`,
    `errorBalanceDest`, `orig_drained`, `dest_balance_both_zero` và
    `consistent`.

Các thống kê dùng để tạo feature như median theo loại giao dịch và
outlier fence được xác định từ Train, sau đó áp dụng cho các tập dữ liệu
còn lại để tránh sử dụng thông tin Validation/Test khi fit các thống kê
này.

### Ba feature sets được xây dựng

  -----------------------------------------------------------------------
  Feature set             Thành phần              Mục đích
  ----------------------- ----------------------- -----------------------
  `A_full`                BASE + SYN + ORIG_BAL + Bộ đầy đủ, dùng để kiểm
                          DEST_BAL                tra ảnh hưởng của
                                                  sender-balance features

  `B_no_orig_balance`     BASE + SYN + DEST_BAL   Loại sender-balance
                                                  features; là bộ feature
                                                  chính trong thí nghiệm
                                                  hiện tại

  `C_no_balance`          BASE + SYN              Loại toàn bộ balance
                                                  features
  -----------------------------------------------------------------------

Trong đó:

-   `BASE`: nhóm feature giao dịch, thời gian, loại giao dịch và
    destination velocity.
-   `SYN`: các synthetic ordinal features và `n_risk_flags`.
-   `ORIG_BAL`: nhóm feature dựa trên số dư tài khoản nguồn.
-   `DEST_BAL`: nhóm feature dựa trên số dư tài khoản đích.

**Feature set được chọn để export:** `B_no_orig_balance`, gồm **18
features**.

Việc loại sender-balance features giúp hạn chế mô hình dựa quá nhiều vào
một nhóm thông tin có thể tạo shortcut. Tuy nhiên, B vẫn chứa synthetic
features phụ thuộc target nên không được xem là bộ feature sạch leakage.

------------------------------------------------------------------------

## 6. Temporal split và xử lý mất cân bằng

Dữ liệu được chia theo `step`, không chia ngẫu nhiên:

  Tập                       Điều kiện     Số dòng   Fraud   Fraud rate
  ------------ ---------------------- ----------- ------- ------------
  Train                 `step <= 504`   6,064,036   5,621      0.0927%
  Validation     `505 <= step <= 624`     208,896   1,264      0.6051%
  Test                  `step >= 625`      89,688   1,328      1.4807%

Tỷ lệ Fraud tăng rõ rệt từ Train sang Validation và Test. Đây là một
thay đổi phân phối nhãn theo thời gian mà nhóm Model Training cần lưu ý
khi đánh giá mô hình.

### Undersampling

Chỉ áp dụng Random Undersampling trên Train:

-   Giữ lại toàn bộ 5,621 giao dịch Fraud.
-   Lấy ngẫu nhiên 10 lần số giao dịch Non-Fraud.
-   Seed được sử dụng cho file bàn giao: `1`.
-   Kết quả: **61,831 dòng**, gồm 5,621 Fraud và 56,210 Non-Fraud.
-   Fraud rate sau undersampling: **9.0909%**.

Validation và Test giữ nguyên phân phối gốc, không undersampling và
không SMOTE.

SMOTENC đã được thử nghiệm trong quá trình FE nhưng không được chọn làm
dữ liệu bàn giao chính. Scaling cũng không được thực hiện ở bước export
cuối cùng.

------------------------------------------------------------------------

## 7. Kết quả kiểm tra và đánh giá FE

### Feature ablation

Ablation được sử dụng để quan sát mức độ đóng góp của các nhóm feature.
Kết quả LightGBM trên Validation:

  Feature set           PR-AUC trung bình   Độ lệch chuẩn
  ------------------- ------------------- ---------------
  A_full                         0.999586        0.000102
  B_no_orig_balance              0.813514        0.003857
  C_no_balance                   0.588255        0.031050

Diễn giải:

-   A_full đạt điểm gần như tuyệt đối, nhưng có thể dựa vào shortcut từ
    sender-balance features; không nên mặc định đây là mô hình tốt nhất.
-   B_no_orig_balance là cấu hình thí nghiệm chính hiện tại.
-   So sánh B và C cho thấy nhóm destination-balance features có ảnh
    hưởng đáng kể trong thí nghiệm.
-   Kết quả trên vẫn chịu giới hạn từ synthetic features phụ thuộc
    target.

### Feature drift

Drift được kiểm tra bằng standardized mean difference giữa Train và
Validation. Các feature được kiểm tra đều có mức drift trung bình thuộc
ngưỡng Low theo heuristic đang sử dụng.

Tuy nhiên, kết quả này chỉ cho biết mức thay đổi về trung bình theo cách
đo đã chọn. Nó **không chứng minh toàn bộ phân phối feature không thay
đổi**, cũng không chứng minh dữ liệu không có leakage.

------------------------------------------------------------------------

## 8. Cấu trúc file bàn giao

Thư mục bàn giao:

``` text
fe_out/
└── model_training/
    ├── train.csv
    ├── valid.csv
    ├── test.csv
    └── metadata.json
```

  -----------------------------------------------------------------------
  File                    Nội dung                Người làm Model
                                                  Training sử dụng
  ----------------------- ----------------------- -----------------------
  `train.csv`             18 features +           Huấn luyện mô hình
                          `isFraud`, đã           
                          undersampling 10:1      

  `valid.csv`             18 features +           So sánh mô hình, tuning
                          `isFraud`, phân phối    và chọn threshold
                          gốc                     

  `test.csv`              18 features +           Đánh giá cuối cùng sau
                          `isFraud`, phân phối    khi chốt mô hình
                          gốc                     

  `metadata.json`         Feature list, feature   Hiểu cách dữ liệu được
                          sets, encoding,         tạo và các giới hạn
                          sampling và limitations 
  -----------------------------------------------------------------------

Tất cả CSV đều chứa:

-   Các cột feature theo đúng thứ tự trong `metadata.json`.
-   Cột target `isFraud`.
-   Không chứa các cột phụ trợ như `step`, `type`, `amount` nếu chúng
    không thuộc feature set đã chọn.

**Không có scaler được export.** Nếu mô hình cần scaling, nhóm Model
Training phải fit scaler trên Train rồi dùng chính scaler đó transform
Validation và Test. Không fit scaler riêng trên từng tập.

------------------------------------------------------------------------

## 9. Người tiếp theo cần làm gì? --- Model Training

Nhóm Model Training có thể tiếp tục theo trình tự sau:

### Bước 1 --- Load và kiểm tra dữ liệu

-   Đọc ba file CSV.
-   Đọc `metadata.json`.
-   Kiểm tra số dòng, số cột, target, missing values và feature
    consistency.
-   Xác nhận Validation/Test chưa bị resampling.

### Bước 2 --- Xây dựng baseline models

Có thể bắt đầu với:

-   Logistic Regression: baseline tuyến tính; cần scaling.
-   Random Forest: mô hình ensemble cây; không bắt buộc scaling.
-   XGBoost hoặc LightGBM: mô hình boosting cho dữ liệu dạng bảng.

Nên giữ cùng feature set và cùng cách chia dữ liệu khi so sánh các mô
hình để kết quả có ý nghĩa.

### Bước 3 --- Đánh giá trên Validation

Ưu tiên các metric phù hợp với dữ liệu mất cân bằng:

-   **PR-AUC:** metric quan trọng để đánh giá khả năng nhận diện lớp
    Fraud trong bối cảnh Fraud hiếm.
-   **ROC-AUC:** đánh giá khả năng phân biệt hai lớp ở nhiều threshold.
-   **Precision:** trong các giao dịch bị dự đoán Fraud, tỷ lệ dự đoán
    đúng.
-   **Recall:** tỷ lệ giao dịch Fraud thực tế được phát hiện.
-   **F1-score:** cân bằng Precision và Recall tại một threshold cụ thể.

Không nên chỉ dùng Accuracy để chọn mô hình.

### Bước 4 --- Tuning và chọn threshold

-   Tuning hyperparameters dựa trên Validation.
-   Chọn threshold theo mục tiêu nghiệp vụ hoặc trade-off
    Precision/Recall.
-   Lưu lại cấu hình, metric và threshold của mô hình được chọn.
-   Lưu ý: Train đã undersampling trong khi Validation giữ phân phối
    gốc, nên xác suất dự đoán có thể chưa được calibration tốt.

### Bước 5 --- Đánh giá cuối cùng trên Test

Sau khi đã chốt mô hình, hyperparameters và threshold:

-   Chạy đánh giá trên `test.csv`.
-   Không dùng Test để tiếp tục tuning hoặc chọn lại mô hình.
-   Báo cáo đầy đủ PR-AUC, ROC-AUC, Precision, Recall, F1 và confusion
    matrix.
-   Đặt kết quả trong bối cảnh tỷ lệ Fraud của Test cao hơn đáng kể so
    với Train.

### Bước 6 --- Kiểm tra độ tin cậy của feature set

Đây là việc quan trọng trước khi đưa ra kết luận:

-   Thực hiện thêm thí nghiệm không sử dụng synthetic features để đánh
    giá mức độ phụ thuộc vào nhóm feature target-dependent.
-   Kiểm tra tính khả dụng của từng feature tại thời điểm dự đoán.
-   Đặc biệt, `dest_prior_tx_count` phải được tính từ các giao dịch xảy
    ra trước thời điểm dự đoán; các giao dịch cùng `step` được xem là
    đồng thời.

------------------------------------------------------------------------

## 10. Kết luận bàn giao

Giai đoạn Feature Engineering đã hoàn thành các công việc: kiểm tra dữ
liệu, tạo và mã hóa feature, xây dựng feature sets, temporal split, phân
tích ablation và drift, xử lý mất cân bằng trên Train, kiểm tra dữ liệu
và export.

**Đầu ra chính:** ba file `train.csv`, `valid.csv`, `test.csv` cùng
`metadata.json` trong thư mục `model_training/`.

Giai đoạn tiếp theo là **Model Training**: xây dựng baseline models, so
sánh bằng metric phù hợp với class imbalance, tuning trên Validation,
chốt threshold và chỉ đánh giá Test ở bước cuối.

> **Cảnh báo xuyên suốt:** các synthetic features hiện tại được sinh có
> sử dụng `isFraud`. Vì vậy, kết quả mô hình sử dụng feature set B chỉ
> nên được xem là kết quả thí nghiệm trên dữ liệu synthetic, không phải
> bằng chứng về hiệu năng triển khai thực tế.
