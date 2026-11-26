# Dữ liệu synthetic cho bài toán phát hiện gian lận

## 1. Mục đích và giới hạn

Đề bài yêu cầu bổ sung dữ liệu ngữ cảnh (tuổi tài khoản, thiết bị, địa chỉ, số lần thanh toán thất bại, khoảng cách IP, giờ giao dịch). Bộ dữ liệu gốc không có các thông tin này nên ta sinh chúng bằng Python.

Giới hạn cần nói rõ:
- Feature synthetic **không thêm thông tin thật** vào dữ liệu. Tín hiệu của chúng là do giả định của generator.
- Vì `isFraud` là input của generator, các cột synthetic mang thông tin nhãn *theo thiết kế*. Vì vậy kết quả trên chúng dùng để kiểm tra pipeline và so sánh phương pháp, không dùng để khẳng định hiệu năng trong thực tế.
- Mục tiêu thiết kế: tín hiệu **yếu đến vừa**, hai phân phối (fraud và non-fraud) **chồng lấn nhiều**, mọi giả định được ghi lại và tái lập được.

## 2. Đầu vào, đầu ra của generator

| | Nội dung |
|---|---|
| Input | `nameOrig`, `nameDest`, `consistent` (0/1), `isFraud` (0/1) |
| Output | Với mỗi cột synthetic `c`: `c` (nhãn category) và `c_p` (xác suất của nhãn được chọn) |

`consistent` = 1 khi `|oldbalanceOrg − newbalanceOrig| == amount`.

## 3. Mô hình sinh

Với mỗi cột có K nhãn, xác suất của nhãn k là một multinomial logit:

```
logit_k = log(base_k) + fraud_scale · w_fraud_k · isFraud
                      + inc_scale   · w_inc_k   · (1 − consistent)
                      + σ · z_{entity,k}
P(nhãn = k) = softmax(logit)_k ,   z ~ N(0, 1)
```

Lý do chọn từng thành phần:
- **`base_k`**: phân phối nền theo lập luận nghiệp vụ (ví dụ đa số thiết bị là `known`). Đây là nơi giả định được ghi rõ.
- **`w_fraud_k`**: dịch log-odds khi giao dịch là gian lận. Giữ nhỏ (khoảng 0 đến 1.8) để người dùng thật vẫn có lúc dùng thiết bị lạ, và kẻ gian vẫn có lúc trông bình thường.
- **`w_inc_k`**: giao dịch không nhất quán số dư thì hơi nghiêng về hành vi bất thường. Hệ số `inc_scale` đặt nhỏ vì `consistent` vốn đã là hệ quả của luật "rút sạch tài khoản" trong dữ liệu gốc.
- **`z_{entity}`**: hiệu ứng ẩn sinh một lần cho mỗi `nameOrig` hoặc `nameDest`, để cùng một thực thể có xu hướng giống nhau giữa các giao dịch và xác suất không bị cứng theo nhãn.
- **softmax**: đảm bảo xác suất hợp lệ (dương, tổng bằng 1) với số nhãn tùy ý.

Nhãn cuối cùng được lấy mẫu từ `P`. Cột `hour_of_day = (step − 1) mod 24` lấy trực tiếp từ dữ liệu thật, không sinh.

## 4. Các cột synthetic

| Cột | Thực thể | Nhãn (từ ít đến nhiều rủi ro) | base | w_fraud | w_inc | Ý nghĩa |
|---|---|---|---|---|---|---|
| `account_age` | nameOrig | `<30d`, `30-180d`, `180-365d`, `>1y` | .05 .15 .20 .60 | 1.0 .4 0 −.2 | .3 .1 0 0 | tài khoản mới thì rủi ro cao hơn |
| `device_status` | nameOrig | `known`, `new`, `suspicious` | .85 .13 .02 | 0 .8 1.5 | 0 .4 .8 | thiết bị khác thiết bị quen dùng |
| `failed_attempts` | nameOrig | `0`, `1`, `2-3`, `4+` | .90 .07 .025 .005 | 0 .6 1.2 1.8 | 0 .2 .4 .6 | số lần thanh toán thất bại trước giao dịch |
| `addr_mismatch` | nameDest | `match`, `mismatch` | .93 .07 | 0 1.0 | 0 .3 | địa chỉ giao hàng khác địa chỉ thanh toán |
| `ip_billing_dist` | nameDest | `same_city`, `same_country`, `foreign_near`, `foreign_far` | .80 .15 .04 .01 | 0 .5 1.0 1.5 | 0 .2 .4 .6 | khoảng cách IP so với nước thanh toán |
| `hour_of_day` | (dữ liệu thật) | 0 đến 23 | | | | khung giờ giao dịch |

Nhãn được xếp theo thứ tự rủi ro tăng dần, nên khi modeling có thể mã hóa ordinal theo vị trí nhãn.

## 5. Tham số mặc định

| Tham số | Giá trị | Vai trò |
|---|---|---|
| `SEED` | 42 | tái lập kết quả |
| `SIGMA` | 0.5 | độ lớn hiệu ứng theo thực thể |
| `FRAUD_SCALE` | 1.0 | độ mạnh tín hiệu fraud (dùng 0.5 / 1.0 / 1.5 cho sensitivity) |
| `INC_SCALE` | 0.25 | độ ảnh hưởng của `consistent` |

## 6. Rủi ro leakage và cách xử lý

1. **Chốt tham số theo lập luận nghiệp vụ trước khi nhìn kết quả model.** Chỉnh tham số để tăng metric là leakage gián tiếp.
2. **Chia train/test theo thời gian (`step`)**, không xáo trộn. Mã hóa (ví dụ CatBoostEncoder cho `type`) chỉ fit trên train.
3. **Kiểm tra AUC từng cột synthetic.** Kỳ vọng khoảng 0.6 đến 0.75. Gần 1.0 nghĩa là tín hiệu quá lộ.
4. **Ablation**: so sánh model có và không có cột synthetic trên cùng bộ feature.
5. **Sensitivity**: đổi `FRAUD_SCALE` (yếu, vừa, mạnh) và kiểm tra kết luận so sánh các model có ổn định không.
6. **Shortcut của dữ liệu gốc**: luật "fraud rút sạch tài khoản" (`oldbalanceOrg == amount` và `newbalanceOrig == 0`) khiến baseline đạt điểm tuyệt đối. Vì vậy phần đánh giá synthetic nên dùng bộ feature đã bỏ thông tin số dư người gửi (bộ B) hoặc bỏ toàn bộ số dư (bộ C).
7. **Không đưa các cột `*_p` vào model.** Chúng chỉ để ghi lại và kiểm tra generator.

## 7. Cấu trúc file đầu ra

`data_synthetic.csv.gz` (hoặc `.parquet`) gồm:
- Các cột gốc: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.
- `consistent`, `hour_of_day`.
- Với mỗi cột synthetic: `<cột>` (nhãn) và `<cột>_p` (xác suất làm tròn 4 chữ số).

Dữ liệu được sắp xếp theo `step` (ổn định). `data_synthetic_meta.json` lưu seed, các hệ số và toàn bộ `SPEC` để tái lập.

## 8. Hạn chế

- `nameOrig` gần như không lặp trong bộ dữ liệu này, nên hiệu ứng `z` theo người gửi chủ yếu đóng vai trò nhiễu. Hiệu ứng theo `nameDest` mới tạo được sự nhất quán giữa các giao dịch.
- Các cột sinh độc lập với nhau (chỉ chung `isFraud` và `consistent`), nên chưa có tương quan chéo thực tế giữa chúng (ví dụ thiết bị lạ thường đi cùng IP xa).
- `failed_attempts` được sinh theo giao dịch, chưa mô phỏng cửa sổ thời gian trước đó của từng tài khoản.
- Các giá trị `base` và `w_*` là giả định chưa hiệu chỉnh với số liệu thật của ngành.
