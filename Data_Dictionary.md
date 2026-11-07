# Complete Data Dictionary — E-Commerce Payment Fraud Detection

**Course**: Business Analytics — Group Project  
**Role**: Data Engineer (Module 1: Synthetic Data Generation & Data Dictionary, Module 3: Data Cleaning)  
**Dataset Scope**: Full PaySim Base Log + Synthetic Contextual Telemetry (`6,362,620` rows across all 5 transaction types: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`)

---

## 1. Base Dataset Fields (Kaggle PaySim Log)

| Column Name | Data Type | Unit / Format | Valid Range / Values | Description & Cleaning / Modeling Role |
|---|---|---|---|---|
| `step` | `int32` | Hours | `[1, 743]` | Maps a unit of time in the real world where `1 step = 1 hour` over a 31-day simulation (`743` total steps). Validated for monotonic order; used for time-ordered splitting and velocity features. |
| `type` | `category` | Categorical string | `{'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'}` | Transaction channel/type. Standardized to uppercase trimmed strings. **All 5 types (`6,362,620` rows) are retained** in the cleaned dataset to preserve complete customer behavioral history and full transaction stream coverage. |
| `amount` | `float64` | Currency unit | `[0.00, 92,445,516.64]` | Transaction value in local currency. `16` zero-amount rows (`amount == 0`, all `isFraud = 1`) and `3xIQR` high-value outliers are retained and explicitly flagged (`is_zero_amount`, `is_amount_outlier_3xIQR`). |
| `nameOrig` | `string` | Account ID (`C...`) | Prefix `C` + digits (`6,353,307` unique) | Originating customer account identifier. Trimmed and verified to start with prefix `C`. Used in downstream feature engineering to track account-level velocity. |
| `oldbalanceOrg` | `float64` | Currency unit | `[0.00, 59,585,040.37]` | Initial balance before the transaction. **Excluded in Module 3 Data Cleaning** due to post-decision cancellation leakage (`[Q1.4]`: fraudulent transactions are cancelled before balances are recorded). |
| `newbalanceOrig` | `float64` | Currency unit | `[0.00, 49,585,040.37]` | Origin account balance after the transaction. **Excluded in Module 3 Data Cleaning** (post-decision state / target leakage). |
| `nameDest` | `string` | Account ID (`C...` / `M...`) | Prefix `C` (customer) or `M` (merchant) | Destination recipient account identifier (`2,722,362` unique). `M` prefix occurs strictly in `PAYMENT` transactions (`2,151,495` rows). |
| `oldbalanceDest` | `float64` | Currency unit | `[0.00, 356,015,889.35]` | Initial destination balance before the transaction (structurally `0` for all `M` merchant accounts). **Excluded in Module 3 Data Cleaning** due to post-decision cancellation leakage and structural missingness. |
| `newbalanceDest` | `float64` | Currency unit | `[0.00, 356,179,278.92]` | Destination balance after the transaction (structurally `0` for all `M` merchant accounts). **Excluded in Module 3 Data Cleaning** (post-decision state / target leakage). |
| `isFraud` | `int8` | Binary label | `{0, 1}` | Ground-truth target label (`1` = fraudulent transaction initiated by an attacker taking over an account via `TRANSFER` and cashing out via `CASH_OUT`; `0` = legitimate). Total fraud cases = `8,213` (`0.1291%`). |
| `isFlaggedFraud` | `int8` | Binary flag | `{0, 1}` | Legacy simulator rule flag (`1` for `16` `TRANSFER` transactions `> 200,000`). Evaluated as a baseline policy in Module 1, then **excluded in Module 3 Data Cleaning** so models learn without rule leakage. |

---

## 2. Synthetically Generated Contextual & Behavioral Fields (Module 1)

Every synthetically generated field below is produced in Python (`seed = 42`) to augment initiation-time transaction records with realistic e-commerce payment telemetry.

| Column Name | Data Type | Unit / Format | Valid Range | Business Assumption & Generation Logic |
|---|---|---|---|---|
| `account_age_days` | `int32` (`Integer`) | Days | `[1, 3650]` | **Assumption**: Newly created or recently registered throwaway accounts carry higher fraud risk, whereas long-tenured accounts are predominantly legitimate.<br>**Generation Logic**:<br>• `isFraud = 0`: Drawn from a `Gamma(shape = 3.0, scale = 150.0)` distribution skewed toward mature accounts (mean `~450` days), rounded and clipped at `[1, 3650]`.<br>• `isFraud = 1`: Mixture distribution where `70%` are young accounts (`Exponential(scale = 25.0)` days) and `30%` are mature accounts (`Gamma(shape = 3.0, scale = 150.0)` representing Account Takeover of established users), clipped at `[1, 3650]`. |
| `device_trust_score` | `float32` (`Float`) | Continuous Score | `[0.0, 1.0]` | **Assumption**: Device/browser fingerprinting measures whether the transaction originates from a recognized, trusted hardware/browser history.<br>**Generation Logic**:<br>• `isFraud = 0`: Sampled from `Beta(alpha = 8.0, beta = 2.0)`, concentrated near `0.70–1.00` (mean `~0.80`) with a small tail for legitimate users on new devices.<br>• `isFraud = 1`: Sampled from `Beta(alpha = 2.0, beta = 6.0)`, concentrated near `0.05–0.45` (mean `~0.25`), reflecting unrecognized devices, headless browsers, or emulators used by attackers. |
| `is_new_device` | `int8` (`Integer Binary`) | Binary Flag | `{0, 1}` | **Assumption**: Account takeover typically occurs from a device never previously associated with the user's account.<br>**Generation Logic**:<br>• `isFraud = 0`: Bernoulli trial with `P(is_new_device = 1) = 0.08` (normal device upgrades or secondary devices).<br>• `isFraud = 1`: Bernoulli trial with `P(is_new_device = 1) = 0.72`. |
| `ip_billing_distance_km` | `float32` (`Float`) | Kilometers (`km`) | `[0.0, 15000.0]` | **Assumption**: A large physical distance between the IP geolocation at transaction initiation and the registered billing address indicates potential VPN/proxy usage or foreign account takeover.<br>**Generation Logic**:<br>• `isFraud = 0`: `LogNormal(mu = 2.5, sigma = 1.2)` centered around local proximity (`~12.8 km` median), mixed with `3%` `Uniform[500, 5000]` km outliers representing legitimate travelers, clipped at `[0.0, 15000.0]`.<br>• `isFraud = 1`: `LogNormal(mu = 6.2, sigma = 1.5)` shifted toward long-distance/cross-border anomalies (`~511 km` median, `~1,415 km` mean), clipped at `[0.0, 15000.0]`. |
| `shipping_billing_mismatch` | `int8` (`Integer Binary`) | Binary Flag | `{0, 1}` | **Assumption**: In e-commerce payments (`PAYMENT`) or account-takeover fund transfers (`TRANSFER` / `CASH_OUT`), fraudsters frequently route goods or cash-out funds to a destination/beneficiary registered region that mismatches the originating account's primary billing address.<br>**Generation Logic**:<br>• `isFraud = 0`: Bernoulli trial with `P(shipping_billing_mismatch = 1) = 0.05` (e.g., legitimate customers sending gifts or paying remote merchants).<br>• `isFraud = 1`: Bernoulli trial with `P(shipping_billing_mismatch = 1) = 0.60`. |
| `failed_auth_attempts_24h` | `int8` (`Integer`) | Count | `[0, 15]` | **Assumption**: Brute-force login attempts, card testing, or repeated PIN/OTP failures in the 24 hours prior to a transaction strongly signal unauthorized access.<br>**Generation Logic**:<br>• `isFraud = 0`: `Poisson(lambda = 0.15)` (`~86%` have `0` failed attempts), clipped at `[0, 15]`.<br>• `isFraud = 1`: `Poisson(lambda = 2.80)`, clipped at `[0, 15]`, reflecting credential stuffing or repeated trial-and-error attempts. |

---

## 3. Data Cleaning Quality & Outlier Treatment Flags (Added in Module 3)

| Column Name | Data Type | Unit / Format | Valid Range | Cleaning Rationale & Definition |
|---|---|---|---|---|
| `is_amount_outlier_3xIQR` | `int8` (`Integer Binary`) | Binary Flag | `{0, 1}` | `1` if `amount > Q3 + 3.0 * IQR` computed within each of the 5 transaction types (`CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`), `0` otherwise. Retained rather than dropped because extreme amounts carry disproportionately high fraud rates in `CASH_OUT` (`43.56%`) and `TRANSFER` (`2.84%`). |
| `is_zero_amount` | `int8` (`Integer Binary`) | Binary Flag | `{0, 1}` | `1` if `amount == 0.0`, `0` otherwise. Captures the `16` simulator artefact transactions (`100%` of which are `isFraud = 1`) without deleting fraudulent records. |

