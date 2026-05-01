# Part 3 — Báo cáo Kỹ thuật: Mô hình Dự báo Doanh thu

## 1. Tổng quan Pipeline

### 1.1 Kiến trúc hệ thống

Pipeline dự báo doanh thu được xây dựng theo kiến trúc **Recursive Multi-Step Forecasting** với ensemble đa mô hình:

```
┌─────────────────────────────────────────────────────┐
│  Data Sources                                       │
│  sales.csv │ web_traffic.csv │ inventory.csv │ ...  │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Feature Engineering (76 features)                  │
│  Lag │ Rolling │ Fourier │ Calendar │ External      │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Recursive Cross-Validation (5-fold, 180-day)       │
│  → Simulates actual test conditions                 │
│  → No actual Revenue in validation lags             │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Ensemble: CatBoost + LightGBM + Ridge              │
│  5 seeds × 2 tree types + 1 Ridge                   │
│  Blend: 60% tree avg + 40% Ridge                    │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  Recursive Forecast (548 days)                      │
│  Day t → predict → feed into Day t+1 lags           │
│  COGS = Revenue × avg_cogs_ratio (0.8746)           │
└─────────────────────────────────────────────────────┘
```

### 1.2 Dữ liệu sử dụng

| Nguồn | Kích thước | Mục đích |
|-------|-----------|----------|
| `sales.csv` | 3,833 rows (2012-07-04 → 2022-12-31) | Target (Revenue, COGS) + lag features |
| `web_traffic.csv` | 3,652 rows | Sessions, unique visitors → traffic signals |
| `inventory.csv` | 60,247 rows | Fill rate, stockout flags → supply signals |
| `promotions.csv` | 50 rows | Active promo periods, discount values |
| `sample_submission.csv` | 548 rows (2023-01-01 → 2024-07-01) | Test period definition |

---

## 2. Feature Engineering (76 features)

Bộ feature được thiết kế theo **10 nhóm chức năng**, mỗi nhóm phục vụ một khía cạnh khác nhau của bài toán dự báo:

### 2.1 Chi tiết các nhóm feature

| Nhóm | Số lượng | Mô tả | Ví dụ |
|------|---------|-------|-------|
| **A. Lag features** | 10 | Giá trị Revenue dịch chuyển theo thời gian | `rev_lag_1`, `rev_lag_7`, `rev_lag_365`, `rev_yoy_ratio` |
| **B. Rolling statistics** | 8 | Trung bình trượt, độ lệch chuẩn, momentum | `rev_roll_mean_7`, `rev_roll_std_30`, `rev_momentum` |
| **C. Fourier seasonality** | 12 | Mã hóa chu kỳ năm/tuần bằng sin/cos | `sin_year_1`, `cos_week`, `is_peak_summer` |
| **D. Trend** | 5 | Xu hướng dài hạn, structural breaks | `time_idx`, `post_2018`, `is_covid` |
| **E. Calendar** | 11 | Đặc trưng lịch | `month`, `dayofweek`, `days_to_month_end` |
| **F. Promotions** | 7 | Khuyến mãi đang hoạt động, hiệu ứng trễ | `has_promo`, `n_active_promos`, `promo_lag1_effect` |
| **G. Web traffic** | 6 | Lượng truy cập website | `sessions_lag_1`, `sessions_roll7`, `rev_per_session` |
| **H. Inventory** | 4 | Tình trạng kho hàng | `fill_rate_mean`, `stockout_risk` |
| **I. Interactions** | 8 | Tương tác giữa các nhóm | `peak_x_promo`, `covid_x_peak`, `post18_x_summer` |
| **J. YoY reference** | 5 | So sánh cùng kỳ năm trước | `rev_yoy_delta`, `rev_2yr_ratio`, `rev_yoy_roll30` |

### 2.2 Nguyên tắc thiết kế feature

1. **Tất cả lag/rolling features đều dùng `shift(≥1)`**: Đảm bảo không sử dụng thông tin tương lai. Ví dụ: `rev_lag_1 = Revenue.shift(1)` chỉ dùng doanh thu ngày hôm trước.

2. **External signals được merge theo Date**: Web traffic, inventory, promotions đều join theo ngày, đảm bảo temporal alignment.

3. **Warm-up period = 365 ngày**: 365 dòng đầu tiên bị loại bỏ (`df_model = df_train_full.iloc[365:]`) để đảm bảo tất cả lag features (đặc biệt `rev_lag_365`) đều có giá trị hợp lệ.

4. **Fourier encoding cho seasonality**: Thay vì one-hot encoding tháng (12 biến), sử dụng sin/cos harmonics (6 biến cho 3 tần số) giúp mô hình nhận diện chu kỳ mượt hơn và tổng quát hóa tốt hơn.

---

## 3. Kiểm soát Leakage

### 3.1 Các loại leakage đã xử lý

| Loại leakage | Biện pháp |
|--------------|-----------|
| **Feature leakage** | Tất cả features dùng `shift(≥1)` — không dùng thông tin ngày hiện tại |
| **CV validation leakage** | Sử dụng **Recursive CV** — validation không dùng actual Revenue cho lag |
| **Temporal leakage** | `TimeSeriesSplit` đảm bảo train luôn trước test theo thời gian |
| **Target leakage** | COGS không dùng làm feature; chỉ tính từ Revenue × fixed ratio khi export |

### 3.2 Recursive Cross-Validation — Chi tiết

Đây là điểm quan trọng nhất trong kiểm soát leakage. Thay vì Direct CV (sử dụng actual Revenue trong validation để tính lag features), chúng tôi implement **Recursive CV** mô phỏng chính xác điều kiện test thật:

```python
# Bước 1: Blank out Revenue trong validation period
val_timeline.loc[val_idx, 'Revenue'] = np.nan

# Bước 2: Với mỗi ngày validation, build features từ predicted history
for vidx in val_idx:
    features = build_features(val_timeline_with_predictions)
    pred = ensemble.predict(features)
    
    # Prediction ngày t → trở thành lag input cho ngày t+1
    val_timeline.loc[vidx, 'Revenue'] = pred
```

**Tại sao quan trọng?** Direct CV cho MAE ~500K nhưng Kaggle score ~860K — gap 72%. Recursive CV cho MAE ~760K, gần Kaggle hơn nhiều (gap ~5%). Điều này chứng minh Direct CV tạo **ảo tưởng hiệu suất** do lag features trong validation dùng actual Revenue thay vì predicted values.

### 3.3 So sánh Direct CV vs Recursive CV

| Phương pháp | Ensemble MAE | Ensemble R² | Gap với Kaggle |
|-------------|-------------|------------|----------------|
| Direct CV | ~500,000 | ~0.75 | ~72% |
| **Recursive CV** | **760,422** | **0.4987** | **~5%** |
| Kaggle test | 802,029 | — | — |

---

## 4. Mô hình & Ensemble

### 4.1 Kiến trúc Ensemble

Ensemble gồm **3 thành phần** với vai trò bổ sung:

| Thành phần | Số models | Vai trò |
|-----------|----------|--------|
| **CatBoost** (depth=4, l2_reg=10) | 5 seeds | Gradient boosting — xử lý tốt categorical, robust với outliers |
| **LightGBM** (num_leaves=31, max_depth=5) | 5 seeds | Histogram-based boosting — nhanh, tốt với high-dim features |
| **Ridge** (alpha=200) | 1 | Linear baseline — ổn định trong recursive, giảm variance |

**Tổng: 10 tree models + 1 Ridge = 11 models**

### 4.2 Chiến lược giảm Overfitting

Model capacity được **giảm có chủ đích** để cải thiện recursive forecast:

| Tham số | Giá trị ban đầu | Giá trị tối ưu | Lý do |
|---------|-----------------|----------------|-------|
| CatBoost depth | 6 | **4** | Giảm complexity, tránh memorize noise |
| CatBoost l2_leaf_reg | 3.0 | **10.0** | Tăng regularization |
| LightGBM num_leaves | 127 | **31** | Giảm 4× capacity |
| LightGBM max_depth | 7 | **5** | Shallower trees |
| LightGBM min_child_samples | 50 | **100** | Leaf cần nhiều data hơn |
| LightGBM reg_lambda | 1.0 | **5.0** | L2 regularization mạnh hơn |
| Subsample | 0.8 | **0.7** | Nhiều randomness hơn |

### 4.3 Multi-Seed Averaging

Mỗi model type được train với **5 random seeds** (42, 123, 456, 789, 2024) rồi average predictions:
- Giảm variance do random initialization
- Tạo implicit ensemble trong mỗi model type

### 4.4 Ensemble Blending

```
Final_prediction = 0.60 × mean(tree_predictions) + 0.40 × ridge_prediction
```

**Lý do chọn 60/40**: Ridge ổn định hơn tree models trong recursive forecast vì:
- Ridge không overfit lag features mạnh như tree models
- Ridge tạo predictions mượt hơn, giảm error accumulation
- Tỷ lệ 60/40 đạt Kaggle score tốt nhất (802K) trong thực nghiệm

### 4.5 Early Stopping & Optimal Iterations

- CV sử dụng **early stopping (100 rounds patience)** trên 80/20 split trong train fold
- `optimal_iters` = mean(best_iterations) × 1.1 = **1,022 iterations**
- Final training dùng fixed `optimal_iters` trên 100% data (không early stopping, vì không có holdout)

---

## 5. Recursive Forecast (Test Period)

### 5.1 Quy trình

```
Cho 548 ngày test (2023-01-01 → 2024-07-01):
  Với mỗi ngày t:
    1. Build features từ actual train + predicted test so far
    2. Predict Revenue(t) = ensemble(features)
    3. COGS(t) = Revenue(t) × 0.8746 (mean ratio từ training)
    4. Ghi Revenue(t) vào timeline → trở thành lag cho ngày t+1
```

### 5.2 COGS Estimation

COGS được tính bằng **fixed ratio** từ training data:
```
avg_cogs_ratio = mean(COGS / Revenue) = 0.8746
```

Lý do: Thực nghiệm cho thấy train COGS model riêng không cải thiện score vì features được thiết kế cho Revenue prediction (lag Revenue, rev_per_session, etc.) — không có COGS-specific lag features.

---

## 6. Giải thích Mô hình (SHAP Analysis)

### 6.1 Top 10 Revenue Drivers

SHAP analysis được thực hiện trên LightGBM (model có R² cao nhất trong CV) với 1,000 samples:

| Rank | Feature | Mean |SHAP| | Giải thích |
|------|---------|-------------|------------|
| 1 | `rev_lag_1` | 703,705 | Doanh thu hôm trước — predictor mạnh nhất |
| 2 | `rev_lag_365` | 416,607 | Cùng kỳ năm trước — seasonal anchor |
| 3 | `rev_lag_7` | 254,731 | Doanh thu cùng ngày tuần trước |
| 4 | `rev_lag_364` | 218,421 | YoY reference (±1 ngày) |
| 5 | `rev_lag_14` | 120,955 | 2-week pattern |
| 6 | `days_to_month_end` | 98,884 | End-of-month effect (revenue tăng cuối tháng) |
| 7 | `rev_roll_mean_7` | 92,733 | Trend ngắn hạn 7 ngày |
| 8 | `rev_2yr_lag` | 92,074 | Reference 2 năm trước |
| 9 | `rev_lag_90` | 91,545 | Quarterly pattern |
| 10 | `rev_roll_std_7` | 89,436 | Volatility ngắn hạn |

### 6.2 Business Insights từ SHAP

1. **Seasonal pattern là driver chính**: `rev_lag_365` + `rev_lag_364` + `rev_lag_366` chiếm ~670K SHAP tổng — mô hình dựa mạnh vào cùng kỳ năm trước.

2. **Short-term momentum quan trọng**: `rev_lag_1` + `rev_lag_7` + `rev_roll_mean_7` chiếm ~1.05M SHAP — gần 60% tổng explanatory power đến từ trend ngắn hạn.

3. **Calendar effects có ý nghĩa**: `days_to_month_end` nằm top 6, phản ánh hành vi mua hàng tăng cuối tháng (có thể liên quan đến ngày nhận lương).

4. **Volatility là signal**: `rev_roll_std_7` nằm top 10 — khi volatility cao, model điều chỉnh prediction, giúp xử lý periods bất thường.

### 6.3 Visualization

Pipeline tự động tạo:
- `vis_shap_summary.png`: SHAP beeswarm plot top 15 features
- `vis_forecast_advanced_comparison.png`: 4-panel diagnostic (timeline, zoom, R² comparison, feature importance)

---

## 7. Kết quả & Hiệu suất

### 7.1 Cross-Validation (Recursive, 5-fold, 180-day test)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| CatBoost | 754,888 | 1,016,840 | 0.4978 |
| LightGBM | 768,600 | 1,024,810 | 0.4980 |
| Ridge | 825,096 | 1,087,234 | 0.4205 |
| **Ensemble** | **760,422** | **1,016,581** | **0.4987** |

### 7.2 Kaggle Leaderboard

| Metric | Giá trị |
|--------|---------|
| **MAE (Kaggle)** | **802,029** |
| Revenue mean (forecast) | 4,205,039 |
| COGS mean (forecast) | 3,677,757 |
| COGS/Revenue ratio | 0.8746 |

### 7.3 Fold-level Analysis

| Fold | Recursive MAE | Ghi chú |
|------|--------------|---------|
| 1 | 441,437 | Ổn định |
| 2 | 1,078,775 | Cao — có thể chứa structural change |
| 3 | 521,290 | Ổn định |
| 4 | 1,231,311 | Cao — tương tự fold 2 |
| 5 | 529,299 | Ổn định |

Fold 2 và 4 có MAE cao hơn đáng kể, có thể do chứa giai đoạn COVID-19 (2020-2021) hoặc structural breaks trong revenue pattern.

---

## 8. Ràng buộc & Tuân thủ

| Ràng buộc | Tuân thủ | Chi tiết |
|-----------|---------|----------|
| Format submission | ✅ | 3 columns (Date, Revenue, COGS), 548 rows |
| Date range | ✅ | 2023-01-01 → 2024-07-01 |
| No future leakage | ✅ | Tất cả features dùng `shift(≥1)` |
| Time-series CV | ✅ | `TimeSeriesSplit` + Recursive validation |
| SHAP/Feature importance | ✅ | TreeExplainer + summary plot |
| Reproducibility | ✅ | Fixed `SEED=42`, 5 deterministic seeds |

---

## 9. Hạn chế & Hướng phát triển

### 9.1 Hạn chế hiện tại

1. **Recursive error accumulation**: `rev_lag_1` (SHAP=703K) quá dominant — lỗi ngày t tích lũy qua ngày t+1, t+2, ..., đặc biệt nghiêm trọng cho 548-day horizon.

2. **Dữ liệu order-level chưa được khai thác**: `orders.csv` (646K rows), `order_items.csv` (714K rows), `returns.csv`, `reviews.csv` chứa thông tin giá trị (n_orders, avg_order_value, return_rate) nhưng chưa tích hợp vào pipeline.

3. **COGS dùng fixed ratio**: Tỷ lệ COGS/Revenue thực tế dao động 0.79–0.92 theo năm, nhưng pipeline dùng global mean 0.8746.

### 9.2 Hướng phát triển

1. **Tích hợp order-level features**: `n_orders` có correlation 0.936 với Revenue — có thể cải thiện đáng kể nếu dùng `n_orders_lag_365` làm feature.

2. **Direct forecasting cho horizon dài**: Thay vì recursive 548 ngày, có thể train separate models cho different horizons (1-30 ngày, 31-180 ngày, 181-548 ngày).

3. **Adaptive COGS ratio**: Sử dụng rolling COGS/Revenue ratio thay vì global mean.
