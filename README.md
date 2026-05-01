# Datathon 2026 — The Gridbreakers (Preliminary Round)

> **VinTelligence — VinUni DS&AI Club**

## 📁 Cấu trúc thư mục

```
datathon-2026-round-1/
│
├── 📊 Data
│   ├── sales.csv                  # Doanh thu & COGS hàng ngày (2012–2022)
│   ├── orders.csv                 # Chi tiết đơn hàng (646K rows)
│   ├── order_items.csv            # Sản phẩm trong đơn (714K rows)
│   ├── payments.csv               # Thanh toán (646K rows)
│   ├── products.csv               # Danh mục sản phẩm (2,412 SKU)
│   ├── customers.csv              # Thông tin khách hàng (121K)
│   ├── geography.csv              # Địa lý theo mã ZIP
│   ├── promotions.csv             # Chương trình khuyến mãi (50 chiến dịch)
│   ├── inventory.csv              # Tồn kho hàng tháng (60K rows)
│   ├── web_traffic.csv            # Lượng truy cập website (3,652 rows)
│   ├── returns.csv                # Đổi trả hàng (39K rows)
│   ├── reviews.csv                # Đánh giá sản phẩm (113K rows)
│   ├── shipments.csv              # Vận chuyển (646K rows)
│   └── sample_submission.csv      # Template nộp bài (548 rows)
│
├── 📈 Part 2 — Data Visualization & Analysis
│   ├── analysis_part2.py                  # Code chính (jupytext format)
│   └── analysis_part2_storytelling.ipynb   # Notebook với output
│
├── 🤖 Part 3 — Revenue Forecasting
│   ├── analysis_part3_advanced.py         # Code chính (jupytext format)
│   ├── analysis_part3_advanced.ipynb      # Notebook với output
│   └── report_part3_forecasting.md        # Báo cáo kỹ thuật
│
├── 📤 Output
│   ├── submission.csv                     # File nộp Kaggle
│   └── analysis_outputs/                  # Biểu đồ xuất ra
│       ├── vis1–vis8_*.png                # 8 biểu đồ Part 2
│       ├── vis_forecast_advanced_comparison.png
│       └── vis_shap_summary.png
│
└── README.md
```

## 🚀 Hướng dẫn chạy lại kết quả

### Yêu cầu hệ thống

- Python 3.9+
- Conda (khuyến nghị) hoặc pip

### Cài đặt dependencies

```bash
pip install pandas numpy matplotlib scikit-learn scipy \
            xgboost lightgbm catboost shap jupytext
```

### Chạy Part 2 — Visualization

```bash
# Chạy trực tiếp
python analysis_part2.py

# Hoặc mở notebook
jupytext --to notebook analysis_part2.py --output analysis_part2_storytelling.ipynb
jupyter notebook analysis_part2_storytelling.ipynb
```

**Output**: 8 biểu đồ trong `analysis_outputs/` (vis1–vis8).

### Chạy Part 3 — Forecasting

```bash
# Chạy trực tiếp (tạo submission.csv)
python analysis_part3_advanced.py

# Hoặc mở notebook
jupytext --to notebook analysis_part3_advanced.py --output analysis_part3_advanced.ipynb
jupyter notebook analysis_part3_advanced.ipynb
```

**Output**:
- `submission.csv` — File nộp Kaggle (548 rows: Date, Revenue, COGS)
- `analysis_outputs/vis_forecast_advanced_comparison.png` — Biểu đồ forecast
- `analysis_outputs/vis_shap_summary.png` — SHAP feature importance

### Thời gian chạy dự kiến

| Bước | Thời gian |
|------|----------|
| Load data + Feature engineering | ~5s |
| Recursive Cross-Validation (5 folds × 180 days) | ~30s |
| Train final models (5 seeds × 2 types) | ~10s |
| Recursive Forecast (548 days) | ~10s |
| SHAP analysis | ~5s |
| **Tổng** | **~60s** |

## 🏗️ Pipeline Part 3 — Tóm tắt

```
sales.csv + web_traffic + inventory + promotions
          ↓
   76 Features (lag, rolling, Fourier, calendar, external)
          ↓
   Recursive CV (5-fold, 180-day, no leakage)
          ↓
   Ensemble: CatBoost + LightGBM + Ridge (60/40 blend)
   × 5 random seeds
          ↓
   Recursive Forecast 548 days → submission.csv
```

### Kết quả

| Metric | Giá trị |
|--------|---------|
| Recursive CV MAE | 760,422 |
| Recursive CV R² | 0.4987 |
| Kaggle MAE | 802,029 |

## 📝 Ghi chú

- File `.py` dùng [jupytext](https://jupytext.readthedocs.io/) format (`py:percent`) — có thể chạy trực tiếp bằng Python hoặc chuyển sang `.ipynb`.
- Tất cả output được lưu trong `analysis_outputs/` (tự tạo nếu chưa có).
- Random seed cố định (`SEED=42`) đảm bảo kết quả tái lập.
