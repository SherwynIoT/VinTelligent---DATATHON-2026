# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Part 3 — Advanced Revenue Forecasting (v2)
# **Strategy**: Rich lag/external features + Recursive forecasting + Diversified ensemble
#
# Combines datathon-model feature engineering with multi-model ensemble.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings, time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')
np.random.seed(42)
SEED = 42

plt.rcParams.update({
    'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cccccc', 'axes.labelcolor': '#333333',
    'text.color': '#333333', 'xtick.color': '#555555', 'ytick.color': '#555555',
    'grid.color': '#e0e0e0', 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'savefig.facecolor': '#ffffff',
})
COLORS = {
    'primary': '#2563eb', 'secondary': '#e63946', 'success': '#2d9f4e',
    'danger': '#d62828', 'warning': '#f59e0b', 'accent': '#7c3aed',
    'teal': '#0d9488', 'text': '#333333',
}
OUT_DIR = 'analysis_outputs/'

def fmt_millions(x, _): return f'{x/1e6:.1f}M' if abs(x) >= 1e6 else f'{x/1e3:.0f}K'

def save_fig(fig, name):
    fig.savefig(f'{OUT_DIR}{name}.png', bbox_inches='tight', pad_inches=0.3)
    print(f'  ✓ Saved {name}.png')

# %% [markdown]
# ## 1 — Load All Data

# %%
sales = pd.read_csv('sales.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
web_traffic = pd.read_csv('web_traffic.csv', parse_dates=['date'])
inventory = pd.read_csv('inventory.csv', parse_dates=['snapshot_date'])
promotions = pd.read_csv('promotions.csv', parse_dates=['start_date', 'end_date'])
sample_sub = pd.read_csv('sample_submission.csv', parse_dates=['Date'])

print(f'Sales: {len(sales)} rows, {sales.Date.min().date()} → {sales.Date.max().date()}')
print(f'Web traffic: {len(web_traffic)} rows | Inventory: {len(inventory)} rows')
print(f'Promotions: {len(promotions)} rows | Test: {len(sample_sub)} rows')

# %% [markdown]
# ## 2 — Prepare External Signals

# %%
# Daily web traffic
traffic_daily = web_traffic.groupby('date').agg(
    sessions=('sessions', 'sum'),
    unique_visitors=('unique_visitors', 'sum')
).reset_index().rename(columns={'date': 'Date'})

# Monthly inventory → forward-fill to daily
inv_monthly = inventory.groupby('snapshot_date').agg(
    fill_rate_mean=('fill_rate', 'mean'),
    stockout_flag_sum=('stockout_flag', 'sum')
).reset_index().rename(columns={'snapshot_date': 'Date'})
all_dates = pd.DataFrame({'Date': pd.date_range('2012-07-04', '2024-07-01')})
inv_daily = all_dates.merge(inv_monthly, on='Date', how='left').sort_values('Date')
inv_daily[['fill_rate_mean', 'stockout_flag_sum']] = inv_daily[['fill_rate_mean', 'stockout_flag_sum']].ffill()

# Promotions → daily active promo counts
promo_dates = pd.DataFrame({'Date': pd.date_range('2012-07-04', '2024-07-01')})
active_counts, total_disc = [], []
for d in promo_dates['Date']:
    mask = (promotions['start_date'] <= d) & (promotions['end_date'] >= d)
    ap = promotions[mask]
    active_counts.append(len(ap))
    total_disc.append(ap['discount_value'].sum() if len(ap) > 0 else 0)
promo_dates['n_active_promos'] = active_counts
promo_dates['has_promo'] = (promo_dates['n_active_promos'] > 0).astype(int)
promo_dates['total_discount_val'] = total_disc
print('✓ External signals ready')

# %% [markdown]
# ## 3 — Feature Engineering

# %%
def build_features(df_sales, traffic_daily, inv_daily, promo_dates):
    """Rich feature set: lag + rolling + calendar + Fourier + external signals."""
    df = df_sales.copy().sort_values('Date').reset_index(drop=True)
    t0 = pd.Timestamp('2012-07-04')

    # A. Lag features (10)
    for lag in [1, 7, 14, 28, 90, 180]:
        df[f'rev_lag_{lag}'] = df['Revenue'].shift(lag)
    df['rev_lag_364'] = df['Revenue'].shift(364)
    df['rev_lag_365'] = df['Revenue'].shift(365)
    df['rev_lag_366'] = df['Revenue'].shift(366)
    df['rev_yoy_ratio'] = df['rev_lag_1'] / df['rev_lag_365'].replace(0, np.nan)

    # B. Rolling stats (8)
    for w in [7, 30, 90]:
        df[f'rev_roll_mean_{w}'] = df['Revenue'].shift(1).rolling(w, min_periods=1).mean()
        df[f'rev_roll_std_{w}'] = df['Revenue'].shift(1).rolling(w, min_periods=1).std()
    df['rev_roll_cv_30'] = df['rev_roll_std_30'] / df['rev_roll_mean_30'].replace(0, np.nan)
    df['rev_momentum'] = df['rev_roll_mean_7'] / df['rev_roll_mean_90'].replace(0, np.nan)

    # C. Fourier seasonality (12)
    doy = df['Date'].dt.dayofyear; dow = df['Date'].dt.dayofweek
    for k in [1, 2, 3]:
        df[f'sin_year_{k}'] = np.sin(2 * np.pi * k * doy / 365.25)
        df[f'cos_year_{k}'] = np.cos(2 * np.pi * k * doy / 365.25)
    df['sin_week'] = np.sin(2 * np.pi * dow / 7)
    df['cos_week'] = np.cos(2 * np.pi * dow / 7)
    df['is_peak_summer'] = df['Date'].dt.month.isin([5, 6]).astype(int)
    df['is_year_end_sale'] = df['Date'].dt.month.isin([11, 12]).astype(int)
    df['is_trough'] = df['Date'].dt.month.isin([1, 2]).astype(int)
    df['is_pre_peak'] = df['Date'].dt.month.isin([3, 4]).astype(int)

    # D. Trend (5)
    df['time_idx'] = (df['Date'] - t0).dt.days
    df['post_2018'] = (df['Date'].dt.year >= 2019).astype(int)
    df['post_2018_t'] = df['time_idx'] * df['post_2018']
    df['year_centered'] = df['Date'].dt.year - 2017
    df['is_covid'] = df['Date'].dt.year.isin([2020, 2021]).astype(int)

    # E. Calendar (11)
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    dim = df['Date'].dt.days_in_month
    df['days_to_month_end'] = dim - df['day']
    df['is_month_end_window'] = (df['days_to_month_end'] <= 4).astype(int)
    df['is_month_start_window'] = (df['day'] <= 5).astype(int)
    df['day_of_month_ratio'] = df['day'] / dim
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

    # F. Promotions (7)
    df = df.merge(promo_dates[['Date', 'has_promo', 'n_active_promos', 'total_discount_val']],
                  on='Date', how='left')
    df['has_promo'] = df['has_promo'].fillna(0)
    df['n_active_promos'] = df['n_active_promos'].fillna(0)
    df['total_discount_val'] = df['total_discount_val'].fillna(0)
    df['is_dec_promo'] = (df['Date'].dt.month == 12).astype(int)
    df['promo_lag2_effect'] = df['has_promo'].shift(60).fillna(0)
    df['promo_lag1_effect'] = df['has_promo'].shift(30).fillna(0)
    df['pre_yearend_cannib'] = df['Date'].dt.month.isin([10, 11]).astype(int)

    # G. Web traffic (6)
    df = df.merge(traffic_daily[['Date', 'sessions']], on='Date', how='left')
    df['sessions'] = df['sessions'].fillna(df['sessions'].median())
    df['sessions_lag_1'] = df['sessions'].shift(1)
    df['sessions_lag_7'] = df['sessions'].shift(7)
    df['sessions_roll7'] = df['sessions'].shift(1).rolling(7, min_periods=1).mean()
    df['sessions_roll30'] = df['sessions'].shift(1).rolling(30, min_periods=1).mean()
    df['rev_per_session'] = df['rev_lag_1'] / df['sessions_lag_1'].replace(0, np.nan)

    # H. Inventory (4)
    df = df.merge(inv_daily[['Date', 'fill_rate_mean', 'stockout_flag_sum']], on='Date', how='left')
    df['fill_rate_mean'] = df['fill_rate_mean'].fillna(df['fill_rate_mean'].median())
    df['stockout_flag_sum'] = df['stockout_flag_sum'].fillna(0)
    df['fill_rate_lag30'] = df['fill_rate_mean'].shift(30)
    df['stockout_risk'] = (df['stockout_flag_sum'] > 10).astype(int)

    # I. Interactions (8)
    df['peak_x_promo'] = df['is_peak_summer'] * df['has_promo']
    df['yearend_x_promo'] = df['is_year_end_sale'] * df['is_dec_promo']
    df['post18_x_summer'] = df['post_2018'] * df['is_peak_summer']
    df['post18_x_promo'] = df['post_2018'] * df['has_promo']
    df['covid_x_peak'] = df['is_covid'] * df['is_peak_summer']
    df['session_momentum'] = df['sessions_lag_1'] / df['sessions_roll30'].replace(0, np.nan)
    df['peak_x_monthend'] = df['is_peak_summer'] * df['is_month_end_window']
    df['yearend_x_monthend'] = df['is_year_end_sale'] * df['is_month_end_window']

    # J. YoY reference (5)
    df['yoy_roll_mean_7'] = df['rev_lag_365'].rolling(7, min_periods=1).mean()
    df['rev_2yr_lag'] = df['Revenue'].shift(365 * 2)
    df['rev_yoy_delta'] = df['rev_lag_1'] - df['rev_lag_365']
    df['rev_2yr_ratio'] = df['rev_lag_365'] / df['rev_2yr_lag'].replace(0, np.nan)
    df['rev_yoy_roll30'] = df['rev_yoy_delta'].rolling(30, min_periods=1).mean()

    return df

# Build and identify feature columns
df_train_full = build_features(sales, traffic_daily, inv_daily, promo_dates)
EXCLUDE_COLS = ['Date', 'Revenue', 'COGS']
FEATURE_COLS = [c for c in df_train_full.columns if c not in EXCLUDE_COLS]
print(f'✓ {len(FEATURE_COLS)} features built')

# Trim first 365 rows (need lag history)
df_model = df_train_full.dropna(subset=['Revenue']).iloc[365:].copy()
print(f'Training rows after lag warmup: {len(df_model)}')

# %% [markdown]
# ## 4 — Model Definitions

# %%
EARLY_STOP_ROUNDS = 100

# Reduced capacity models (prevent overfit in recursive forecast)
MODELS = {
    'CatBoost': {
        'class': cb.CatBoostRegressor,
        'params': {'iterations': 1500, 'depth': 4, 'learning_rate': 0.03,
                   'l2_leaf_reg': 10.0, 'random_seed': SEED, 'verbose': 0,
                   'subsample': 0.7},
        'iter_key': 'iterations', 'seed_key': 'random_seed',
    },
    'LightGBM': {
        'class': lgb.LGBMRegressor,
        'params': {'objective': 'regression', 'metric': 'mae', 'n_estimators': 1500,
                   'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 5,
                   'min_child_samples': 100, 'subsample': 0.7, 'subsample_freq': 1,
                   'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
                   'random_state': SEED, 'n_jobs': -1, 'verbose': -1},
        'iter_key': 'n_estimators', 'seed_key': 'random_state',
    },
}
# Ensemble weights: 60% tree (equal split) + 40% Ridge
TREE_WEIGHT = 0.60  # 60% tree + 40% Ridge — best Kaggle score

def train_model(model_class, params, X_tr, y_tr, X_eval, y_eval):
    """Train a single model with early stopping."""
    model = model_class(**params)
    if isinstance(model, xgb.XGBRegressor):
        model.fit(X_tr, y_tr, eval_set=[(X_eval, y_eval)], verbose=False)
    elif isinstance(model, lgb.LGBMRegressor):
        model.fit(X_tr, y_tr, eval_set=[(X_eval, y_eval)],
                  callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                             lgb.log_evaluation(period=0)])
    elif isinstance(model, cb.CatBoostRegressor):
        model.fit(X_tr, y_tr, eval_set=(X_eval, y_eval),
                  early_stopping_rounds=EARLY_STOP_ROUNDS, verbose=0)
    elif isinstance(model, HistGradientBoostingRegressor):
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)
    return model

def evaluate(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# %% [markdown]
# ## 5 — Recursive Cross-Validation (simulates actual test conditions)

# %%
from sklearn.model_selection import TimeSeriesSplit

X_all = df_model[FEATURE_COLS].fillna(0)
y_all = df_model['Revenue']

RIDGE_WEIGHT = 1 - TREE_WEIGHT

tscv = TimeSeriesSplit(n_splits=5, test_size=180)
cv_results = {m: [] for m in MODELS}
cv_results['Ridge'] = []
cv_results['Ensemble'] = []
cv_best_iters = []

print('=== Recursive Cross-Validation (5-fold, 180-day test) ===')
for fold, (ti, vi) in enumerate(tscv.split(X_all)):
    train_idx = df_model.index[ti]
    val_idx = df_model.index[vi]
    y_vl = y_all.iloc[vi]

    # Train with early stopping (80/20 split within train)
    X_tr, y_tr = X_all.iloc[ti], y_all.iloc[ti]
    split_pt = int(len(X_tr) * 0.8)
    X_t, X_e = X_tr.iloc[:split_pt], X_tr.iloc[split_pt:]
    y_t, y_e = y_tr.iloc[:split_pt], y_tr.iloc[split_pt:]

    fold_models = {}
    for mname, mcfg in MODELS.items():
        model = train_model(mcfg['class'], mcfg['params'], X_t, y_t, X_e, y_e)
        fold_models[mname] = model
        if hasattr(model, 'best_iteration_'):
            cv_best_iters.append(model.best_iteration_)

    scaler_cv = StandardScaler()
    scaler_cv.fit(X_tr)
    ridge_cv = Ridge(alpha=200).fit(scaler_cv.transform(X_tr), y_tr)

    # === RECURSIVE VALIDATION (no actual revenue in validation) ===
    val_timeline = sales.iloc[:val_idx[-1]+1].copy()
    val_timeline.loc[val_idx, 'Revenue'] = np.nan
    val_timeline.loc[val_idx, 'COGS'] = np.nan

    fold_preds = {m: [] for m in list(MODELS.keys()) + ['Ridge', 'Ensemble']}

    for vidx in val_idx:
        df_feat = build_features(val_timeline, traffic_daily, inv_daily, promo_dates)
        row = df_feat.loc[vidx:vidx, FEATURE_COLS].fillna(0)

        tree_preds = []
        for mname, model in fold_models.items():
            pred = max(0, model.predict(row)[0])
            fold_preds[mname].append(pred)
            tree_preds.append(pred)
        tree_avg = np.mean(tree_preds)

        row_scaled = scaler_cv.transform(row)
        ridge_p = max(0, ridge_cv.predict(row_scaled)[0])
        fold_preds['Ridge'].append(ridge_p)

        ens_p = (1 - RIDGE_WEIGHT) * tree_avg + RIDGE_WEIGHT * ridge_p
        fold_preds['Ensemble'].append(ens_p)

        # Recursive: prediction feeds into next day's lags
        val_timeline.loc[vidx, 'Revenue'] = ens_p
        val_timeline.loc[vidx, 'COGS'] = ens_p * 0.87

    y_vl_arr = y_vl.values
    for mname in list(MODELS.keys()) + ['Ridge', 'Ensemble']:
        cv_results[mname].append(evaluate(y_vl_arr, np.array(fold_preds[mname])))

    ens_mae = cv_results['Ensemble'][-1]['MAE']
    print(f'  Fold {fold+1}: Recursive MAE={ens_mae:,.0f}')

optimal_iters = int(np.mean(cv_best_iters) * 1.1) if cv_best_iters else 500
print(f'\n  optimal_iters={optimal_iters} (mean={np.mean(cv_best_iters):.0f} × 1.1)')

print('\n=== Recursive CV Summary (5-fold avg) ===')
print(f'{"Model":<12} {"MAE":>10} {"RMSE":>10} {"R²":>8}')
print('-' * 45)
for mname in list(MODELS.keys()) + ['Ridge', 'Ensemble']:
    avg_mae = np.mean([r['MAE'] for r in cv_results[mname]])
    avg_rmse = np.mean([r['RMSE'] for r in cv_results[mname]])
    avg_r2 = np.mean([r['R2'] for r in cv_results[mname]])
    marker = ' ⭐' if mname == 'Ensemble' else ''
    print(f'  {mname:<10} {avg_mae:>10,.0f} {avg_rmse:>10,.0f} {avg_r2:>8.4f}{marker}')

# %% [markdown]
# ## 6 — Train Final Models (Multi-Seed, ALL data, fixed iterations)

# %%
SEEDS = [42, 123, 456, 789, 2024]

final_models = {}
for mname, mcfg in MODELS.items():
    seed_models = []
    iter_key = mcfg.get('iter_key', 'n_estimators')
    seed_key = mcfg.get('seed_key', 'random_state')
    for sd in SEEDS:
        params = mcfg['params'].copy()
        params[iter_key] = optimal_iters
        params[seed_key] = sd
        # Remove early_stopping_rounds for final training (fixed iters)
        params.pop('early_stopping_rounds', None)
        m = mcfg['class'](**params)
        m.fit(X_all, y_all)  # 100% data, no eval set
        seed_models.append(m)
        tr_mae = mean_absolute_error(y_all, np.clip(m.predict(X_all), 0, None))
        print(f'  {mname} seed {sd}: train MAE={tr_mae:,.0f}')
    final_models[mname] = seed_models

final_scaler = StandardScaler()
X_all_scaled = final_scaler.fit_transform(X_all)
final_ridge = Ridge(alpha=200).fit(X_all_scaled, y_all)
ridge_tr = np.clip(final_ridge.predict(X_all_scaled), 0, None)
print(f'  Ridge: train MAE={mean_absolute_error(y_all, ridge_tr):,.0f}')
total_tree = sum(len(v) for v in final_models.values())
print(f'\n  Ensemble: {total_tree} tree models ({len(MODELS)} types × {len(SEEDS)} seeds) + 1 Ridge')

# %% [markdown]
# ## 7 — Recursive Forecast for Test Period

# %%
test_df = pd.DataFrame({'Date': sample_sub['Date'], 'Revenue': np.nan, 'COGS': np.nan})
timeline = pd.concat([sales, test_df], ignore_index=True).sort_values('Date').reset_index(drop=True)
test_mask = timeline['Revenue'].isna()
test_indices = timeline[test_mask].index.tolist()

avg_cogs_ratio = (sales['COGS'] / sales['Revenue'].replace(0, np.nan)).dropna().mean()
print(f'COGS ratio: {avg_cogs_ratio:.4f}')
print(f'Forecasting {len(test_indices)} days ({len(SEEDS)} seeds × {len(MODELS)} types + Ridge)...')

t0 = time.time()
predictions_rev = {}

for i, idx in enumerate(test_indices):
    df_feat = build_features(timeline, traffic_daily, inv_daily, promo_dates)
    row = df_feat.loc[idx:idx, FEATURE_COLS].fillna(0)

    # Multi-seed: average all seeds within each model type, then average across types
    tree_preds = []
    for mname, seed_models in final_models.items():
        model_avg = np.mean([max(0, m.predict(row)[0]) for m in seed_models])
        tree_preds.append(model_avg)
    tree_avg = np.mean(tree_preds)

    # Ridge
    row_scaled = final_scaler.transform(row)
    ridge_p = max(0, final_ridge.predict(row_scaled)[0])

    # Ensemble: 60% tree + 40% Ridge
    pred = (1 - RIDGE_WEIGHT) * tree_avg + RIDGE_WEIGHT * ridge_p

    predictions_rev[idx] = pred
    timeline.loc[idx, 'Revenue'] = pred
    timeline.loc[idx, 'COGS'] = pred * avg_cogs_ratio

    if (i + 1) % 50 == 0 or i == len(test_indices) - 1:
        elapsed = time.time() - t0
        print(f'  [{i+1}/{len(test_indices)}] {timeline.loc[idx, "Date"].date()} → {pred:,.0f}'
              f'  ({elapsed:.0f}s)')

print(f'✓ Recursive forecast complete in {time.time()-t0:.0f}s')

# %% [markdown]
# ## 8 — Export Submission

# %%
rows = []
for idx, pred in predictions_rev.items():
    rows.append({
        'Date': timeline.loc[idx, 'Date'],
        'Revenue': round(pred, 2),
        'COGS': round(pred * avg_cogs_ratio, 2)
    })
submission = pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)
submission = sample_sub[['Date']].merge(submission, on='Date', how='left')
submission.to_csv('submission.csv', index=False)
print(f'✓ Saved submission.csv: {len(submission)} rows')
print(f'  Revenue mean: {submission["Revenue"].mean():,.0f}')
print(f'  COGS mean: {submission["COGS"].mean():,.0f}')

# %% [markdown]
# ## 9 — Diagnostic Plots

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('Advanced Forecasting — Recursive + Rich Features', fontsize=16, fontweight='bold', y=1.01)

# (0,0) — Full timeline
ax = axes[0, 0]
ax.plot(sales.Date, sales.Revenue, lw=0.5, color=COLORS['text'], alpha=0.4, label='Train')
ax.plot(submission.Date, submission.Revenue, lw=0.9, color=COLORS['primary'], label='Forecast')
ax.axvline(pd.Timestamp('2023-01-01'), color=COLORS['danger'], lw=1.5, ls='--', alpha=0.6)
ax.set_title('Revenue: Train + Recursive Forecast')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
ax.legend(fontsize=9); ax.grid(alpha=0.2)

# (0,1) — Zoom 2022+
ax = axes[0, 1]
tail = sales[sales.Date >= '2022-01-01']
ax.plot(tail.Date, tail.Revenue, lw=0.8, color=COLORS['text'], alpha=0.5, label='Train 2022')
ax.plot(submission.Date, submission.Revenue, lw=0.9, color=COLORS['primary'], label='Forecast')
ax.axvline(pd.Timestamp('2023-01-01'), color=COLORS['danger'], lw=1.5, ls='--', alpha=0.6)
ax.set_title('Zoom: 2022 → 2024')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
ax.legend(fontsize=9); ax.grid(alpha=0.2)

# (1,0) — CV R² comparison
ax = axes[1, 0]
model_labels = list(MODELS.keys()) + ['Ensemble']
avg_r2s = [np.mean([r['R2'] for r in cv_results[m]]) for m in model_labels]
bars = ax.barh(model_labels, avg_r2s, color=[COLORS['primary']]*len(MODELS) + [COLORS['accent']])
ax.set_xlabel('Average R² (5-fold CV)')
ax.set_title('Model R² Comparison')
for bar, r2 in zip(bars, avg_r2s):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{r2:.4f}',
            va='center', fontsize=10, color=COLORS['text'])
ax.set_xlim(0, 1); ax.grid(alpha=0.2, axis='x')

# (1,1) — Feature importance
ax = axes[1, 1]
best_mname = max(MODELS.keys(), key=lambda m: np.mean([r['R2'] for r in cv_results[m]]))
best_model = final_models[best_mname][0]  # First seed
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
    idx_top = np.argsort(importance)[-15:]
    ax.barh([FEATURE_COLS[i] for i in idx_top], importance[idx_top], color=COLORS['teal'])
    ax.set_title(f'Top 15 Features ({best_mname})')
    ax.grid(alpha=0.2, axis='x')

plt.tight_layout()
save_fig(fig, 'vis_forecast_advanced_comparison')
plt.show()

# %% [markdown]
# ## 10 — SHAP Explainability

# %%
import shap

best_mname = max(MODELS.keys(), key=lambda m: np.mean([r['R2'] for r in cv_results[m]]))
print(f'SHAP analysis on: {best_mname} (seed 0)')
shap_model = final_models[best_mname][0]  # Use first seed model

X_shap = X_all.sample(n=min(1000, len(X_all)), random_state=SEED)
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_shap)

fig_shap, _ = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, feature_names=FEATURE_COLS, show=False, max_display=15)
plt.title(f'SHAP Feature Impact ({best_mname})', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
save_fig(plt.gcf(), 'vis_shap_summary')
plt.show()

# Business interpretation
print('\n' + '=' * 70)
print('KEY REVENUE DRIVERS')
print('=' * 70)
mean_abs = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs)[::-1][:10]
for rank, i in enumerate(top_idx, 1):
    print(f'  {rank:2d}. {FEATURE_COLS[i]:25s} |SHAP|={mean_abs[i]:>12,.0f}')

# %% [markdown]
# ## 11 — Submission Verification

# %%
sample = pd.read_csv('sample_submission.csv')
sub = pd.read_csv('submission.csv')
assert list(sub.columns) == list(sample.columns), 'Column mismatch!'
assert len(sub) == len(sample), f'Row count mismatch!'
assert list(sub.Date) == list(sample.Date), 'Date order mismatch!'
print('✓ Submission verified: columns, row count, and date order match.')
print(f'  Rows: {len(sub)} | Date range: {sub.Date.iloc[0]} → {sub.Date.iloc[-1]}')

# %%
print('\n' + '=' * 60)
print('Advanced forecasting pipeline v2 complete!')
print('=' * 60)
