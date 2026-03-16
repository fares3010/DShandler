# dspipeline

A stateful, chainable data-science pipeline for the full tabular ML workflow —
from raw data to model-ready arrays — in a single class.

## Installation

```bash
pip install dspipeline
```

## Quick start

```python
import pandas as pd
from dspipeline import DataSciencePipeline

df  = pd.read_csv("your_dataset.csv")
dsp = DataSciencePipeline(df, target_col="Churn", task_type="classification", exclude_cols=["customer_id", "signup_date"])

# One-liner: diagnostics → cleaning → preprocessing
dsp.run_diagnostics().run_cleaning().run_preprocessing()

# Leakproof split
X_train, X_test, y_train, y_test, preprocessor = dsp.split(test_size=0.2)

# Fit your model on processed arrays
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(dsp.results["split"]["X_train_processed"], y_train)
```

---

## What it covers

| Phase | Methods |
|---|---|
| **Diagnostics** | `profile_missing`, `detect_structural`, `detect_dimensional`, `detect_categorical`, `detect_predictive`, `detect_anomaly_scan`, `detect_leakage` |
| **Cleaning** | `format_structure`, `drop_duplicates`, `standardize_text`, `impute_numeric`, `impute_categorical` |
| **Anomaly handling** | `handle_outliers` |
| **Transformation** | `transform_shape` |
| **Encoding** | `encode` |
| **Feature selection** | `select_features`, `vif_optimize` |
| **Split** | `split` |
| **EDA** | `analyze_distribution`, `evaluate_distribution`, `analyze_relationship`, `test_hypothesis` |
| **Time-series** | `enforce_stationarity`, `analyze_autocorrelation` |

---

## Method chaining

Every mutating method returns `self`, so you can chain calls:

```python
(dsp
  .profile_missing()
  .drop_duplicates(subset=["user_id"], sort_by="updated_at")
  .impute_numeric(strategy="knn")
  .impute_categorical(strategy="mode")
  .handle_outliers(method="iqr", action="clip", threshold=1.5)
  .transform_shape(scale_method="robust")
  .encode(nominal_cols=["color"], ordinal_maps={"size": ["S", "M", "L"]})
  .select_features(multi_corr_threshold=0.85)
  .vif_optimize(threshold=5.0)
)
X_train, X_test, y_train, y_test, preprocessor = dsp.split(stratify=True)
```

---

## State inspection

```python
dsp.summary()           # prints shape history for every step
dsp.results.keys()      # all stored reports and artefacts
dsp.history             # list of {method, shape_before, shape_after}
df_snapshot = dsp.snapshot()   # copy of current working DataFrame
dsp.reset()             # restore to original raw DataFrame
```

---

## Individual functions

Every function is also importable directly:

```python
from dspipeline import (
    advanced_missing_profiler,
    detect_anomalies,
    handle_numerical_missing,
    advanced_knn_impute,
    transform_data_shape,
    encode_categorical_data,
    optimize_vif,
    setup_leakproof_environment,
    analyze_distribution,
    test_hypothesis,
    enforce_stationarity,
    analyze_autocorrelation,
)
```

---

## Time-series example

```python
dsp = DataSciencePipeline(df, target_col="demand")

# Check and fix stationarity
series, d, report = dsp.enforce_stationarity("revenue", seasonal_period=12)
print(f"Applied d={d} differencing steps")

# ACF / PACF + ARIMA order hints
acf_report = dsp.analyze_autocorrelation("revenue", lags=40)
print(f"Suggested ARIMA({acf_report['arima_hint_p']}, {d}, {acf_report['arima_hint_q']})")
```

---

## Requirements

- Python ≥ 3.9
- pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib, seaborn

---

## License

MIT
