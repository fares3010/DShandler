"""
Diagnostic functions: missing value profiling, structural anomaly detection,
dimensional analysis, categorical issues, predictive issues, and leakage detection.
"""
import json
import warnings
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import pointbiserialr, spearmanr
from sklearn.ensemble import IsolationForest


def detect_anomalies(df, columns, contamination=0.05):
    """
    Detects and categorizes anomalies into:
      - Statistical Outlier  → mild univariate outlier (1.5x IQR)
      - Extreme Outlier      → severe univariate outlier (3.0x IQR)
      - Inlier Anomaly       → normal univariate but anomalous multivariate combination (Isolation Forest)

    Parameters
    ----------
    df            : pd.DataFrame       — input data
    columns       : list[str]          — numeric columns to scan
    contamination : float | dict       — expected anomaly fraction for Isolation Forest.
                                         Pass a float (global) or a dict keyed by column
                                         name for per-column control (dict only affects
                                         which rows are passed to IF scoring groups).
                                         Default: 0.05

    Returns
    -------
    anomaly_df : pd.DataFrame   — only the anomalous rows, with an 'Anomaly_Type' column
    report     : dict           — summary counts
    """

    # ── 0. Validate inputs ──────────────────────────────────────────────────
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Resolve contamination to a single float for Isolation Forest
    if isinstance(contamination, dict):
        # Use the mean of provided values as the global IF contamination
        if_contamination = float(np.mean(list(contamination.values())))
    else:
        if_contamination = float(contamination)

    if not (0.0 < if_contamination < 0.5):
        raise ValueError(f"contamination must be between 0 and 0.5, got {if_contamination}")

    print(f"🔍 Scanning {len(columns)} columns for anomalies...\n")

    # ── 1. Severity mapping (higher = worse) ─────────────────────────────────
    SEVERITY = {
        "Normal":             0,
        "Statistical Outlier": 1,
        "Inlier Anomaly":      2,
        "Extreme Outlier":     3,
    }

    work_df = df.copy()
    # Numeric severity tracker — lets us always keep the *worst* label per row
    work_df["_severity"] = 0
    work_df["Anomaly_Type"] = "Normal"

    skipped_cols = []

    # ── 2. IQR-based univariate outlier detection ────────────────────────────
    for col in columns:
        Q1  = work_df[col].quantile(0.25)
        Q3  = work_df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Skip constant or near-constant columns — IQR = 0 would flag everything
        if IQR == 0:
            skipped_cols.append(col)
            continue

        lower_mild    = Q1 - 1.5 * IQR
        upper_mild    = Q3 + 1.5 * IQR
        lower_extreme = Q1 - 3.0 * IQR
        upper_extreme = Q3 + 3.0 * IQR

        extreme_mask = (work_df[col] < lower_extreme) | (work_df[col] > upper_extreme)
        mild_mask = (
            ((work_df[col] < lower_mild)  & (work_df[col] >= lower_extreme)) |
            ((work_df[col] > upper_mild)  & (work_df[col] <= upper_extreme))
        )

        # Apply only if this column's label is *worse* than what's already recorded
        for mask, label in [(extreme_mask, "Extreme Outlier"), (mild_mask, "Statistical Outlier")]:
            upgrade = mask & (SEVERITY[label] > work_df["_severity"])
            work_df.loc[upgrade, "_severity"]   = SEVERITY[label]
            work_df.loc[upgrade, "Anomaly_Type"] = label

    if skipped_cols:
        print(f"⚠️  Skipped {len(skipped_cols)} zero-IQR (constant/binary) column(s): {skipped_cols}\n")

    # ── 3. Isolation Forest — multivariate inlier anomaly detection ──────────
    # Work only on rows with complete data to avoid median-imputation bias
    complete_mask = work_df[columns].notna().all(axis=1)
    n_complete    = complete_mask.sum()

    if n_complete < 10:
        print(f"⚠️  Only {n_complete} complete rows — skipping Isolation Forest.\n")
    else:
        iso_forest = IsolationForest(
            contamination=if_contamination,
            random_state=42,
            n_estimators=200,
        )
        if_scores = iso_forest.fit_predict(work_df.loc[complete_mask, columns])

        # "Inlier Anomaly": flagged by IF but NOT already flagged by IQR
        inlier_label    = "Inlier Anomaly"
        inlier_severity = SEVERITY[inlier_label]

        if_anomaly_index = work_df.loc[complete_mask].index[if_scores == -1]
        upgrade_mask = (
            work_df.index.isin(if_anomaly_index) &
            (inlier_severity > work_df["_severity"])
        )
        work_df.loc[upgrade_mask, "_severity"]   = inlier_severity
        work_df.loc[upgrade_mask, "Anomaly_Type"] = inlier_label

    # ── 4. Build output ───────────────────────────────────────────────────────
    work_df = work_df.drop(columns=["_severity"])
    anomaly_df = work_df[work_df["Anomaly_Type"] != "Normal"].copy()

    report = {
        "Total_Rows_Scanned":   len(df),
        "Normal_Rows":          len(df) - len(anomaly_df),
        "Statistical_Outliers": (anomaly_df["Anomaly_Type"] == "Statistical Outlier").sum(),
        "Extreme_Outliers":     (anomaly_df["Anomaly_Type"] == "Extreme Outlier").sum(),
        "Inlier_Anomalies":     (anomaly_df["Anomaly_Type"] == "Inlier Anomaly").sum(),
        "Total_Anomalies":      len(anomaly_df),
    }

    print("--- 🚨 Anomaly Detection Report ---")
    for key, val in report.items():
        print(f"  {key.replace('_', ' '):<25} {val}")
    print("-" * 35)

    return anomaly_df, report


def detect_structural_anomalies(df, shifted_threshold: float = 0.01):
    """
    Scans a DataFrame to detect hidden structural issues and isolates shifted rows.

    Detects:
      - Dominant Python data types per column (actual contents, not just dtype)
      - Shifted / misaligned rows (cells whose type doesn't match the column's majority)
      - Nested JSON or dict/list columns
      - Date-like string columns
      - Timezone information in date columns

    Parameters
    ----------
    df                 : pd.DataFrame
    shifted_threshold  : float
        Minimum fraction of mismatched cells in a column before rows are flagged
        as shifted. Default 0.01 (1 %). Set lower to be more aggressive.

    Returns
    -------
    report     : dict         — structured diagnostic summary
    shifted_df : pd.DataFrame — only the rows suspected of being shifted
    """

    # ── 0. Guard ──────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty — nothing to scan.")

    print("🔍 Scanning dataset for structural anomalies...\n")

    report = {
        "Total_Rows":          len(df),
        "Total_Columns":       len(df.columns),
        "Actual_Data_Types":   {},
        "Type_Distributions":  {},   # NEW: full type breakdown per column
        "Date_Columns":        [],
        "Timezones_Detected":  {},
        "Nested_JSON_Columns": [],
        "Shifted_Row_Count":   0,
        "Shifted_Columns":     [],   # NEW: which columns triggered shifts
        "All_Null_Columns":    [],   # NEW: columns skipped due to being fully null
    }

    shifted_row_indices: set = set()

    for col in df.columns:
        non_null = df[col].dropna()

        # ── Fully null column ─────────────────────────────────────────────────
        if non_null.empty:
            report["All_Null_Columns"].append(col)
            report["Actual_Data_Types"][col] = "all_null"
            continue

        # ── 1. Dominant type (actual Python type, not pandas dtype) ───────────
        type_counts = Counter(type(x).__name__ for x in non_null)
        dominant_type, dominant_count = type_counts.most_common(1)[0]
        report["Actual_Data_Types"][col]  = dominant_type
        report["Type_Distributions"][col] = dict(type_counts)

        # ── 2. Shifted-row detection ──────────────────────────────────────────
        total_non_null = len(non_null)

        if dominant_type in ("int", "float"):
            mismatch_mask = pd.Series(
                [type(x).__name__ not in ("int", "float") for x in non_null],
                index=non_null.index,
            )
        else:
            mismatch_mask = pd.Series(
                [type(x).__name__ != dominant_type for x in non_null],
                index=non_null.index,
            )

        mismatch_fraction = mismatch_mask.sum() / total_non_null
        if mismatch_mask.any() and mismatch_fraction >= shifted_threshold:
            bad_indices = non_null[mismatch_mask].index.tolist()
            shifted_row_indices.update(bad_indices)
            report["Shifted_Columns"].append(
                {
                    "column":            col,
                    "dominant_type":     dominant_type,
                    "mismatch_count":    int(mismatch_mask.sum()),
                    "mismatch_fraction": round(mismatch_fraction, 4),
                }
            )

        # ── 3. Nested JSON / dict / list detection ────────────────────────────
        first_valid = non_null.iloc[0]
        if isinstance(first_valid, (dict, list)):
            report["Nested_JSON_Columns"].append(col)
            continue  # Not a date — skip further checks

        if dominant_type == "str":
            sample = non_null.head(20).astype(str).str.strip()

            json_candidates = sample[
                sample.str.startswith(("{", "[")) & sample.str.endswith(("}", "]"))
            ]
            if not json_candidates.empty:
                parsed_ok = 0
                for raw in json_candidates:
                    try:
                        json.loads(raw)
                        parsed_ok += 1
                    except (json.JSONDecodeError, ValueError):
                        # Single-quote dicts (Python reprs) — try normalising
                        try:
                            json.loads(raw.replace("'", '"'))
                            parsed_ok += 1
                        except (json.JSONDecodeError, ValueError):
                            pass
                if parsed_ok > 0:
                    report["Nested_JSON_Columns"].append(col)
                    continue  # Confirmed JSON — skip date check

            # ── 4 & 5. Date and timezone detection ───────────────────────────
            # Use a small sample; errors='coerce' avoids ValueError on bad rows
            parsed = pd.to_datetime(sample, errors="coerce", utc=False)
            success_rate = parsed.notna().sum() / len(sample)

            # Require ≥ 80 % of the sample to parse as a date before labelling
            if success_rate >= 0.80:
                report["Date_Columns"].append(col)

                # Timezone-aware parsed datetimes
                if parsed.dt.tz is not None:
                    report["Timezones_Detected"][col] = str(parsed.dt.tz)

                # Timezone offsets still encoded as strings (e.g. +05:00, Z, UTC)
                elif sample.str.contains(
                    r"(?:[-+]\d{2}:\d{2}|\bZ\b|\bUTC\b)", regex=True
                ).any():
                    # Collect the distinct offset tokens found
                    offsets = (
                        sample.str.extract(r"([-+]\d{2}:\d{2}|\bZ\b|\bUTC\b)")[0]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                    report["Timezones_Detected"][col] = (
                        offsets[0] if len(offsets) == 1 else f"Mixed: {offsets}"
                    )

    # ── 4. Compile shifted rows ───────────────────────────────────────────────
    shifted_df = df.loc[sorted(shifted_row_indices)].copy() if shifted_row_indices else df.iloc[0:0].copy()
    report["Shifted_Row_Count"] = len(shifted_df)

    # ── 5. Print summary ──────────────────────────────────────────────────────
    print("--- 📊 Structural Diagnostic Report ---")
    print(f"  {'Rows scanned':<28} {report['Total_Rows']}")
    print(f"  {'Columns scanned':<28} {report['Total_Columns']}")
    print(f"  {'All-null columns':<28} {report['All_Null_Columns'] or 'None'}")
    print(f"  {'Nested JSON columns':<28} {report['Nested_JSON_Columns'] or 'None'}")
    print(f"  {'Date columns':<28} {report['Date_Columns'] or 'None'}")

    if report["Timezones_Detected"]:
        print("  Timezones detected:")
        for k, v in report["Timezones_Detected"].items():
            print(f"    - {k}: {v}")

    if report["Shifted_Columns"]:
        print("  Columns triggering shifted-row flags:")
        for entry in report["Shifted_Columns"]:
            print(
                f"    - {entry['column']}: {entry['mismatch_count']} mismatched cells "
                f"({entry['mismatch_fraction']*100:.1f} %) — dominant type: {entry['dominant_type']}"
            )

    print(f"  {'⚠️  Suspected shifted rows':<28} {report['Shifted_Row_Count']}")
    print("-" * 41)

    return report, shifted_df


def detect_dimensional_issues(
    df,
    skew_threshold: float = 1.0,
    moderate_skew_threshold: float = 0.5,
    sparse_threshold: float = 0.7,
    scale_ratio_threshold: float = 1000.0,
    near_zero_variance_threshold: float = 1e-6,
):
    """
    Scans numeric columns to detect vast scale differences, high skewness,
    extreme sparsity, and near-zero-variance features.

    Parameters
    ----------
    df                          : pd.DataFrame
    skew_threshold              : float  — |skew| above this → "Highly Skewed"   (default 1.0)
    moderate_skew_threshold     : float  — |skew| above this → "Moderately Skewed" (default 0.5)
    sparse_threshold            : float  — zero-fraction [0,1] above this → sparse (default 0.7)
    scale_ratio_threshold       : float  — max_range / min_range above this → scale alert (default 1000)
    near_zero_variance_threshold: float  — std below this → flagged as constant-like (default 1e-6)

    Returns
    -------
    report   : dict           — structured summary of every flagged category
    stats_df : pd.DataFrame   — per-column metrics, sorted by severity
    """

    # ── 0. Guard ──────────────────────────────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found. Dimensional analysis requires numbers.")

    numeric_cols = numeric_df.columns.tolist()
    print(f"📐 Profiling {len(numeric_cols)} numeric columns for dimensional anomalies...\n")

    # ── 1. Per-column metrics ─────────────────────────────────────────────────
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()

        if series.empty:
            rows.append({
                "Column": col, "Min": np.nan, "Max": np.nan, "Range": np.nan,
                "Std": np.nan, "Skewness": np.nan, "Kurtosis": np.nan,
                "Zero_Fraction (%)": np.nan, "Null_Fraction (%)": round(100.0, 2),
                "Unique_Count": 0, "Near_Zero_Variance": True,
            })
            continue

        min_val  = series.min()
        max_val  = series.max()
        col_range = max_val - min_val
        std_val  = series.std()

        # Fisher-Pearson adjusted skewness (same as pandas .skew())
        skewness = series.skew()

        # Excess kurtosis — large positive → heavy tails / outlier-prone
        kurtosis = series.kurt()

        # Sparsity: fraction of EXACT zeros relative to total rows (including NaNs)
        # Using len(df) keeps NaN rows in the denominator — a deliberate choice so
        # sparsity reflects the true memory footprint, not just non-null values.
        zero_fraction = (df[col] == 0).sum() / len(df)

        # Null fraction — useful context alongside sparsity
        null_fraction = df[col].isna().sum() / len(df)

        rows.append({
            "Column":              col,
            "Min":                 round(float(min_val),  4),
            "Max":                 round(float(max_val),  4),
            "Range":               round(float(col_range), 4),
            "Std":                 round(float(std_val),  4) if pd.notna(std_val) else np.nan,
            "Skewness":            round(float(skewness), 4) if pd.notna(skewness) else 0.0,
            "Kurtosis":            round(float(kurtosis), 4) if pd.notna(kurtosis) else 0.0,
            "Zero_Fraction (%)":   round(zero_fraction * 100, 2),
            "Null_Fraction (%)":   round(null_fraction * 100, 2),
            "Unique_Count":        int(series.nunique()),
            "Near_Zero_Variance":  (std_val is not None) and (float(std_val) < near_zero_variance_threshold),
        })

    stats_df = pd.DataFrame(rows)

    # ── 2. Flag categories ────────────────────────────────────────────────────

    # — Skewness (two-tier) ————————————————————————————————————————————————
    highly_skewed = (
        stats_df[stats_df["Skewness"].abs() > skew_threshold]["Column"].tolist()
    )
    moderately_skewed = (
        stats_df[
            (stats_df["Skewness"].abs() > moderate_skew_threshold) &
            (stats_df["Skewness"].abs() <= skew_threshold)
        ]["Column"].tolist()
    )

    # — Sparsity ——————————————————————————————————————————————————————————
    sparse_cols = (
        stats_df[stats_df["Zero_Fraction (%)"] > sparse_threshold * 100]["Column"].tolist()
    )

    # — Scale disparity ————————————————————————————————————————————————————
    valid_ranges = stats_df.loc[stats_df["Range"] > 0, "Range"]
    vast_scale_issue = False
    scale_ratio_val  = np.nan
    widest_col       = None
    narrowest_col    = None

    if len(valid_ranges) >= 2:
        max_range = valid_ranges.max()
        min_range = valid_ranges.min()
        scale_ratio_val = max_range / min_range

        if scale_ratio_val > scale_ratio_threshold:
            vast_scale_issue = True
            widest_col    = stats_df.loc[stats_df["Range"] == max_range, "Column"].iloc[0]
            narrowest_col = stats_df.loc[stats_df["Range"] == min_range, "Column"].iloc[0]

    # — Near-zero variance ————————————————————————————————————————————————
    near_zero_variance_cols = stats_df[stats_df["Near_Zero_Variance"] == True]["Column"].tolist()

    # — High kurtosis (outlier-prone features) ————————————————————————————
    high_kurtosis_cols = stats_df[stats_df["Kurtosis"] > 3]["Column"].tolist()

    # ── 3. Build report ───────────────────────────────────────────────────────
    report = {
        "Total_Numeric_Columns":    len(numeric_cols),
        "Vast_Scale_Disparity":     vast_scale_issue,
        "Scale_Ratio":              f"{scale_ratio_val:,.1f}x" if pd.notna(scale_ratio_val) else "N/A",
        "Widest_Column":            widest_col,
        "Narrowest_Column":         narrowest_col,
        "Highly_Skewed_Columns":    highly_skewed,
        "Moderately_Skewed_Columns": moderately_skewed,
        "Sparse_Columns":           sparse_cols,
        "Near_Zero_Variance_Columns": near_zero_variance_cols,
        "High_Kurtosis_Columns":    high_kurtosis_cols,
    }

    # ── 4. Print summary ──────────────────────────────────────────────────────
    print("--- 📏 Dimensionality Report ---")

    # Scale
    if vast_scale_issue:
        print(
            f"⚠️  SCALE ISSUE: Widest feature ({widest_col}) is {report['Scale_Ratio']} larger "
            f"than narrowest ({narrowest_col})."
        )
        print("     → Apply StandardScaler / MinMaxScaler before any distance-based model.\n")
    else:
        print(f"✅  Scales relatively balanced (ratio: {report['Scale_Ratio']}).\n")

    # Near-zero variance
    if near_zero_variance_cols:
        print(f"🚫  Near-Zero Variance Columns ({len(near_zero_variance_cols)}): {near_zero_variance_cols}")
        print("     → These are near-constant and should be dropped before training.\n")

    # Skewness
    print(f"⚠️  Highly Skewed    (|skew| > {skew_threshold}):           {len(highly_skewed)} column(s)")
    if highly_skewed:
        print(f"     → {highly_skewed}")
        print("     → Apply np.log1p or Yeo-Johnson power transform.\n")

    print(f"⚠️  Moderately Skewed ({moderate_skew_threshold} < |skew| ≤ {skew_threshold}): {len(moderately_skewed)} column(s)")
    if moderately_skewed:
        print(f"     → {moderately_skewed}")
        print("     → Consider square-root transform or monitor after scaling.\n")

    # Sparsity
    print(f"⚠️  Sparse Columns   (> {sparse_threshold*100:.0f}% zeros):         {len(sparse_cols)} column(s)")
    if sparse_cols:
        print(f"     → {sparse_cols}")
        print("     → Convert to pd.SparseDtype; use MaxAbsScaler.\n")

    # High kurtosis
    print(f"⚠️  High-Kurtosis Columns (excess kurtosis > 3):  {len(high_kurtosis_cols)} column(s)")
    if high_kurtosis_cols:
        print(f"     → {high_kurtosis_cols}")
        print("     → Heavy tails detected — robust scalers (RobustScaler) recommended.\n")

    print("-" * 34)

    # ── 5. Sort stats for readability ─────────────────────────────────────────
    # Primary: sparsity ↓, Secondary: |skewness| ↓, Tertiary: range ↓
    stats_df["_abs_skew"] = stats_df["Skewness"].abs()
    stats_df = (
        stats_df
        .sort_values(
            by=["Zero_Fraction (%)", "_abs_skew", "Range"],
            ascending=[False, False, False],
        )
        .drop(columns=["_abs_skew"])
        .reset_index(drop=True)
    )

    return report, stats_df


def detect_categorical_issues(
    df,
    card_threshold: int = 50,
    rare_freq_threshold: float = 0.01,
    id_uniqueness_threshold: float = 0.95,
):
    """
    Scans categorical columns to detect high cardinality, rare categories
    (which cause unseen-category errors in production), and classifies
    columns as Nominal, Likely Ordinal, or Boolean.

    Parameters
    ----------
    df                      : pd.DataFrame
    card_threshold          : int   — unique-value count above which a column is
                                      flagged as High Cardinality. Default 50.
    rare_freq_threshold     : float — categories appearing less than this fraction
                                      are flagged as rare (default 0.01 = 1 %).
    id_uniqueness_threshold : float — columns whose uniqueness ratio exceeds this
                                      are flagged as probable ID columns (default 0.95).

    Returns
    -------
    report   : dict           — structured summary of every flagged category
    stats_df : pd.DataFrame   — per-column metrics, sorted by severity
    """

    # ── 0. Guard ──────────────────────────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        raise ValueError("No categorical columns found. Detection requires text/categorical data.")

    print(f"📝 Profiling {len(cat_cols)} categorical columns for structural issues...\n")

    # ── Ordinal keyword list (extendable) ─────────────────────────────────────
    ORDINAL_KEYWORDS = {
        "low", "medium", "med", "high",
        "small", "medium", "large", "xl", "xxl",
        "poor", "fair", "good", "better", "best", "excellent",
        "beginner", "intermediate", "advanced", "expert",
        "cold", "warm", "hot",
        "never", "rarely", "sometimes", "often", "always",
        "none", "mild", "moderate", "severe", "critical",
        "q1", "q2", "q3", "q4",
        "bronze", "silver", "gold", "platinum",
        "junior", "senior", "lead", "principal",
    }

    report = {
        "Total_Categorical_Columns": len(cat_cols),
        "High_Cardinality":          [],
        "Probable_ID_Columns":       [],   # NEW: near-100 % unique → almost certainly IDs
        "Rare_Categories_Risk":      [],
        "Likely_Ordinal":            [],
        "Likely_Boolean":            [],   # NEW: exactly 2 distinct values
        "Likely_Nominal":            [],
        "All_Null_Columns":          [],   # NEW: skipped columns surfaced explicitly
    }

    rows = []

    # ── 1. Per-column analysis ────────────────────────────────────────────────
    for col in cat_cols:
        series      = df[col].dropna()
        total_rows  = len(df)
        total_valid = len(series)

        if series.empty:
            report["All_Null_Columns"].append(col)
            continue

        n_unique        = series.nunique()
        uniqueness_ratio = n_unique / total_valid

        # ── Null fraction ──────────────────────────────────────────────────
        null_fraction = df[col].isna().sum() / total_rows

        # ── 1a. Probable ID ───────────────────────────────────────────────
        is_probable_id = uniqueness_ratio >= id_uniqueness_threshold
        if is_probable_id:
            report["Probable_ID_Columns"].append(col)

        # ── 1b. High cardinality (excluding IDs — they get their own bucket) ─
        is_high_card = (n_unique > card_threshold) and not is_probable_id
        if is_high_card:
            report["High_Cardinality"].append(col)

        # ── 1c. Rare categories ────────────────────────────────────────────
        # Normalise against non-null values so NaN rows don't inflate rarity
        freqs     = series.value_counts(normalize=True)
        rare_cats = freqs[freqs < rare_freq_threshold].index.tolist()
        rare_count = len(rare_cats)
        if rare_cats:
            report["Rare_Categories_Risk"].append(col)

        # ── 1d. Boolean heuristic ─────────────────────────────────────────
        # Exactly 2 distinct values (e.g., Yes/No, True/False, M/F)
        unique_lower = {str(v).strip().lower() for v in series.unique()}
        is_boolean = n_unique == 2

        # ── 1e. Ordinal heuristic ─────────────────────────────────────────
        # Intersection with keyword list; skip if boolean (bool takes priority)
        is_ordinal = (not is_boolean) and bool(unique_lower.intersection(ORDINAL_KEYWORDS))

        # ── 1f. Assign category type ──────────────────────────────────────
        if is_boolean:
            report["Likely_Boolean"].append(col)
            category_type = "Likely Boolean"
        elif is_ordinal:
            report["Likely_Ordinal"].append(col)
            category_type = "Likely Ordinal"
        else:
            report["Likely_Nominal"].append(col)
            category_type = "Likely Nominal"

        # ── 1g. Top-5 values with their frequencies ───────────────────────
        top5 = freqs.head(5)
        top5_repr = [f"{v} ({p*100:.1f}%)" for v, p in top5.items()]

        rows.append({
            "Column":             col,
            "Unique_Values":      n_unique,
            "Uniqueness_Ratio":   round(uniqueness_ratio, 4),
            "Null_Fraction (%)":  round(null_fraction * 100, 2),
            "Probable_ID":        is_probable_id,
            "High_Cardinality":   is_high_card,
            "Rare_Category_Count": rare_count,
            "Rare_Categories":    rare_cats[:10],   # cap list for readability
            "Category_Type":      category_type,
            "Top_5_Values":       top5_repr,
        })

    stats_df = pd.DataFrame(rows)

    # ── 2. Print summary ──────────────────────────────────────────────────────
    print("--- 🏷️  Categorical Diagnostics ---")

    # All-null
    if report["All_Null_Columns"]:
        print(f"🚫  All-Null Columns (skipped): {report['All_Null_Columns']}\n")

    # Probable IDs
    print(f"🪪  Probable ID Columns (≥ {id_uniqueness_threshold*100:.0f}% unique): "
          f"{len(report['Probable_ID_Columns'])}")
    if report["Probable_ID_Columns"]:
        print(f"     → {report['Probable_ID_Columns']}")
        print("     → Drop before training — these carry no generalizable signal.\n")

    # High cardinality
    print(f"⚠️   High Cardinality (> {card_threshold} unique, not an ID): "
          f"{len(report['High_Cardinality'])}")
    if report["High_Cardinality"]:
        print(f"     → {report['High_Cardinality']}")
        print("     → Avoid One-Hot Encoding. Use Target Encoding or frequency encoding.\n")

    # Rare categories
    print(f"⚠️   Rare Category Risk (< {rare_freq_threshold*100:.1f}% frequency): "
          f"{len(report['Rare_Categories_Risk'])}")
    if report["Rare_Categories_Risk"]:
        print(f"     → {report['Rare_Categories_Risk']}")
        print("     → Group rare values into an 'Other' bucket to prevent unseen-category crashes.\n")

    # Ordinal
    print(f"🧠  Likely Ordinal: {len(report['Likely_Ordinal'])}")
    if report["Likely_Ordinal"]:
        print(f"     → {report['Likely_Ordinal']}")
        print("     → Map to integers (e.g., Low→1, Medium→2, High→3) to preserve rank logic.\n")

    # Boolean
    print(f"🔘  Likely Boolean (2 distinct values): {len(report['Likely_Boolean'])}")
    if report["Likely_Boolean"]:
        print(f"     → {report['Likely_Boolean']}")
        print("     → Label-encode directly (0/1). One-Hot Encoding is redundant here.\n")

    print("-" * 37)

    # ── 3. Sort stats for readability ─────────────────────────────────────────
    # Primary: probable ID ↓, High cardinality ↓, Rare count ↓
    stats_df = (
        stats_df
        .sort_values(
            by=["Probable_ID", "High_Cardinality", "Rare_Category_Count", "Unique_Values"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )

    return report, stats_df


def advanced_missing_profiler(
    df,
    show_heatmap: bool = True,
    show_bar: bool = True,
    heatmap_row_limit: int = 5_000,
    warn_threshold: float = 0.20,
    critical_threshold: float = 0.50,
):
    """
    Profiles missing data across every column: counts, percentages, severity
    tiers, missingness type hints, correlated-missingness detection, and
    optional visualisations.

    Parameters
    ----------
    df                 : pd.DataFrame
    show_heatmap       : bool  — row-level missingness heatmap (default True)
    show_bar           : bool  — sorted horizontal bar chart (default True)
    heatmap_row_limit  : int   — max rows rendered in heatmap; larger frames
                                 are sampled to avoid memory pressure (default 5 000)
    warn_threshold     : float — missing fraction flagged as Warning  (default 0.20)
    critical_threshold : float — missing fraction flagged as Critical (default 0.50)

    Returns
    -------
    summary_df : pd.DataFrame — one row per affected column, sorted by missing
                                count descending, with Severity, Missingness_Hint,
                                and Correlated_With columns.
    """

    # ── 0. Guard ──────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty — nothing to profile.")

    total_rows = len(df)
    total_cols = len(df.columns)

    # ── 1. Core missingness statistics ───────────────────────────────────────
    nan_count = df.isna().sum()
    nan_pct   = (nan_count / total_rows * 100).round(2)

    summary_df = pd.DataFrame({
        "NaN_Count":              nan_count,
        "Missing_Percentage (%)": nan_pct,
        "Dtype":                  df.dtypes.astype(str),
    })

    summary_df = (
        summary_df[summary_df["NaN_Count"] > 0]
        .copy()
        .sort_values("NaN_Count", ascending=False)
        .reset_index(names="Column")
    )

    if summary_df.empty:
        print(f"✅  No missing data found across {total_cols} columns ({total_rows:,} rows).")
        return summary_df

    # ── 2. Severity tier ──────────────────────────────────────────────────────
    def _severity(pct: float) -> str:
        frac = pct / 100
        if frac >= critical_threshold:
            return "🔴 Critical"
        if frac >= warn_threshold:
            return "🟡 Warning"
        return "🟢 Low"

    summary_df["Severity"] = summary_df["Missing_Percentage (%)"].apply(_severity)

    # ── 3. Missingness-type hint ──────────────────────────────────────────────
    # Heuristic only — labelled explicitly so users don't treat it as statistical proof.
    MNAR_KEYWORDS = (
        "comment", "note", "remark", "optional", "extra",
        "secondary", "alt", "address2", "phone2", "fax",
        "description", "suffix", "middle",
    )

    def _missing_hint(col: str) -> str:
        col_lower = col.lower()
        if any(kw in col_lower for kw in MNAR_KEYWORDS):
            return "Likely MNAR"
        if df[col].dtype in (np.float64, np.float32, np.int64, np.int32):
            return "Possibly MAR"
        return "Unknown"

    summary_df["Missingness_Hint"] = summary_df["Column"].apply(_missing_hint)

    # ── 4. Correlated-missingness detection ───────────────────────────────────
    # Columns whose NA masks are highly correlated (r > 0.70) likely share a cause.
    missing_cols = summary_df["Column"].tolist()
    corr_groups: dict[str, list[str]] = {}

    if len(missing_cols) > 1:
        na_mask     = df[missing_cols].isna().astype(int)
        corr_matrix = na_mask.corr()
        for i, c1 in enumerate(missing_cols):
            partners = [
                c2 for c2 in missing_cols[i + 1:]
                if corr_matrix.loc[c1, c2] > 0.70
            ]
            if partners:
                corr_groups[c1] = partners

    summary_df["Correlated_With"] = summary_df["Column"].map(
        lambda c: corr_groups.get(c, [])
    )

    # ── 5. Print console summary ──────────────────────────────────────────────
    n_critical = (summary_df["Severity"] == "🔴 Critical").sum()
    n_warn     = (summary_df["Severity"] == "🟡 Warning").sum()
    n_low      = (summary_df["Severity"] == "🟢 Low").sum()
    total_cells   = total_rows * total_cols
    total_missing = int(summary_df["NaN_Count"].sum())

    print(f"Dataset Shape  : {total_rows:,} rows × {total_cols} columns  "
          f"({total_cells:,} total cells)")
    print(f"Missing Cells  : {total_missing:,}  "
          f"({total_missing / total_cells * 100:.2f}% of all cells)")
    print(f"Affected Cols  : {len(summary_df)} of {total_cols}  "
          f"[ 🔴 Critical: {n_critical}  🟡 Warning: {n_warn}  🟢 Low: {n_low} ]")

    if corr_groups:
        print("\n🔗 Correlated missingness detected (r > 0.70):")
        for col, partners in corr_groups.items():
            print(f"   {col}  ↔  {', '.join(partners)}")

    print("-" * 50)

    # ── 6. Visualisations ─────────────────────────────────────────────────────
    if show_bar or show_heatmap:
        n_plots = int(show_bar) + int(show_heatmap)
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6), squeeze=False)
        axes = axes.flatten()
        ax_idx = 0

        # ── 6a. Horizontal bar chart ──────────────────────────────────────
        if show_bar:
            ax = axes[ax_idx]; ax_idx += 1

            COLOR_MAP = {
                "🔴 Critical": "#e05c5c",
                "🟡 Warning":  "#f5c542",
                "🟢 Low":      "#4caf82",
            }
            colors = summary_df["Severity"].map(COLOR_MAP)

            # Reversed so the worst offender appears at the top
            ax.barh(
                summary_df["Column"][::-1].values,
                summary_df["Missing_Percentage (%)"][::-1].values,
                color=colors[::-1].values,
                edgecolor="white",
                linewidth=0.5,
            )

            # Inline percentage labels
            for i, (pct, col) in enumerate(
                zip(
                    summary_df["Missing_Percentage (%)"][::-1],
                    summary_df["Column"][::-1],
                )
            ):
                ax.text(
                    pct + 0.4, i, f"{pct:.1f}%",
                    va="center", ha="left", fontsize=8,
                )

            # Threshold reference lines
            ax.axvline(
                warn_threshold * 100, color="#f5c542",
                linestyle="--", linewidth=1.2,
                label=f"Warn {warn_threshold*100:.0f}%",
            )
            ax.axvline(
                critical_threshold * 100, color="#e05c5c",
                linestyle="--", linewidth=1.2,
                label=f"Critical {critical_threshold*100:.0f}%",
            )

            ax.set_xlabel("Missing (%)")
            ax.set_xlim(0, min(summary_df["Missing_Percentage (%)"].max() + 8, 105))
            ax.set_title("Missing Data by Column", fontsize=13, fontweight="bold")
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(axis="x", linestyle=":", alpha=0.5)

        # ── 6b. Heatmap ───────────────────────────────────────────────────
        if show_heatmap:
            ax = axes[ax_idx]; ax_idx += 1
            plot_df = df[missing_cols]
            sampled = False

            if len(plot_df) > heatmap_row_limit:
                plot_df = plot_df.sample(heatmap_row_limit, random_state=42)
                sampled = True

            sns.heatmap(
                plot_df.isna(),
                cmap="viridis",
                cbar=False,
                yticklabels=False,
                ax=ax,
            )

            title_note = f" — {heatmap_row_limit:,}-row sample" if sampled else ""
            ax.set_title(
                f"Missing Data Matrix{title_note}\n(Yellow = Missing · Purple = Present)",
                fontsize=12, fontweight="bold",
            )
            ax.set_xlabel("Columns")
            ax.set_ylabel("Rows")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    return summary_df


def detect_predictive_issues(
    df,
    target_col=None,
    task_type: str = "classification",
    imbalance_threshold: float = 0.10,
    multi_corr_threshold: float = 0.85,
    irrelevant_corr_threshold: float = 0.05,
    near_zero_variance_threshold: float = 0.01,
):
    """
    Scans a DataFrame for predictive modelling roadblocks:
      - Zero / Near-Zero Variance features
      - Multicollinearity between features
      - Class Imbalance (classification) or Target Skew (regression)
      - Irrelevant numeric features (low linear correlation with target)
      - Perfect / near-perfect target leakage

    Parameters
    ----------
    df                          : pd.DataFrame
    target_col                  : str   — column to predict (optional)
    task_type                   : str   — "classification" or "regression"
    imbalance_threshold         : float — minority class fraction below which
                                          imbalance is flagged (default 0.10)
    multi_corr_threshold        : float — |corr| above which a feature pair is
                                          flagged as multicollinear (default 0.85)
    irrelevant_corr_threshold   : float — |corr with target| below which a numeric
                                          feature is flagged as irrelevant (default 0.05)
    near_zero_variance_threshold: float — coefficient of variation below which a
                                          numeric column is flagged as near-constant
                                          (default 0.01)

    Returns
    -------
    report : dict — structured diagnostic summary
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    valid_tasks = {"classification", "regression"}
    if task_type not in valid_tasks:
        raise ValueError(f"task_type must be one of {valid_tasks}, got '{task_type}'.")

    if target_col and target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame columns.")

    print("🎯 Profiling predictive signals and feature relationships...\n")

    report = {
        "Zero_Variance_Columns":       [],
        "Near_Zero_Variance_Columns":  [],   # NEW: almost-constant numeric columns
        "Multicollinear_Pairs":        [],
        "Leaky_Features":              [],   # NEW: suspiciously high target correlation
        "Class_Imbalance":             None,
        "Target_Skew":                 None, # NEW: for regression targets
        "Irrelevant_Numeric_Features": [],
    }

    # ── 1. Zero Variance ──────────────────────────────────────────────────────
    # nunique(dropna=False) catches constant columns of any dtype
    nunique_counts = df.nunique(dropna=False)
    zero_var_cols  = nunique_counts[nunique_counts <= 1].index.tolist()
    report["Zero_Variance_Columns"] = zero_var_cols

    # ── 2. Near-Zero Variance (numeric only) ──────────────────────────────────
    # Coefficient of variation = std / |mean| — scale-invariant measure of spread.
    # Skip columns that are already flagged as zero-variance or are the target.
    numeric_df = df.select_dtypes(include=[np.number])
    nzv_candidates = numeric_df.drop(
        columns=[c for c in zero_var_cols + ([target_col] if target_col else [])
                 if c in numeric_df.columns],
        errors="ignore",
    )
    near_zero_var_cols = []
    for col in nzv_candidates.columns:
        s    = nzv_candidates[col].dropna()
        mean = s.mean()
        if mean == 0 or s.empty:
            continue
        cv = s.std() / abs(mean)
        if cv < near_zero_variance_threshold:
            near_zero_var_cols.append({"Column": col, "CV": round(cv, 6)})

    report["Near_Zero_Variance_Columns"] = near_zero_var_cols

    # ── 3. Build clean feature matrix for correlation work ────────────────────
    all_problematic = zero_var_cols + [r["Column"] for r in near_zero_var_cols]
    drop_from_features = all_problematic + ([target_col] if target_col else [])

    numeric_features = numeric_df.drop(
        columns=[c for c in drop_from_features if c in numeric_df.columns],
        errors="ignore",
    )

    # ── 4. Multicollinearity ──────────────────────────────────────────────────
    if len(numeric_features.columns) > 1:
        corr_matrix = numeric_features.corr().abs()
        upper_tri   = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )
        collinear_pairs = []
        for col in upper_tri.columns:
            flagged = upper_tri.index[upper_tri[col] > multi_corr_threshold].tolist()
            for partner in flagged:
                collinear_pairs.append({
                    "Feature_A":   partner,
                    "Feature_B":   col,
                    "Correlation": round(float(upper_tri.loc[partner, col]), 4),
                })
        collinear_pairs.sort(key=lambda x: x["Correlation"], reverse=True)
        report["Multicollinear_Pairs"] = collinear_pairs

    # ── 5. Target-dependent checks ────────────────────────────────────────────
    if target_col:
        target_series = df[target_col].dropna()

        # ── 5a. Leakage detection ─────────────────────────────────────────
        # Features that are almost perfectly correlated with the target are
        # suspicious — they may be derived from it (data leakage).
        LEAKAGE_THRESHOLD = 0.95
        if pd.api.types.is_numeric_dtype(target_series) and not numeric_features.empty:
            target_corr_all = numeric_features.corrwith(target_series).abs()
            leaky = target_corr_all[target_corr_all >= LEAKAGE_THRESHOLD]
            report["Leaky_Features"] = [
                {"Feature": col, "Target_Correlation": round(float(val), 4)}
                for col, val in leaky.items()
            ]

        # ── 5b. Classification: class imbalance ───────────────────────────
        if task_type == "classification":
            class_freqs    = target_series.value_counts(normalize=True)
            n_classes      = len(class_freqs)
            minority_class = class_freqs.idxmin()
            minority_freq  = float(class_freqs.min())

            # Compute imbalance ratio (majority / minority)
            imbalance_ratio = float(class_freqs.max() / class_freqs.min()) if minority_freq > 0 else np.inf

            report["Class_Imbalance"] = {
                "Is_Imbalanced":      minority_freq < imbalance_threshold,
                "N_Classes":          n_classes,
                "Minority_Class":     minority_class,
                "Minority_Fraction":  round(minority_freq, 4),
                "Imbalance_Ratio":    round(imbalance_ratio, 2),
                "Class_Distribution": class_freqs.round(4).to_dict(),
            }

        # ── 5c. Regression: target skewness ──────────────────────────────
        elif task_type == "regression":
            if pd.api.types.is_numeric_dtype(target_series):
                skew = float(target_series.skew())
                report["Target_Skew"] = {
                    "Skewness":         round(skew, 4),
                    "Is_Highly_Skewed": abs(skew) > 1.0,
                }

        # ── 5d. Irrelevant numeric features ───────────────────────────────
        if pd.api.types.is_numeric_dtype(target_series) and not numeric_features.empty:
            target_corr = numeric_features.corrwith(target_series).abs()

            # Exclude already-flagged leaky features from the irrelevant list
            leaky_set = {r["Feature"] for r in report["Leaky_Features"]}
            irrelevant_mask = (target_corr < irrelevant_corr_threshold) & \
                              (~target_corr.index.isin(leaky_set))

            report["Irrelevant_Numeric_Features"] = sorted(
                [
                    {"Feature": col, "Target_Correlation": round(float(target_corr[col]), 4)}
                    for col in target_corr[irrelevant_mask].index
                ],
                key=lambda x: x["Target_Correlation"],
            )

    else:
        print("ℹ️  No target_col provided — skipping Imbalance, Leakage & Relevance checks.\n")

    # ── 6. Print summary ──────────────────────────────────────────────────────
    print("--- 📡 Predictive Signal Report ---")

    # Zero variance
    print(f"🛑  Zero Variance Columns: {len(report['Zero_Variance_Columns'])}")
    if report["Zero_Variance_Columns"]:
        print(f"     → {report['Zero_Variance_Columns']}")
        print("     → Drop immediately — no information content.\n")

    # Near-zero variance
    print(f"⚠️   Near-Zero Variance Columns (CV < {near_zero_variance_threshold}): "
          f"{len(report['Near_Zero_Variance_Columns'])}")
    if report["Near_Zero_Variance_Columns"]:
        for entry in report["Near_Zero_Variance_Columns"][:5]:
            print(f"     → {entry['Column']} (CV: {entry['CV']})")
        if len(report["Near_Zero_Variance_Columns"]) > 5:
            print(f"     → ... and {len(report['Near_Zero_Variance_Columns']) - 5} more.")
        print("     → Near-constant; consider dropping or binning.\n")

    # Leakage
    if report["Leaky_Features"]:
        print(f"🚨  Potential Data Leakage (|corr| ≥ 0.95 with target): "
              f"{len(report['Leaky_Features'])}")
        for item in report["Leaky_Features"]:
            print(f"     → {item['Feature']} (corr: {item['Target_Correlation']})")
        print("     → Investigate before training — these may be derived from the target.\n")

    # Class imbalance
    if report.get("Class_Imbalance"):
        ci = report["Class_Imbalance"]
        print(f"⚖️   Class Imbalance: {'⚠️ IMBALANCED' if ci['Is_Imbalanced'] else '✅ Balanced'}")
        if ci["Is_Imbalanced"]:
            print(f"     → Minority class '{ci['Minority_Class']}' = "
                  f"{ci['Minority_Fraction']*100:.1f}%  "
                  f"(ratio {ci['Imbalance_Ratio']}:1)")
            print("     → Use SMOTE, class_weight='balanced', or stratified sampling.\n")
        else:
            print()

    # Target skew
    if report.get("Target_Skew"):
        ts = report["Target_Skew"]
        label = "⚠️ Highly Skewed" if ts["Is_Highly_Skewed"] else "✅ Acceptable"
        print(f"📈  Regression Target Skewness: {ts['Skewness']}  [{label}]")
        if ts["Is_Highly_Skewed"]:
            print("     → Apply np.log1p or Box-Cox transform to the target.\n")
        else:
            print()

    # Multicollinearity
    print(f"👯  Multicollinear Pairs (|corr| > {multi_corr_threshold}): "
          f"{len(report['Multicollinear_Pairs'])}")
    if report["Multicollinear_Pairs"]:
        for pair in report["Multicollinear_Pairs"][:5]:
            print(f"     → {pair['Feature_A']} & {pair['Feature_B']} "
                  f"(corr: {pair['Correlation']})")
        if len(report["Multicollinear_Pairs"]) > 5:
            print(f"     → ... and {len(report['Multicollinear_Pairs']) - 5} more.")
        print("     → Drop one feature from each pair, or apply PCA.\n")

    # Irrelevant features
    if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"🗑️   Irrelevant Numeric Features (|corr| < {irrelevant_corr_threshold}): "
              f"{len(report['Irrelevant_Numeric_Features'])}")
        if report["Irrelevant_Numeric_Features"]:
            for item in report["Irrelevant_Numeric_Features"][:5]:
                print(f"     → {item['Feature']} (corr: {item['Target_Correlation']})")
            if len(report["Irrelevant_Numeric_Features"]) > 5:
                print(f"     → ... and {len(report['Irrelevant_Numeric_Features']) - 5} more.")
            print("     → Consider dropping; note that tree models may still extract non-linear value.\n")

    print("-" * 39)

    return report


def detect_leakage_risks(
    df,
    target_col: str,
    time_col: str = None,
    proxy_threshold: float = 0.95,
    index_corr_threshold: float = 0.80,
    name_leakage_patterns: tuple = (
        "target", "label", "outcome", "response",
        "cancel", "churn", "default", "fraud", "status",
        "result", "flag", "indicator",
    ),
):
    """
    Scans a DataFrame for the mathematical and structural fingerprints of
    Data Leakage.

    Checks for:
    - Target Proxies      — features with suspiciously high correlation to the target
    - Name-Based Leakage  — column names that semantically echo the target
    - Time-Series Risk    — unsorted temporal data that causes future-peeking
    - Index Leakage       — row order that encodes the target
    - Post-Aggregation    — features whose names suggest they were computed from the target

    Parameters
    ----------
    df                     : pd.DataFrame
    target_col             : str   — column to predict
    time_col               : str   — datetime column for temporal ordering check
    proxy_threshold        : float — |corr| above which a feature is flagged as a proxy
                                     (default 0.95)
    index_corr_threshold   : float — |corr(row_index, target)| above which row-order
                                     leakage is flagged (default 0.80)
    name_leakage_patterns  : tuple — substrings that suggest a column was derived from
                                     the target (case-insensitive)

    Returns
    -------
    report : dict — structured leakage diagnostic summary
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame.")
    if time_col and time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in DataFrame.")

    print(f"🕵️  Scanning for Data Leakage Risks (Target: '{target_col}')...\n")

    report = {
        "Target_Proxies":          [],
        "Name_Based_Leakage_Risk": [],   # NEW
        "Post_Aggregation_Risk":   [],   # NEW
        "Time_Series_Risk":        False,
        "Time_Sort_Details":       None, # NEW
        "Index_Leakage_Risk":      False,
        "Index_Correlation":       None, # NEW
    }

    target_series = df[target_col].dropna()
    is_binary     = target_series.nunique() == 2
    is_numeric_target = pd.api.types.is_numeric_dtype(target_series)

    # ── 1. Target Proxy Detection ─────────────────────────────────────────────
    # Uses the most appropriate correlation measure for the target type:
    #   - Numeric target         → Pearson on all numeric features
    #   - Binary target          → Point-biserial for continuous features
    # Both are computed; the higher of the two is stored.
    numeric_df    = df.select_dtypes(include=[np.number])
    feature_cols  = [c for c in numeric_df.columns if c != target_col]

    if is_numeric_target and feature_cols:
        target_vec = df[target_col].fillna(df[target_col].median())

        for col in feature_cols:
            col_vec = numeric_df[col].fillna(numeric_df[col].median())

            # Pearson
            pearson_r = abs(col_vec.corr(target_vec))

            # Spearman (catches monotonic but non-linear relationships)
            try:
                spearman_r, _ = spearmanr(col_vec, target_vec)
                spearman_r = abs(spearman_r) if not np.isnan(spearman_r) else 0.0
            except Exception:
                spearman_r = 0.0

            max_corr = max(pearson_r, spearman_r)

            if max_corr >= proxy_threshold:
                report["Target_Proxies"].append({
                    "Feature":       col,
                    "Pearson_Corr":  round(float(pearson_r),   4),
                    "Spearman_Corr": round(float(spearman_r),  4),
                    "Max_Corr":      round(float(max_corr),    4),
                })

        report["Target_Proxies"].sort(key=lambda x: x["Max_Corr"], reverse=True)

    # ── 2. Name-Based Leakage ─────────────────────────────────────────────────
    # Columns whose names semantically echo the target or common leakage patterns.
    # Catches things like "cancel_date" when target is "churn", even if the correlation
    # is not yet visible (e.g., many NaNs masking the signal).
    flagged_proxy_names = {r["Feature"] for r in report["Target_Proxies"]}
    target_lower        = target_col.lower()

    for col in df.columns:
        if col == target_col or col in flagged_proxy_names:
            continue
        col_lower = col.lower()
        # Matches if the column name contains the target name OR any leakage keyword
        matches = [
            pat for pat in name_leakage_patterns
            if pat in col_lower or pat in target_lower and pat in col_lower
        ]
        # Also flag if the column name directly contains the target column name
        if target_lower in col_lower and col_lower != target_lower:
            matches.append(f"(contains target name '{target_col}')")
        if matches:
            report["Name_Based_Leakage_Risk"].append({
                "Column":          col,
                "Matched_Patterns": list(set(matches)),
            })

    # ── 3. Post-Aggregation Risk ──────────────────────────────────────────────
    # Features computed FROM the target after the fact: cumulative totals,
    # rolling means, running counts — all of which encode future knowledge.
    AGGREGATE_KEYWORDS = (
        "cumul", "running", "rolling", "total_so_far", "ytd",
        "lifetime", "historical", "avg_to_date", "cum_",
    )
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower()
        hits = [kw for kw in AGGREGATE_KEYWORDS if kw in col_lower]
        if hits:
            report["Post_Aggregation_Risk"].append({
                "Column":   col,
                "Keywords": hits,
            })

    # ── 4. Time-Series Shuffling Risk ─────────────────────────────────────────
    if time_col:
        raw_col = df[time_col]

        # Parse to datetime if necessary — handle failure gracefully
        if not pd.api.types.is_datetime64_any_dtype(raw_col):
            try:
                time_series = pd.to_datetime(raw_col, errors="coerce")
                n_failed    = time_series.isna().sum() - raw_col.isna().sum()
                if n_failed > 0:
                    print(f"⚠️  {n_failed} values in '{time_col}' could not be parsed as dates.\n")
            except Exception as e:
                print(f"⚠️  Could not parse '{time_col}' as datetime: {e}\n")
                time_series = raw_col
        else:
            time_series = raw_col

        is_sorted        = bool(time_series.is_monotonic_increasing)
        has_duplicates   = bool(time_series.duplicated().any())
        n_null_times     = int(time_series.isna().sum())

        report["Time_Series_Risk"]    = not is_sorted
        report["Time_Sort_Details"]   = {
            "Is_Chronologically_Sorted": is_sorted,
            "Has_Duplicate_Timestamps":  has_duplicates,
            "Null_Timestamp_Count":      n_null_times,
            "Earliest":  str(time_series.min()),
            "Latest":    str(time_series.max()),
        }

    # ── 5. Index / Row-Order Leakage ──────────────────────────────────────────
    # Data sorted by target before export → row index predicts the target.
    # Use Spearman to catch non-linear monotonic sorting (e.g. partially sorted blocks).
    if is_numeric_target:
        row_numbers  = pd.Series(np.arange(len(df)), index=df.index)
        target_reset = df[target_col].reset_index(drop=True)
        row_reset    = row_numbers.reset_index(drop=True)

        try:
            spearman_idx, _ = spearmanr(row_reset, target_reset.fillna(target_reset.median()))
            index_corr      = abs(float(spearman_idx)) if not np.isnan(spearman_idx) else 0.0
        except Exception:
            index_corr = 0.0

        report["Index_Correlation"]  = round(index_corr, 4)
        report["Index_Leakage_Risk"] = index_corr > index_corr_threshold

    # ── 6. Print summary ──────────────────────────────────────────────────────
    print("--- 🚨 Leakage Risk Report ---")

    # Target proxies
    print(f"🦊  Target Proxies (max |corr| ≥ {proxy_threshold}): {len(report['Target_Proxies'])}")
    if report["Target_Proxies"]:
        for p in report["Target_Proxies"]:
            print(f"     → '{p['Feature']}'  Pearson={p['Pearson_Corr']}  "
                  f"Spearman={p['Spearman_Corr']}")
        print("     → Drop these — they give the model the answer key.\n")

    # Name-based leakage
    print(f"🏷️   Name-Based Leakage Risk: {len(report['Name_Based_Leakage_Risk'])}")
    if report["Name_Based_Leakage_Risk"]:
        for entry in report["Name_Based_Leakage_Risk"]:
            print(f"     → '{entry['Column']}'  matched: {entry['Matched_Patterns']}")
        print("     → Investigate whether these columns were derived from or after the target.\n")

    # Post-aggregation
    print(f"📊  Post-Aggregation Risk: {len(report['Post_Aggregation_Risk'])}")
    if report["Post_Aggregation_Risk"]:
        for entry in report["Post_Aggregation_Risk"]:
            print(f"     → '{entry['Column']}'  keywords: {entry['Keywords']}")
        print("     → Cumulative/rolling features may encode future information.\n")

    # Time series
    if time_col and report["Time_Sort_Details"]:
        td = report["Time_Sort_Details"]
        print(f"⏱️   Time-Series Shuffling Risk: "
              f"{'⚠️ FAILED' if report['Time_Series_Risk'] else '✅ PASSED'}")
        print(f"     Sorted: {td['Is_Chronologically_Sorted']}  |  "
              f"Duplicate timestamps: {td['Has_Duplicate_Timestamps']}  |  "
              f"Null timestamps: {td['Null_Timestamp_Count']}")
        if report["Time_Series_Risk"]:
            print(f"     → '{time_col}' is NOT chronologically ordered.")
            print("     → Sort by this column BEFORE splitting. Never use random shuffle.\n")
        else:
            print("     → Data is chronologically sorted. Safe for time-series split.\n")

    # Index leakage
    if report["Index_Correlation"] is not None:
        print(f"🔢  Index/Row-Order Leakage Risk: "
              f"{'⚠️ FAILED' if report['Index_Leakage_Risk'] else '✅ PASSED'}  "
              f"(Spearman r = {report['Index_Correlation']})")
        if report["Index_Leakage_Risk"]:
            print("     → Row numbers are predictive of the target — dataset was pre-sorted.")
            print("     → Shuffle before splitting (unless this is a time-series task).\n")

    print("-" * 32)
    return report
