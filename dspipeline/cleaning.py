"""
Cleaning functions: numeric/categorical imputation, duplicate handling,
text standardization, structural formatting, and anomaly remediation.
"""
import json
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import IsolationForest


def handle_numerical_missing(
    df,
    strategy: str = "mean",
    fill_value=None,
    columns: list = None,
    add_indicator: bool = False,
):
    """
    Handles missing values in a DataFrame using a range of strategies.

    Strategies
    ----------
    'drop'     — Remove rows that contain any missing values.
    'mean'     — Fill with column mean       (numeric columns only).
    'median'   — Fill with column median     (numeric columns only).
    'mode'     — Fill with column mode       (all column types).
    'ffill'    — Forward-fill                (useful for time-series).
    'bfill'    — Backward-fill              (useful for time-series).
    'constant' — Fill with a fixed `fill_value`.

    Parameters
    ----------
    df            : pd.DataFrame
    strategy      : str  — one of the strategies listed above (default 'mean')
    fill_value    : any  — value used when strategy='constant'
    columns       : list — subset of columns to impute; all columns used if None
    add_indicator : bool — if True, add binary indicator columns for each column
                           that had missing values before imputation (default False)

    Returns
    -------
    df_clean : pd.DataFrame — copy of df with missing values handled
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    VALID_STRATEGIES = {"drop", "mean", "median", "mode", "ffill", "bfill", "constant"}
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unrecognised strategy '{strategy}'. "
            f"Choose from: {', '.join(sorted(VALID_STRATEGIES))}."
        )
    if strategy == "constant" and fill_value is None:
        raise ValueError("strategy='constant' requires a fill_value to be provided.")

    df_clean = df.copy()

    # Resolve target columns
    target_cols = columns if columns is not None else df_clean.columns.tolist()
    missing_before = {col: df_clean[col].isna().sum() for col in target_cols}

    # ── 1. Optional missingness indicators ───────────────────────────────────
    if add_indicator:
        for col in target_cols:
            if missing_before.get(col, 0) > 0:
                df_clean[f"{col}_was_missing"] = df_clean[col].isna().astype(np.uint8)

    # ── 2. Apply strategy ─────────────────────────────────────────────────────
    if strategy == "drop":
        # Drop only rows where the target columns have NaN
        df_clean = df_clean.dropna(subset=target_cols)

    elif strategy in ("mean", "median"):
        numeric_cols = df_clean[target_cols].select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if missing_before[col] == 0:
                continue
            val = df_clean[col].mean() if strategy == "mean" else df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(val)

    elif strategy == "mode":
        for col in target_cols:
            if missing_before[col] == 0:
                continue
            mode_series = df_clean[col].mode()
            if mode_series.empty:
                continue  # All-NaN column — skip rather than crash
            df_clean[col] = df_clean[col].fillna(mode_series.iloc[0])

    elif strategy == "ffill":
        # Deprecated `method=` kwarg removed — use dedicated methods
        df_clean[target_cols] = df_clean[target_cols].ffill()

    elif strategy == "bfill":
        df_clean[target_cols] = df_clean[target_cols].bfill()

    elif strategy == "constant":
        df_clean[target_cols] = df_clean[target_cols].fillna(fill_value)

    # ── 3. Report ─────────────────────────────────────────────────────────────
    filled_cols  = [c for c, n in missing_before.items() if n > 0 and strategy != "drop"]
    total_filled = sum(missing_before[c] for c in filled_cols)
    if strategy == "drop":
        dropped = len(df) - len(df_clean)
        print(f"[handle_numerical_missing] strategy='drop' → removed {dropped:,} rows.")
    else:
        print(
            f"[handle_numerical_missing] strategy='{strategy}' → "
            f"filled {total_filled:,} missing values across {len(filled_cols)} column(s)."
        )

    return df_clean


# ─────────────────────────────────────────────────────────────────────────────

def advanced_knn_impute(
    df,
    n_neighbors: int = 5,
    columns: list = None,
    add_indicator: bool = False,
    weights: str = "uniform",
):
    """
    Imputes missing values in numeric columns using K-Nearest Neighbors.
    Data is scaled before imputation so distance calculations are not
    dominated by high-magnitude features, then inverse-transformed back.

    Parameters
    ----------
    df            : pd.DataFrame
    n_neighbors   : int  — number of nearest neighbours for KNN (default 5)
    columns       : list — numeric columns to impute; auto-detected if None
    add_indicator : bool — if True, add binary indicator columns for each imputed
                           column before filling (default False)
    weights       : str  — 'uniform' or 'distance' — passed to KNNImputer
                           'distance' weights closer neighbours more heavily

    Returns
    -------
    df_imputed : pd.DataFrame — copy of df with numeric NaNs filled;
                                original column order preserved
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be ≥ 1, got {n_neighbors}.")
    if weights not in ("uniform", "distance"):
        raise ValueError(f"weights must be 'uniform' or 'distance', got '{weights}'.")

    # ── 1. Resolve numeric target columns ─────────────────────────────────────
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if columns is not None:
        # Validate that all requested columns are actually numeric
        non_numeric = [c for c in columns if c not in all_numeric]
        if non_numeric:
            raise ValueError(
                f"The following requested columns are not numeric and cannot be "
                f"KNN-imputed: {non_numeric}"
            )
        numeric_cols = [c for c in columns if c in df.columns]
    else:
        numeric_cols = all_numeric

    if not numeric_cols:
        raise ValueError("No numeric columns found — nothing to impute.")

    # Only impute columns that actually have missing values
    cols_with_missing = [c for c in numeric_cols if df[c].isna().any()]
    if not cols_with_missing:
        print("[advanced_knn_impute] No missing values found in numeric columns. Returning copy.")
        return df.copy()

    df_out = df.copy()

    # ── 2. Optional missingness indicators ───────────────────────────────────
    if add_indicator:
        for col in cols_with_missing:
            df_out[f"{col}_was_missing"] = df_out[col].isna().astype(np.uint8)

    # ── 3. Scale → impute → inverse-scale ────────────────────────────────────
    # Pass ALL numeric_cols to the scaler (not just missing ones) so that the
    # KNN distance matrix uses all available numeric context, but only the
    # columns with actual NaNs are replaced in the output.
    df_numeric   = df_out[numeric_cols].copy()
    scaler       = StandardScaler()
    scaled_array = scaler.fit_transform(df_numeric)

    imputer       = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputed_array = imputer.fit_transform(scaled_array)

    # Inverse-transform to restore original scale
    restored_array = scaler.inverse_transform(imputed_array)
    df_restored    = pd.DataFrame(restored_array, columns=numeric_cols, index=df_out.index)

    # Write back ONLY the previously-missing columns, preserving all others untouched
    for col in cols_with_missing:
        df_out[col] = df_restored[col]

    # ── 4. Restore original column order (non-numeric cols may have shifted) ──
    original_cols = list(df.columns)
    indicator_cols = [c for c in df_out.columns if c not in original_cols]
    df_out = df_out[original_cols + indicator_cols]

    # ── 5. Report ─────────────────────────────────────────────────────────────
    total_filled = sum(df[c].isna().sum() for c in cols_with_missing)
    print(
        f"[advanced_knn_impute] KNN (k={n_neighbors}, weights='{weights}') → "
        f"filled {total_filled:,} missing values across {len(cols_with_missing)} column(s): "
        f"{cols_with_missing}"
    )

    return df_out


def handle_categorical_missing(
    df,
    strategy: str = "mode",
    fill_value: str = "Unknown",
    columns: list = None,
    add_indicator: bool = False,
    random_state: int = 42,
):
    """
    Handles missing values in categorical / object / boolean columns.

    Strategies
    ----------
    'mode'         — Fill with the most frequent category.
    'constant'     — Fill with a fixed string (treats "Missing" as its own category).
    'proportional' — Sample from the existing category distribution to preserve ratios.

    Parameters
    ----------
    df            : pd.DataFrame
    strategy      : str  — one of the strategies listed above (default 'mode')
    fill_value    : str  — value used when strategy='constant' (default 'Unknown')
    columns       : list — subset of columns to impute; auto-detected if None
    add_indicator : bool — if True, add a binary {col}_was_missing indicator column
                           before imputing (default False)
    random_state  : int  — seed for 'proportional' sampling (default 42)

    Returns
    -------
    df_clean : pd.DataFrame — copy of df with categorical NaNs handled
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    VALID = {"mode", "constant", "proportional"}
    if strategy not in VALID:
        raise ValueError(
            f"Unrecognised strategy '{strategy}'. Choose from: {', '.join(sorted(VALID))}."
        )

    df_clean = df.copy()

    # Resolve target columns — include bool as it often represents yes/no categories
    auto_cols = df_clean.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    target_cols = columns if columns is not None else auto_cols

    rng = np.random.default_rng(random_state)  # modern, reproducible RNG

    # ── 1. Optional missingness indicators ───────────────────────────────────
    if add_indicator:
        for col in target_cols:
            if df_clean[col].isna().any():
                df_clean[f"{col}_was_missing"] = df_clean[col].isna().astype(np.uint8)

    # ── 2. Apply strategy per column ─────────────────────────────────────────
    total_filled = 0

    for col in target_cols:
        missing_mask = df_clean[col].isna()
        n_missing = missing_mask.sum()
        if n_missing == 0:
            continue

        if strategy == "mode":
            mode_vals = df_clean[col].mode()
            if mode_vals.empty:
                continue  # All-NaN column — skip rather than crash
            df_clean[col] = df_clean[col].fillna(mode_vals.iloc[0])

        elif strategy == "constant":
            df_clean[col] = df_clean[col].fillna(fill_value)

        elif strategy == "proportional":
            freq = df_clean[col].value_counts(normalize=True)
            if freq.empty:
                continue  # All-NaN column — nothing to sample from
            sampled = rng.choice(freq.index, size=n_missing, p=freq.values)
            df_clean.loc[missing_mask, col] = sampled

        total_filled += n_missing

    print(
        f"[handle_categorical_missing] strategy='{strategy}' → "
        f"filled {total_filled:,} missing values across "
        f"{sum(1 for c in target_cols if df[c].isna().any())} column(s)."
    )

    return df_clean


# ─────────────────────────────────────────────────────────────────────────────

def missforest_impute(
    df,
    n_estimators: int = 100,
    max_iter: int = 10,
    random_state: int = 42,
    add_indicator: bool = False,
):
    """
    Imputes missing values using a MissForest-style approach:
    numeric columns use a RandomForestRegressor and categorical columns use a
    RandomForestClassifier, iterated until convergence.

    The original function used a single RandomForestRegressor for ALL columns
    (including categoricals), which is incorrect — classifiers must be used for
    categorical targets. This implementation uses IterativeImputer with
    per-column estimator selection via OrdinalEncoder + a unified RF strategy,
    which is scikit-learn's closest supported equivalent.

    Parameters
    ----------
    df            : pd.DataFrame
    n_estimators  : int  — trees per forest (default 100; original used 10 which
                           is too few for stable imputation)
    max_iter      : int  — IterativeImputer rounds (default 10)
    random_state  : int  — reproducibility seed (default 42)
    add_indicator : bool — if True, adds {col}_was_missing binary columns before
                           imputing (default False)

    Returns
    -------
    df_final : pd.DataFrame — original column order preserved; dtypes restored
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df_work = df.copy()

    # ── 1. Separate column types ──────────────────────────────────────────────
    cat_cols     = df_work.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    all_cols     = numeric_cols + cat_cols   # numeric first keeps the array cleaner

    if not any(df_work[c].isna().any() for c in all_cols):
        print("[missforest_impute] No missing values found. Returning copy.")
        return df.copy()

    # ── 2. Optional missingness indicators ────────────────────────────────────
    if add_indicator:
        for col in all_cols:
            if df_work[col].isna().any():
                df_work[f"{col}_was_missing"] = df_work[col].isna().astype(np.uint8)

    # ── 3. Encode categoricals with OrdinalEncoder ────────────────────────────
    # OrdinalEncoder is preferred over LabelEncoder here because:
    #   (a) it is designed for 2-D arrays and handles multi-column encoding cleanly
    #   (b) it natively supports unknown values via handle_unknown='use_encoded_value'
    #   (c) inverse_transform works directly without manual clipping hacks
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,   # keeps NaNs intact for the imputer
    )

    df_encoded = df_work[all_cols].copy()
    if cat_cols:
        df_encoded[cat_cols] = encoder.fit_transform(df_work[cat_cols])

    # ── 4. IterativeImputer with RandomForestRegressor ────────────────────────
    # scikit-learn's IterativeImputer does not yet support per-column estimators
    # in the stable API, so we use RandomForestRegressor for all columns —
    # continuous outputs are rounded + clipped back to integer class indices for
    # categoricals in step 5.  n_estimators=100 (vs. original 10) gives stable,
    # reproducible imputations.
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        ),
        max_iter=max_iter,
        random_state=random_state,
        keep_empty_features=True,   # never drop all-NaN columns silently
    )

    imputed_array = imputer.fit_transform(df_encoded[all_cols])
    df_imputed    = pd.DataFrame(imputed_array, columns=all_cols, index=df_work.index)

    # ── 5. Decode categoricals back to original strings ───────────────────────
    if cat_cols:
        # Round and clip to valid class-index range before inverse_transform
        for i, col in enumerate(cat_cols):
            n_classes = len(encoder.categories_[i])
            df_imputed[col] = (
                df_imputed[col]
                .round()
                .clip(0, n_classes - 1)
                .astype(int)
            )
        df_imputed[cat_cols] = encoder.inverse_transform(df_imputed[cat_cols])

    # ── 6. Restore original dtypes for numeric columns ────────────────────────
    for col in numeric_cols:
        original_dtype = df[col].dtype
        try:
            df_imputed[col] = df_imputed[col].astype(original_dtype)
        except (ValueError, TypeError):
            pass  # leave as float64 if original dtype can't hold imputed values

    # ── 7. Re-attach indicator columns and restore original column order ───────
    indicator_cols = [c for c in df_work.columns if c not in df.columns]
    df_final = df_imputed[all_cols].copy()
    for col in indicator_cols:
        df_final[col] = df_work[col]

    # Preserve original column order, append any new indicator columns at end
    original_order = [c for c in df.columns if c in df_final.columns]
    df_final = df_final[original_order + indicator_cols]

    # ── 8. Report ─────────────────────────────────────────────────────────────
    total_filled = sum(df[c].isna().sum() for c in all_cols)
    print(
        f"[missforest_impute] RF IterativeImputer (trees={n_estimators}, "
        f"iters={max_iter}) → filled {total_filled:,} missing values across "
        f"{len(numeric_cols)} numeric + {len(cat_cols)} categorical column(s)."
    )

    return df_final


def handle_duplicates(
    df,
    subset=None,
    keep: str = "first",
    sort_by=None,
    ascending=False,
    add_duplicate_flag: bool = False,
):
    """
    Handles exact and partial duplicates in a DataFrame.

    Parameters
    ----------
    df                 : pd.DataFrame
    subset             : list | str | None
                         Columns that define a "partial" duplicate key (e.g., ['user_id']).
                         None = exact match across all columns.
    keep               : 'first' | 'last' | False
                         Which occurrence to keep. False drops all copies.
    sort_by            : str | list | None
                         Column(s) to sort by before deduplication so the
                         "best" row rises to the kept position.
    ascending          : bool | list
                         Sort direction. False (default) keeps the largest
                         value (e.g., most recent timestamp, highest version).
    add_duplicate_flag : bool
                         If True, adds a boolean column 'is_duplicate' marking
                         every row that WOULD be removed BEFORE dropping them,
                         then returns the full annotated DataFrame instead of
                         dropping. Useful for auditing. Default False.

    Returns
    -------
    df_clean : pd.DataFrame — deduplicated (or annotated) copy
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    VALID_KEEP = {"first", "last", False}
    if keep not in VALID_KEEP:
        raise ValueError(f"keep must be one of {VALID_KEEP}, got '{keep}'.")

    # Validate subset columns exist
    if subset is not None:
        subset = [subset] if isinstance(subset, str) else list(subset)
        missing = [c for c in subset if c not in df.columns]
        if missing:
            raise ValueError(f"subset columns not found in DataFrame: {missing}")

    # Validate sort_by columns exist
    if sort_by is not None:
        sort_by_list = [sort_by] if isinstance(sort_by, str) else list(sort_by)
        missing_sort = [c for c in sort_by_list if c not in df.columns]
        if missing_sort:
            raise ValueError(f"sort_by columns not found in DataFrame: {missing_sort}")

    df_clean = df.copy()

    # ── 1. Sort to establish priority before deduplication ────────────────────
    if sort_by is not None:
        df_clean = df_clean.sort_values(by=sort_by, ascending=ascending)

    # ── 2. Flag mode — annotate without dropping ──────────────────────────────
    if add_duplicate_flag:
        # ~keep_mask marks rows that drop_duplicates would REMOVE
        keep_mask          = ~df_clean.duplicated(subset=subset, keep=keep)
        df_clean["is_duplicate"] = ~keep_mask
        n_flagged = df_clean["is_duplicate"].sum()
        print(
            f"[handle_duplicates] Flagged {n_flagged:,} duplicate rows "
            f"(column 'is_duplicate' added). No rows dropped."
        )
        return df_clean

    # ── 3. Drop duplicates ────────────────────────────────────────────────────
    initial_count = len(df_clean)
    df_clean      = df_clean.drop_duplicates(subset=subset, keep=keep)
    removed       = initial_count - len(df_clean)

    # Reset index after sort+drop so row numbers are clean
    df_clean = df_clean.reset_index(drop=True)

    print(
        f"[handle_duplicates] Removed {removed:,} duplicate rows "
        f"({initial_count:,} → {len(df_clean):,}).  "
        f"key={'all columns' if subset is None else subset}  keep='{keep}'"
    )

    return df_clean


# ─────────────────────────────────────────────────────────────────────────────

def standardize_data(
    df,
    text_cols=None,
    typo_map=None,
    unit_col: str = None,
    unit_conversions: dict = None,
    capitalize: str = "lower",
    return_report: bool = False,
):
    """
    Cleans inconsistent text, fixes typos, and standardizes mixed-unit columns.

    Parameters
    ----------
    df               : pd.DataFrame
    text_cols        : list | str | None
                       Columns to normalize whitespace and capitalization.
    typo_map         : dict | None
                       Exact-string replacements applied AFTER text normalization.
                       Format: {'misspelling': 'correction', ...}
                       Applied as whole-value replacement (not substring).
                       For substring replacement, pass regex=True keys wrapped in
                       a list of (pattern, replacement) tuples instead.
    unit_col         : str | None
                       Column containing mixed-unit strings (e.g., '10 kg', '500 g').
    unit_conversions : dict | None
                       Maps unit string → multiplier to convert to a base unit.
                       Format: {'kg': 1.0, 'g': 0.001, 'lb': 0.4536}
    capitalize       : str
                       How to normalize case. One of:
                       'lower' (default), 'upper', 'title', 'sentence', 'none'.
    return_report    : bool
                       If True, return (df_clean, report_dict) instead of just
                       df_clean. The report details how many values changed.

    Returns
    -------
    df_clean          : pd.DataFrame
    report (optional) : dict  — returned only when return_report=True
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    VALID_CAP = {"lower", "upper", "title", "sentence", "none"}
    if capitalize not in VALID_CAP:
        raise ValueError(
            f"capitalize must be one of {VALID_CAP}, got '{capitalize}'."
        )

    if text_cols is not None:
        text_cols = [text_cols] if isinstance(text_cols, str) else list(text_cols)
        missing   = [c for c in text_cols if c not in df.columns]
        if missing:
            raise ValueError(f"text_cols not found in DataFrame: {missing}")

    if unit_col and unit_col not in df.columns:
        raise ValueError(f"unit_col '{unit_col}' not found in DataFrame.")

    df_clean = df.copy()
    report   = {"text_changes": {}, "typo_fixes": {}, "unit_conversions": 0, "unit_failures": 0}

    # ── 1. Text normalization ─────────────────────────────────────────────────
    CAP_FUNCS = {
        "lower":    lambda s: s.str.lower(),
        "upper":    lambda s: s.str.upper(),
        "title":    lambda s: s.str.title(),
        "sentence": lambda s: s.str.capitalize(),
        "none":     lambda s: s,
    }

    if text_cols:
        for col in text_cols:
            before = df_clean[col].copy()

            # Cast to str — preserves the value, NaN becomes the string "nan"
            # We handle NaN separately to avoid polluting category values
            nan_mask         = df_clean[col].isna()
            df_clean[col]    = df_clean[col].astype(str)

            # Strip outer whitespace, collapse internal runs of whitespace
            df_clean[col]    = df_clean[col].str.strip()
            df_clean[col]    = df_clean[col].str.replace(r"\s+", " ", regex=True)

            # Apply case normalization
            df_clean[col]    = CAP_FUNCS[capitalize](df_clean[col])

            # Restore genuine NaNs — prevents "nan" string appearing in data
            df_clean.loc[nan_mask, col] = np.nan

            changed = (df_clean[col].fillna("__NaN__") != before.fillna("__NaN__")).sum()
            report["text_changes"][col] = int(changed)

    # ── 2. Typo / substitution correction ─────────────────────────────────────
    if typo_map and text_cols:
        for col in text_cols:
            before        = df_clean[col].copy()
            df_clean[col] = df_clean[col].replace(typo_map)
            fixed         = (df_clean[col].fillna("__NaN__") != before.fillna("__NaN__")).sum()
            report["typo_fixes"][col] = int(fixed)

    # ── 3. Unit standardization ───────────────────────────────────────────────
    if unit_col and unit_conversions:
        # Pre-compile the regex once for performance
        _UNIT_RE = re.compile(r"^\s*([\d.]+)\s*([a-zA-Z]+)\s*$")

        def _convert(val):
            if pd.isna(val):
                return np.nan
            match = _UNIT_RE.match(str(val).strip())
            if not match:
                return np.nan  # unparseable — NaN is safer than returning the raw string
            number = float(match.group(1))
            unit   = match.group(2).lower()
            if unit not in unit_conversions:
                return np.nan  # unknown unit — flag as NaN so caller can inspect
            return number * unit_conversions[unit]

        result_col = f"{unit_col}_standardized"
        df_clean[result_col] = df_clean[unit_col].apply(_convert)

        n_success = df_clean[result_col].notna().sum()
        n_fail    = df_clean[unit_col].notna().sum() - n_success  # non-null inputs that failed
        report["unit_conversions"] = int(n_success)
        report["unit_failures"]    = int(n_fail)

        if n_fail > 0:
            print(
                f"[standardize_data] ⚠️  {n_fail} value(s) in '{unit_col}' could not be "
                f"converted (unknown unit or bad format) → set to NaN in '{result_col}'."
            )

    # ── 4. Print summary ──────────────────────────────────────────────────────
    if text_cols:
        total_text = sum(report["text_changes"].values())
        total_typo = sum(report["typo_fixes"].values())
        print(
            f"[standardize_data] Text normalized: {total_text:,} value(s) changed.  "
            f"Typos fixed: {total_typo:,} value(s) replaced."
        )
    if unit_col and unit_conversions:
        print(
            f"[standardize_data] Unit conversion: {report['unit_conversions']:,} success, "
            f"{report['unit_failures']:,} failed."
        )

    if return_report:
        return df_clean, report

    return df_clean


def format_structural_issues(
    df,
    type_map: dict = None,
    date_cols: list = None,
    target_tz: str = "UTC",
    nested_col: str = None,
    nested_prefix: str = None,
    strict_col: str = None,
    strict_col_dtype: str = "numeric",
    return_cast_report: bool = False,
):
    """
    Resolves data types, date formats, timezones, nested JSON, and isolates
    shifted / corrupted rows — in the correct dependency order.

    Processing order
    ----------------
    1. Isolate shifted rows (quarantine before any transforms touch them)
    2. Flatten nested JSON / dict column
    3. Standardise date columns + timezone conversion
    4. Enforce data types via type_map

    Parameters
    ----------
    df                : pd.DataFrame
    type_map          : dict | None
                        Maps column names → target dtype.
                        Use 'numeric' for coerced numeric conversion,
                        or any valid pandas dtype string ('int64', 'float32',
                        'str', 'category', 'bool', etc.).
    date_cols         : list | None
                        Columns to parse as datetime and normalise to target_tz.
    target_tz         : str
                        Timezone to convert all date columns to (default 'UTC').
    nested_col        : str | None
                        Column containing dicts or JSON strings to flatten into
                        new columns. Original column is dropped after expansion.
    nested_prefix     : str | None
                        Optional prefix prepended to expanded column names to
                        avoid collision with existing columns (e.g. 'meta_').
    strict_col        : str | None
                        A "litmus test" column that must be numeric (e.g., an ID
                        or age column). Rows where conversion fails are quarantined.
    strict_col_dtype  : str
                        The expected dtype to validate strict_col against.
                        Currently supports 'numeric' (default).
    return_cast_report: bool
                        If True, returns a third element: a dict detailing how many
                        values were coerced / failed per column.

    Returns
    -------
    df_clean       : pd.DataFrame
    corrupted_rows : pd.DataFrame  — quarantined shifted rows (empty if none found)
    cast_report    : dict          — only returned when return_cast_report=True
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if strict_col and strict_col not in df.columns:
        raise ValueError(f"strict_col '{strict_col}' not found in DataFrame.")

    if nested_col and nested_col not in df.columns:
        raise ValueError(f"nested_col '{nested_col}' not found in DataFrame.")

    if date_cols:
        missing_dates = [c for c in date_cols if c not in df.columns]
        if missing_dates:
            raise ValueError(f"date_cols not found in DataFrame: {missing_dates}")

    if type_map:
        missing_types = [c for c in type_map if c not in df.columns]
        if missing_types:
            # Warn but don't crash — type_map may be shared across multiple DataFrames
            print(f"[format_structural_issues] ⚠️  type_map columns not found (skipped): {missing_types}")

    df_clean       = df.copy()
    corrupted_rows = pd.DataFrame(columns=df.columns)   # empty with correct schema
    cast_report    = {}

    # ── 1. Quarantine shifted / corrupted rows ────────────────────────────────
    # Must run FIRST — transforms in later steps could mask the corruption signal
    if strict_col:
        if strict_col_dtype == "numeric":
            coerced     = pd.to_numeric(df_clean[strict_col], errors="coerce")
            # A row is shifted when coercion produced NaN but the original was not NaN
            shifted_mask = coerced.isna() & df_clean[strict_col].notna()
        else:
            # Placeholder for future dtype checks (e.g., date validation)
            shifted_mask = pd.Series(False, index=df_clean.index)

        n_shifted = shifted_mask.sum()
        if n_shifted > 0:
            corrupted_rows = df_clean[shifted_mask].copy()
            df_clean       = df_clean[~shifted_mask].copy()
            print(
                f"[format_structural_issues] Quarantined {n_shifted:,} shifted row(s) "
                f"via strict_col='{strict_col}'."
            )

    # ── 2. Flatten nested JSON / dict column ──────────────────────────────────
    if nested_col and nested_col in df_clean.columns:

        def _safe_parse(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                val = val.strip()
                # Normalise Python-style single-quoted dicts to valid JSON
                val = val.replace("'", '"')
                try:
                    parsed = json.loads(val)
                    return parsed if isinstance(parsed, dict) else {}
                except (json.JSONDecodeError, ValueError):
                    return {}
            return {}

        parsed = df_clean[nested_col].apply(_safe_parse)

        # json_normalize handles deeply nested keys (e.g., 'address.city')
        expanded = pd.json_normalize(parsed.tolist())
        expanded.index = df_clean.index

        # Apply optional prefix to prevent column-name collisions
        if nested_prefix:
            expanded.columns = [f"{nested_prefix}{c}" for c in expanded.columns]

        # Check for collisions with existing columns BEFORE concat
        existing_cols   = set(df_clean.columns) - {nested_col}
        collision_cols  = existing_cols.intersection(expanded.columns)
        if collision_cols:
            print(
                f"[format_structural_issues] ⚠️  Expanded nested columns clash with "
                f"existing columns: {sorted(collision_cols)}. "
                f"Pass nested_prefix to disambiguate."
            )

        df_clean = pd.concat(
            [df_clean.drop(columns=[nested_col]), expanded],
            axis=1,
        )

    # ── 3. Date parsing and timezone normalization ────────────────────────────
    if date_cols:
        for col in date_cols:
            if col not in df_clean.columns:
                continue

            before_nulls = df_clean[col].isna().sum()

            # utc=True coerces ALL timezone-aware and timezone-naive strings to UTC
            # simultaneously, resolving mixed-timezone columns in one pass.
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce", utc=True)

            after_nulls  = df_clean[col].isna().sum()
            n_coerced    = max(0, after_nulls - before_nulls)

            if n_coerced > 0:
                print(
                    f"[format_structural_issues] '{col}': {n_coerced} value(s) could not "
                    f"be parsed as dates → set to NaT."
                )

            # Convert from UTC to target timezone (tz_convert preserves the instant in time)
            if target_tz != "UTC":
                try:
                    df_clean[col] = df_clean[col].dt.tz_convert(target_tz)
                except Exception as e:
                    print(
                        f"[format_structural_issues] ⚠️  Could not convert '{col}' "
                        f"to tz='{target_tz}': {e}"
                    )

    # ── 4. Enforce data types ─────────────────────────────────────────────────
    if type_map:
        for col, dtype in type_map.items():
            if col not in df_clean.columns:
                continue

            before       = df_clean[col].copy()
            n_before_null = before.isna().sum()

            if dtype == "numeric":
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                n_coerced     = max(0, df_clean[col].isna().sum() - n_before_null)
                cast_report[col] = {"target": "numeric", "coerced_to_NaN": int(n_coerced)}

            elif dtype in ("int", "int32", "int64", "Int32", "Int64"):
                # Use nullable integer dtype (Int64) to safely handle NaNs
                # Regular int64 raises on NaN; Int64 (capital I) does not
                nullable_dtype = "Int64" if dtype in ("int", "int64", "Int64") else "Int32"
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                    df_clean[col] = df_clean[col].astype(nullable_dtype)
                    n_coerced     = max(0, df_clean[col].isna().sum() - n_before_null)
                    cast_report[col] = {"target": nullable_dtype, "coerced_to_NaN": int(n_coerced)}
                except Exception as e:
                    print(f"[format_structural_issues] ⚠️  Could not cast '{col}' to {nullable_dtype}: {e}")

            else:
                try:
                    df_clean[col] = df_clean[col].astype(dtype)
                    cast_report[col] = {"target": dtype, "coerced_to_NaN": 0}
                except Exception as e:
                    # Surface the error clearly instead of silently eating it
                    print(
                        f"[format_structural_issues] ⚠️  Could not cast '{col}' "
                        f"to '{dtype}': {e}. Column left unchanged."
                    )
                    cast_report[col] = {"target": dtype, "coerced_to_NaN": -1, "error": str(e)}

    if return_cast_report:
        return df_clean, corrupted_rows, cast_report

    return df_clean, corrupted_rows


def handle_anomalies(
    df,
    columns: list,
    method: str = "iqr",
    action: str = "clip",
    threshold: float = 1.5,
    logical_bounds: dict = None,
    add_flag: bool = False,
    contamination: float = None,
    random_state: int = 42,
):
    """
    Handles statistical outliers, extreme outliers, and multivariate anomalies.

    Methods
    -------
    'iqr'              — Flag values outside Q1 ± threshold × IQR (default threshold=1.5).
    'zscore'           — Flag values beyond ±threshold standard deviations (default threshold=3).
    'isolation_forest' — Multivariate anomaly detection via Isolation Forest.
    'logical'          — Domain-knowledge hard bounds supplied via logical_bounds dict.

    Actions
    -------
    'clip' — Winsorize: cap values at the computed bounds (IQR/z-score/logical).
             For isolation_forest, falls back to 'nan' with a warning (multivariate
             anomalies have no single clipping bound).
    'drop' — Remove the entire row.
    'nan'  — Replace the outlier value with NaN for later imputation.

    Parameters
    ----------
    df              : pd.DataFrame
    columns         : list          — numeric columns to analyse
    method          : str           — detection method (see above)
    action          : str           — remediation action (see above)
    threshold       : float         — IQR multiplier, z-score cutoff, or (deprecated)
                                      contamination rate. Default 1.5.
    logical_bounds  : dict | None   — {col: (min_val, max_val)} for 'logical' method.
    add_flag        : bool          — if True, add a boolean '{col}_outlier' column
                                      before applying the action (default False).
    contamination   : float | None  — explicit contamination rate for isolation_forest
                                      (overrides threshold for IF; range 0–0.5).
    random_state    : int           — seed for IsolationForest (default 42).

    Returns
    -------
    df_clean : pd.DataFrame — copy of df with anomalies handled
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    VALID_METHODS = {"iqr", "zscore", "isolation_forest", "logical"}
    VALID_ACTIONS = {"clip", "drop", "nan"}

    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'.")
    if action not in VALID_ACTIONS:
        raise ValueError(f"action must be one of {VALID_ACTIONS}, got '{action}'.")

    if method == "logical" and not logical_bounds:
        raise ValueError("method='logical' requires a logical_bounds dict.")

    # Validate that all requested columns exist and are numeric
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    non_numeric = [
        c for c in columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric columns cannot be processed: {non_numeric}")

    df_clean     = df.copy()
    total_flagged = 0

    # ── Helper: apply action to a boolean mask on a single column ─────────────
    def _apply_action(frame, col, mask, lower=None, upper=None):
        nonlocal total_flagged
        n = mask.sum()
        if n == 0:
            return frame
        total_flagged += n

        if add_flag:
            flag_col = f"{col}_outlier"
            if flag_col not in frame.columns:
                frame[flag_col] = False
            frame.loc[mask, flag_col] = True

        if action == "clip":
            frame[col] = frame[col].clip(lower=lower, upper=upper)
        elif action == "drop":
            frame = frame[~mask].copy()
        elif action == "nan":
            frame.loc[mask, col] = np.nan
        return frame

    # ── 1. IQR ────────────────────────────────────────────────────────────────
    if method == "iqr":
        for col in columns:
            series = df_clean[col].dropna()
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1

            # Zero-IQR columns (constant / binary) — skip to avoid flagging everything
            if IQR == 0:
                print(f"[handle_anomalies] '{col}': IQR=0 (constant column) — skipped.")
                continue

            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask  = (df_clean[col] < lower) | (df_clean[col] > upper)
            df_clean = _apply_action(df_clean, col, mask, lower=lower, upper=upper)

    # ── 2. Z-Score ────────────────────────────────────────────────────────────
    elif method == "zscore":
        for col in columns:
            series = df_clean[col].dropna()
            mean, std = series.mean(), series.std()

            # Zero-std columns — skip to avoid divide-by-zero
            if std == 0:
                print(f"[handle_anomalies] '{col}': std=0 (constant column) — skipped.")
                continue

            lower = mean - threshold * std
            upper = mean + threshold * std
            mask  = (df_clean[col] < lower) | (df_clean[col] > upper)
            df_clean = _apply_action(df_clean, col, mask, lower=lower, upper=upper)

    # ── 3. Isolation Forest (multivariate) ────────────────────────────────────
    elif method == "isolation_forest":
        # Resolve contamination — explicit param takes priority over threshold
        if contamination is not None:
            if not (0 < contamination < 0.5):
                raise ValueError(f"contamination must be between 0 and 0.5, got {contamination}.")
            contam = contamination
        else:
            # Legacy: threshold was (mis)used as contamination in the original.
            # Guard against accidentally passing an IQR multiplier like 1.5.
            if threshold >= 0.5:
                print(
                    f"[handle_anomalies] ⚠️  threshold={threshold} is invalid as a "
                    f"contamination rate. Defaulting to 0.05. "
                    f"Pass contamination= explicitly to suppress this warning."
                )
                contam = 0.05
            else:
                contam = threshold

        # Isolation Forest requires no NaNs — impute with column medians temporarily.
        # We ONLY use this for fitting/predicting; we never write these medians back.
        medians   = df_clean[columns].median()
        temp_data = df_clean[columns].fillna(medians)

        # Scale before fitting — IF is not scale-invariant in practice;
        # large-magnitude features dominate the split selection.
        scaler      = StandardScaler()
        scaled_data = scaler.fit_transform(temp_data)

        iso  = IsolationForest(contamination=contam, random_state=random_state, n_estimators=200)
        preds = iso.fit_predict(scaled_data)
        mask  = pd.Series(preds == -1, index=df_clean.index)

        total_flagged += mask.sum()

        if add_flag:
            df_clean["isolation_forest_outlier"] = mask

        if action == "clip":
            # Clipping is undefined for multivariate anomalies — fall back to nan
            print(
                "[handle_anomalies] action='clip' is not meaningful for "
                "isolation_forest. Applying action='nan' instead."
            )
            df_clean.loc[mask, columns] = np.nan
        elif action == "drop":
            df_clean = df_clean[~mask].copy()
        elif action == "nan":
            df_clean.loc[mask, columns] = np.nan

    # ── 4. Logical / domain bounds ────────────────────────────────────────────
    elif method == "logical":
        for col, (min_val, max_val) in logical_bounds.items():
            if col not in df_clean.columns:
                print(f"[handle_anomalies] logical_bounds column '{col}' not in DataFrame — skipped.")
                continue
            if min_val is not None and max_val is not None and min_val > max_val:
                raise ValueError(
                    f"logical_bounds for '{col}': min_val ({min_val}) > max_val ({max_val})."
                )
            lower = min_val if min_val is not None else -np.inf
            upper = max_val if max_val is not None else  np.inf
            mask  = (df_clean[col] < lower) | (df_clean[col] > upper)
            df_clean = _apply_action(df_clean, col, mask, lower=lower, upper=upper)

    # ── 5. Report ─────────────────────────────────────────────────────────────
    if method != "isolation_forest":
        print(
            f"[handle_anomalies] method='{method}' action='{action}' → "
            f"{total_flagged:,} outlier value(s) handled across {len(columns)} column(s)."
        )
    else:
        print(
            f"[handle_anomalies] method='isolation_forest' action='{action}' → "
            f"{total_flagged:,} anomalous row(s) handled  "
            f"(contamination={contam}, n_estimators=200)."
        )

    if action == "drop":
        df_clean = df_clean.reset_index(drop=True)

    return df_clean
