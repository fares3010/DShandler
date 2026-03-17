"""
Preprocessing functions: scaling, skew correction, sparse conversion,
categorical encoding, feature selection, VIF optimisation, and train/test splitting.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor


def transform_data_shape(
    df,
    scale_cols: list = None,
    scale_method: str = "standard",
    skew_cols: list = None,
    skew_method: str = "yeo-johnson",
    sparse_cols: list = None,
    return_transformers: bool = False,
):
    """
    Handles vastly different scales, highly skewed distributions, and
    memory-heavy sparse columns.

    Processing order
    ----------------
    1. Skew correction   — applied BEFORE scaling so the scaler sees
                           the already-normalized distribution.
    2. Scaling           — applied to scale_cols (which may overlap skew_cols).
    3. Sparse conversion — purely a memory optimization; done last.

    Parameters
    ----------
    df                  : pd.DataFrame
    scale_cols          : list | None
                          Numeric columns to scale.
    scale_method        : str
                          'standard'  — zero mean, unit variance (StandardScaler).
                          'minmax'    — rescale to [0, 1] (MinMaxScaler).
                          'maxabs'    — divide by max absolute value; preserves sparsity
                                        (MaxAbsScaler).
                          'robust'    — uses median + IQR; best when outliers remain
                                        after anomaly handling (RobustScaler).
    skew_cols           : list | None
                          Highly skewed numeric columns to normalize.
    skew_method         : str
                          'log'          — np.log1p; safe for zero values, requires
                                           all values ≥ 0.
                          'sqrt'         — np.sqrt; milder than log; requires all ≥ 0.
                          'yeo-johnson'  — PowerTransformer; handles negatives and zeros.
                          'box-cox'      — PowerTransformer; strictly positive data only.
    sparse_cols         : list | None
                          Columns with mostly zeros to convert to pd.SparseDtype.
    return_transformers : bool
                          If True, return (df_clean, transformers_dict) where
                          transformers_dict holds fitted sklearn objects keyed by
                          column/method, so they can be reused on test data.

    Returns
    -------
    df_clean             : pd.DataFrame
    transformers (opt.)  : dict  — only when return_transformers=True
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    VALID_SCALE = {"standard", "minmax", "maxabs", "robust"}
    VALID_SKEW  = {"log", "sqrt", "yeo-johnson", "box-cox"}

    if scale_cols and scale_method not in VALID_SCALE:
        raise ValueError(f"scale_method must be one of {VALID_SCALE}, got '{scale_method}'.")
    if skew_cols and skew_method not in VALID_SKEW:
        raise ValueError(f"skew_method must be one of {VALID_SKEW}, got '{skew_method}'.")

    def _validate_cols(cols, label):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{label} columns not found in DataFrame: {missing}")
        non_num = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_num:
            raise ValueError(f"{label} contains non-numeric columns: {non_num}")

    if scale_cols:
        _validate_cols(scale_cols, "scale_cols")
    if skew_cols:
        _validate_cols(skew_cols, "skew_cols")
    if sparse_cols:
        missing = [c for c in sparse_cols if c not in df.columns]
        if missing:
            raise ValueError(f"sparse_cols not found in DataFrame: {missing}")

    df_clean     = df.copy()
    transformers = {}

    # ── 1. Skew Correction ────────────────────────────────────────────────────
    # Applied BEFORE scaling so the scaler operates on the corrected distribution.
    if skew_cols:
        for col in skew_cols:
            series = df_clean[col].dropna()

            if skew_method == "log":
                if (series < 0).any():
                    raise ValueError(
                        f"skew_method='log' requires all values ≥ 0. "
                        f"Column '{col}' contains negative values. "
                        f"Use skew_method='yeo-johnson' instead."
                    )
                df_clean[col] = np.log1p(df_clean[col])
                transformers[f"{col}_skew"] = "log1p"

            elif skew_method == "sqrt":
                if (series < 0).any():
                    raise ValueError(
                        f"skew_method='sqrt' requires all values ≥ 0. "
                        f"Column '{col}' contains negative values."
                    )
                df_clean[col] = np.sqrt(df_clean[col])
                transformers[f"{col}_skew"] = "sqrt"

            elif skew_method in ("yeo-johnson", "box-cox"):
                if skew_method == "box-cox" and (series <= 0).any():
                    raise ValueError(
                        f"skew_method='box-cox' requires all values > 0. "
                        f"Column '{col}' contains zeros or negatives. "
                        f"Use 'yeo-johnson' instead."
                    )
                pt = PowerTransformer(method=skew_method, standardize=False)
                # fit_transform requires 2-D input; reshape single column
                transformed = pt.fit_transform(df_clean[[col]])
                df_clean[col] = transformed.ravel()
                transformers[f"{col}_skew"] = pt

    # ── 2. Scaling ────────────────────────────────────────────────────────────
    if scale_cols:
        SCALER_MAP = {
            "standard": StandardScaler(),
            "minmax":   MinMaxScaler(),
            "maxabs":   MaxAbsScaler(),
            "robust":   RobustScaler(),
        }
        scaler = SCALER_MAP[scale_method]

        # Fit only on rows with no NaN in any scale_col to avoid silent median-imputation
        # by sklearn — the caller is responsible for imputing before scaling.
        complete_mask = df_clean[scale_cols].notna().all(axis=1)
        n_incomplete  = (~complete_mask).sum()
        if n_incomplete > 0:
            print(
                f"[transform_data_shape] ⚠️  {n_incomplete} row(s) have NaN in scale_cols "
                f"and will remain NaN after scaling. Impute before calling this function."
            )

        # Fit on complete rows, transform all rows (NaN rows pass through as NaN)
        scaler.fit(df_clean.loc[complete_mask, scale_cols])
        scaled_vals = scaler.transform(df_clean[scale_cols])
        df_clean[scale_cols] = scaled_vals
        transformers["scaler"] = scaler

    # ── 3. Sparse Conversion ──────────────────────────────────────────────────
    if sparse_cols:
        for col in sparse_cols:
            # Fill NaN with 0 before converting — NaN in sparse arrays creates
            # ambiguity between "missing" and "zero" that downstream code rarely handles
            n_nan = df_clean[col].isna().sum()
            if n_nan > 0:
                print(
                    f"[transform_data_shape] '{col}': {n_nan} NaN(s) filled with 0 "
                    f"before sparse conversion."
                )
            df_clean[col] = df_clean[col].fillna(0)

            # Determine the actual numeric dtype of the column before converting
            base_dtype = df_clean[col].dtype if df_clean[col].dtype != object else float
            df_clean[col] = df_clean[col].astype(
                pd.SparseDtype(base_dtype, fill_value=0)
            )

            zero_frac = (df_clean[col] == 0).sum() / len(df_clean)
            print(
                f"[transform_data_shape] '{col}' → SparseDtype "
                f"({zero_frac*100:.1f}% zeros compressed)."
            )

    # ── 4. Report ─────────────────────────────────────────────────────────────
    parts = []
    if skew_cols:
        parts.append(f"{len(skew_cols)} skew-corrected ({skew_method})")
    if scale_cols:
        parts.append(f"{len(scale_cols)} scaled ({scale_method})")
    if sparse_cols:
        parts.append(f"{len(sparse_cols)} sparsified")
    if parts:
        print(f"[transform_data_shape] Done — {', '.join(parts)}.")

    if return_transformers:
        return df_clean, transformers

    return df_clean


def encode_categorical_data(
    df,
    nominal_cols: list = None,
    ordinal_maps: dict = None,
    high_card_cols: list = None,
    target_col: str = None,
    rare_threshold: float = 0.01,
    drop_first: bool = True,
    target_smoothing: float = 0.0,
    return_encoding_report: bool = False,
):
    """
    Transforms categorical columns into model-ready numbers safely.

    Processing order
    ----------------
    1. Rare-category grouping  — collapse infrequent labels to 'Other_Rare'
    2. Ordinal encoding        — integer mapping that preserves rank
    3. Target encoding         — mean-target substitution for high-cardinality cols
    4. One-hot encoding        — binary expansion for low-cardinality nominal cols

    Parameters
    ----------
    df                   : pd.DataFrame
    nominal_cols         : list | None
                           Low-cardinality columns with no natural order (e.g., 'Color').
                           Encoded via One-Hot Encoding.
    ordinal_maps         : dict | None
                           Ordered mapping for ordinal columns.
                           Format: {'Size': ['Small', 'Medium', 'Large']}
    high_card_cols       : list | None
                           High-cardinality columns (e.g., 'ZipCode').
                           Encoded via Target Encoding (requires target_col).
    target_col           : str | None
                           Column to predict. Required for target encoding.
    rare_threshold       : float
                           Categories below this frequency fraction are grouped
                           into 'Other_Rare' (default 0.01 = 1 %).
    drop_first           : bool
                           Drop the first dummy column in OHE to avoid the
                           dummy-variable trap (default True).
    target_smoothing     : float
                           Smoothing strength for target encoding.
                           0.0 = no smoothing (original behaviour).
                           Positive values (e.g., 10.0) blend the category mean
                           toward the global mean, which reduces overfitting on
                           rare categories: smoothed = (n × cat_mean + k × global_mean) / (n + k).
    return_encoding_report: bool
                           If True, return (df_encoded, report_dict) where report_dict
                           documents what was done to each column.

    Returns
    -------
    df_encoded       : pd.DataFrame
    report (optional): dict  — only when return_encoding_report=True
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    all_cat_cols = (
        list(nominal_cols or [])
        + list((ordinal_maps or {}).keys())
        + list(high_card_cols or [])
    )

    missing_cols = [c for c in all_cat_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    if high_card_cols and not target_col:
        raise ValueError(
            "target_col must be provided when high_card_cols is specified "
            "(target encoding requires a target)."
        )
    if target_col and target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame.")

    df_encoded = df.copy()
    report: dict = {}

    # ── 1. Rare-category grouping ─────────────────────────────────────────────
    # Runs on ALL categorical columns before any encoding step so that every
    # encoder downstream sees a clean, consolidated vocabulary.
    for col in all_cat_cols:
        if col not in df_encoded.columns:
            continue
        freqs     = df_encoded[col].value_counts(normalize=True)
        rare_cats = freqs[freqs < rare_threshold].index.tolist()
        if rare_cats:
            df_encoded[col]  = df_encoded[col].replace(rare_cats, "Other_Rare")
            report[col]      = report.get(col, {})
            report[col]["rare_grouped"] = rare_cats

    # ── 2. Ordinal encoding ───────────────────────────────────────────────────
    if ordinal_maps:
        for col, ordered_list in ordinal_maps.items():
            ordered_list = list(ordered_list)  # defensive copy

            # Ensure 'Other_Rare' has a position (lowest rank = 0)
            if "Other_Rare" not in ordered_list:
                ordered_list = ["Other_Rare"] + ordered_list

            # Detect any values present in the column that are NOT in the map
            unique_vals    = set(df_encoded[col].dropna().unique())
            unmapped_vals  = unique_vals - set(ordered_list)
            if unmapped_vals:
                print(
                    f"[encode_categorical_data] '{col}': {len(unmapped_vals)} value(s) "
                    f"not in ordinal_maps → mapped to 0 ('Other_Rare'): {unmapped_vals}"
                )

            mapping        = {cat: i for i, cat in enumerate(ordered_list)}
            df_encoded[col] = df_encoded[col].map(mapping)
            # Unmapped values become NaN after map(); fill with 0 = 'Other_Rare'
            df_encoded[col] = df_encoded[col].fillna(0).astype(int)

            report[col]              = report.get(col, {})
            report[col]["encoding"]  = "ordinal"
            report[col]["mapping"]   = mapping

    # ── 3. Target encoding (high cardinality) ─────────────────────────────────
    # Target encoding is computed BEFORE One-Hot Encoding because OHE expands
    # the DataFrame and can push the high-card text columns out of alignment.
    if high_card_cols and target_col:
        global_mean = df_encoded[target_col].mean()

        for col in high_card_cols:
            if col not in df_encoded.columns:
                continue

            # Compute per-category stats
            stats = df_encoded.groupby(col)[target_col].agg(["mean", "count"])

            if target_smoothing > 0:
                # Smoothed target encoding: blends category mean toward global mean
                # based on category size. Prevents overfitting on rare categories.
                k = target_smoothing
                stats["smoothed"] = (
                    (stats["count"] * stats["mean"] + k * global_mean)
                    / (stats["count"] + k)
                )
                target_map = stats["smoothed"]
            else:
                target_map = stats["mean"]

            encoded_col = f"{col}_te"
            df_encoded[encoded_col] = df_encoded[col].map(target_map)

            # Unseen categories (NaN after map) fall back to global mean
            n_unseen = df_encoded[encoded_col].isna().sum()
            if n_unseen > 0:
                df_encoded[encoded_col] = df_encoded[encoded_col].fillna(global_mean)

            # Drop original high-cardinality text column
            df_encoded = df_encoded.drop(columns=[col])

            report[col] = report.get(col, {})
            report[col]["encoding"]    = "target"
            report[col]["output_col"]  = encoded_col
            report[col]["global_mean"] = round(float(global_mean), 6)
            report[col]["unseen_filled"] = int(n_unseen)
            report[col]["smoothing"]   = target_smoothing

    # ── 4. One-Hot Encoding (nominal / low cardinality) ───────────────────────
    if nominal_cols:
        # Only encode columns that still exist (they may have been dropped upstream)
        present_nominal = [c for c in nominal_cols if c in df_encoded.columns]
        if present_nominal:
            before_cols = set(df_encoded.columns)
            df_encoded  = pd.get_dummies(
                df_encoded,
                columns=present_nominal,
                drop_first=drop_first,
                dtype=np.uint8,   # saves memory vs default bool/int64
            )
            new_cols = set(df_encoded.columns) - before_cols
            for col in present_nominal:
                report[col]              = report.get(col, {})
                report[col]["encoding"]  = "one-hot"
                report[col]["drop_first"] = drop_first
                report[col]["new_columns"] = sorted(
                    [c for c in new_cols if c.startswith(col)]
                )

    # ── 5. Print summary ──────────────────────────────────────────────────────
    ordinal_done  = [c for c in (ordinal_maps or {}) if report.get(c, {}).get("encoding") == "ordinal"]
    target_done   = [c for c in (high_card_cols or []) if report.get(c, {}).get("encoding") == "target"]
    ohe_done      = [c for c in (nominal_cols or []) if report.get(c, {}).get("encoding") == "one-hot"]
    rare_done     = [c for c, v in report.items() if "rare_grouped" in v]

    print(
        f"[encode_categorical_data] "
        f"Rare-grouped: {len(rare_done)} col(s)  |  "
        f"Ordinal: {len(ordinal_done)}  |  "
        f"Target-encoded: {len(target_done)}  |  "
        f"One-Hot: {len(ohe_done)}  |  "
        f"Output shape: {df_encoded.shape}"
    )

    if return_encoding_report:
        return df_encoded, report

    return df_encoded


def optimize_features(
    df,
    target_col: str = None,
    var_threshold: float = 0.0,
    multi_corr_threshold: float = 0.85,
    target_corr_threshold: float = 0.05,
    protected_cols: list = None,
    return_report: bool = False,
):
    """
    Cleans up the feature space by removing zero-variance, irrelevant,
    and highly multicollinear columns — in the correct dependency order.

    Processing order
    ----------------
    1. Zero / near-zero variance  — drop constants first (cheapest check)
    2. Target irrelevance         — drop weak features before collinearity
    3. Multicollinearity          — drop the WEAKER of each correlated pair
                                    (the original always dropped the second
                                    column alphabetically, regardless of which
                                    had lower target correlation)

    Parameters
    ----------
    df                     : pd.DataFrame
    target_col             : str | None  — column to predict
    var_threshold          : float       — variance at or below this is dropped
                                           (0.0 = strictly constant only)
    multi_corr_threshold   : float       — |corr| above this triggers collinearity
                                           drop (default 0.85)
    target_corr_threshold  : float       — |corr with target| below this drops the
                                           feature (default 0.05); ignored if no
                                           target_col is provided
    protected_cols         : list | None — columns that must never be dropped
                                           (e.g., IDs kept for joining later)
    return_report          : bool        — if True, return (df_clean, report_dict)

    Returns
    -------
    df_clean        : pd.DataFrame
    report (opt.)   : dict  — only when return_report=True
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if target_col and target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame.")

    protected = set(protected_cols or [])
    if target_col:
        protected.add(target_col)

    invalid_protected = protected - set(df.columns)
    if invalid_protected:
        raise ValueError(f"protected_cols not found in DataFrame: {invalid_protected}")

    df_clean     = df.copy()
    numeric_cols = (
        df_clean
        .select_dtypes(include=[np.number])
        .columns
        .difference(list(protected))   # never consider protected cols for dropping
        .tolist()
    )

    report = {
        "initial_numeric_features": len(numeric_cols),
        "low_variance_dropped":     [],
        "irrelevant_dropped":       [],
        "multicollinear_dropped":   [],
        "retained_features":        [],
    }

    print(f"[optimize_features] Starting with {len(numeric_cols)} numeric feature(s).")

    # ── 1. Zero / Near-Zero Variance ──────────────────────────────────────────
    # Variance is computed on non-null values to avoid deflation from NaNs.
    variances    = df_clean[numeric_cols].var(ddof=1)   # sample variance, consistent with sklearn
    low_var_cols = variances[variances <= var_threshold].index.tolist()

    if low_var_cols:
        df_clean     = df_clean.drop(columns=low_var_cols)
        numeric_cols = [c for c in numeric_cols if c not in low_var_cols]
        report["low_variance_dropped"] = low_var_cols
        print(
            f"  Dropped {len(low_var_cols)} low/zero-variance feature(s): {low_var_cols}"
        )

    # ── 2. Target Irrelevance ─────────────────────────────────────────────────
    # Uses Pearson correlation — only a linear signal check.  Non-linear features
    # may still be valuable to tree-based models; the threshold should be kept low.
    # Skip if the target is non-numeric (e.g. string labels like 'Yes'/'No') —
    # corrwith would try to cast strings to float and crash.
    _target_is_numeric = (
        target_col
        and target_col in df_clean.columns
        and pd.api.types.is_numeric_dtype(df_clean[target_col])
    )
    if _target_is_numeric and numeric_cols:
        target_corrs = df_clean[numeric_cols].corrwith(df_clean[target_col]).abs()
        irrelevant   = target_corrs[target_corrs < target_corr_threshold].index.tolist()

        if irrelevant:
            df_clean     = df_clean.drop(columns=irrelevant)
            numeric_cols = [c for c in numeric_cols if c not in irrelevant]
            report["irrelevant_dropped"] = irrelevant
            print(
                f"  Dropped {len(irrelevant)} irrelevant feature(s) "
                f"(|corr with target| < {target_corr_threshold}): {irrelevant}"
            )
    elif target_col and not _target_is_numeric:
        print(
            f"  Irrelevance check skipped — target '{target_col}' is non-numeric "
            f"(dtype: {df_clean[target_col].dtype}). "
            f"Encode the target to 0/1 before running if you want correlation-based pruning."
        )

    # ── 3. Multicollinearity ──────────────────────────────────────────────────
    # FIX: the original always dropped the SECOND column (alphabetically) from
    # each correlated pair.  The correct strategy is to drop the column with the
    # LOWER correlation to the target, because it carries less unique signal.
    # When no target is available, fall back to dropping the second column.
    if len(numeric_cols) > 1:
        corr_matrix   = df_clean[numeric_cols].corr().abs()
        upper_tri     = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )

        # Pre-compute target correlations for tie-breaking
        # Only possible when the target is numeric; fall back to 0 otherwise.
        if _target_is_numeric and target_col in df_clean.columns:
            target_corrs = df_clean[numeric_cols].corrwith(df_clean[target_col]).abs()
        else:
            target_corrs = pd.Series(0.0, index=numeric_cols)

        to_drop: set = set()

        # Iterate over all pairs that exceed the threshold
        for col_b in upper_tri.columns:
            for col_a in upper_tri.index:
                if pd.isna(upper_tri.loc[col_a, col_b]):
                    continue
                if upper_tri.loc[col_a, col_b] <= multi_corr_threshold:
                    continue
                # Skip if one of them is already queued for removal
                if col_a in to_drop or col_b in to_drop:
                    continue

                # Drop whichever has the LOWER target correlation
                corr_a = target_corrs.get(col_a, 0.0)
                corr_b = target_corrs.get(col_b, 0.0)
                drop   = col_a if corr_a <= corr_b else col_b
                to_drop.add(drop)

        to_drop_list = sorted(to_drop)
        if to_drop_list:
            df_clean     = df_clean.drop(columns=to_drop_list)
            numeric_cols = [c for c in numeric_cols if c not in to_drop_list]
            report["multicollinear_dropped"] = to_drop_list
            print(
                f"  Dropped {len(to_drop_list)} multicollinear feature(s) "
                f"(|corr| > {multi_corr_threshold}): {to_drop_list}"
            )

    # ── 4. Summary ────────────────────────────────────────────────────────────
    retained  = [
        c for c in df_clean.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df_clean[c])
    ]
    report["retained_features"] = retained

    total_dropped = (
        len(report["low_variance_dropped"])
        + len(report["irrelevant_dropped"])
        + len(report["multicollinear_dropped"])
    )
    print(
        f"[optimize_features] Done — dropped {total_dropped} feature(s) total.  "
        f"{len(df_clean.columns)} column(s) remain "
        f"({len(retained)} numeric features + "
        f"{len(df_clean.columns) - len(retained) - int(bool(target_col and target_col in df_clean.columns))} "
        f"non-numeric + "
        f"{int(bool(target_col and target_col in df_clean.columns))} target)."
    )

    if return_report:
        return df_clean, report

    return df_clean


def optimize_vif(
    df,
    threshold: float = 5.0,
    protected_cols: list = None,
    target_col: str = None,
    max_iterations: int = 500,
    return_vif_history: bool = False,
):
    """
    Iteratively removes features with a Variance Inflation Factor (VIF)
    above the specified threshold to eliminate multicollinearity.

    Each iteration drops only the SINGLE worst offender — this is critical
    because VIF is a joint property: dropping one high-VIF column changes
    the VIF of every other column. Dropping multiple columns per iteration
    (as some implementations do) can remove features that would have been
    fine after the primary collinear column was gone.

    Parameters
    ----------
    df                  : pd.DataFrame
                          Must be numeric and free of NaN / infinite values.
    threshold           : float — maximum acceptable VIF (default 5.0 strict;
                          10.0 is commonly used as a lenient cutoff).
    protected_cols      : list | None
                          Columns that must never be dropped regardless of VIF
                          (e.g., a key predictor, one-hot encoded dummies).
    target_col          : str | None
                          If provided, excluded from VIF computation and never
                          dropped. Convenience alias for adding to protected_cols.
    max_iterations      : int — safety cap to prevent infinite loops on
                          pathological inputs (default 500).
    return_vif_history  : bool — if True, return (df_clean, history) where
                          history is a list of dicts recording each drop decision.

    Returns
    -------
    df_clean       : pd.DataFrame — DataFrame with multicollinear columns removed
    history (opt.) : list[dict]   — only when return_vif_history=True
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if threshold <= 1.0:
        raise ValueError(
            f"threshold must be > 1.0 (VIF = 1 means zero collinearity). Got {threshold}."
        )

    protected = set(protected_cols or [])
    if target_col:
        protected.add(target_col)

    invalid_protected = protected - set(df.columns)
    if invalid_protected:
        raise ValueError(f"protected_cols / target_col not found in DataFrame: {invalid_protected}")

    # ── 1. Isolate numeric feature columns ────────────────────────────────────
    numeric_cols = (
        df.select_dtypes(include=[np.number])
        .columns
        .difference(list(protected))
        .tolist()
    )

    if len(numeric_cols) < 2:
        raise ValueError(
            "VIF requires at least 2 numeric feature columns (after excluding "
            "protected columns). Nothing to optimize."
        )

    df_clean = df[numeric_cols].copy()

    # ── 2. Pre-flight data quality check ─────────────────────────────────────
    # statsmodels raises cryptic LinAlgError on NaN or Inf — catch it cleanly upfront.
    nan_cols = df_clean.columns[df_clean.isnull().any()].tolist()
    inf_cols  = df_clean.columns[np.isinf(df_clean).any()].tolist()
    problems  = []
    if nan_cols:
        problems.append(f"NaN in: {nan_cols}")
    if inf_cols:
        problems.append(f"Inf in: {inf_cols}")
    if problems:
        raise ValueError(
            "VIF requires clean numeric data. Fix the following before calling optimize_vif:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    # ── 3. Detect perfectly collinear (constant or duplicate) columns ─────────
    # variance_inflation_factor() returns inf or raises on rank-deficient matrices.
    # Drop zero-variance columns and near-duplicate columns before iterating.
    zero_var = df_clean.columns[df_clean.var(ddof=1) == 0].tolist()
    if zero_var:
        print(
            f"[optimize_vif] ⚠️  Dropping {len(zero_var)} zero-variance column(s) before VIF: {zero_var}"
        )
        df_clean = df_clean.drop(columns=zero_var)

    # Detect perfectly correlated pairs (correlation = ±1.0) which produce infinite VIF
    corr_matrix   = df_clean.corr().abs()
    upper_tri      = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    perfect_pairs  = [(c1, c2) for c2 in upper_tri.columns for c1 in upper_tri.index
                      if upper_tri.loc[c1, c2] >= 0.9999]
    if perfect_pairs:
        # Drop the second column of each perfectly correlated pair
        to_drop_perfect = list({pair[1] for pair in perfect_pairs})
        print(
            f"[optimize_vif] ⚠️  Dropping {len(to_drop_perfect)} perfectly collinear column(s): "
            f"{to_drop_perfect}"
        )
        df_clean = df_clean.drop(columns=to_drop_perfect)

    print(
        f"[optimize_vif] Starting with {len(df_clean.columns)} feature(s). "
        f"Threshold: {threshold}  |  Protected: {sorted(protected) or 'None'}"
    )

    # ── 4. Iterative VIF elimination ──────────────────────────────────────────
    history: list = []
    iteration  = 0

    while iteration < max_iterations:
        if len(df_clean.columns) < 2:
            print("[optimize_vif] ⚠️  Only 1 feature remains — stopping to preserve it.")
            break

        # VIF requires an intercept column to be computed correctly.
        # Without it, VIF assumes data is centred at zero and returns inflated values.
        X_const = sm.add_constant(df_clean, has_constant="add")

        # Compute VIF for every column including the constant, then strip the constant.
        col_names = X_const.columns.tolist()
        vif_values = []
        for i in range(len(col_names)):
            try:
                v = variance_inflation_factor(X_const.values, i)
            except Exception:
                v = np.inf   # treat computational failure as infinite VIF
            vif_values.append(v)

        vif_df = pd.DataFrame({"Feature": col_names, "VIF": vif_values})
        vif_df = vif_df[vif_df["Feature"] != "const"].reset_index(drop=True)

        # Sort for readability; find the worst offender
        vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)
        max_vif   = float(vif_df.loc[0, "VIF"])
        max_feat  = str(vif_df.loc[0, "Feature"])

        if max_vif <= threshold:
            break   # all features are within tolerance

        # Skip protected columns — find the worst non-protected offender
        non_protected_vif = vif_df[~vif_df["Feature"].isin(protected)]
        if non_protected_vif.empty:
            print(
                "[optimize_vif] ⚠️  All remaining high-VIF features are protected. "
                "Cannot reduce further."
            )
            break

        drop_feat = str(non_protected_vif.iloc[0]["Feature"])
        drop_vif  = float(non_protected_vif.iloc[0]["VIF"])

        print(f"  Iteration {iteration + 1:>3}: Drop '{drop_feat}'  VIF={drop_vif:.2f}")
        history.append({
            "iteration": iteration + 1,
            "dropped":   drop_feat,
            "vif":       round(drop_vif, 4),
            "remaining": len(df_clean.columns) - 1,
        })

        df_clean  = df_clean.drop(columns=[drop_feat])
        iteration += 1

    else:
        print(
            f"[optimize_vif] ⚠️  max_iterations={max_iterations} reached. "
            f"Check your data for extreme collinearity."
        )

    # ── 5. Final VIF table ────────────────────────────────────────────────────
    if len(df_clean.columns) >= 2:
        X_final   = sm.add_constant(df_clean, has_constant="add")
        final_vif = pd.DataFrame({
            "Feature": df_clean.columns,
            "VIF":     [
                round(float(variance_inflation_factor(X_final.values, i + 1)), 4)
                for i in range(len(df_clean.columns))
            ],
        }).sort_values("VIF", ascending=False).reset_index(drop=True)
    else:
        final_vif = pd.DataFrame({"Feature": df_clean.columns, "VIF": [1.0] * len(df_clean.columns)})

    dropped_total = len(numeric_cols) - len(df_clean.columns) + len(zero_var) + len(
        {pair[1] for pair in perfect_pairs}
    )
    print(
        f"\n[optimize_vif] Complete — {len(df_clean.columns)} feature(s) retained, "
        f"{dropped_total} dropped.\n"
    )
    print(final_vif.to_string(index=False))
    print()

    if return_vif_history:
        return df_clean, history

    return df_clean


def setup_leakproof_environment(
    df,
    target_col: str,
    time_col: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
    drop_cols: list = None,
    numeric_scaler=None,
    impute_strategy_numeric: str = "median",
    ohe_max_categories: int = None,
):
    """
    Prevents data leakage by strictly separating train/test data BEFORE
    preprocessing, then returning an UNFITTED preprocessor pipeline that
    must be fitted on training data only.

    Leakage prevention contract
    ---------------------------
    - The returned preprocessor is NEVER fitted here — fitting on test data
      would leak test-set statistics into scaling and imputation.
    - Time-series data is split chronologically (no shuffle).
    - The time column is dropped from X after sorting, so it cannot be used
      as a feature proxy for the target.
    - Column lists are derived from X_train (post-split) so test-set columns
      never influence the pipeline structure.

    Parameters
    ----------
    df                       : pd.DataFrame  — raw, un-preprocessed data
    target_col               : str           — column to predict
    time_col                 : str | None    — datetime column; triggers chronological split
    test_size                : float         — fraction of data held out (default 0.2)
    random_state             : int           — seed for reproducible random split (default 42)
    stratify                 : bool          — stratify random split by target class distribution
                                               (classification only; ignored for time-series splits)
    drop_cols                : list | None   — columns to exclude from X entirely (e.g., ID cols)
    numeric_scaler           : sklearn scaler | None
                               Custom scaler to use instead of StandardScaler.
                               Must implement fit/transform (e.g., RobustScaler()).
    impute_strategy_numeric  : str           — SimpleImputer strategy for numeric columns
                                               ('median', 'mean', 'most_frequent'; default 'median')
    ohe_max_categories       : int | None    — max_categories passed to OneHotEncoder to cap
                                               one-hot expansion on high-cardinality columns.
                                               None = no cap.

    Returns
    -------
    X_train      : pd.DataFrame
    X_test       : pd.DataFrame
    y_train      : pd.Series
    y_test       : pd.Series
    preprocessor : ColumnTransformer  — UNFITTED; call .fit_transform(X_train) then
                                        .transform(X_test) in your model pipeline.
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame.")
    if time_col and time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in DataFrame.")
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    VALID_IMPUTE = {"mean", "median", "most_frequent", "constant"}
    if impute_strategy_numeric not in VALID_IMPUTE:
        raise ValueError(
            f"impute_strategy_numeric must be one of {VALID_IMPUTE}, "
            f"got '{impute_strategy_numeric}'."
        )

    if drop_cols:
        missing_drops = [c for c in drop_cols if c not in df.columns]
        if missing_drops:
            print(f"[setup_leakproof_environment] ⚠️  drop_cols not found (skipped): {missing_drops}")

    df_safe = df.copy()

    # Columns to exclude from X (target + time + user-specified drops)
    exclude = {target_col}
    if time_col:
        exclude.add(time_col)
    if drop_cols:
        exclude.update(c for c in drop_cols if c in df_safe.columns)

    # ── 1. Split ──────────────────────────────────────────────────────────────
    if time_col:
        # Chronological split — no shuffle, no stratification
        # Sort BEFORE extracting X/y so indices stay aligned
        df_safe = df_safe.sort_values(by=time_col).reset_index(drop=True)

        X = df_safe.drop(columns=list(exclude))
        y = df_safe[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            shuffle=False,        # crucial: no future-peeking
        )
        split_type = "Time-Series (chronological, no shuffle)"

    else:
        X = df_safe.drop(columns=list(exclude))
        y = df_safe[target_col]

        # Stratification is only valid when the target has enough samples per class
        stratify_vec = None
        if stratify:
            min_class_count = y.value_counts().min()
            if min_class_count < 2:
                print(
                    "[setup_leakproof_environment] ⚠️  stratify=True requested but the "
                    "rarest class has fewer than 2 samples — stratification disabled."
                )
            else:
                stratify_vec = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_vec,
        )
        split_type = f"Random (stratified={stratify_vec is not None}, seed={random_state})"

    print(f"[setup_leakproof_environment] Split: {split_type}")
    print(
        f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows  "
        f"({test_size*100:.0f}% held out)"
    )

    # ── 2. Derive column lists from X_train ONLY ──────────────────────────────
    # CRITICAL: never inspect X_test to build the pipeline — that leaks test
    # structure into the preprocessor design.
    numeric_cols     = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # ── 3. Build unfitted preprocessor ───────────────────────────────────────
    scaler = numeric_scaler if numeric_scaler is not None else StandardScaler()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=impute_strategy_numeric)),
        ("scaler",  scaler),
    ])

    ohe_kwargs = dict(handle_unknown="ignore", sparse_output=False)
    if ohe_max_categories is not None:
        ohe_kwargs["max_categories"] = ohe_max_categories

    categorical_transformer = Pipeline(steps=[
        # 'constant' fill with 'Missing' treats absence as its own category —
        # more informative than 'most_frequent' which injects a false signal.
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot",  OneHotEncoder(**ohe_kwargs)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",   # FIX: 'passthrough' silently forwards raw object columns
                            # (e.g., free-text) which downstream models cannot handle.
                            # 'drop' is the safe default; use remainder='passthrough'
                            # explicitly if you have columns that need no transformation.
        verbose_feature_names_out=False,  # keeps output column names clean
    )

    print(
        f"  Numeric features : {len(numeric_cols)}\n"
        f"  Categorical features : {len(categorical_cols)}\n"
        f"  Preprocessor: UNFITTED — call .fit_transform(X_train) before .transform(X_test)."
    )

    return X_train, X_test, y_train, y_test, preprocessor
