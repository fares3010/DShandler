"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DataSciencePipeline
  A single class wrapping every diagnostic, cleaning, and
  modelling-prep function from this project as chainable methods.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DESIGN PRINCIPLES
  ─────────────────
  • self.df        — the live working DataFrame; mutated by every
                     cleaning / transform method.
  • self.results   — dict accumulating every report / artefact
                     produced so the caller can inspect later.
  • self.history   — ordered list of (method_name, shape_before,
                     shape_after) for auditing data shape changes.
  • Method chaining — every mutating method returns `self` so calls
                     can be strung together:
                       dsp.handle_duplicates().impute_numeric().encode()
  • Non-mutating methods (diagnostics, EDA, splits) return their
    result directly AND store it in self.results.

  QUICK START
  ───────────
  from ds_pipeline_class import DataSciencePipeline

  dsp = DataSciencePipeline(df, target_col="Churn")

  # Run everything in one chain:
  (dsp
    .profile_missing()
    .detect_structural()
    .detect_dimensional()
    .detect_categorical()
    .detect_predictive()
    .detect_anomaly_scan()
    .detect_leakage()
    .format_structure()
    .drop_duplicates()
    .impute_numeric()
    .impute_categorical()
    .handle_outliers()
    .transform_shape()
    .encode()
    .select_features()
    .vif_optimize()
  )

  X_train, X_test, y_train, y_test, preprocessor = dsp.split()
  dsp.analyze_distribution("Age")
  dsp.analyze_relationship("Age", "Fare")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ── Package-relative imports ──────────────────────────────────────────────────
from .diagnostics import (
    detect_anomalies,
    detect_structural_anomalies,
    detect_dimensional_issues,
    detect_categorical_issues,
    advanced_missing_profiler,
    detect_predictive_issues,
    detect_leakage_risks,
)
from .cleaning import (
    handle_numerical_missing,
    advanced_knn_impute,
    handle_categorical_missing,
    missforest_impute,
    handle_duplicates,
    standardize_data,
    format_structural_issues,
    handle_anomalies,
)
from .preprocessing import (
    transform_data_shape,
    encode_categorical_data,
    optimize_features,
    optimize_vif,
    setup_leakproof_environment,
)
from .statistics import (
    analyze_distribution,
    evaluate_distribution,
    analyze_relationship,
    test_hypothesis,
    enforce_stationarity,
    analyze_autocorrelation,
)


class DataSciencePipeline:
    """
    Stateful pipeline class. Every method operates on self.df and returns
    self for chaining. Results are accumulated in self.results.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # CONSTRUCTION & STATE
    # ══════════════════════════════════════════════════════════════════════════

    def __init__(self, df: pd.DataFrame, target_col: str = None, task_type: str = "classification"):
        """
        Parameters
        ----------
        df         : pd.DataFrame — raw input data
        target_col : str | None   — column to predict
        task_type  : str          — "classification" | "regression"
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        self.df         = df.copy()
        self._df_raw    = df.copy()          # immutable snapshot of original
        self.target_col = target_col
        self.task_type  = task_type
        self.results    = {}
        self.history    = []
        self._transformers       = {}        # fitted sklearn objects
        self._encoding_report    = {}
        self._split_result       = None      # cached from split()
        print(f"[DataSciencePipeline] Loaded {self.df.shape[0]:,} rows × {self.df.shape[1]} columns.")
        if target_col:
            print(f"  Target: '{target_col}'  |  Task: {task_type}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _record(self, method: str, shape_before: tuple):
        """Append an entry to self.history after a mutation."""
        self.history.append({
            "method":        method,
            "shape_before":  shape_before,
            "shape_after":   self.df.shape,
            "rows_delta":    self.df.shape[0] - shape_before[0],
            "cols_delta":    self.df.shape[1] - shape_before[1],
        })

    def _densify_sparse(self):
        """Convert any SparseDtype columns to dense in-place (needed before math ops)."""
        sparse = [c for c in self.df.columns
                  if isinstance(self.df[c].dtype, pd.SparseDtype)]
        if sparse:
            for col in sparse:
                self.df[col] = self.df[col].sparse.to_dense()

    def reset(self):
        """Restore self.df to the original raw DataFrame."""
        self.df = self._df_raw.copy()
        self.results = {}
        self.history = []
        print("[DataSciencePipeline] Reset to original DataFrame.")
        return self

    def snapshot(self) -> pd.DataFrame:
        """Return a copy of the current working DataFrame."""
        return self.df.copy()

    def summary(self):
        """Print a concise state summary."""
        print(f"\n{'─'*52}")
        print(f"  DataSciencePipeline State")
        print(f"  Shape       : {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"  Target      : {self.target_col or 'not set'}")
        print(f"  Task        : {self.task_type}")
        print(f"  Results keys: {list(self.results.keys()) or '(none yet)'}")
        if self.history:
            print(f"  History     : {len(self.history)} step(s)")
            for h in self.history[-5:]:
                dr = h['rows_delta']; dc = h['cols_delta']
                print(f"    {h['method']:<35} "
                      f"rows {h['shape_before'][0]}→{h['shape_after'][0]} ({dr:+d})  "
                      f"cols {h['shape_before'][1]}→{h['shape_after'][1]} ({dc:+d})")
        print(f"{'─'*52}\n")
        return self

    def __repr__(self) -> str:
        return (f"DataSciencePipeline("
                f"shape={self.df.shape}, "
                f"target='{self.target_col}', "
                f"steps={len(self.history)})")

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 1 — DIAGNOSTICS  (non-mutating → return self, store in results)
    # ══════════════════════════════════════════════════════════════════════════

    def profile_missing(self, show_heatmap: bool = False, show_bar: bool = True,
                        warn_threshold: float = 0.20, critical_threshold: float = 0.50):
        """Profile missing values across all columns."""
        result = advanced_missing_profiler(
            self.df,
            show_heatmap=show_heatmap,
            show_bar=show_bar,
            warn_threshold=warn_threshold,
            critical_threshold=critical_threshold,
        )
        self.results["missing_profile"] = result
        return self

    def detect_structural(self, shifted_threshold: float = 0.01):
        """Detect structural issues: type mismatches, shifted rows, JSON columns."""
        report, shifted_df = detect_structural_anomalies(self.df, shifted_threshold=shifted_threshold)
        self.results["structural_report"]  = report
        self.results["structural_shifted"] = shifted_df
        return self

    def detect_dimensional(self, skew_threshold: float = 1.0,
                           sparse_threshold: float = 0.7,
                           scale_ratio_threshold: float = 1000):
        """Detect scale disparity, skewness, sparsity, and near-zero variance."""
        report, stats_df = detect_dimensional_issues(
            self.df,
            skew_threshold=skew_threshold,
            sparse_threshold=sparse_threshold,
            scale_ratio_threshold=scale_ratio_threshold,
        )
        self.results["dimensional_report"] = report
        self.results["dimensional_stats"]  = stats_df
        return self

    def detect_categorical(self, card_threshold: int = 50,
                           rare_freq_threshold: float = 0.01,
                           id_uniqueness_threshold: float = 0.95):
        """Detect cardinality, rare categories, probable IDs, and ordinal columns."""
        report, stats_df = detect_categorical_issues(
            self.df,
            card_threshold=card_threshold,
            rare_freq_threshold=rare_freq_threshold,
            id_uniqueness_threshold=id_uniqueness_threshold,
        )
        self.results["categorical_report"] = report
        self.results["categorical_stats"]  = stats_df
        return self

    def detect_predictive(self, imbalance_threshold: float = 0.10,
                          multi_corr_threshold: float = 0.85,
                          irrelevant_corr_threshold: float = 0.05):
        """Scan for zero-variance, multicollinearity, class imbalance, and leaky features."""
        report = detect_predictive_issues(
            self.df,
            target_col=self.target_col,
            task_type=self.task_type,
            imbalance_threshold=imbalance_threshold,
            multi_corr_threshold=multi_corr_threshold,
            irrelevant_corr_threshold=irrelevant_corr_threshold,
        )
        self.results["predictive_report"] = report
        return self

    def detect_anomaly_scan(self, columns: list = None, contamination: float = 0.05):
        """Run IQR + Isolation Forest anomaly scan (diagnostic only — does not modify df)."""
        cols = columns or self.df.select_dtypes(include=[np.number]).columns.difference(
            [self.target_col] if self.target_col else []
        ).tolist()
        anomaly_df, report = detect_anomalies(self.df, cols, contamination=contamination)
        self.results["anomaly_scan_df"]     = anomaly_df
        self.results["anomaly_scan_report"] = report
        return self

    def detect_leakage(self, time_col: str = None, proxy_threshold: float = 0.95):
        """Scan for target proxies, time-series leakage, and index leakage."""
        if not self.target_col:
            print("[detect_leakage] target_col not set — skipping.")
            return self
        report = detect_leakage_risks(
            self.df,
            target_col=self.target_col,
            time_col=time_col,
            proxy_threshold=proxy_threshold,
        )
        self.results["leakage_report"] = report
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 2 — CLEANING  (mutating → modify self.df, return self)
    # ══════════════════════════════════════════════════════════════════════════

    def format_structure(self, type_map: dict = None, date_cols: list = None,
                         target_tz: str = "UTC", nested_col: str = None,
                         nested_prefix: str = None, strict_col: str = None):
        """Fix data types, dates, timezones, nested JSON, and quarantine shifted rows."""
        before = self.df.shape
        self.df, shifted_rows = format_structural_issues(
            self.df,
            type_map=type_map,
            date_cols=date_cols,
            target_tz=target_tz,
            nested_col=nested_col,
            nested_prefix=nested_prefix,
            strict_col=strict_col,
        )
        self.results["format_shifted_rows"] = shifted_rows
        self._record("format_structure", before)
        return self

    def drop_duplicates(self, subset=None, keep: str = "first",
                        sort_by=None, ascending: bool = False,
                        add_duplicate_flag: bool = False):
        """Remove or flag duplicate rows."""
        before = self.df.shape
        self.df = handle_duplicates(
            self.df, subset=subset, keep=keep,
            sort_by=sort_by, ascending=ascending,
            add_duplicate_flag=add_duplicate_flag,
        )
        self._record("drop_duplicates", before)
        return self

    def standardize_text(self, text_cols: list = None, typo_map: dict = None,
                         unit_col: str = None, unit_conversions: dict = None,
                         capitalize: str = "lower"):
        """Normalize whitespace, capitalization, typos, and unit mismatches."""
        before = self.df.shape
        self.df, report = standardize_data(
            self.df,
            text_cols=text_cols,
            typo_map=typo_map,
            unit_col=unit_col,
            unit_conversions=unit_conversions,
            capitalize=capitalize,
            return_report=True,
        )
        self.results["standardize_report"] = report
        self._record("standardize_text", before)
        return self

    def impute_numeric(self, strategy: str = "median", columns: list = None,
                       fill_value=None, add_indicator: bool = False):
        """
        Impute missing numeric values.
        strategy: 'mean' | 'median' | 'mode' | 'ffill' | 'bfill' | 'constant' | 'knn'
        """
        before = self.df.shape
        cols = columns or [
            c for c in self.df.select_dtypes(include=[np.number]).columns
            if c != self.target_col and self.df[c].isna().any()
        ]
        if not cols:
            print("[impute_numeric] No numeric columns with missing values — skipping.")
            return self
        if strategy == "knn":
            self.df = advanced_knn_impute(self.df, columns=cols, add_indicator=add_indicator)
        else:
            self.df = handle_numerical_missing(
                self.df, strategy=strategy, columns=cols,
                fill_value=fill_value, add_indicator=add_indicator,
            )
        self._record("impute_numeric", before)
        return self

    def impute_categorical(self, strategy: str = "mode", columns: list = None,
                           fill_value: str = "Unknown", add_indicator: bool = False,
                           use_missforest: bool = False):
        """
        Impute missing categorical values.
        strategy: 'mode' | 'constant' | 'proportional'
        Set use_missforest=True for Random Forest imputation (handles mixed types).
        """
        before = self.df.shape
        if use_missforest:
            self.df = missforest_impute(self.df, add_indicator=add_indicator)
        else:
            cols = columns or [
                c for c in self.df.select_dtypes(include=["object", "category", "bool"]).columns
                if self.df[c].isna().any()
            ]
            if not cols:
                print("[impute_categorical] No categorical columns with missing values — skipping.")
                return self
            self.df = handle_categorical_missing(
                self.df, strategy=strategy, columns=cols,
                fill_value=fill_value, add_indicator=add_indicator,
            )
        self._record("impute_categorical", before)
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 3 — ANOMALY HANDLING  (mutating)
    # ══════════════════════════════════════════════════════════════════════════

    def handle_outliers(self, columns: list = None, method: str = "iqr",
                        action: str = "clip", threshold: float = 1.5,
                        logical_bounds: dict = None, add_flag: bool = False,
                        contamination: float = None):
        """
        Detect and handle outliers in numeric columns.
        method: 'iqr' | 'zscore' | 'isolation_forest' | 'logical'
        action: 'clip' | 'drop' | 'nan'
        """
        before = self.df.shape
        cols = columns or self.df.select_dtypes(include=[np.number]).columns.difference(
            [self.target_col] if self.target_col else []
        ).tolist()
        self.df = handle_anomalies(
            self.df, columns=cols, method=method, action=action,
            threshold=threshold, logical_bounds=logical_bounds,
            add_flag=add_flag, contamination=contamination,
        )
        self._record("handle_outliers", before)
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 4 — TRANSFORMATION  (mutating)
    # ══════════════════════════════════════════════════════════════════════════

    def transform_shape(self, scale_cols: list = None, scale_method: str = "standard",
                        skew_cols: list = None, skew_method: str = "yeo-johnson",
                        sparse_cols: list = None, skew_threshold: float = 1.0):
        """
        Apply skew correction, scaling, and sparse conversion.
        Auto-detects skewed and sparse columns when not specified.
        """
        before = self.df.shape
        num_cols = self.df.select_dtypes(include=[np.number]).columns.difference(
            [self.target_col] if self.target_col else []
        ).tolist()

        if skew_cols is None:
            skew_cols = [c for c in num_cols if abs(self.df[c].skew()) > skew_threshold]
        if sparse_cols is None:
            sparse_cols = [c for c in num_cols if (self.df[c] == 0).sum() / len(self.df) > 0.70]
        if scale_cols is None:
            scale_cols = [c for c in num_cols if c not in sparse_cols]

        self.df, transformers = transform_data_shape(
            self.df,
            scale_cols=scale_cols   or None,
            scale_method=scale_method,
            skew_cols=skew_cols     or None,
            skew_method=skew_method,
            sparse_cols=sparse_cols or None,
            return_transformers=True,
        )
        self._transformers.update(transformers)
        self.results["transformers"] = self._transformers
        self._record("transform_shape", before)
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 5 — ENCODING  (mutating)
    # ══════════════════════════════════════════════════════════════════════════

    def encode(self, nominal_cols: list = None, ordinal_maps: dict = None,
               high_card_cols: list = None, rare_threshold: float = 0.01,
               drop_first: bool = True, target_smoothing: float = 0.0):
        """
        Encode categorical columns: OHE (nominal), ordinal mapping, target encoding.
        Auto-detects nominal and high-cardinality columns when not specified.
        """
        before = self.df.shape

        # Auto-detect if not provided
        if nominal_cols is None and high_card_cols is None:
            try:
                cat_report, _ = detect_categorical_issues(self.df)
                nominal_cols   = [
                    c for c in cat_report.get("Likely_Nominal", []) + cat_report.get("Likely_Boolean", [])
                    if c != self.target_col
                ]
                high_card_cols = [
                    c for c in cat_report.get("High_Cardinality", [])
                    if c != self.target_col
                ]
            except Exception:
                pass

        self.df, self._encoding_report = encode_categorical_data(
            self.df,
            nominal_cols=nominal_cols   or None,
            ordinal_maps=ordinal_maps,
            high_card_cols=high_card_cols or None,
            target_col=self.target_col,
            rare_threshold=rare_threshold,
            drop_first=drop_first,
            target_smoothing=target_smoothing,
            return_encoding_report=True,
        )
        self.results["encoding_report"] = self._encoding_report
        self._record("encode", before)
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 6 — FEATURE SELECTION  (mutating)
    # ══════════════════════════════════════════════════════════════════════════

    def select_features(self, var_threshold: float = 0.0,
                        multi_corr_threshold: float = 0.85,
                        target_corr_threshold: float = 0.05,
                        protected_cols: list = None):
        """Remove zero-variance, irrelevant, and multicollinear features."""
        before = self.df.shape
        self._densify_sparse()

        protected = list(protected_cols or [])
        if self.target_col:
            protected.append(self.target_col)

        # Skip target-correlation check when target is non-numeric
        effective_target = (
            self.target_col
            if (self.target_col and pd.api.types.is_numeric_dtype(self.df.get(self.target_col, pd.Series(dtype=float))))
            else None
        )

        self.df, report = optimize_features(
            self.df,
            target_col=effective_target,
            var_threshold=var_threshold,
            multi_corr_threshold=multi_corr_threshold,
            target_corr_threshold=target_corr_threshold,
            protected_cols=protected,
            return_report=True,
        )

        # Reattach target if it was dropped (it shouldn't be, but guard anyway)
        if self.target_col and self.target_col not in self.df.columns:
            self.df[self.target_col] = before[1]   # type: ignore

        self.results["feature_selection_report"] = report
        self._record("select_features", before)
        return self

    def vif_optimize(self, threshold: float = 5.0, protected_cols: list = None):
        """Iteratively remove features with VIF above threshold."""
        before = self.df.shape
        self._densify_sparse()

        protected = list(protected_cols or [])
        if self.target_col:
            protected.append(self.target_col)

        num_cols = self.df.select_dtypes(include=[np.number]).columns.difference(protected).tolist()

        if len(num_cols) < 2:
            print("[vif_optimize] Fewer than 2 unprotected numeric columns — skipping.")
            return self

        if self.df[num_cols].isnull().values.any():
            print("[vif_optimize] NaN values present — impute before VIF. Skipping.")
            return self

        df_vif, vif_history = optimize_vif(
            self.df[num_cols + ([self.target_col] if self.target_col else [])],
            threshold=threshold,
            target_col=self.target_col,
            return_vif_history=True,
        )
        # Drop VIF-eliminated columns from the full working DataFrame
        dropped = set(num_cols) - set(df_vif.columns)
        if dropped:
            self.df = self.df.drop(columns=list(dropped))

        self.results["vif_history"] = vif_history
        self._record("vif_optimize", before)
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 7 — SPLIT  (non-mutating → returns arrays, stores in results)
    # ══════════════════════════════════════════════════════════════════════════

    def split(self, test_size: float = 0.2, time_col: str = None,
              stratify: bool = False, random_state: int = 42,
              drop_cols: list = None):
        """
        Leakproof train/test split with a fitted preprocessor.

        Returns
        -------
        X_train, X_test, y_train, y_test, preprocessor
        (also stores them in self.results["split"])
        """
        if not self.target_col:
            raise ValueError("target_col must be set before calling split().")

        self._densify_sparse()

        X_train, X_test, y_train, y_test, preprocessor = setup_leakproof_environment(
            self.df,
            target_col=self.target_col,
            time_col=time_col,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify and self.task_type == "classification",
            drop_cols=drop_cols,
        )

        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc  = preprocessor.transform(X_test)

        split_result = {
            "X_train": X_train,  "X_test": X_test,
            "y_train": y_train,  "y_test": y_test,
            "X_train_processed": X_train_proc,
            "X_test_processed":  X_test_proc,
            "preprocessor":      preprocessor,
        }
        self._split_result            = split_result
        self.results["split"]         = split_result
        self._transformers["preprocessor"] = preprocessor

        return X_train, X_test, y_train, y_test, preprocessor

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 8 — EDA & STATISTICS  (non-mutating → return result, store)
    # ══════════════════════════════════════════════════════════════════════════

    def analyze_distribution(self, column: str, bins="auto",
                             show_plot: bool = True, show_normality: bool = True,
                             percentiles: list = None) -> dict:
        """Full statistical profile: central tendency, dispersion, shape, normality."""
        result = analyze_distribution(
            self.df, column, bins=bins,
            show_plot=show_plot, show_normality=show_normality,
            percentiles=percentiles,
        )
        self.results[f"distribution_{column}"] = result
        return result

    def evaluate_distribution(self, column: str, alpha: float = 0.05,
                              use_kpss: bool = False) -> dict:
        """Normality tests, parametric/non-parametric recommendation, modality detection."""
        result = evaluate_distribution(self.df, column, alpha=alpha)
        self.results[f"eval_distribution_{column}"] = result
        return result

    def analyze_relationship(self, col1: str, col2: str,
                             alpha: float = 0.05) -> dict:
        """
        Pearson, Spearman, Mutual Information (numeric vs numeric);
        Cramér's V (cat vs cat); Point-Biserial / Eta (mixed).
        """
        result = analyze_relationship(self.df, col1, col2, alpha=alpha)
        self.results[f"relationship_{col1}_vs_{col2}"] = result
        return result

    def test_hypothesis(self, group_col: str, value_col: str,
                        alpha: float = 0.05, post_hoc: bool = True) -> dict:
        """
        Auto-select T-Test / ANOVA / Mann-Whitney / Kruskal-Wallis based on
        normality; compute effect size and optional post-hoc pairwise tests.
        """
        p, test_name, report = test_hypothesis(
            self.df, group_col=group_col, value_col=value_col,
            alpha=alpha, post_hoc=post_hoc, return_report=True,
        )
        self.results[f"hypothesis_{group_col}_{value_col}"] = report
        return report

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 9 — TIME-SERIES  (non-mutating → return result, store)
    # ══════════════════════════════════════════════════════════════════════════

    def enforce_stationarity(self, column: str, alpha: float = 0.05,
                             max_diff: int = 2, seasonal_period: int = None,
                             use_kpss: bool = True) -> tuple:
        """
        ADF + KPSS stationarity tests with iterative differencing.
        Returns (stationary_series, d, report).
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found.")
        series = self.df[column]
        stationary_series, d, report = enforce_stationarity(
            series, alpha=alpha, max_diff=max_diff,
            seasonal_period=seasonal_period, use_kpss=use_kpss,
        )
        self.results[f"stationarity_{column}"] = report
        return stationary_series, d, report

    def analyze_autocorrelation(self, column: str, lags: int = 30,
                                alpha: float = 0.05,
                                ljungbox_lags: list = None) -> dict:
        """
        ACF / PACF plots, Ljung-Box test, and ARIMA order hints.
        Pass the stationary series from enforce_stationarity for best results.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found.")
        lb_df, report = analyze_autocorrelation(
            self.df[column], lags=lags, alpha=alpha,
            ljungbox_lags=ljungbox_lags, return_report=True,
        )
        self.results[f"autocorrelation_{column}"] = report
        return report

    # ══════════════════════════════════════════════════════════════════════════
    # CONVENIENCE RUNNERS
    # ══════════════════════════════════════════════════════════════════════════

    def run_diagnostics(self, time_col: str = None):
        """Run all seven diagnostic scanners in one call."""
        return (self
                .profile_missing()
                .detect_structural()
                .detect_dimensional()
                .detect_categorical()
                .detect_predictive()
                .detect_anomaly_scan()
                .detect_leakage(time_col=time_col))

    def run_cleaning(self, num_strategy: str = "median", cat_strategy: str = "mode"):
        """Run the standard cleaning sequence in one call."""
        return (self
                .format_structure()
                .drop_duplicates()
                .impute_numeric(strategy=num_strategy)
                .impute_categorical(strategy=cat_strategy))

    def run_preprocessing(self, scale_method: str = "standard",
                          skew_method: str = "yeo-johnson"):
        """Run the standard pre-modelling sequence in one call."""
        return (self
                .handle_outliers()
                .transform_shape(scale_method=scale_method, skew_method=skew_method)
                .encode()
                .select_features()
                .vif_optimize())


# ══════════════════════════════════════════════════════════════════════════════
# QUICK START EXAMPLE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv("your_dataset.csv")

    # ── 2. Instantiate ────────────────────────────────────────────────────────
    dsp = DataSciencePipeline(df, target_col="target", task_type="classification")

    # ── 3A. Full auto-pipeline (one chain) ────────────────────────────────────
    dsp.run_diagnostics().run_cleaning().run_preprocessing()
    X_train, X_test, y_train, y_test, preprocessor = dsp.split(test_size=0.2)

    # ── 3B. Fine-grained control ──────────────────────────────────────────────
    # (dsp
    #   .profile_missing()
    #   .detect_structural()
    #   .detect_leakage()
    #   .format_structure(type_map={"age": "Int64"})
    #   .drop_duplicates(subset=["user_id"], sort_by="updated_at")
    #   .standardize_text(text_cols=["city"], typo_map={"usa": "united states"})
    #   .impute_numeric(strategy="knn")
    #   .impute_categorical(strategy="mode")
    #   .handle_outliers(method="iqr", action="clip", threshold=1.5)
    #   .transform_shape(scale_method="robust", skew_method="yeo-johnson")
    #   .encode(nominal_cols=["color"], ordinal_maps={"size": ["S", "M", "L"]})
    #   .select_features(var_threshold=0.01, multi_corr_threshold=0.85)
    #   .vif_optimize(threshold=5.0)
    # )
    # X_train, X_test, y_train, y_test, pre = dsp.split(stratify=True)

    # ── 4. EDA ────────────────────────────────────────────────────────────────
    # dsp.analyze_distribution("Age")
    # dsp.analyze_relationship("Age", "Fare")
    # dsp.test_hypothesis("Pclass", "Fare")

    # ── 5. Time-series ────────────────────────────────────────────────────────
    # series, d, report = dsp.enforce_stationarity("revenue", seasonal_period=12)
    # dsp.analyze_autocorrelation("revenue", lags=40)

    # ── 6. Inspect state ──────────────────────────────────────────────────────
    # dsp.summary()
    # print(dsp.results.keys())
    # print(dsp.history)
