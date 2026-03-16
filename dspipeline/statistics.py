"""
Statistical analysis: distribution profiling, normality evaluation,
relationship analysis, hypothesis testing, stationarity, and autocorrelation.
"""
import warnings
from itertools import combinations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats, stats as scipy_stats
from scipy.signal import find_peaks
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss


def analyze_distribution(
    df,
    column: str,
    bins: int = "auto",
    show_plot: bool = True,
    show_normality: bool = True,
    percentiles: list = None,
):
    """
    Calculates and visualizes the full statistical profile of a numeric column:
    Central Tendency, Dispersion, Shape, Normality Tests, and Percentiles.

    Parameters
    ----------
    df               : pd.DataFrame
    column           : str    — numeric column to analyze
    bins             : int | 'auto'
                       Histogram bin count. 'auto' uses Sturges' rule: ⌈log₂(n)⌉ + 1,
                       which is more stable than the original hardcoded 30 (default 'auto').
    show_plot        : bool   — render the visualization (default True)
    show_normality   : bool   — run and print Shapiro-Wilk / D'Agostino normality
                               tests (default True)
    percentiles      : list   — additional percentiles to include in the report
                               (default [1, 5, 25, 50, 75, 95, 99])

    Returns
    -------
    stats_dict : dict — all computed statistics
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' is not numeric. analyze_distribution requires numbers.")

    data       = df[column].dropna()
    n          = len(data)
    n_null     = df[column].isna().sum()

    if n == 0:
        raise ValueError(f"Column '{column}' has no non-null values to analyze.")

    percentiles = percentiles or [1, 5, 25, 50, 75, 95, 99]

    # ── 1. Central tendency ───────────────────────────────────────────────────
    mean_val   = float(data.mean())
    median_val = float(data.median())
    mode_vals  = data.mode()
    # Mode can be multimodal — store all modes; use first for plotting
    mode_val   = float(mode_vals.iloc[0]) if not mode_vals.empty else np.nan
    is_multimodal = len(mode_vals) > 1

    # ── 2. Dispersion ─────────────────────────────────────────────────────────
    variance   = float(data.var(ddof=1))  # sample variance (ddof=1)
    std_dev    = float(data.std(ddof=1))
    data_range = float(data.max() - data.min())
    iqr        = float(data.quantile(0.75) - data.quantile(0.25))
    cv         = (std_dev / abs(mean_val) * 100) if mean_val != 0 else np.nan  # coefficient of variation

    # ── 3. Shape ──────────────────────────────────────────────────────────────
    skewness = float(data.skew())
    kurtosis = float(data.kurtosis())     # excess kurtosis (Fisher; normal = 0)

    # Skewness interpretation
    if abs(skewness) < 0.5:
        skew_label = "Approximately Symmetric"
    elif abs(skewness) < 1.0:
        skew_label = "Moderately Skewed " + ("Right ▶" if skewness > 0 else "Left ◀")
    else:
        skew_label = "Highly Skewed " + ("Right ▶" if skewness > 0 else "Left ◀")

    # Kurtosis interpretation (excess kurtosis)
    if kurtosis < -1:
        kurt_label = "Platykurtic (flat, thin tails)"
    elif kurtosis > 1:
        kurt_label = "Leptokurtic (peaked, heavy tails)"
    else:
        kurt_label = "Mesokurtic (near-normal)"

    # ── 4. Percentiles ────────────────────────────────────────────────────────
    pct_dict = {f"P{p}": float(data.quantile(p / 100)) for p in percentiles}

    # ── 5. Normality tests ────────────────────────────────────────────────────
    normality = {}
    if show_normality:
        # Shapiro-Wilk: best for n ≤ 5000; more powerful for small samples
        if n <= 5_000:
            sw_stat, sw_p = scipy_stats.shapiro(data)
            normality["Shapiro-Wilk"] = {"statistic": round(float(sw_stat), 5),
                                          "p_value":   round(float(sw_p),   6)}
        else:
            normality["Shapiro-Wilk"] = {"note": "Skipped (n > 5 000 — use D'Agostino)"}

        # D'Agostino-Pearson: works well for large samples
        dp_stat, dp_p = scipy_stats.normaltest(data)
        normality["D'Agostino-Pearson"] = {"statistic": round(float(dp_stat), 5),
                                            "p_value":   round(float(dp_p),   6)}

    # ── 6. Package results ────────────────────────────────────────────────────
    stats_dict = {
        "n":              n,
        "n_null":         n_null,
        "Mean":           round(mean_val,   6),
        "Median":         round(median_val, 6),
        "Mode":           round(mode_val,   6) if not np.isnan(mode_val) else np.nan,
        "Is_Multimodal":  is_multimodal,
        "Variance":       round(variance,   6),
        "Std_Dev":        round(std_dev,    6),
        "CV (%)":         round(cv,         4)  if not np.isnan(cv) else np.nan,
        "Range":          round(data_range, 6),
        "IQR":            round(iqr,        6),
        "Skewness":       round(skewness,   6),
        "Kurtosis":       round(kurtosis,   6),
        "Skew_Label":     skew_label,
        "Kurt_Label":     kurt_label,
        **pct_dict,
        "Normality":      normality,
    }

    # ── 7. Visualization ──────────────────────────────────────────────────────
    if show_plot:
        # Dynamic bin count via Sturges' rule — avoids over/under-binning
        n_bins = int(np.ceil(np.log2(n)) + 1) if bins == "auto" else int(bins)

        fig = plt.figure(figsize=(14, 8))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_hist = fig.add_subplot(gs[0, :])   # top: histogram spans full width
        ax_box  = fig.add_subplot(gs[1, 0])   # bottom-left: box plot
        ax_qq   = fig.add_subplot(gs[1, 1])   # bottom-right: Q-Q plot

        # ── Histogram + KDE ───────────────────────────────────────────────
        sns.histplot(
            data, bins=n_bins, kde=True, ax=ax_hist,
            color="#4a90d9", edgecolor="white", linewidth=0.4,
            alpha=0.75, line_kws={"linewidth": 2, "color": "#1a3a5c"},
        )

        # Vertical lines for central tendency
        ax_hist.axvline(mean_val,   color="#e74c3c", linestyle="--", lw=2,
                        label=f"Mean   {mean_val:.3f}")
        ax_hist.axvline(median_val, color="#2ecc71", linestyle="-",  lw=2,
                        label=f"Median {median_val:.3f}")
        if not np.isnan(mode_val):
            ax_hist.axvline(mode_val, color="#f39c12", linestyle="-.", lw=2,
                            label=f"Mode   {mode_val:.3f}"
                                  + (" (multimodal)" if is_multimodal else ""))

        # IQR shading
        q25, q75 = float(data.quantile(0.25)), float(data.quantile(0.75))
        ax_hist.axvspan(q25, q75, alpha=0.10, color="#9b59b6", label=f"IQR [{q25:.2f}, {q75:.2f}]")

        # Annotation box — skewness / kurtosis / CV
        shape_text = (
            f"Skewness : {skewness:+.3f}  {skew_label}\n"
            f"Kurtosis : {kurtosis:+.3f}  {kurt_label}\n"
            f"CV       : {cv:.1f}%   n={n:,}   null={n_null:,}"
        )
        ax_hist.annotate(
            shape_text,
            xy=(0.98, 0.95), xycoords="axes fraction",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9),
            ha="right", va="top",
        )

        ax_hist.set_title(
            f"Distribution of  {column}", fontsize=13, fontweight="bold", pad=10
        )
        ax_hist.set_xlabel(column, fontsize=10)
        ax_hist.set_ylabel("Frequency", fontsize=10)
        ax_hist.legend(fontsize=9, framealpha=0.9)
        ax_hist.grid(axis="y", alpha=0.3, linestyle=":")

        # ── Box plot ──────────────────────────────────────────────────────
        sns.boxplot(
            x=data, ax=ax_box,
            color="#4a90d9", linewidth=1.2,
            flierprops=dict(marker="o", markersize=3, markerfacecolor="#e74c3c", alpha=0.5),
        )
        ax_box.set_title("Box Plot", fontsize=11, fontweight="bold")
        ax_box.set_xlabel(column, fontsize=9)
        ax_box.grid(axis="x", alpha=0.3, linestyle=":")

        # ── Q-Q plot (tests normality visually) ───────────────────────────
        (osm, osr), (slope, intercept, r) = scipy_stats.probplot(data, dist="norm")
        ax_qq.scatter(osm, osr, s=8, alpha=0.4, color="#4a90d9", label="Data")
        ax_qq.plot(
            [min(osm), max(osm)],
            [slope * min(osm) + intercept, slope * max(osm) + intercept],
            color="#e74c3c", lw=1.5, label=f"Normal line  r={r:.3f}",
        )
        ax_qq.set_title("Q-Q Plot (vs Normal)", fontsize=11, fontweight="bold")
        ax_qq.set_xlabel("Theoretical Quantiles", fontsize=9)
        ax_qq.set_ylabel("Sample Quantiles",      fontsize=9)
        ax_qq.legend(fontsize=8)
        ax_qq.grid(alpha=0.3, linestyle=":")

        plt.suptitle(
            f"Statistical Profile  ·  {column}",
            fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        plt.show()

    # ── 8. Print report ───────────────────────────────────────────────────────
    w = 18
    print(f"\n{'─'*44}")
    print(f"  Statistical Summary  ·  '{column}'")
    print(f"{'─'*44}")
    print(f"  {'Observations':<{w}} {n:,}  (null: {n_null:,})")
    print(f"  {'Mean':<{w}} {mean_val:.6f}")
    print(f"  {'Median':<{w}} {median_val:.6f}")
    print(f"  {'Mode':<{w}} {mode_val:.6f}"
          + ("  ← multimodal" if is_multimodal else ""))
    print(f"  {'Std Dev':<{w}} {std_dev:.6f}")
    print(f"  {'Variance':<{w}} {variance:.6f}")
    print(f"  {'CV (%)':<{w}} {cv:.4f}")
    print(f"  {'Range':<{w}} {data_range:.6f}")
    print(f"  {'IQR':<{w}} {iqr:.6f}")
    print(f"  {'Skewness':<{w}} {skewness:+.6f}  ({skew_label})")
    print(f"  {'Kurtosis':<{w}} {kurtosis:+.6f}  ({kurt_label})")

    print(f"\n  Percentiles")
    for p in percentiles:
        print(f"  {'P' + str(p):<{w}} {pct_dict['P' + str(p)]:.6f}")

    if show_normality and normality:
        print(f"\n  Normality Tests  (H₀: data is normally distributed)")
        for test, result in normality.items():
            if "note" in result:
                print(f"  {test:<22} {result['note']}")
            else:
                sig = "✅ Normal (p > 0.05)" if result["p_value"] > 0.05 else "❌ Not Normal (p ≤ 0.05)"
                print(f"  {test:<22} stat={result['statistic']:.5f}  p={result['p_value']:.6f}  {sig}")

    print(f"{'─'*44}\n")

    return stats_dict


def evaluate_distribution(
    df,
    column: str,
    alpha: float = 0.05,
    kde_points: int = 1000,
    peak_prominence: float = None,
    show_both_tests: bool = True,
):
    """
    Evaluates normality, selects appropriate statistical tests, and detects
    modality (number of peaks) via KDE.

    Normality testing strategy
    --------------------------
    - Shapiro-Wilk  → n ≤ 5 000  (most powerful for small samples)
    - D'Agostino-Pearson → n > 5 000  (robust for large samples)
    - When show_both_tests=True, both are run and a consensus verdict is formed.

    Parameters
    ----------
    df                : pd.DataFrame
    column            : str    — numeric column to evaluate
    alpha             : float  — significance level (default 0.05)
    kde_points        : int    — resolution of the KDE grid for peak detection
                                 (default 1 000)
    peak_prominence   : float | None
                        Minimum prominence for a KDE peak to be counted as a mode.
                        If None, auto-calibrated to 5 % of the KDE height range,
                        which adapts to the actual distribution scale rather than
                        using a hardcoded 0.01 that mismatches KDE-normalized density.
    show_both_tests   : bool   — run both Shapiro-Wilk and D'Agostino-Pearson and
                                 report both (default True)

    Returns
    -------
    results : dict — full diagnostic summary
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' is not numeric.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")

    data   = df[column].dropna()
    n      = len(data)
    n_null = df[column].isna().sum()

    if n < 8:
        raise ValueError(
            f"Column '{column}' has only {n} non-null values. "
            f"Normality tests require at least 8 observations."
        )

    results = {
        "column":        column,
        "n":             n,
        "n_null":        n_null,
        "alpha":         alpha,
        "normality":     {},
        "Is_Normal":     None,
        "Modality":      None,
        "n_peaks":       None,
        "Recommended_Math":  None,
        "Compare_2_Groups":  None,
        "Compare_3+_Groups": None,
        "Correlation":       None,
        "Effect_Size_Metric": None,
    }

    # ── 1. Normality tests ────────────────────────────────────────────────────
    normality_verdicts = []

    # Shapiro-Wilk — most powerful for n ≤ 5 000
    if n <= 5_000 or show_both_tests:
        if n <= 5_000:
            sw_stat, sw_p = stats.shapiro(data)
            sw_result = {
                "statistic": round(float(sw_stat), 6),
                "p_value":   round(float(sw_p),   6),
                "verdict":   "Normal" if sw_p > alpha else "Not Normal",
                "note":      "",
            }
        else:
            sw_result = {
                "note": f"Skipped — n={n:,} > 5 000. Shapiro-Wilk loses power at this scale.",
                "verdict": "Skipped",
            }
        results["normality"]["Shapiro-Wilk"] = sw_result
        if sw_result["verdict"] not in ("Skipped",):
            normality_verdicts.append(sw_p > alpha)

    # D'Agostino-Pearson — robust for all sample sizes
    dp_stat, dp_p = stats.normaltest(data)
    dp_result = {
        "statistic": round(float(dp_stat), 6),
        "p_value":   round(float(dp_p),   6),
        "verdict":   "Normal" if dp_p > alpha else "Not Normal",
        "note":      "Best for n > 5 000" if n > 5_000 else "",
    }
    results["normality"]["D'Agostino-Pearson"] = dp_result
    normality_verdicts.append(dp_p > alpha)

    # Consensus: normal only if ALL run tests agree it is normal
    # (conservative — avoids falsely clearing a non-normal distribution)
    is_normal = all(normality_verdicts)
    results["Is_Normal"] = is_normal

    # ── 2. Statistical test recommendations ───────────────────────────────────
    if is_normal:
        results["Recommended_Math"]   = "Parametric"
        results["Compare_2_Groups"]   = "Independent T-Test (scipy.stats.ttest_ind)"
        results["Compare_3+_Groups"]  = "One-Way ANOVA (scipy.stats.f_oneway)"
        results["Correlation"]        = "Pearson r (scipy.stats.pearsonr)"
        results["Effect_Size_Metric"] = "Cohen's d  /  η² (eta-squared)"
        results["Paired_Test"]        = "Paired T-Test (scipy.stats.ttest_rel)"
    else:
        results["Recommended_Math"]   = "Non-Parametric"
        results["Compare_2_Groups"]   = "Mann-Whitney U (scipy.stats.mannwhitneyu)"
        results["Compare_3+_Groups"]  = "Kruskal-Wallis H (scipy.stats.kruskal)"
        results["Correlation"]        = "Spearman ρ (scipy.stats.spearmanr)"
        results["Effect_Size_Metric"] = "Rank-Biserial r  /  ε² (epsilon-squared)"
        results["Paired_Test"]        = "Wilcoxon Signed-Rank (scipy.stats.wilcoxon)"

    # ── 3. Modality — KDE peak detection ─────────────────────────────────────
    # FIX: the original used a hardcoded prominence=0.01, which is calibrated
    # to KDE density curves where the y-axis is in probability/unit space.
    # For some distributions (e.g., counts in the millions) the KDE y-values
    # are tiny (e.g., 1e-7) and 0.01 would flag ZERO peaks even for unimodal data.
    # Auto-calibrating to 5 % of the KDE height range makes this scale-invariant.
    kde    = stats.gaussian_kde(data)
    x_grid = np.linspace(float(data.min()), float(data.max()), kde_points)
    y_grid = kde(x_grid)

    if peak_prominence is None:
        # 5 % of the KDE's height range — adapts to any scale
        auto_prominence  = 0.05 * (y_grid.max() - y_grid.min())
        peak_prominence  = max(auto_prominence, 1e-10)  # never let it be 0

    peaks, peak_props = find_peaks(y_grid, prominence=peak_prominence)
    n_peaks           = len(peaks)

    if n_peaks == 0:
        # Pathological case — flat distribution; treat as unimodal
        modality = "Uniform / Flat (0 distinct peaks)"
    elif n_peaks == 1:
        modality = "Unimodal (1 peak)"
    elif n_peaks == 2:
        modality = "Bimodal (2 peaks)"
    else:
        modality = f"Multimodal ({n_peaks} peaks)"

    results["Modality"] = modality
    results["n_peaks"]  = n_peaks

    # Peak locations in original data units
    results["Peak_Locations"] = [round(float(x_grid[p]), 4) for p in peaks]

    # ── 4. Print report ───────────────────────────────────────────────────────
    w = 24
    print(f"\n{'─'*50}")
    print(f"  Distribution Evaluation  ·  '{column}'")
    print(f"{'─'*50}")
    print(f"  {'Observations':<{w}} {n:,}  (null: {n_null:,})")
    print(f"\n  Normality Tests  (α = {alpha})")
    for test, res in results["normality"].items():
        if "note" in res and "verdict" not in res:
            print(f"  {test:<{w}} {res['note']}")
        else:
            note = f"  ← {res['note']}" if res.get("note") else ""
            verdict_icon = "✅" if res["verdict"] == "Normal" else ("⏭️" if res["verdict"] == "Skipped" else "❌")
            if res["verdict"] == "Skipped":
                print(f"  {test:<{w}} {res['note']}")
            else:
                print(
                    f"  {test:<{w}} "
                    f"stat={res['statistic']:.5f}  p={res['p_value']:.6f}  "
                    f"{verdict_icon} {res['verdict']}{note}"
                )

    consensus_icon = "✅" if is_normal else "❌"
    print(f"\n  {'Consensus Verdict':<{w}} {consensus_icon} {'Normal' if is_normal else 'Not Normal'}")
    print(f"  {'Recommended Math':<{w}} {results['Recommended_Math']}")
    print(f"\n  Recommended Tests")
    print(f"  {'2-Group Comparison':<{w}} {results['Compare_2_Groups']}")
    print(f"  {'Paired Comparison':<{w}} {results['Paired_Test']}")
    print(f"  {'3+ Group Comparison':<{w}} {results['Compare_3+_Groups']}")
    print(f"  {'Correlation':<{w}} {results['Correlation']}")
    print(f"  {'Effect Size':<{w}} {results['Effect_Size_Metric']}")
    print(f"\n  {'Modality':<{w}} {results['Modality']}")
    if results["Peak_Locations"]:
        print(f"  {'Peak Locations':<{w}} {results['Peak_Locations']}")
    print(f"{'─'*50}\n")

    return results


def analyze_relationship(
    df,
    col1: str,
    col2: str,
    numeric_unique_threshold: int = 10,
    mi_trap_threshold: float = 0.1,
    alpha: float = 0.05,
    return_report: bool = False,
):
    """
    Intelligently analyzes the statistical relationship between two variables
    based on their data types.

    Handles:
      - Numeric   vs Numeric     → Pearson, Spearman, Mutual Information
      - Categorical vs Categorical → Cramér's V (bias-corrected), Chi-square
      - Numeric   vs Categorical  → Point-Biserial (binary) or Correlation Ratio η (multi-class)
                                    + ANOVA / Kruskal-Wallis significance test

    Also detects the Zero Correlation Trap: Pearson ≈ 0 but MI is high,
    indicating a strong non-linear relationship that linear metrics would miss.

    Parameters
    ----------
    df                      : pd.DataFrame
    col1, col2              : str   — columns to compare
    numeric_unique_threshold: int   — columns with fewer unique values than this
                                      are treated as categorical even if numeric
                                      (default 10; handles binary/ordinal integers)
    mi_trap_threshold       : float — MI above this while |Pearson| < 0.1 triggers
                                      the Zero Correlation Trap warning (default 0.1)
    alpha                   : float — significance level for hypothesis tests (default 0.05)
    return_report           : bool  — if True, return (print output stays) + results dict

    Returns
    -------
    results : dict  — all computed statistics (always)
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    for col in (col1, col2):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    temp_df = df[[col1, col2]].dropna()
    n_dropped = len(df) - len(temp_df)

    if len(temp_df) < 8:
        raise ValueError(
            f"Only {len(temp_df)} complete rows — too few for reliable analysis."
        )

    x, y = temp_df[col1], temp_df[col2]

    # ── Type classification ────────────────────────────────────────────────────
    # A column is "numeric" for relationship purposes only when it is both
    # dtype-numeric AND has enough distinct values to be truly continuous.
    # This prevents integers like 0/1/2 (encoded categories) being treated as
    # continuous, which would distort Pearson and Spearman calculations.
    def _is_continuous(series: pd.Series) -> bool:
        return (
            pd.api.types.is_numeric_dtype(series)
            and series.nunique() > numeric_unique_threshold
        )

    is_x_num = _is_continuous(x)
    is_y_num = _is_continuous(y)

    results = {
        "col1": col1, "col2": col2,
        "n": len(temp_df), "n_dropped": n_dropped,
        "scenario": None,
    }

    print(f"\n{'─'*54}")
    print(f"  Relationship Analysis  ·  '{col1}'  vs  '{col2}'")
    print(f"  n={len(temp_df):,}  (dropped {n_dropped:,} rows with nulls)")
    print(f"{'─'*54}")

    # ── SCENARIO 1: Numeric vs Numeric ────────────────────────────────────────
    if is_x_num and is_y_num:
        results["scenario"] = "Numeric vs Numeric"
        print(f"  Type: {results['scenario']}\n")

        # Pearson — linear correlation
        pearson_r, pearson_p = stats.pearsonr(x, y)
        pearson_r = float(pearson_r)

        # Spearman — monotonic correlation (rank-based, outlier-robust)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        spearman_r = float(spearman_r)

        # Mutual Information — captures ANY dependency (linear + non-linear)
        # MI requires 2-D input; reshape single column.
        mi = float(
            mutual_info_regression(
                x.values.reshape(-1, 1), y.values, random_state=42
            )[0]
        )

        results.update({
            "Pearson_r":    round(pearson_r,  4),
            "Pearson_p":    round(float(pearson_p),  6),
            "Spearman_r":   round(spearman_r, 4),
            "Spearman_p":   round(float(spearman_p), 6),
            "Mutual_Info":  round(mi, 4),
            "Zero_Corr_Trap": False,
        })

        _sig = lambda p: "✅ significant" if p < alpha else "❌ not significant"

        print(f"  {'Pearson r (linear)':<32} {pearson_r:+.4f}   p={pearson_p:.5f}  {_sig(pearson_p)}")
        print(f"  {'Spearman ρ (monotonic)':<32} {spearman_r:+.4f}   p={spearman_p:.5f}  {_sig(spearman_p)}")
        print(f"  {'Mutual Information (any shape)':<32} {mi:.4f}")

        # Zero Correlation Trap
        if abs(pearson_r) < mi_trap_threshold and mi > mi_trap_threshold:
            results["Zero_Corr_Trap"] = True
            print(
                f"\n  🚨 ZERO CORRELATION TRAP DETECTED!\n"
                f"     Pearson ≈ 0 ({pearson_r:+.3f}) but MI = {mi:.3f} — "
                f"strong non-linear relationship exists.\n"
                f"     Possible shape: U-curve, wave, or interaction. Do NOT drop this feature."
            )

        # Pearson vs Spearman divergence hint
        if abs(abs(pearson_r) - abs(spearman_r)) > 0.15:
            print(
                f"\n  ⚠️  Pearson ({pearson_r:+.3f}) and Spearman ({spearman_r:+.3f}) "
                f"diverge significantly — outliers may be distorting the linear signal."
            )

    # ── SCENARIO 2: Categorical vs Categorical ────────────────────────────────
    elif not is_x_num and not is_y_num:
        results["scenario"] = "Categorical vs Categorical"
        print(f"  Type: {results['scenario']}\n")

        contingency = pd.crosstab(x, y)
        chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)
        n   = int(contingency.sum().sum())
        r, k = contingency.shape

        # Bias-corrected Cramér's V (Bergsma 2013)
        # The original formula underestimates association for small samples or
        # tables with many cells. The corrected form adjusts phi2 and the
        # denominator using sample-size corrections.
        phi2      = chi2 / n
        phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        k_corr    = k - (k - 1) ** 2 / (n - 1)
        r_corr    = r - (r - 1) ** 2 / (n - 1)
        cramers_v = float(np.sqrt(phi2_corr / max(min(k_corr - 1, r_corr - 1), 1e-10)))

        results.update({
            "Cramers_V":      round(cramers_v, 4),
            "Chi2_statistic": round(float(chi2),   4),
            "Chi2_p":         round(float(chi2_p), 6),
            "Degrees_of_Freedom": int(dof),
            "Contingency_Shape":  (r, k),
        })

        _sig = "✅ significant" if chi2_p < alpha else "❌ not significant"
        print(f"  {'Chi-square statistic':<32} {chi2:.4f}   p={chi2_p:.6f}  {_sig}")
        print(f"  {'Cramér\'s V (bias-corrected)':<32} {cramers_v:.4f}   (0 = no association, 1 = perfect)")
        print(f"  {'Contingency table shape':<32} {r} rows × {k} cols")

        # Effect size interpretation for Cramér's V
        min_dim = min(r, k) - 1
        if min_dim >= 1:
            if cramers_v < 0.10:
                strength = "Negligible"
            elif cramers_v < 0.20:
                strength = "Weak"
            elif cramers_v < 0.40:
                strength = "Moderate"
            else:
                strength = "Strong"
            print(f"  {'Association Strength':<32} {strength}")
            results["Association_Strength"] = strength

    # ── SCENARIO 3: Numeric vs Categorical ───────────────────────────────────
    else:
        results["scenario"] = "Numeric vs Categorical"
        print(f"  Type: {results['scenario']}\n")

        # Ensure correct assignment regardless of col order
        if is_x_num:
            num_col, cat_col = x, y
            num_name, cat_name = col1, col2
        else:
            num_col, cat_col = y, x
            num_name, cat_name = col2, col1

        n_cats = cat_col.nunique()

        # ── Binary categorical: Point-Biserial ────────────────────────────
        if n_cats == 2:
            cat_encoded       = pd.factorize(cat_col)[0]
            pb_r, pb_p        = stats.pointbiserialr(cat_encoded, num_col)
            pb_r              = float(pb_r)
            results.update({
                "Point_Biserial_r": round(abs(pb_r), 4),
                "Point_Biserial_p": round(float(pb_p), 6),
            })
            _sig = "✅ significant" if pb_p < alpha else "❌ not significant"
            print(f"  {'Point-Biserial r (binary)':<32} {abs(pb_r):.4f}   p={pb_p:.6f}  {_sig}")
            # Note: report |r| — direction is arbitrary based on encoding order
            print(f"  (Direction is encoding-dependent; magnitude is meaningful.)")

        # ── Multi-class categorical: Correlation Ratio η + ANOVA ─────────
        else:
            overall_mean = float(num_col.mean())
            ss_between   = sum(
                len(group) * (float(group.mean()) - overall_mean) ** 2
                for _, group in num_col.groupby(cat_col)
            )
            ss_total = float(((num_col - overall_mean) ** 2).sum())

            # Guard against degenerate all-same-value case
            if ss_total == 0:
                eta = 0.0
            else:
                eta = float(np.sqrt(ss_between / ss_total))

            results["Correlation_Ratio_Eta"] = round(eta, 4)
            print(f"  {'Correlation Ratio η (multi-cat)':<32} {eta:.4f}   (η² = {eta**2:.4f} variance explained)")

            # ANOVA significance test (parametric)
            groups    = [grp.values for _, grp in num_col.groupby(cat_col)]
            f_stat, anova_p = stats.f_oneway(*groups)
            results.update({
                "ANOVA_F":  round(float(f_stat),  4),
                "ANOVA_p":  round(float(anova_p), 6),
            })
            _sig_anova = "✅ significant" if anova_p < alpha else "❌ not significant"
            print(f"  {'ANOVA F-statistic':<32} {f_stat:.4f}   p={anova_p:.6f}  {_sig_anova}")

            # Kruskal-Wallis (non-parametric fallback — always reported for robustness)
            kw_stat, kw_p = stats.kruskal(*groups)
            results.update({
                "Kruskal_Wallis_H": round(float(kw_stat), 4),
                "Kruskal_Wallis_p": round(float(kw_p),    6),
            })
            _sig_kw = "✅ significant" if kw_p < alpha else "❌ not significant"
            print(f"  {'Kruskal-Wallis H (non-param.)':<32} {kw_stat:.4f}   p={kw_p:.6f}  {_sig_kw}")

        # Mutual Information (categorical target)
        cat_encoded_mi = pd.factorize(cat_col)[0]
        mi_cat = float(
            mutual_info_classif(
                num_col.values.reshape(-1, 1), cat_encoded_mi, random_state=42
            )[0]
        )
        results["Mutual_Info"] = round(mi_cat, 4)
        print(f"  {'Mutual Information':<32} {mi_cat:.4f}")

    print(f"{'─'*54}\n")

    return results


def test_hypothesis(
    df,
    group_col: str,
    value_col: str,
    alpha: float = 0.05,
    equal_var: bool = False,
    post_hoc: bool = True,
    return_report: bool = False,
):
    """
    Automatically selects and runs the correct hypothesis test for comparing
    groups, computes effect size, and optionally runs post-hoc pairwise tests.

    Decision logic
    --------------
    Normality  → Shapiro-Wilk per group (n ≤ 5 000) or D'Agostino-Pearson (n > 5 000).
    Variance   → Levene's test to decide equal_var for T-test (Welch's by default).
    2 groups   → Welch's T-Test (parametric) / Mann-Whitney U (non-parametric).
    3+ groups  → One-Way ANOVA (parametric) / Kruskal-Wallis H (non-parametric).
    Post-hoc   → Pairwise Welch T-Tests with Bonferroni correction (parametric)
                 / Pairwise Mann-Whitney U with Bonferroni correction (non-parametric).
    Effect size → Cohen's d (2-group) / η² eta-squared (multi-group) for parametric;
                  Rank-Biserial r (2-group) / ε² epsilon-squared (multi-group) for non-parametric.

    Parameters
    ----------
    df           : pd.DataFrame
    group_col    : str   — categorical column defining groups
    value_col    : str   — numeric column being measured
    alpha        : float — significance level (default 0.05)
    equal_var    : bool  — force equal-variance T-test instead of Welch's (default False)
    post_hoc     : bool  — run pairwise post-hoc tests when ≥ 3 groups (default True)
    return_report: bool  — if True, return (p_value, test_name, full_report dict)

    Returns
    -------
    p_value   : float
    test_name : str
    report    : dict  — only when return_report=True
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    for col in (group_col, value_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise TypeError(f"value_col '{value_col}' must be numeric.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")

    temp_df = df[[group_col, value_col]].dropna()
    n_dropped = len(df) - len(temp_df)

    groups      = temp_df[group_col].unique()
    n_groups    = len(groups)

    if n_groups < 2:
        raise ValueError(
            f"At least 2 groups required; found {n_groups} in '{group_col}'."
        )

    data_arrays = [
        temp_df.loc[temp_df[group_col] == g, value_col].values
        for g in groups
    ]

    # ── 1. Normality check ────────────────────────────────────────────────────
    # Use Shapiro-Wilk for small groups; D'Agostino-Pearson for large.
    # Fall back to non-parametric if ANY group fails normality.
    is_parametric    = True
    normality_detail = {}

    for g, arr in zip(groups, data_arrays):
        n_g = len(arr)
        if n_g < 3:
            # Too few points for any normality test — assume non-normal
            normality_detail[str(g)] = {"test": "N/A", "p": None, "normal": False}
            is_parametric = False
            continue
        if n_g <= 5_000:
            stat_n, p_n = stats.shapiro(arr)
            test_name_n  = "Shapiro-Wilk"
        else:
            stat_n, p_n = stats.normaltest(arr)
            test_name_n  = "D'Agostino-Pearson"

        is_normal = bool(p_n > alpha)
        normality_detail[str(g)] = {
            "test": test_name_n, "statistic": round(float(stat_n), 5),
            "p": round(float(p_n), 6), "normal": is_normal,
        }
        if not is_normal:
            is_parametric = False

    # ── 2. Variance homogeneity (for T-test decision) ─────────────────────────
    # Levene's test is robust to non-normality, so it works as a check
    # even when we'll ultimately use non-parametric tests.
    levene_p = None
    if n_groups >= 2:
        _, levene_p = stats.levene(*data_arrays)
        levene_p    = float(levene_p)
        # If variances are unequal, force Welch's (equal_var=False)
        if levene_p < alpha:
            equal_var = False

    # ── 3. Primary hypothesis test ────────────────────────────────────────────
    if n_groups == 2:
        if is_parametric:
            test_name = "Welch's T-Test" if not equal_var else "Student's T-Test"
            stat, p_value = stats.ttest_ind(
                data_arrays[0], data_arrays[1], equal_var=equal_var
            )
        else:
            test_name = "Mann-Whitney U Test"
            stat, p_value = stats.mannwhitneyu(
                data_arrays[0], data_arrays[1], alternative="two-sided"
            )
    else:
        if is_parametric:
            test_name = "One-Way ANOVA"
            stat, p_value = stats.f_oneway(*data_arrays)
        else:
            test_name = "Kruskal-Wallis H Test"
            stat, p_value = stats.kruskal(*data_arrays)

    stat, p_value = float(stat), float(p_value)

    # ── 4. Effect size ────────────────────────────────────────────────────────
    effect_size_name  = None
    effect_size_value = None
    effect_strength   = None

    if n_groups == 2:
        a, b = data_arrays[0], data_arrays[1]
        if is_parametric:
            # Cohen's d — pooled standard deviation
            pooled_std = np.sqrt(
                ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
                / (len(a) + len(b) - 2)
            )
            effect_size_value = float(abs(np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else 0.0
            effect_size_name  = "Cohen's d"
            # Cohen's d benchmarks: small=0.2, medium=0.5, large=0.8
            if effect_size_value < 0.2:    effect_strength = "Negligible"
            elif effect_size_value < 0.5:  effect_strength = "Small"
            elif effect_size_value < 0.8:  effect_strength = "Medium"
            else:                           effect_strength = "Large"
        else:
            # Rank-Biserial r = 1 - (2 × U) / (n₁ × n₂)
            u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
            effect_size_value = float(1 - (2 * u_stat) / (len(a) * len(b)))
            effect_size_name  = "Rank-Biserial r"
            # Absolute value for strength; sign indicates direction
            abs_r = abs(effect_size_value)
            if abs_r < 0.1:    effect_strength = "Negligible"
            elif abs_r < 0.3:  effect_strength = "Small"
            elif abs_r < 0.5:  effect_strength = "Medium"
            else:               effect_strength = "Large"
    else:
        total_n = sum(len(a) for a in data_arrays)
        if is_parametric:
            # η² (eta-squared) = SS_between / SS_total
            overall_mean = np.concatenate(data_arrays).mean()
            ss_between   = sum(len(a) * (a.mean() - overall_mean) ** 2 for a in data_arrays)
            ss_total     = sum(((v - overall_mean) ** 2).sum() for v in data_arrays)
            effect_size_value = float(ss_between / ss_total) if ss_total > 0 else 0.0
            effect_size_name  = "η² (eta-squared)"
            if effect_size_value < 0.01:   effect_strength = "Negligible"
            elif effect_size_value < 0.06: effect_strength = "Small"
            elif effect_size_value < 0.14: effect_strength = "Medium"
            else:                           effect_strength = "Large"
        else:
            # ε² (epsilon-squared) for Kruskal-Wallis
            h_stat, _ = stats.kruskal(*data_arrays)
            effect_size_value = float((h_stat - n_groups + 1) / (total_n - n_groups))
            effect_size_value = max(0.0, effect_size_value)   # clip to 0
            effect_size_name  = "ε² (epsilon-squared)"
            if effect_size_value < 0.01:   effect_strength = "Negligible"
            elif effect_size_value < 0.06: effect_strength = "Small"
            elif effect_size_value < 0.14: effect_strength = "Medium"
            else:                           effect_strength = "Large"

    # ── 5. Post-hoc pairwise tests ────────────────────────────────────────────
    post_hoc_results = []
    if post_hoc and n_groups >= 3:
        pairs          = list(combinations(range(n_groups), 2))
        n_comparisons  = len(pairs)   # Bonferroni correction
        alpha_adjusted = alpha / n_comparisons

        for i, j in pairs:
            gi, gj = groups[i], groups[j]
            ai, aj = data_arrays[i], data_arrays[j]

            if is_parametric:
                _, ph_p = stats.ttest_ind(ai, aj, equal_var=equal_var)
            else:
                _, ph_p = stats.mannwhitneyu(ai, aj, alternative="two-sided")

            ph_p = float(ph_p)
            post_hoc_results.append({
                "Group_A":           str(gi),
                "Group_B":           str(gj),
                "p_value_raw":       round(ph_p, 6),
                "p_value_bonferroni": round(min(ph_p * n_comparisons, 1.0), 6),
                "significant":       ph_p < alpha_adjusted,
            })

    # ── 6. Assemble report ────────────────────────────────────────────────────
    report = {
        "group_col":        group_col,
        "value_col":        value_col,
        "n_total":          len(temp_df),
        "n_dropped":        n_dropped,
        "n_groups":         n_groups,
        "groups":           [str(g) for g in groups],
        "group_sizes":      {str(g): int(len(a)) for g, a in zip(groups, data_arrays)},
        "group_means":      {str(g): round(float(a.mean()), 4) for g, a in zip(groups, data_arrays)},
        "group_medians":    {str(g): round(float(np.median(a)), 4) for g, a in zip(groups, data_arrays)},
        "is_parametric":    is_parametric,
        "normality_detail": normality_detail,
        "levene_p":         round(levene_p, 6) if levene_p is not None else None,
        "test_name":        test_name,
        "statistic":        round(stat, 5),
        "p_value":          round(p_value, 6),
        "significant":      bool(p_value < alpha),
        "effect_size_name":  effect_size_name,
        "effect_size_value": round(effect_size_value, 4) if effect_size_value is not None else None,
        "effect_strength":   effect_strength,
        "post_hoc":         post_hoc_results,
    }

    # ── 7. Print summary ──────────────────────────────────────────────────────
    w = 26
    print(f"\n{'─'*56}")
    print(f"  Hypothesis Test  ·  '{value_col}'  grouped by  '{group_col}'")
    print(f"{'─'*56}")
    print(f"  {'n (complete rows)':<{w}} {len(temp_df):,}  (dropped {n_dropped:,})")
    print(f"  {'Groups':<{w}} {[str(g) for g in groups]}")
    print(f"  {'Group sizes':<{w}} { {str(g): len(a) for g, a in zip(groups, data_arrays)} }")
    print(f"  {'Group means':<{w}} { {str(g): round(float(a.mean()),2) for g, a in zip(groups, data_arrays)} }")

    print(f"\n  Assumption Checks")
    norm_result = "✅ Passed" if is_parametric else "❌ Failed → Non-Parametric"
    print(f"  {'Normality (per group)':<{w}} {norm_result}")
    if levene_p is not None:
        lev_result = f"p={levene_p:.4f}  {'✅ Equal variances' if levene_p >= alpha else '⚠️ Unequal → Welch\'s'}"
        print(f"  {'Levene\'s Variance Test':<{w}} {lev_result}")

    print(f"\n  Test Selected        {test_name}")
    print(f"  {'Statistic':<{w}} {stat:.5f}")
    print(f"  {'P-Value':<{w}} {p_value:.6f}")
    print(f"  {'Effect Size':<{w}} {effect_size_name} = {effect_size_value:.4f}  [{effect_strength}]")

    sig_icon = "🚨" if p_value < alpha else "⚖️"
    conclusion = "SIGNIFICANT" if p_value < alpha else "NOT SIGNIFICANT"
    h0_action  = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"
    print(f"\n  {sig_icon}  {conclusion}  (p {'<' if p_value < alpha else '>='} {alpha})  →  {h0_action}")

    if p_value < alpha:
        print(
            f"  There IS a statistically significant difference in '{value_col}' "
            f"across '{group_col}' groups.\n"
            f"  Effect size is {effect_strength.lower()} ({effect_size_name} = {effect_size_value:.3f}).\n"
            f"  Probability this occurred by chance: < {alpha*100:.1f}%."
        )
    else:
        print(
            f"  There is NO statistically significant difference in '{value_col}' "
            f"across '{group_col}' groups.\n"
            f"  Observed differences are consistent with random variation."
        )

    if post_hoc_results:
        n_sig = sum(r["significant"] for r in post_hoc_results)
        print(f"\n  Post-Hoc Pairwise Tests  (Bonferroni α = {alpha}/{len(pairs)} = {alpha_adjusted:.4f})")
        for r in post_hoc_results:
            icon = "🚨" if r["significant"] else "  "
            print(
                f"  {icon} {r['Group_A']} vs {r['Group_B']:<16} "
                f"p_raw={r['p_value_raw']:.5f}  "
                f"p_bonf={r['p_value_bonferroni']:.5f}  "
                f"{'*sig*' if r['significant'] else ''}"
            )
        print(f"  {n_sig} of {len(post_hoc_results)} pairs are significant after correction.")

    print(f"{'─'*56}\n")

    if return_report:
        return p_value, test_name, report

    return p_value, test_name


def enforce_stationarity(
    series,
    alpha: float = 0.05,
    max_diff: int = 2,
    seasonal_period: int = None,
    use_kpss: bool = True,
    regression: str = "c",
):
    """
    Tests for stationarity using ADF (and optionally KPSS), then applies the
    minimum number of differencing steps needed to achieve stationarity.

    ADF / KPSS consensus logic
    --------------------------
    ADF alone has low power against near-unit-root alternatives, which means
    it can fail to reject non-stationarity even when differencing is needed.
    KPSS has the opposite null hypothesis (H₀: stationary), so using both
    provides a robust consensus:

      ADF rejects H₀ (p < α) AND KPSS fails to reject H₀ (p > α) → Stationary ✅
      ADF fails       OR  KPSS rejects                              → Non-Stationary ❌

    This dual-test approach avoids both false positives (ADF alone) and
    false negatives (KPSS alone).

    Parameters
    ----------
    series          : pd.Series — time-series data (index should be datetime or ordered)
    alpha           : float     — significance level (default 0.05)
    max_diff        : int       — maximum regular differencing steps before giving up (default 2)
    seasonal_period : int | None — if provided, applies seasonal differencing (lag=period)
                                   BEFORE regular differencing when regular diff fails
    use_kpss        : bool      — also run KPSS test for consensus verdict (default True)
    regression      : str       — ADF regression type: 'c' (constant, default),
                                  'ct' (constant+trend), 'n' (none)

    Returns
    -------
    stationary_series : pd.Series — transformed series (index preserved where possible)
    d                 : int       — number of regular differencing steps applied
    report            : dict      — full diagnostic summary
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Series must be numeric.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
    if max_diff < 0:
        raise ValueError(f"max_diff must be ≥ 0, got {max_diff}.")

    VALID_REGRESSION = {"c", "ct", "n", "ctt"}
    if regression not in VALID_REGRESSION:
        raise ValueError(f"regression must be one of {VALID_REGRESSION}, got '{regression}'.")

    ts_data  = series.dropna().copy()
    n_dropped = len(series) - len(ts_data)
    col_name  = series.name if series.name else "series"

    # ADF requires at least ~20 observations for the test to be meaningful
    MIN_OBS = 20
    if len(ts_data) < MIN_OBS:
        raise ValueError(
            f"Series '{col_name}' has only {len(ts_data)} observations after dropping NaN. "
            f"ADF requires at least {MIN_OBS} for reliable results."
        )

    report = {
        "column":           col_name,
        "n_original":       len(series),
        "n_after_dropna":   len(ts_data),
        "n_null_dropped":   n_dropped,
        "alpha":            alpha,
        "iterations":       [],
        "d":                0,
        "seasonal_diff_applied": False,
        "converged":        False,
        "final_recommendation": None,
    }

    w = 22
    print(f"\n{'─'*52}")
    print(f"  Stationarity Check  ·  '{col_name}'")
    print(f"  n={len(ts_data):,}  (dropped {n_dropped:,} NaN)  |  α={alpha}  |  max_diff={max_diff}")
    print(f"{'─'*52}")

    def _run_adf(data):
        result      = adfuller(data, regression=regression, autolag="AIC")
        return {
            "statistic":       round(float(result[0]), 6),
            "p_value":         round(float(result[1]), 6),
            "n_lags_used":     int(result[2]),
            "n_obs_used":      int(result[3]),
            "critical_values": {k: round(v, 4) for k, v in result[4].items()},
            "is_stationary":   result[1] < alpha,
        }

    def _run_kpss(data):
        # KPSS: H₀ = stationary. Suppress the truncation warning that fires when
        # n_lags is set to 'legacy' — use 'auto' instead.
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p, n_lags, crits = kpss(data, regression="c", nlags="auto")
            return {
                "statistic":     round(float(stat),   6),
                "p_value":       round(float(p),      6),   # p ≤ 0.1 only (KPSS p is bounded)
                "n_lags_used":   int(n_lags),
                "is_stationary": float(p) > alpha,           # H₀=stationary, so large p = stationary
            }
        except Exception as e:
            return {"error": str(e), "is_stationary": None}

    def _consensus(adf_res, kpss_res):
        """Both ADF (reject non-stationarity) and KPSS (fail to reject stationarity) must agree."""
        if kpss_res.get("is_stationary") is None:
            return adf_res["is_stationary"]   # fallback to ADF only
        return adf_res["is_stationary"] and kpss_res["is_stationary"]

    # ── 1. Optional seasonal differencing ─────────────────────────────────────
    # Applied once BEFORE regular differencing if a period is provided.
    if seasonal_period and seasonal_period > 1:
        adf0 = _run_adf(ts_data)
        if not adf0["is_stationary"]:
            print(f"  Applying seasonal differencing (period={seasonal_period}) first...")
            ts_data = ts_data.diff(seasonal_period).dropna()
            report["seasonal_diff_applied"] = True
            if len(ts_data) < MIN_OBS:
                print(f"  ⚠️  Too few observations after seasonal differencing — skipping.")
                ts_data = series.dropna().copy()
                report["seasonal_diff_applied"] = False

    # ── 2. Iterative regular differencing ─────────────────────────────────────
    for d in range(max_diff + 1):
        adf_res  = _run_adf(ts_data)
        kpss_res = _run_kpss(ts_data) if use_kpss else {"is_stationary": None}
        is_stat  = _consensus(adf_res, kpss_res)

        iter_record = {
            "d":    d,
            "ADF":  adf_res,
            "KPSS": kpss_res if use_kpss else None,
            "consensus_stationary": is_stat,
        }
        report["iterations"].append(iter_record)

        # Print iteration results
        print(f"\n  Diff d={d}")
        print(f"  {'ADF Statistic':<{w}} {adf_res['statistic']:+.4f}")
        print(f"  {'ADF P-Value':<{w}} {adf_res['p_value']:.6f}  "
              f"{'✅ reject H₀ (stationary)' if adf_res['is_stationary'] else '❌ fail to reject H₀'}")
        print(f"  {'ADF Critical Values':<{w}} {adf_res['critical_values']}")

        if use_kpss and "error" not in kpss_res:
            print(f"  {'KPSS Statistic':<{w}} {kpss_res['statistic']:+.4f}")
            print(f"  {'KPSS P-Value':<{w}} {kpss_res['p_value']:.6f}  "
                  f"{'✅ H₀ holds (stationary)' if kpss_res['is_stationary'] else '❌ reject H₀ (non-stationary)'}")

        print(f"  {'Consensus':<{w}} {'✅ STATIONARY' if is_stat else '❌ NON-STATIONARY'}")

        if is_stat:
            report["d"]                    = d
            report["converged"]            = True
            report["final_recommendation"] = (
                f"Use d={d} in ARIMA/SARIMA. "
                f"{'Apply log transform first if variance is growing.' if d >= 2 else ''}"
            )
            print(f"\n  ✅ Stationary achieved at d={d}. Differencing complete.")
            break

        # Not yet stationary — difference if budget allows
        if d < max_diff:
            ts_data = ts_data.diff(1).dropna()
        else:
            # Exhausted differencing budget
            report["d"] = d
            report["converged"] = False
            report["final_recommendation"] = (
                f"Still non-stationary after d={d} differences. "
                f"Try: log transform (if variance grows), "
                f"seasonal differencing (period=12/4/7), "
                f"or KPSS/PP test for structural breaks."
            )
            print(
                f"\n  🚨 max_diff={max_diff} reached — series remains non-stationary.\n"
                f"  Suggestions:\n"
                f"    • Log-transform if variance is increasing (np.log1p)\n"
                f"    • Try seasonal differencing (seasonal_period=12/4/7)\n"
                f"    • Check for structural breaks (Zivot-Andrews test)\n"
                f"    • Consider fractional differencing (ARFIMA)"
            )

    print(f"{'─'*52}\n")
    return ts_data, report["d"], report


def analyze_autocorrelation(
    series,
    lags: int = 30,
    alpha: float = 0.05,
    ljungbox_lags: list = None,
    show_plot: bool = True,
    return_report: bool = False,
):
    """
    Analyzes time-series autocorrelation using ACF/PACF plots and the
    Ljung-Box test, then provides ARIMA order hints based on the patterns.

    Parameters
    ----------
    series         : pd.Series — stationary time-series data
    lags           : int       — number of lags for ACF/PACF plots (default 30)
    alpha          : float     — significance level for confidence bounds and
                                 Ljung-Box test (default 0.05)
    ljungbox_lags  : list | None
                     Specific lags to test with Ljung-Box. If None, tests at
                     [lags//4, lags//2, lags] to catch autocorrelation at
                     multiple horizons rather than just a single lag.
    show_plot      : bool — render the ACF/PACF visualization (default True)
    return_report  : bool — if True, return (lb_df, report_dict) instead of lb_df

    Returns
    -------
    lb_df          : pd.DataFrame — Ljung-Box test results
    report (opt.)  : dict         — only when return_report=True
    """

    # ── 0. Guards ─────────────────────────────────────────────────────────────
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Series must be numeric.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
    if lags < 1:
        raise ValueError(f"lags must be ≥ 1, got {lags}.")

    ts_data  = series.dropna()
    n        = len(ts_data)
    n_null   = len(series) - n
    col_name = series.name if series.name else "series"

    # PACF requires at least lags + 2 observations
    # ACF/Ljung-Box are unreliable below ~20 observations
    MIN_OBS = max(lags + 2, 20)
    if n < MIN_OBS:
        raise ValueError(
            f"Series has {n} observations after dropping NaN — need at least "
            f"{MIN_OBS} (lags={lags} + buffer). Reduce lags or provide more data."
        )

    # Clamp lags to n//2 — ACF/PACF values beyond n/2 are unreliable
    max_safe_lags = n // 2 - 1
    if lags > max_safe_lags:
        print(
            f"[analyze_autocorrelation] ⚠️  lags={lags} exceeds n//2 - 1 = {max_safe_lags}. "
            f"Clamped to {max_safe_lags}."
        )
        lags = max_safe_lags

    # ── 1. Ljung-Box test at multiple lags ────────────────────────────────────
    # The original tested only at a single lag. Testing at multiple horizons
    # catches autocorrelation that only manifests at shorter or longer lags.
    if ljungbox_lags is None:
        ljungbox_lags = sorted(set([
            max(1, lags // 4),
            max(1, lags // 2),
            lags,
        ]))

    lb_df     = acorr_ljungbox(ts_data, lags=ljungbox_lags, return_df=True)
    lb_df.index.name = "lag"

    # Consensus: significant if ANY tested lag shows autocorrelation
    any_significant  = (lb_df["lb_pvalue"] < alpha).any()
    min_p            = float(lb_df["lb_pvalue"].min())
    min_p_lag        = int(lb_df["lb_pvalue"].idxmin())

    # ── 2. ACF / PACF pattern interpretation → ARIMA order hints ─────────────
    # This heuristic uses the classical Box-Jenkins pattern-reading rules.
    from statsmodels.tsa.stattools import acf, pacf

    acf_vals  = acf(ts_data,  nlags=lags,  alpha=None, fft=True)
    # PACF: 'ywm' is Yule-Walker with bias correction — more stable than 'ywmle'
    # for short series. It is the safest default across series lengths.
    pacf_vals = pacf(ts_data, nlags=lags, alpha=None, method="ywm")

    # Significance bound ≈ ±1.96 / √n for 95 % confidence
    bound = 1.96 / np.sqrt(n)

    # Count significant lags (excluding lag 0 for ACF)
    acf_sig_lags  = [i for i in range(1, lags + 1) if abs(acf_vals[i])  > bound]
    pacf_sig_lags = [i for i in range(1, lags + 1) if abs(pacf_vals[i]) > bound]

    n_acf_sig  = len(acf_sig_lags)
    n_pacf_sig = len(pacf_sig_lags)

    # Box-Jenkins pattern rules:
    # AR(p):  ACF tails off | PACF cuts off at lag p
    # MA(q):  ACF cuts off at lag q | PACF tails off
    # ARMA:   Both tail off
    # White Noise: Neither significant
    if not acf_sig_lags and not pacf_sig_lags:
        pattern       = "White Noise — no AR or MA structure detected"
        arima_hint_p  = 0
        arima_hint_q  = 0
    elif acf_sig_lags and not pacf_sig_lags:
        pattern       = "MA(q) — ACF cuts off, PACF tails off"
        arima_hint_p  = 0
        arima_hint_q  = acf_sig_lags[-1]   # last significant ACF lag as q bound
    elif pacf_sig_lags and not acf_sig_lags:
        pattern       = "AR(p) — PACF cuts off, ACF tails off"
        arima_hint_p  = pacf_sig_lags[-1]  # last significant PACF lag as p bound
        arima_hint_q  = 0
    else:
        pattern       = "ARMA(p,q) — both ACF and PACF tail off"
        arima_hint_p  = min(pacf_sig_lags[-1], 3)   # cap at 3 to avoid over-fitting
        arima_hint_q  = min(acf_sig_lags[-1],  3)

    # ── 3. Print report ───────────────────────────────────────────────────────
    w = 26
    print(f"\n{'─'*56}")
    print(f"  Autocorrelation Analysis  ·  '{col_name}'")
    print(f"  n={n:,}  (dropped {n_null:,} NaN)  |  lags={lags}  |  α={alpha}")
    print(f"{'─'*56}")

    print(f"\n  Ljung-Box Test  (H₀: white noise — no autocorrelation)")
    for lag_val, row in lb_df.iterrows():
        sig = "🚨 significant" if row["lb_pvalue"] < alpha else "  not significant"
        print(
            f"  {'Lag ' + str(lag_val):<{w}} "
            f"stat={row['lb_stat']:.4f}  p={row['lb_pvalue']:.6f}  {sig}"
        )

    print(
        f"\n  {'Consensus Verdict':<{w}} "
        f"{'🚨 AUTOCORRELATION DETECTED' if any_significant else '✅ WHITE NOISE (no autocorrelation)'}"
    )
    if any_significant:
        print(
            f"  Past values significantly predict future values (strongest at lag {min_p_lag}, "
            f"p={min_p:.5f}).\n  An AR/MA/ARMA model is appropriate."
        )
    else:
        print(
            f"  No significant autocorrelation found. "
            f"Past values do not predict future values."
        )

    print(f"\n  ACF/PACF Pattern Analysis")
    print(f"  {'ACF significant lags':<{w}} {acf_sig_lags[:10]}{'...' if n_acf_sig > 10 else ''}")
    print(f"  {'PACF significant lags':<{w}} {pacf_sig_lags[:10]}{'...' if n_pacf_sig > 10 else ''}")
    print(f"  {'Pattern':<{w}} {pattern}")
    if arima_hint_p > 0 or arima_hint_q > 0:
        print(f"  {'ARIMA Order Hint':<{w}} ARIMA(p≤{arima_hint_p}, d=?, q≤{arima_hint_q})")
        print(f"  (d comes from enforce_stationarity; use auto_arima to confirm p, q)")

    print(f"{'─'*56}\n")

    # ── 4. Visualization ──────────────────────────────────────────────────────
    if show_plot:
        fig  = plt.figure(figsize=(15, 10))
        gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_acf  = fig.add_subplot(gs[0, :])   # ACF spans full top row
        ax_pacf = fig.add_subplot(gs[1, 0])   # PACF bottom-left
        ax_lb   = fig.add_subplot(gs[1, 1])   # Ljung-Box p-values bottom-right

        # ACF
        plot_acf(ts_data, lags=lags, ax=ax_acf, alpha=alpha, color="#4a90d9",
                 vlines_kwargs={"colors": "#4a90d9"})
        ax_acf.set_title(
            f"ACF  ·  '{col_name}'\n"
            f"(Moving Average 'q' indicator — bars outside bounds suggest MA order)",
            fontsize=11, fontweight="bold",
        )
        ax_acf.set_xlabel("Lag", fontsize=9)
        ax_acf.set_ylabel("Correlation", fontsize=9)
        ax_acf.grid(axis="y", alpha=0.3, linestyle=":")
        ax_acf.axhline(0, color="black", linewidth=0.8)

        # PACF — method='ywm' is bias-corrected Yule-Walker, safe for all series lengths
        plot_pacf(ts_data, lags=lags, ax=ax_pacf, alpha=alpha, method="ywm",
                  color="#2ecc71", vlines_kwargs={"colors": "#2ecc71"})
        ax_pacf.set_title(
            "PACF\n(Auto-Regressive 'p' indicator)",
            fontsize=11, fontweight="bold",
        )
        ax_pacf.set_xlabel("Lag", fontsize=9)
        ax_pacf.set_ylabel("Direct Correlation", fontsize=9)
        ax_pacf.grid(axis="y", alpha=0.3, linestyle=":")
        ax_pacf.axhline(0, color="black", linewidth=0.8)

        # Ljung-Box p-value plot — visualizes significance across lags
        lb_all = acorr_ljungbox(ts_data, lags=list(range(1, lags + 1)), return_df=True)
        ax_lb.plot(lb_all.index, lb_all["lb_pvalue"], color="#e74c3c", linewidth=1.5,
                   label="Ljung-Box p-value")
        ax_lb.axhline(alpha, color="gray", linestyle="--", linewidth=1.2,
                      label=f"α = {alpha}")
        ax_lb.fill_between(lb_all.index, lb_all["lb_pvalue"], alpha,
                           where=lb_all["lb_pvalue"] < alpha,
                           color="#e74c3c", alpha=0.15, label="Significant region")
        ax_lb.set_title("Ljung-Box P-Values by Lag", fontsize=11, fontweight="bold")
        ax_lb.set_xlabel("Lag", fontsize=9)
        ax_lb.set_ylabel("P-Value", fontsize=9)
        ax_lb.set_ylim(0, 1)
        ax_lb.legend(fontsize=8)
        ax_lb.grid(alpha=0.3, linestyle=":")

        plt.suptitle(
            f"Autocorrelation Diagnostics  ·  '{col_name}'",
            fontsize=13, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        plt.show()

    # ── 5. Assemble report ────────────────────────────────────────────────────
    report = {
        "column":             col_name,
        "n":                  n,
        "n_null_dropped":     n_null,
        "lags_plotted":       lags,
        "ljungbox_lags":      ljungbox_lags,
        "any_significant":    any_significant,
        "min_p_value":        round(min_p, 6),
        "min_p_lag":          min_p_lag,
        "acf_significant_lags":  acf_sig_lags,
        "pacf_significant_lags": pacf_sig_lags,
        "pattern":            pattern,
        "arima_hint_p":       arima_hint_p,
        "arima_hint_q":       arima_hint_q,
    }

    if return_report:
        return lb_df, report

    return lb_df
