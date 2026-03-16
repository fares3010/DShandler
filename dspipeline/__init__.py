"""
dspipeline
──────────
A stateful, chainable data science pipeline class covering every stage of
the tabular ML workflow: diagnostics, cleaning, anomaly handling, transformation,
encoding, feature selection, leakproof splitting, EDA, and time-series analysis.

Quick start
-----------
>>> from dspipeline import DataSciencePipeline
>>> dsp = DataSciencePipeline(df, target_col="Churn", task_type="classification")
>>> dsp.run_diagnostics().run_cleaning().run_preprocessing()
>>> X_train, X_test, y_train, y_test, preprocessor = dsp.split()
"""

from .pipeline import DataSciencePipeline

# Expose individual function groups for direct import
from .diagnostics import (
    advanced_missing_profiler,
    detect_anomalies,
    detect_categorical_issues,
    detect_dimensional_issues,
    detect_leakage_risks,
    detect_predictive_issues,
    detect_structural_anomalies,
)
from .cleaning import (
    advanced_knn_impute,
    format_structural_issues,
    handle_anomalies,
    handle_categorical_missing,
    handle_duplicates,
    handle_numerical_missing,
    missforest_impute,
    standardize_data,
)
from .preprocessing import (
    encode_categorical_data,
    optimize_features,
    optimize_vif,
    setup_leakproof_environment,
    transform_data_shape,
)
from .statistics import (
    analyze_autocorrelation,
    analyze_distribution,
    analyze_relationship,
    enforce_stationarity,
    evaluate_distribution,
    test_hypothesis,
)

__version__ = "0.1.2"
__author__  = "Your Name"
__email__   = "your@email.com"

__all__ = [
    # Main class
    "DataSciencePipeline",
    # Diagnostics
    "advanced_missing_profiler",
    "detect_anomalies",
    "detect_categorical_issues",
    "detect_dimensional_issues",
    "detect_leakage_risks",
    "detect_predictive_issues",
    "detect_structural_anomalies",
    # Cleaning
    "advanced_knn_impute",
    "format_structural_issues",
    "handle_anomalies",
    "handle_categorical_missing",
    "handle_duplicates",
    "handle_numerical_missing",
    "missforest_impute",
    "standardize_data",
    # Preprocessing
    "encode_categorical_data",
    "optimize_features",
    "optimize_vif",
    "setup_leakproof_environment",
    "transform_data_shape",
    # Statistics
    "analyze_autocorrelation",
    "analyze_distribution",
    "analyze_relationship",
    "enforce_stationarity",
    "evaluate_distribution",
    "test_hypothesis",
]
