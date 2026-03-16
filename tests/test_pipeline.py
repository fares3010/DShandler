"""
Basic smoke tests — verify all imports resolve and the class instantiates.
Run with:  pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age":      np.random.randint(18, 80, n).astype(float),
        "salary":   np.random.normal(50_000, 15_000, n),
        "score":    np.random.uniform(0, 1, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "city":     np.random.choice(["london", "paris", "berlin"], n),
        "target":   np.random.randint(0, 2, n),
    })


# ── Import tests ──────────────────────────────────────────────────────────────
def test_top_level_imports():
    from dspipeline import DataSciencePipeline
    from dspipeline import advanced_missing_profiler, detect_anomalies
    from dspipeline import handle_numerical_missing, transform_data_shape
    from dspipeline import analyze_distribution, test_hypothesis


def test_submodule_imports():
    from dspipeline.diagnostics   import detect_anomalies, detect_categorical_issues
    from dspipeline.cleaning      import handle_numerical_missing, handle_duplicates
    from dspipeline.preprocessing import optimize_features, encode_categorical_data
    from dspipeline.statistics    import analyze_distribution, evaluate_distribution


# ── Class instantiation ───────────────────────────────────────────────────────
def test_instantiation(sample_df):
    from dspipeline import DataSciencePipeline
    dsp = DataSciencePipeline(sample_df, target_col="target", task_type="classification")
    assert dsp.df.shape == sample_df.shape
    assert dsp.target_col == "target"
    assert dsp.task_type == "classification"
    assert len(dsp.history) == 0


def test_repr(sample_df):
    from dspipeline import DataSciencePipeline
    dsp = DataSciencePipeline(sample_df, target_col="target")
    assert "DataSciencePipeline" in repr(dsp)


# ── Diagnostics ───────────────────────────────────────────────────────────────
def test_profile_missing(sample_df):
    from dspipeline import DataSciencePipeline
    df_nulls = sample_df.copy()
    df_nulls.loc[:10, "age"] = np.nan
    dsp = DataSciencePipeline(df_nulls, target_col="target")
    dsp.profile_missing(show_heatmap=False, show_bar=False)
    assert "missing_profile" in dsp.results


def test_detect_categorical(sample_df):
    from dspipeline import DataSciencePipeline
    dsp = DataSciencePipeline(sample_df, target_col="target")
    dsp.detect_categorical()
    assert "categorical_report" in dsp.results


# ── Cleaning ──────────────────────────────────────────────────────────────────
def test_impute_numeric(sample_df):
    from dspipeline import DataSciencePipeline
    df_nulls = sample_df.copy()
    df_nulls.loc[:5, "age"] = np.nan
    dsp = DataSciencePipeline(df_nulls, target_col="target")
    before_nulls = dsp.df["age"].isna().sum()
    dsp.impute_numeric(strategy="median")
    assert dsp.df["age"].isna().sum() == 0
    assert before_nulls > 0


def test_drop_duplicates(sample_df):
    from dspipeline import DataSciencePipeline
    df_duped = pd.concat([sample_df, sample_df.iloc[:10]], ignore_index=True)
    dsp = DataSciencePipeline(df_duped, target_col="target")
    dsp.drop_duplicates()
    assert len(dsp.df) == len(sample_df)


# ── Chaining ──────────────────────────────────────────────────────────────────
def test_method_chaining(sample_df):
    from dspipeline import DataSciencePipeline
    dsp = DataSciencePipeline(sample_df, target_col="target")
    result = dsp.impute_numeric(strategy="median").drop_duplicates()
    assert result is dsp   # must return self


# ── Transformation ────────────────────────────────────────────────────────────
def test_transform_shape(sample_df):
    from dspipeline import DataSciencePipeline
    dsp = DataSciencePipeline(sample_df, target_col="target")
    dsp.transform_shape(scale_method="standard")
    assert "transformers" in dsp.results


# ── Split ─────────────────────────────────────────────────────────────────────
def test_split(sample_df):
    from dspipeline import DataSciencePipeline
    dsp = DataSciencePipeline(sample_df, target_col="target")
    dsp.impute_numeric().impute_categorical()
    X_train, X_test, y_train, y_test, preprocessor = dsp.split(test_size=0.2)
    assert len(X_train) + len(X_test) == len(sample_df)
    assert len(y_train) == len(X_train)


# ── Reset ─────────────────────────────────────────────────────────────────────
def test_reset(sample_df):
    from dspipeline import DataSciencePipeline
    dsp = DataSciencePipeline(sample_df, target_col="target")
    dsp.drop_duplicates()
    dsp.reset()
    assert dsp.df.shape == sample_df.shape
    assert len(dsp.history) == 0
