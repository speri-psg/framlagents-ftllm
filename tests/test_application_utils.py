"""Tests for application.py pure utility functions and lambda_rule_analysis.compute_2d_drilldown."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# ── _json_safe ─────────────────────────────────────────────────────────────────

# Import at module level — no side effects from application for this function
# conftest.py stubs dash/make_figures permanently — application can be imported directly.
# lambda_ds_performance is mocked per-test where needed (it IS tested in its own file).
with patch("lambda_ds_performance.discover_dims", return_value=[]), \
     patch("sar_scorer.train", return_value=(None, None)):
    from application import _json_safe, compute_segment_stats, compute_sar_backtest, _filter_by_cluster
    import application as _app


class TestJsonSafe:
    def test_plain_dict_passes_through(self):
        d = {"a": 1, "b": "hello"}
        assert _json_safe(d) == d

    def test_numpy_int_converted(self):
        result = _json_safe(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float_converted(self):
        result = _json_safe(np.float32(3.14))
        assert isinstance(result, float)

    def test_numpy_array_converted_to_list(self):
        arr = np.array([1, 2, 3])
        result = _json_safe(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_nested_dict_with_numpy(self):
        d = {"count": np.int64(5), "value": np.float64(1.5)}
        result = _json_safe(d)
        assert result == {"count": 5, "value": 1.5}
        assert isinstance(result["count"], int)

    def test_list_of_numpy_ints(self):
        lst = [np.int64(1), np.int64(2), np.int64(3)]
        result = _json_safe(lst)
        assert result == [1, 2, 3]
        assert all(isinstance(x, int) for x in result)

    def test_tuple_treated_as_list(self):
        t = (np.int64(10), np.int64(20))
        result = _json_safe(t)
        assert result == [10, 20]

    def test_plain_string_unchanged(self):
        assert _json_safe("hello") == "hello"

    def test_none_unchanged(self):
        assert _json_safe(None) is None

    def test_nested_list_with_dict(self):
        obj = [{"x": np.int64(1)}, {"x": np.int64(2)}]
        result = _json_safe(obj)
        assert result == [{"x": 1}, {"x": 2}]


# ── compute_segment_stats ──────────────────────────────────────────────────────

def _seg_df(n_biz=100, n_ind=50, seed=42):
    """Build a minimal DataFrame matching the schema expected by compute_segment_stats."""
    rng = np.random.default_rng(seed)
    n = n_biz + n_ind
    seg = [0] * n_biz + [1] * n_ind
    return pd.DataFrame({
        "dynamic_segment": seg,
        "alerts":          rng.integers(0, 2, size=n).astype(float),
        "false_positives": rng.integers(0, 2, size=n).astype(float),
        "false_negatives": rng.integers(0, 2, size=n).astype(float),
    })


class TestComputeSegmentStats:
    def test_returns_string(self):
        assert isinstance(compute_segment_stats(_seg_df()), str)

    def test_contains_pre_computed_header(self):
        result = compute_segment_stats(_seg_df())
        assert "PRE-COMPUTED SEGMENT STATS" in result

    def test_contains_end_marker(self):
        result = compute_segment_stats(_seg_df())
        assert "END PRE-COMPUTED SEGMENT STATS" in result

    def test_contains_business_and_individual(self):
        result = compute_segment_stats(_seg_df())
        assert "Business" in result
        assert "Individual" in result

    def test_account_counts_match_input(self):
        result = compute_segment_stats(_seg_df(n_biz=80, n_ind=20))
        assert "80" in result
        assert "20" in result

    def test_fp_rate_present(self):
        result = compute_segment_stats(_seg_df())
        assert "FP rate=" in result

    def test_false_negatives_present(self):
        result = compute_segment_stats(_seg_df())
        assert "False Negatives" in result

    def test_zero_alerts_does_not_crash(self):
        df = _seg_df()
        df["alerts"] = 0
        result = compute_segment_stats(df)
        assert isinstance(result, str)
        assert "0%" in result or "0.0%" in result

    def test_percentage_sums_are_reasonable(self):
        result = compute_segment_stats(_seg_df(n_biz=75, n_ind=25))
        assert "75.0% of total" in result or "75" in result


# ── compute_sar_backtest ───────────────────────────────────────────────────────

def _sar_df(n_sar=20, n_non=80, col="avg_trxn_amt", seed=42):
    rng = np.random.default_rng(seed)
    n = n_sar + n_non
    is_sar = np.array([1] * n_sar + [0] * n_non)
    rng.shuffle(is_sar)
    values = np.where(is_sar == 1,
                      rng.uniform(50000, 200000, n),   # SARs tend to have higher amounts
                      rng.uniform(1000, 80000, n))
    return pd.DataFrame({"is_sar": is_sar, col: values})


class TestComputeSarBacktest:
    def test_returns_string(self):
        assert isinstance(compute_sar_backtest(_sar_df(), "avg_trxn_amt", "Business"), str)

    def test_contains_pre_computed_header(self):
        result = compute_sar_backtest(_sar_df(), "avg_trxn_amt", "Business")
        assert "PRE-COMPUTED SAR BACKTEST" in result

    def test_contains_end_marker(self):
        result = compute_sar_backtest(_sar_df(), "avg_trxn_amt", "Business")
        assert "END PRE-COMPUTED SAR BACKTEST" in result

    def test_contains_segment_name(self):
        result = compute_sar_backtest(_sar_df(), "avg_trxn_amt", "Individual")
        assert "Individual" in result

    def test_contains_sar_count(self):
        result = compute_sar_backtest(_sar_df(n_sar=20), "avg_trxn_amt", "Business")
        assert "20" in result

    def test_no_sars_returns_message(self):
        df = _sar_df()
        df["is_sar"] = 0
        result = compute_sar_backtest(df, "avg_trxn_amt", "Business")
        assert "No simulated SARs" in result

    def test_90pct_catch_rate_line_present(self):
        result = compute_sar_backtest(_sar_df(n_sar=30), "avg_trxn_amt", "Business")
        assert "90%" in result

    def test_at_lowest_threshold_all_caught(self):
        result = compute_sar_backtest(_sar_df(n_sar=10), "avg_trxn_amt", "Business")
        assert "100%" in result

    def test_nan_rows_dropped(self):
        df = _sar_df(n_sar=15)
        df.loc[df.index[:5], "avg_trxn_amt"] = np.nan
        result = compute_sar_backtest(df, "avg_trxn_amt", "Business")
        assert isinstance(result, str)
        assert "PRE-COMPUTED SAR BACKTEST" in result


# ── _filter_by_cluster ─────────────────────────────────────────────────────────

def _rule_sweep_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customer_id": range(n),
        "risk_factor": ["Elder Abuse"] * n,
        "is_sar":      rng.integers(0, 2, size=n),
        "trigger_amt": rng.uniform(1000, 100000, size=n),
        "cluster":     rng.integers(1, 5, size=n),
    })


class TestFilterByCluster:
    def test_none_cluster_returns_full_df(self):
        df = _rule_sweep_df()
        result = _filter_by_cluster(df, None)
        assert len(result) == len(df)

    def test_cluster_filter_narrows_rows(self):
        df = _rule_sweep_df(100)
        cluster_labels = pd.DataFrame({
            "customer_id": range(100),
            "cluster":     [1] * 25 + [2] * 25 + [3] * 25 + [4] * 25,
        })
        with patch.object(_app, "DF_CLUSTER_LABELS", cluster_labels):
            result = _filter_by_cluster(df, 1)
        assert len(result) == 25

    def test_nonexistent_cluster_returns_empty(self):
        df = _rule_sweep_df(50)
        cluster_labels = pd.DataFrame({
            "customer_id": range(50),
            "cluster":     [1] * 50,
        })
        with patch.object(_app, "DF_CLUSTER_LABELS", cluster_labels):
            result = _filter_by_cluster(df, 9)
        assert len(result) == 0

    def test_none_cluster_labels_returns_full_df(self):
        df = _rule_sweep_df(50)
        with patch.object(_app, "DF_CLUSTER_LABELS", None):
            result = _filter_by_cluster(df, 2)
        assert len(result) == len(df)


# ── compute_2d_drilldown (lambda_rule_analysis) ────────────────────────────────

from lambda_rule_analysis import compute_2d_drilldown, RULE_CATALOGUE
import numpy as np


def _rule_df_for_drilldown(rule_name, n_sar=20, n_fp=80, seed=7):
    rng = np.random.default_rng(seed)
    n = n_sar + n_fp
    is_sar = np.array([1] * n_sar + [0] * n_fp)
    rng.shuffle(is_sar)
    return pd.DataFrame({
        "risk_factor":  [rule_name] * n,
        "is_sar":       is_sar,
        "customer_id":  range(n),
        "trigger_amt":  rng.uniform(1000, 100000, size=n),
        "z_score":      rng.uniform(0, 10, size=n),
        "age":          rng.integers(30, 90, size=n).astype(float),
        "pair_total":   rng.uniform(1000, 50000, size=n),
        "ratio":        rng.uniform(0.8, 1.2, size=n),
        "days_observed":rng.integers(1, 10, size=n).astype(float),
        "txn_count":    rng.integers(1, 20, size=n).astype(float),
        "cluster":      rng.integers(1, 5, size=n),
    })


class TestCompute2dDrilldown:
    def test_valid_call_returns_six_values(self):
        rf = "Activity Deviation (ACH)"
        df = _rule_df_for_drilldown(rf)
        result = compute_2d_drilldown(df, "Activity Deviation (ACH)", "floor_amount", 5000, "z_threshold", 2.0)
        assert len(result) == 6

    def test_tp_fp_fn_tn_are_dataframes(self):
        rf = "Activity Deviation (ACH)"
        df = _rule_df_for_drilldown(rf)
        tp, fp, fn, tn, col1, col2 = compute_2d_drilldown(df, "Activity Deviation (ACH)", "floor_amount", 5000, "z_threshold", 2.0)
        for frame in [tp, fp, fn, tn]:
            assert isinstance(frame, pd.DataFrame)

    def test_col_names_returned(self):
        rf = "Activity Deviation (ACH)"
        df = _rule_df_for_drilldown(rf)
        tp, fp, fn, tn, col1, col2 = compute_2d_drilldown(df, "Activity Deviation (ACH)", "floor_amount", 5000, "z_threshold", 2.0)
        assert isinstance(col1, str)
        assert isinstance(col2, str)

    def test_unknown_rule_returns_none_tuple(self):
        df = _rule_df_for_drilldown("Activity Deviation (ACH)")
        result = compute_2d_drilldown(df, "nonexistent rule xyz", "floor_amount", 5000, "z_threshold", 2.0)
        assert result == (None,) * 6

    def test_invalid_param_returns_none_tuple(self):
        rf = "Activity Deviation (ACH)"
        df = _rule_df_for_drilldown(rf)
        result = compute_2d_drilldown(df, "Activity Deviation (ACH)", "bad_param", 5000, "z_threshold", 2.0)
        assert result == (None,) * 6

    def test_tp_plus_fn_equals_total_sars(self):
        rf = "Activity Deviation (ACH)"
        n_sar = 20
        df = _rule_df_for_drilldown(rf, n_sar=n_sar)
        tp, fp, fn, tn, _, _ = compute_2d_drilldown(df, "Activity Deviation (ACH)", "floor_amount", 5000, "z_threshold", 2.0)
        assert len(tp) + len(fn) == n_sar

    def test_fp_plus_tn_equals_total_non_sars(self):
        rf = "Activity Deviation (ACH)"
        n_fp = 80
        df = _rule_df_for_drilldown(rf, n_fp=n_fp)
        tp, fp, fn, tn, _, _ = compute_2d_drilldown(df, "Activity Deviation (ACH)", "floor_amount", 5000, "z_threshold", 2.0)
        assert len(fp) + len(tn) == n_fp

    def test_elder_abuse_with_age_threshold(self):
        rf = "Elder Abuse"
        df = _rule_df_for_drilldown(rf)
        tp, fp, fn, tn, col1, col2 = compute_2d_drilldown(df, "Elder Abuse", "floor_amount", 3000, "age_threshold", 65)
        assert isinstance(tp, pd.DataFrame)
        assert col1 is not None and col2 is not None
