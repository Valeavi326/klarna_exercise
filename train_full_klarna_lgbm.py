# train_full_klarna_lgbm.py
# End-to-end Klarna PD pipeline (single script):
# - Reads dat/mlcasestudy.csv (or --data_path)
# - Leakage-safe target definition
# - Data quality checks + descriptive stats
# - Feature engineering:
#     * time features (month/dow/day)
#     * card_months_to_expiry
#     * first-time borrower handling (days_since_first_loan == -1 is valid)
#     * delta features (exposure/payments/repaid)
#     * flags for range/consistency violations (USED as model features)
# - Leakage control (outstanding_14d/21d excluded from X)
# - Time-based holdout + TimeSeriesSplit CV
# - LightGBM model + calibration
# - Deciles + thresholding + calibration plot
# - Saves artifacts to outputs/ and models/
#
# Usage:
#   python train_full_klarna_lgbm.py --data_path dat/mlcasestudy.csv --out_dir outputs --models_dir models
#
# If lightgbm isn't installed:
#   pip install lightgbm

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise ImportError(
        "lightgbm is not installed. Run: pip install lightgbm"
    ) from e


# -----------------------
# Config
# -----------------------
@dataclass
class Config:
    data_path: str = "dat/mlcasestudy.csv"
    out_dir: str = "outputs"
    models_dir: str = "models"

    date_col: str = "loan_issue_date"
    id_col: str = "loan_id"
    target_col: str = "default_21d"

    # Leakage columns (post-issuance). Used for target, excluded from features.
    leakage_cols: Tuple[str, ...] = ("amount_outstanding_14d", "amount_outstanding_21d")

    # Categoricals
    cat_cols: Tuple[str, ...] = ("merchant_group", "merchant_category")

    # Base numeric columns expected in the dataset
    base_num_cols: Tuple[str, ...] = (
        "loan_amount",
        "existing_klarna_debt",
        "num_active_loans",
        "days_since_first_loan",
        "new_exposure_7d",
        "new_exposure_14d",
        "num_confirmed_payments_3m",
        "num_confirmed_payments_6m",
        "num_failed_payments_3m",
        "num_failed_payments_6m",
        "num_failed_payments_1y",
        "amount_repaid_14d",
        "amount_repaid_1m",
        "amount_repaid_3m",
        "amount_repaid_6m",
        "amount_repaid_1y",
        "card_expiry_month",
        "card_expiry_year",
    )

    # Business range rules
    # Note: days_since_first_loan allows -1 as "first-time borrower"
    non_negative_cols: Tuple[str, ...] = (
        "loan_amount",
        "existing_klarna_debt",
        "num_active_loans",
        "new_exposure_7d",
        "new_exposure_14d",
        "num_confirmed_payments_3m",
        "num_confirmed_payments_6m",
        "num_failed_payments_3m",
        "num_failed_payments_6m",
        "num_failed_payments_1y",
        "amount_repaid_14d",
        "amount_repaid_1m",
        "amount_repaid_3m",
        "amount_repaid_6m",
        "amount_repaid_1y",
        "amount_outstanding_14d",
        "amount_outstanding_21d",
    )

    card_month_col: str = "card_expiry_month"
    card_year_col: str = "card_expiry_year"
    card_month_min: int = 1
    card_month_max: int = 12

    # Split / CV
    test_frac: float = 0.2
    n_splits_cv: int = 5

    # Calibration
    calib_method: str = "sigmoid"  # "sigmoid" or "isotonic"
    calib_cv: int = 3

    # Thresholds
    thresholds: Tuple[float, ...] = (0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)

    # EDA plotting
    max_unique_for_bar: int = 20


# -----------------------
# IO helpers
# -----------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def assert_csv_ok(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    size = os.path.getsize(path)
    if size < 10:
        raise ValueError(f"CSV seems empty/invalid (size={size} bytes): {path}")


# -----------------------
# Parsing + target (leakage-safe)
# -----------------------
def parse_and_cast(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()

    # Date
    if cfg.date_col in out.columns:
        out[cfg.date_col] = pd.to_datetime(out[cfg.date_col], errors="coerce")

    # Numerics
    for c in cfg.base_num_cols + cfg.leakage_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Categoricals
    for c in cfg.cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("string")

    return out


def make_target(df: pd.DataFrame, cfg: Config) -> pd.Series:
    if "amount_outstanding_21d" not in df.columns:
        raise ValueError("Missing amount_outstanding_21d (required to define target).")
    x = pd.to_numeric(df["amount_outstanding_21d"], errors="coerce").fillna(0)
    return (x > 0).astype(int)


# -----------------------
# Flags (always) + conservative fixes (no dropping)
# -----------------------
def add_violation_flags(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Creates binary flags for out-of-range / consistency issues.
    These flags are kept as model features.
    """
    out = df.copy()

    # Generic negative flags for columns that must be >= 0
    for c in cfg.non_negative_cols:
        if c in out.columns:
            out[f"flag_neg_{c}"] = (out[c].notna() & (out[c] < 0)).astype(int)

    # days_since_first_loan: -1 is VALID (first-time borrower); other negatives are suspicious
    if "days_since_first_loan" in out.columns:
        d = pd.to_numeric(out["days_since_first_loan"], errors="coerce")
        out["flag_days_since_first_loan_lt_minus1"] = (d.notna() & (d < -1)).astype(int)
        out["is_first_time_borrower"] = (d == -1).astype(int)

    # Card month range
    if cfg.card_month_col in out.columns:
        m = pd.to_numeric(out[cfg.card_month_col], errors="coerce")
        out["flag_bad_card_month"] = (
            m.notna() & ((m < cfg.card_month_min) | (m > cfg.card_month_max))
        ).astype(int)

    # Exposure monotonicity (if cumulative windows expected)
    if "new_exposure_7d" in out.columns and "new_exposure_14d" in out.columns:
        a7 = pd.to_numeric(out["new_exposure_7d"], errors="coerce")
        a14 = pd.to_numeric(out["new_exposure_14d"], errors="coerce")
        out["flag_exposure_14_lt_7"] = (
            a7.notna() & a14.notna() & (a14 < a7)
        ).astype(int)

    # Confirmed payments monotonicity: 6m >= 3m expected
    if "num_confirmed_payments_6m" in out.columns and "num_confirmed_payments_3m" in out.columns:
        c6 = pd.to_numeric(out["num_confirmed_payments_6m"], errors="coerce")
        c3 = pd.to_numeric(out["num_confirmed_payments_3m"], errors="coerce")
        out["flag_confirmed_6_lt_3"] = (c6.notna() & c3.notna() & (c6 < c3)).astype(int)

    # Failed payments monotonicity: 1y >= 6m >= 3m expected
    if "num_failed_payments_6m" in out.columns and "num_failed_payments_3m" in out.columns:
        f6 = pd.to_numeric(out["num_failed_payments_6m"], errors="coerce")
        f3 = pd.to_numeric(out["num_failed_payments_3m"], errors="coerce")
        out["flag_failed_6_lt_3"] = (f6.notna() & f3.notna() & (f6 < f3)).astype(int)

    if "num_failed_payments_1y" in out.columns and "num_failed_payments_6m" in out.columns:
        f1 = pd.to_numeric(out["num_failed_payments_1y"], errors="coerce")
        f6 = pd.to_numeric(out["num_failed_payments_6m"], errors="coerce")
        out["flag_failed_1y_lt_6m"] = (f1.notna() & f6.notna() & (f1 < f6)).astype(int)

    # Amount repaid monotonicity: 1y >= 6m >= 3m >= 1m >= 14d expected
    rep_cols = ["amount_repaid_1y", "amount_repaid_6m", "amount_repaid_3m", "amount_repaid_1m", "amount_repaid_14d"]
    if all(c in out.columns for c in rep_cols):
        r1 = pd.to_numeric(out["amount_repaid_1y"], errors="coerce")
        r6 = pd.to_numeric(out["amount_repaid_6m"], errors="coerce")
        r3 = pd.to_numeric(out["amount_repaid_3m"], errors="coerce")
        r1m = pd.to_numeric(out["amount_repaid_1m"], errors="coerce")
        r14 = pd.to_numeric(out["amount_repaid_14d"], errors="coerce")

        out["flag_repaid_1y_lt_6m"] = (r1.notna() & r6.notna() & (r1 < r6)).astype(int)
        out["flag_repaid_6m_lt_3m"] = (r6.notna() & r3.notna() & (r6 < r3)).astype(int)
        out["flag_repaid_3m_lt_1m"] = (r3.notna() & r1m.notna() & (r3 < r1m)).astype(int)
        out["flag_repaid_1m_lt_14d"] = (r1m.notna() & r14.notna() & (r1m < r14)).astype(int)

    return out


def apply_conservative_value_fixes(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Fixes values only where we have a strong business rule.
    We do NOT drop rows. We keep flags.
    - new_exposure_14d is clipped to >= new_exposure_7d (conservative)
    - bad card month -> NaN
    - days_since_first_loan: -1 treated as valid; other negatives set to NaN (then imputed)
    - other non-negative cols: negatives -> NaN (then imputed)
    """
    out = df.copy()

    # Exposure monotonicity fix: set 14d to 7d when violated
    if "new_exposure_7d" in out.columns and "new_exposure_14d" in out.columns:
        mask = out["new_exposure_7d"].notna() & out["new_exposure_14d"].notna() & (out["new_exposure_14d"] < out["new_exposure_7d"])
        out.loc[mask, "new_exposure_14d"] = out.loc[mask, "new_exposure_7d"]

    # Card month invalid -> NaN
    if cfg.card_month_col in out.columns:
        m = pd.to_numeric(out[cfg.card_month_col], errors="coerce")
        bad = m.notna() & ((m < cfg.card_month_min) | (m > cfg.card_month_max))
        out.loc[bad, cfg.card_month_col] = np.nan

    # days_since_first_loan handling: -1 valid; other negatives -> NaN
    if "days_since_first_loan" in out.columns:
        d = pd.to_numeric(out["days_since_first_loan"], errors="coerce")
        d = d.mask(d == -1, 0)        # first-time borrower => 0
        d = d.mask(d < 0, np.nan)     # any other negative => NaN
        out["days_since_first_loan"] = d

    # Other non-negative columns: negatives -> NaN (keep flags)
    for c in cfg.non_negative_cols:
        if c in out.columns:
            out.loc[out[c].notna() & (out[c] < 0), c] = np.nan

    return out


# -----------------------
# Feature engineering (time, card months to expiry, deltas)
# -----------------------
def add_time_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[cfg.date_col], errors="coerce")
    out["issue_month"] = d.dt.month.astype("Int64")
    out["issue_dow"] = d.dt.dayofweek.astype("Int64")
    out["issue_day"] = d.dt.day.astype("Int64")
    return out


def add_card_months_to_expiry(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    issue = pd.to_datetime(out[cfg.date_col], errors="coerce")
    exp_y = pd.to_numeric(out.get(cfg.card_year_col), errors="coerce")
    exp_m = pd.to_numeric(out.get(cfg.card_month_col), errors="coerce")

    issue_mi = issue.dt.year * 12 + issue.dt.month
    exp_mi = exp_y * 12 + exp_m
    out["card_months_to_expiry"] = (exp_mi - issue_mi).astype("Float64")
    return out


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Exposure: 14d - 7d
    if "new_exposure_14d" in out.columns and "new_exposure_7d" in out.columns:
        out["delta_new_exposure_14_7"] = out["new_exposure_14d"] - out["new_exposure_7d"]

    # Confirmed: 6m - 3m
    if "num_confirmed_payments_6m" in out.columns and "num_confirmed_payments_3m" in out.columns:
        out["delta_confirmed_payments_6m_3m"] = out["num_confirmed_payments_6m"] - out["num_confirmed_payments_3m"]

    # Failed: 6m - 3m, 1y - 6m
    if "num_failed_payments_6m" in out.columns and "num_failed_payments_3m" in out.columns:
        out["delta_failed_payments_6m_3m"] = out["num_failed_payments_6m"] - out["num_failed_payments_3m"]
    if "num_failed_payments_1y" in out.columns and "num_failed_payments_6m" in out.columns:
        out["delta_failed_payments_1y_6m"] = out["num_failed_payments_1y"] - out["num_failed_payments_6m"]

    # Repaid deltas: 1y-6m, 6m-3m, 3m-1m, 1m-14d
    if "amount_repaid_1y" in out.columns and "amount_repaid_6m" in out.columns:
        out["delta_amount_repaid_1y_6m"] = out["amount_repaid_1y"] - out["amount_repaid_6m"]
    if "amount_repaid_6m" in out.columns and "amount_repaid_3m" in out.columns:
        out["delta_amount_repaid_6m_3m"] = out["amount_repaid_6m"] - out["amount_repaid_3m"]
    if "amount_repaid_3m" in out.columns and "amount_repaid_1m" in out.columns:
        out["delta_amount_repaid_3m_1m"] = out["amount_repaid_3m"] - out["amount_repaid_1m"]
    if "amount_repaid_1m" in out.columns and "amount_repaid_14d" in out.columns:
        out["delta_amount_repaid_1m_14d"] = out["amount_repaid_1m"] - out["amount_repaid_14d"]

    return out


# -----------------------
# EDA: stats + plots (optional but produced)
# -----------------------
def column_statistics(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = pd.to_numeric(df[col], errors="coerce")

        mode_val = np.nan
        m = s.mode(dropna=True)
        if len(m) > 0:
            mode_val = m.iloc[0]

        row = {
            "column": col,
            "count_non_null": int(s.notna().sum()),
            "missing_pct": float(s.isna().mean()),
            "min": float(s.min()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "median": float(s.median()) if s.notna().any() else np.nan,
            "mode": float(mode_val) if pd.notna(mode_val) else np.nan,
        }

        # Counts vs expected ranges
        if col in cfg.non_negative_cols:
            row["num_below_zero"] = int((s < 0).sum())
        elif col == "days_since_first_loan":
            row["num_below_minus1"] = int((s < -1).sum())

        if col == cfg.card_month_col:
            row["num_outside_valid_range"] = int(((s < cfg.card_month_min) | (s > cfg.card_month_max)).sum())

        rows.append(row)

    return pd.DataFrame(rows)


def plot_distributions(df: pd.DataFrame, cols: List[str], cfg: Config, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in cols:
        if col not in df.columns:
            continue

        s = df[col]
        plt.figure()

        if col in cfg.cat_cols or s.dtype == "object" or str(s.dtype).startswith("string"):
            vc = s.astype("string").fillna("NA").value_counts()
            if len(vc) > cfg.max_unique_for_bar:
                vc = vc.head(cfg.max_unique_for_bar)
            plt.bar(vc.index.astype(str), vc.values)
            plt.xticks(rotation=90)
            plt.title(f"Distribution (top categories) - {col}")
            plt.tight_layout()
        else:
            x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) == 0:
                plt.close()
                continue
            plt.hist(x, bins=50)
            plt.title(f"Distribution - {col}")
            plt.tight_layout()

        plt.savefig(out_dir / f"{col}.png", dpi=160)
        plt.close()


def plot_univariate_vs_target(df: pd.DataFrame, feature_cols: List[str], cfg: Config, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    y = df[cfg.target_col].astype(int)

    for col in feature_cols:
        if col not in df.columns:
            continue

        plt.figure()
        s = df[col]

        if col in cfg.cat_cols or s.dtype == "object" or str(s.dtype).startswith("string"):
            tmp = pd.DataFrame({col: s.astype("string").fillna("NA"), cfg.target_col: y})
            grp = tmp.groupby(col)[cfg.target_col].mean().sort_values(ascending=False)
            if len(grp) > cfg.max_unique_for_bar:
                grp = grp.head(cfg.max_unique_for_bar)
            plt.bar(grp.index.astype(str), grp.values)
            plt.xticks(rotation=90)
            plt.ylabel("P(default)")
            plt.title(f"Default rate by {col} (top)")
            plt.tight_layout()
        else:
            x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            tmp = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(tmp) < 50:
                plt.close()
                continue
            try:
                tmp["bin"] = pd.qcut(tmp["x"], q=10, duplicates="drop")
                bin_rate = tmp.groupby("bin")["y"].mean()
                plt.plot(range(len(bin_rate)), bin_rate.values, marker="o")
                plt.xlabel("Quantile bin (low → high)")
                plt.ylabel("P(default)")
                plt.title(f"P(default) across quantile bins - {col}")
                plt.tight_layout()
            except Exception:
                sample = tmp.sample(min(2000, len(tmp)), random_state=42)
                plt.scatter(sample["x"], sample["y"], s=6)
                plt.title(f"Scatter (sample) y vs {col}")
                plt.tight_layout()

        plt.savefig(out_dir / f"{col}_vs_{cfg.target_col}.png", dpi=160)
        plt.close()


# -----------------------
# Split & pipeline
# -----------------------
def time_split(df: pd.DataFrame, cfg: Config):
    df = df.sort_values(cfg.date_col)
    cut = int(len(df) * (1 - cfg.test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def build_pipeline(feature_cols: List[str], cfg: Config) -> Pipeline:
    # Determine num/cat
    cat_cols = [c for c in feature_cols if c in cfg.cat_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

    model = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight="balanced",
    )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


# -----------------------
# Evaluation utilities
# -----------------------
def threshold_report_credit(
    y_true: np.ndarray,
    proba: np.ndarray,
    loan_amount: np.ndarray,
    thresholds: Tuple[float, ...],
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    loan_amount = np.asarray(loan_amount).astype(float)

    n = len(y_true)
    total_defaults = int((y_true == 1).sum())

    rows = []
    for t in thresholds:
        approved = proba <= t
        declined = ~approved

        n_appr = int(approved.sum())
        appr_rate = n_appr / n if n > 0 else np.nan

        obs_dr_appr = float((y_true[approved] == 1).mean()) if n_appr > 0 else np.nan
        exp_defaults = float(proba[approved].sum()) if n_appr > 0 else 0.0
        exp_loss_proxy = float((proba[approved] * loan_amount[approved]).sum()) if n_appr > 0 else 0.0

        def_capture = float((y_true[declined] == 1).sum() / total_defaults) if total_defaults > 0 else np.nan

        rows.append(
            {
                "pd_threshold": float(t),
                "approved_n": n_appr,
                "approval_rate": appr_rate,
                "observed_default_rate_approved": obs_dr_appr,
                "expected_defaults_approved": exp_defaults,
                "expected_loss_proxy_approved": exp_loss_proxy,
                "defaulter_capture_by_decline": def_capture,
            }
        )

    out = pd.DataFrame(rows)
    out["avg_pd_approved"] = out["expected_defaults_approved"] / out["approved_n"].replace(0, np.nan)
    out["avg_loss_proxy_per_approved"] = out["expected_loss_proxy_approved"] / out["approved_n"].replace(0, np.nan)
    return out


def decile_table(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.astype(int), "pd": proba.astype(float)})
    df["decile"] = pd.qcut(df["pd"], 10, labels=False, duplicates="drop") + 1

    overall_dr = df["y"].mean()

    agg = (
        df.groupby("decile")
        .agg(
            n=("y", "size"),
            avg_pd=("pd", "mean"),
            default_rate=("y", "mean"),
            defaults=("y", "sum"),
        )
        .reset_index()
        .sort_values("decile", ascending=False)
        .reset_index(drop=True)
    )

    agg["lift_vs_overall"] = agg["default_rate"] / overall_dr if overall_dr > 0 else np.nan
    agg["cum_defaults"] = agg["defaults"].cumsum()
    agg["cum_default_capture"] = agg["cum_defaults"] / agg["defaults"].sum() if agg["defaults"].sum() > 0 else np.nan
    agg["cum_volume"] = agg["n"].cumsum() / agg["n"].sum() if agg["n"].sum() > 0 else np.nan
    return agg


def save_calibration_plot(y_true: np.ndarray, proba: np.ndarray, out_path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Mean predicted PD")
    plt.ylabel("Observed default rate")
    plt.title("Calibration curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def time_cv_metrics(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, cfg: Config) -> Dict[str, float]:
    """
    TimeSeriesSplit out-of-fold evaluation (ranking + probability quality pre-calibration).
    """
    tss = TimeSeriesSplit(n_splits=cfg.n_splits_cv)
    oof = np.full(shape=len(y), fill_value=np.nan, dtype=float)

    for train_idx, val_idx in tss.split(X):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va = X.iloc[val_idx]

        #pipe_fold = joblib.loads(joblib.dumps(pipe))  # clone
        pipe_fold = clone(pipe)
        pipe_fold.fit(X_tr, y_tr)
        oof[val_idx] = pipe_fold.predict_proba(X_va)[:, 1]

    mask = ~np.isnan(oof)
    y_m = y[mask]
    p_m = oof[mask]

    return {
        "cv_roc_auc": float(roc_auc_score(y_m, p_m)),
        "cv_pr_auc": float(average_precision_score(y_m, p_m)),
        "cv_brier": float(brier_score_loss(y_m, p_m)),
        "cv_coverage": float(mask.mean()),
    }


# -----------------------
# Main
# -----------------------


# -----------------------
# Calibrazione per segmenti (merchant)
# -----------------------
def build_merchant_segments(
    s: pd.Series,
    min_count: int = 500,
    coverage: float = 0.8,
    other_label: str = "Other",
) -> Tuple[pd.Series, List[str]]:
    """Crea segmenti 'sensati' per merchant.

    Logica:
    - prendo i merchant più frequenti (per numerosità) finché:
      (a) ogni merchant ha almeno min_count osservazioni
      (b) la copertura cumulata arriva a ~coverage (es. 80%)
    - tutto il resto va in 'Other'

    Ritorna:
    - serie con segment label per ogni riga
    - lista dei merchant tenuti esplicitamente (top merchants)
    """
    x = s.astype("string").fillna(other_label)
    vc = x.value_counts(dropna=False)
    total = vc.sum()

    kept = []
    cum = 0
    for m, cnt in vc.items():
        if cnt < min_count:
            break
        kept.append(str(m))
        cum += cnt
        if total > 0 and (cum / total) >= coverage:
            break

    seg = x.where(x.isin(kept), other_label)
    return seg, kept


def fit_isotonic_by_segment(
    p_hat: np.ndarray,
    y: np.ndarray,
    segments: pd.Series,
    min_count: int = 200,
) -> Dict[str, IsotonicRegression]:
    """Fitta una calibrazione isotonic separata per segmento.

    - Se un segmento ha pochi esempi, non fittiamo (si userà fallback globale).
    """
    seg = segments.astype("string").fillna("Other")
    out: Dict[str, IsotonicRegression] = {}
    for g in seg.unique():
        mask = (seg == g).values
        if mask.sum() < min_count:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_hat[mask], y[mask])
        out[str(g)] = iso
    return out


def apply_segment_calibration(
    p_hat: np.ndarray,
    segments: pd.Series,
    calibrators: Dict[str, IsotonicRegression],
    fallback: IsotonicRegression | None = None,
    other_label: str = "Other",
) -> np.ndarray:
    """Applica la calibrazione per segmento ai punteggi probabilistici.

    - Se il segmento non ha calibratore dedicato, usa fallback (se presente) altrimenti lascia invariato.
    """
    seg = segments.astype("string").fillna(other_label)
    p_out = p_hat.copy()

    for g in seg.unique():
        mask = (seg == g).values
        iso = calibrators.get(str(g), fallback)
        if iso is None:
            continue
        p_out[mask] = iso.transform(p_hat[mask])
    return p_out


# -----------------------
# Interpretabilità: SHAP + PDP
# -----------------------
def save_shap_plots_if_available(
    fitted_pipe: Pipeline,
    X_sample: pd.DataFrame,
    out_dir: Path,
    max_samples: int = 5000,
) -> None:
    """Crea plot SHAP (summary + bar) se shap è disponibile.

    Note pratiche:
    - Con OneHotEncoder, lavoriamo nello spazio "trasformato" (feature after preprocessing).
    - Per non rallentare troppo, campioniamo max_samples righe.
    """
    if shap is None:
        return

    try:
        pre = fitted_pipe.named_steps["preprocess"]
        model = fitted_pipe.named_steps["model"]
        feature_names = pre.get_feature_names_out()

        Xs = X_sample.copy()
        if len(Xs) > max_samples:
            Xs = Xs.sample(max_samples, random_state=42)

        Xt = pre.transform(Xs)

        # TreeExplainer funziona bene con LightGBM
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xt)

        # shap_values può essere lista [class0, class1] in classificazione
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Summary plot
        plt.figure()
        shap.summary_plot(sv, Xt, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=180)
        plt.close()

        # Bar plot (importanza media)
        plt.figure()
        shap.summary_plot(sv, Xt, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_bar.png", dpi=180)
        plt.close()

    except Exception:
        # Non vogliamo che SHAP blocchi il training
        return


def save_pdp_plots(
    estimator,
    X: pd.DataFrame,
    features: List[str],
    out_dir: Path,
) -> None:
    """Salva PDP (1 plot per feature) per evitare figure enormi e mantenere leggibilità."""
    for f in features:
        if f not in X.columns:
            continue
        plt.figure()
        try:
            PartialDependenceDisplay.from_estimator(estimator, X, [f])
            plt.tight_layout()
            plt.savefig(out_dir / f"pdp_{f}.png", dpi=180)
        except Exception:
            pass
        finally:
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dat/mlcasestudy.csv")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    cfg = Config(data_path=args.data_path, out_dir=args.out_dir, models_dir=args.models_dir)

    assert_csv_ok(cfg.data_path)

    out_dir = Path(cfg.out_dir)
    models_dir = Path(cfg.models_dir)

    reports_dir = out_dir / "reports"
    plots_dist_dir = out_dir / "plots" / "distributions"
    plots_uni_dir = out_dir / "plots" / "univariate_vs_target"
    ensure_dir(reports_dir)
    ensure_dir(plots_dist_dir)
    ensure_dir(plots_uni_dir)
    ensure_dir(models_dir)

    # Load + parse
    df_raw = pd.read_csv(cfg.data_path)
    df = parse_and_cast(df_raw, cfg)

    # Target (leakage-safe)
    df[cfg.target_col] = make_target(df, cfg)

    # Flags (always, used as features)
    df = add_violation_flags(df, cfg)

    # Conservative fixes (no dropping)
    df = apply_conservative_value_fixes(df, cfg)

    # Drop rows without date or target (this is not "range dropping": it's unusable for time split)
    df = df.dropna(subset=[cfg.date_col, cfg.target_col]).copy()

    # Feature engineering
    df = add_time_features(df, cfg)
    df = add_card_months_to_expiry(df, cfg)
    df = add_delta_features(df)

    # Statistics
    stats = column_statistics(df, cfg)
    stats.to_csv(reports_dir / "column_statistics.csv", index=False)

    # Save a compact flag summary (optional but useful)
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    flag_summary = pd.DataFrame(
        {"flag": flag_cols, "rate": [float(df[c].mean()) for c in flag_cols]}
    ).sort_values("rate", ascending=False)
    flag_summary.to_csv(reports_dir / "flag_rates.csv", index=False)

    # Determine features (exclude leakage + id/date/target + exclude abs if delta exists)
    drop_cols = [cfg.id_col, cfg.date_col, cfg.target_col] + list(cfg.leakage_cols)
    feature_cols = [c for c in df.columns if c not in drop_cols]


    DROP_ABSOLUTE_IF_DELTA = [
    "new_exposure_14d",
    "num_confirmed_payments_6m",
    "num_failed_payments_6m",
    "num_failed_payments_1y",
    "amount_repaid_1m",
    "amount_repaid_3m",
    "amount_repaid_6m",
    "amount_repaid_1y",
    ]

    feature_cols = [c for c in feature_cols if c not in DROP_ABSOLUTE_IF_DELTA]

    DROP_CALENDAR = ["issue_day", "issue_dow"]
    feature_cols = [c for c in feature_cols if c not in DROP_CALENDAR]


    # EDA plots (distributions include leakage columns for understanding)
    plot_distributions(df, feature_cols + list(cfg.leakage_cols), cfg, plots_dist_dir)
    plot_univariate_vs_target(df, feature_cols, cfg, plots_uni_dir)

    # Sort by time
    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    X = df[feature_cols].copy()
    y = df[cfg.target_col].astype(int).values

    # Holdout
    train_df, test_df = time_split(df[[cfg.date_col]].copy(), cfg)
    X_train = X.loc[train_df.index]
    y_train = y[train_df.index]
    X_test = X.loc[test_df.index]
    y_test = y[test_df.index]

    # Model pipeline
    pipe = build_pipeline(feature_cols, cfg)

    # CV metrics (pre-calibration)
    cvm = time_cv_metrics(pipe, X_train, y_train, cfg)

    # Fit + calibrate
    pipe.fit(X_train, y_train)
    cal = CalibratedClassifierCV(pipe, method=cfg.calib_method, cv=cfg.calib_cv)
    cal.fit(X_train, y_train)

    # Predict
    proba = cal.predict_proba(X_test)[:, 1]

    metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_pr_auc": float(average_precision_score(y_test, proba)),
        "test_brier": float(brier_score_loss(y_test, proba)),
        "test_default_rate": float(np.mean(y_test)),
        "model": f"LGBMClassifier + {cfg.calib_method} calibration",
        "split": "time-based holdout",
        **cvm,
    }

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(cal, models_dir / "model.joblib")

    # -----------------------
    # Feature importance (LightGBM) - sklearn-version safe
    # -----------------------
    # We need the fitted pipeline inside CalibratedClassifierCV.
    # Depending on sklearn version, it can be:
    # - cal.estimator
    # - cal.calibrated_classifiers_[0].estimator

    fitted_pipe = getattr(cal, "estimator", None)
    if fitted_pipe is None:
        # fallback for older sklearn
        fitted_pipe = cal.calibrated_classifiers_[0].estimator

    lgbm = fitted_pipe.named_steps["model"]
    pre = fitted_pipe.named_steps["preprocess"]

    feature_names = pre.get_feature_names_out()

    booster = lgbm.booster_
    fi_gain = booster.feature_importance(importance_type="gain")
    fi_split = booster.feature_importance(importance_type="split")

    fi_df = pd.DataFrame(
        {"feature": feature_names, "gain": fi_gain, "split": fi_split}
    ).sort_values("gain", ascending=False)

    fi_df.to_csv(models_dir / "feature_importance_gain.csv", index=False)
    fi_df[["feature", "split"]].sort_values("split", ascending=False).to_csv(
        models_dir / "feature_importance_split.csv", index=False
    )


    # Threshold table (loan_amount required)
    if "loan_amount" in X_test.columns:
        thr = threshold_report_credit(y_test, proba, X_test["loan_amount"].values, cfg.thresholds)
        thr.to_csv(models_dir / "threshold_table_credit.csv", index=False)

    # Deciles
    dec = decile_table(y_test, proba)
    dec.to_csv(models_dir / "decile_table.csv", index=False)

    # Calibration plot
    save_calibration_plot(y_test, proba, models_dir / "calibration.png")

    # Save feature list (useful for API parity)
    pd.DataFrame({"feature": feature_cols}).to_csv(models_dir / "feature_list.csv", index=False)

    print("Training completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
