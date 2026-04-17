import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats

# USER-LEVEL AGGREGATION

def prepare_user_level_data(users_df: pd.DataFrame,
                            events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates event-level data to user-level revenue.
    """
    purchases = events_df[events_df["event_type"] == "purchase"]

    user_rev = (
        purchases.groupby("user_id")["revenue"]
        .sum()
        .reset_index()
    )

    df = users_df[["user_id", "group"]].merge(
        user_rev, on="user_id", how="left"
    )

    df["revenue"] = df["revenue"].fillna(0)

    return df

# CORE METRICS

def calculate_arpu(df: pd.DataFrame,
                   group_col: str = "group",
                   revenue_col: str = "revenue") -> pd.Series:
    """
    Returns ARPU per group.
    """
    return df.groupby(group_col)[revenue_col].mean()


def calculate_uplift(control: float,
                     treatment: float) -> Dict[str, float]:
    """
    Absolute and relative uplift.
    """
    absolute = treatment - control
    relative = absolute / control if control != 0 else 0.0

    return {
        "absolute_uplift": absolute,
        "relative_uplift": relative
    }

# iRPU (CORE BUSINESS METRIC)

def calculate_irpu(impact_df: pd.DataFrame) -> float:
    """
    Calculates true incremental revenue per user.
    """
    return impact_df["true_irpu"].mean()

# STATISTICAL TESTING

def ttest_ab(df: pd.DataFrame,
             metric: str = "revenue",
             group_col: str = "group") -> Dict[str, float]:
    """
    Welch's t-test for A/B comparison.
    """
    A = df[df[group_col] == "A"][metric]
    B = df[df[group_col] == "B"][metric]

    t_stat, p_value = stats.ttest_ind(B, A, equal_var=False)

    return {
        "t_stat": t_stat,
        "p_value": p_value
    }


def bootstrap_uplift_ci(df: pd.DataFrame,
                        n_bootstrap: int = 1000,
                        alpha: float = 0.05,
                        seed: int = 42) -> Dict[str, float]:
    """
    Bootstrap uplift with confidence intervals.
    """
    np.random.seed(seed)

    A = df[df["group"] == "A"]["revenue"].values
    B = df[df["group"] == "B"]["revenue"].values

    uplift_dist = []

    for _ in range(n_bootstrap):
        sample_A = np.random.choice(A, size=len(A), replace=True)
        sample_B = np.random.choice(B, size=len(B), replace=True)

        uplift = sample_B.mean() - sample_A.mean()
        uplift_dist.append(uplift)

    uplift_dist = np.array(uplift_dist)

    lower = np.percentile(uplift_dist, 100 * (alpha / 2))
    upper = np.percentile(uplift_dist, 100 * (1 - alpha / 2))
    mean_uplift = uplift_dist.mean()

    return {
        "mean_uplift": mean_uplift,
        "ci_lower": lower,
        "ci_upper": upper
    }

# FUNNEL METRICS

def calculate_funnel(events_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates CTR and CVR.
    """
    counts = events_df["event_type"].value_counts()

    impressions = counts.get("impression", 0)
    clicks = counts.get("click", 0)
    purchases = counts.get("purchase", 0)

    ctr = clicks / impressions if impressions > 0 else 0.0
    cvr = purchases / clicks if clicks > 0 else 0.0

    return {
        "CTR": ctr,
        "CVR": cvr,
        "impressions": impressions,
        "clicks": clicks,
        "purchases": purchases
    }

# SRM CHECK

def check_srm(df: pd.DataFrame,
              group_col: str = "group") -> Dict[str, float]:
    """
    Checks Sample Ratio Mismatch (SRM).
    """
    counts = df[group_col].value_counts()
    chi2, p_value = stats.chisquare(counts)

    return {
        "chi2_stat": chi2,
        "p_value": p_value
    }

# CUPED

def apply_cuped(df: pd.DataFrame,
                metric_col: str,
                covariate_col: str) -> Tuple[pd.DataFrame, float]:
    """
    Applies CUPED with theta estimated on control group.
    """
    control_df = df[df["group"] == "A"]

    cov_matrix = np.cov(
        control_df[covariate_col],
        control_df[metric_col]
    )

    theta = cov_matrix[0, 1] / cov_matrix[0, 0]

    mean_x = df[covariate_col].mean()

    df[f"{metric_col}_cuped"] = (
        df[metric_col] - theta * (df[covariate_col] - mean_x)
    )

    return df, theta

# ROBUSTNESS (HEAVY TAILS)

def winsorize_series(s: pd.Series,
                     upper_quantile: float = 0.99) -> pd.Series:
    """
    Caps extreme values to reduce impact of outliers.
    """
    cap = s.quantile(upper_quantile)
    return s.clip(upper=cap)

# GROUND TRUTH EVALUATION

def evaluate_ground_truth(estimated_uplift: float,
                         impact_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compares estimated uplift vs true causal effect.
    """
    true_irpu = impact_df["true_irpu"].mean()

    bias = estimated_uplift - true_irpu

    relative_error = (
        bias / true_irpu if true_irpu != 0 else 0.0
    )

    return {
        "estimated_uplift": estimated_uplift,
        "true_irpu": true_irpu,
        "bias": bias,
        "relative_error": relative_error
    }