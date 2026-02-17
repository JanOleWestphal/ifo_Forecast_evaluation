
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Judgemental Evaluation Helper Functions
#
# Author:       Jan Ole Westphal
# Date:         2026-01
#
# Description:  Common helper functions for judgemental derivation analysis.
#               Used by both nowcasting (8_) and forecasting (9_) analysis modules.
# 
# ==================================================================================================
# --------------------------------------------------------------------------------------------------


from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


# =================================================================================================#
#                                   ERROR AND DERIVATION MEASURES
# =================================================================================================#

def add_error_columns(df, prefix="error"):
    """
    Add error columns by subtracting forecast columns from the first column (realized).
    """
    first_col = df.columns[0]

    for col in df.columns[1:]:
        df[f"{prefix}_{first_col}_minus_{col}"] = df[col] - df[first_col] 

    return df


def _classify_derivations(df, b_col: str, suffix: str, r_col: str = "realized", j_col: str = "judgemental"):
    """
    Classify derivations based on shock direction and adjustment direction.
    
    Creates three boolean columns:
    - r_less_{suffix}: whether realized < baseline
    - j_less_{suffix}: whether judgemental < baseline
    - j_diff_less_{suffix}_diff: whether |judgemental - realized| < |baseline - realized|
    """
    r = df[r_col]
    j = df[j_col]
    b = df[b_col]

    # Ensure NA-safe comparisons: keep pd.NA where any input is missing
    valid = r.notna() & j.notna() & b.notna()

    df[f"r_less_{suffix}"] = pd.Series(np.where(valid, r < b, pd.NA), index=df.index, dtype="boolean")
    df[f"j_less_{suffix}"] = pd.Series(np.where(valid, j < b, pd.NA), index=df.index, dtype="boolean")
    df[f"j_diff_less_{suffix}_diff"] = pd.Series(
        np.where(valid, (j - r).abs() < (b - r).abs(), pd.NA),
        index=df.index,
        dtype="boolean",
    )


# =================================================================================================#
#                                   BASELINE SPECIFICATION INFERENCE
# =================================================================================================#

def _infer_baseline_spec(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer which baseline the df refers to (ifoCast vs AR2 vs AVERAGE etc.) and return column spec.
    """
    cols = set(df.columns)

    # ifoCast
    if "ifoCast" in cols or any("ifoCast" in c for c in cols):
        return {
            "baseline_label": "ifoCast",
            "shock_col": "r_less_ifoCast",
            "derivation_col": "derivation_from_ifoCast",
            "ni_lin_col": "net_improvement_jdg_ifoCast_lin",
            "ni_quad_col": "net_improvement_jdg_ifoCast_quad",
            "error_baseline_col": "error_realized_minus_ifoCast",
            "adjustment_col": "j_less_ifoCast",
            "improvement_col": "j_diff_less_ifoCast_diff",
        }

    # Naive Models
    # Look for derivation_from_{Model} column
    for c in cols:
        if c.startswith("derivation_from_"):
             model_part = c.replace("derivation_from_", "")
             if model_part == "ifoCast": continue
             
             # Check if corresponding shock classification exists to confirm it's a valid baseline set
             if f"r_less_{model_part}" in cols:
                 return {
                    "baseline_label": model_part,
                    "shock_col": f"r_less_{model_part}",
                    "derivation_col": f"derivation_from_{model_part}",
                    "ni_lin_col": f"net_improvement_jdg_{model_part}_lin",
                    "ni_quad_col": f"net_improvement_jdg_{model_part}_quad",
                    "error_baseline_col": f"error_realized_minus_naive{model_part}",
                    "adjustment_col": f"j_less_{model_part}",
                    "improvement_col": f"j_diff_less_{model_part}_diff",
                 }

    raise ValueError(f"Could not infer baseline. Columns found: {cols}")


# =================================================================================================#
#                                   SUMMARY STATISTICS
# =================================================================================================#

def _generate_summary_statistics(df: pd.DataFrame, baseline_label: str, shock_filter: Optional[bool] = None) -> dict:
    """
    Generate summary statistics for a judgemental evaluation dataframe.
    """
    spec = _infer_baseline_spec(df)
    shock_col = spec["shock_col"]
    ni_lin_col = spec["ni_lin_col"]
    ni_quad_col = spec["ni_quad_col"]
    error_baseline_col = spec.get("error_baseline_col") 
    
    # If not in spec (old logic compat), standardizing inference
    if not error_baseline_col:
        # Fallback for ifoCast
        if baseline_label == "ifoCast":
            error_baseline_col = "error_realized_minus_ifoCast"
        else: # AR2 old fallback
            error_baseline_col = "error_realized_minus_naiveAR2"

    error_jdg_col = "error_realized_minus_judgemental"
    adjustment_col = spec.get("adjustment_col")
    improvement_col = spec.get("improvement_col")
    
    # Remove rows with NaN values for calculations
    df_clean = df.dropna(subset=[shock_col, error_jdg_col, error_baseline_col, 
                                   ni_lin_col, ni_quad_col, adjustment_col, improvement_col])
    
    # Apply shock filter if specified
    if shock_filter is not None:
        shock_series_bool = df_clean[shock_col].astype(bool)
        if shock_filter:
            # Negative shocks (r < baseline)
            df_clean = df_clean[shock_series_bool]
            subsample_label = "Negative Shocks"
        else:
            # Positive shocks (r >= baseline)
            df_clean = df_clean[~shock_series_bool]
            subsample_label = "Positive Shocks"
    else:
        subsample_label = "Overall"
    
    # Shock counts
    shock_series = df_clean[shock_col].astype(bool)
    negative_shocks = shock_series.sum()
    positive_shocks = (~shock_series).sum()
    
    # Average shock size (absolute error)
    avg_jdg_error = df_clean[error_jdg_col].abs().mean()
    avg_baseline_error = df_clean[error_baseline_col].abs().mean()
    
    # Average net improvements
    avg_ni_lin = df_clean[ni_lin_col].mean()
    avg_ni_quad = df_clean[ni_quad_col].mean()
    
    # Adjustment counts (j_less_baseline)
    adjustment_series = df_clean[adjustment_col].astype(bool)
    adjustments_below_baseline = adjustment_series.sum()
    adjustments_above_baseline = (~adjustment_series).sum()
    
    # Improvement counts (times adjustments led to improvements)
    improvement_series = df_clean[improvement_col].astype(bool)
    successful_improvements = improvement_series.sum()
    unsuccessful_adjustments = (~improvement_series).sum()
    
    return {
        "Baseline": baseline_label,
        "Subsample": subsample_label,
        "Negative Shocks (r < baseline)": int(negative_shocks),
        "Positive Shocks (r >= baseline)": int(positive_shocks),
        "Avg Judgemental Error (abs)": round(avg_jdg_error, 4),
        "Avg Baseline Error (abs)": round(avg_baseline_error, 4),
        "Avg Net Improvement (Linear)": round(avg_ni_lin, 4),
        "Avg Net Improvement (Quadratic)": round(avg_ni_quad, 4),
        "Adjustments Below Baseline (j < b)": int(adjustments_below_baseline),
        "Adjustments Above Baseline (j >= b)": int(adjustments_above_baseline),
        "Adjustments Reducing Error (|j-r| < |b-r|)": int(successful_improvements),
        "Adjustments Increasing Error": int(unsuccessful_adjustments),
        "Total Observations": len(df_clean),
    }


# =================================================================================================#
#                                   VISUALIZATION HELPERS
# =================================================================================================#

def _format_quarterly_index(dt_index) -> list[str]:
    """
    Convert a datetime index to yyyy-Qx format for display.
    """
    def to_quarter_str(ts):
        if pd.isna(ts):
            return "NaN"
        # Determine quarter from month
        quarter = (ts.month - 1) // 3 + 1
        return f"{ts.year}-Q{quarter}"
    
    return [to_quarter_str(ts) for ts in dt_index]


def _format_quarterly_index_with_horizon(dt_index, horizon: int) -> list[str]:
    """
    Convert a datetime index to yyyy-Qx format with horizon notation (h=0, h=1, etc.).
    """
    def to_quarter_str(ts):
        if pd.isna(ts):
            return "NaN"
        quarter = (ts.month - 1) // 3 + 1
        return f"{ts.year}-Q{quarter}(h={horizon})"
    
    return [to_quarter_str(ts) for ts in dt_index]


def _apply_percentile_truncation(ax, data: np.ndarray, percentile: float) -> None:
    """
    Truncate y-axis symmetrically based on percentile.
    
    Args:
        ax: Matplotlib axis object
        data: Numeric data array
        percentile: Percentile threshold (0-100). E.g., 95 means truncate tails beyond 95th percentile.
    """
    # Handle NaN values
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) == 0:
        return
    
    # Calculate bounds symmetrically
    lower_bound = np.percentile(clean_data, 100 - percentile)
    upper_bound = np.percentile(clean_data, percentile)
    
    # Ensure symmetric margins around zero if both bounds have same sign
    if lower_bound >= 0:
        margin = upper_bound * (1 - percentile / 100)
        lower_bound = -margin
    elif upper_bound <= 0:
        margin = abs(lower_bound) * (1 - percentile / 100)
        upper_bound = margin
    
    ax.set_ylim(lower_bound, upper_bound)


def visualize_summary_statistics(df: pd.DataFrame, save_folder: str | Path) -> None:
    """
    Create comprehensive visualizations of summary statistics.
    Dynamics version: adapts to available baselines.
    """
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    
    baselines = df['Baseline'].unique()
    num_baselines = len(baselines)
    
    preferred_subsample_order = ["Overall", "Negative Shocks", "Positive Shocks"]
    present_subsamples = df["Subsample"].dropna().unique().tolist()
    subsamples = [s for s in preferred_subsample_order if s in present_subsamples]
    subsamples.extend([s for s in present_subsamples if s not in subsamples])
    
    # helper to get data for a baseline aligned to subsamples
    def get_data(baseline, col):
        sub_df = df[df['Baseline'] == baseline].set_index("Subsample").reindex(subsamples)
        return sub_df[col].values

    # Colors for baselines
    cmap = plt.colormaps['tab10']
    colors = {b: cmap(i) for i, b in enumerate(baselines)}
    
    x_pos = np.arange(len(subsamples))
    width = 0.8 / num_baselines
    
    # PLOT 1: Sample sizes
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, base in enumerate(baselines):
        offset = (i - (num_baselines - 1) / 2) * width
        vals = get_data(base, 'Total Observations')
        ax.bar(x_pos + offset, vals, width, label=base, alpha=0.8, color=colors[base])

    ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Observations', fontsize=11, fontweight='bold')
    ax.set_title('Sample Sizes by Subsample and Baseline', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subsamples)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_sample_sizes.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # PLOT 2: Average Errors
    fig, axes = plt.subplots(1, num_baselines, figsize=(7*num_baselines, 5), sharey=True)
    if num_baselines == 1: axes = [axes]
    
    sub_width = 0.25
    for i, base in enumerate(baselines):
        ax = axes[i]
        err_jdg = get_data(base, 'Avg Judgemental Error (abs)')
        err_base = get_data(base, 'Avg Baseline Error (abs)')
        ni_lin = get_data(base, 'Avg Net Improvement (Linear)')
        
        ax.bar(x_pos - sub_width, err_jdg, sub_width, label='Avg |judgemental - realized|', alpha=0.85, color='steelblue')
        ax.bar(x_pos, err_base, sub_width, label='Avg |baseline - realized|', alpha=0.85, color='darkorange')
        ax.bar(x_pos + sub_width, ni_lin, sub_width, label='Avg net improvement (linear)', alpha=0.85, color='seagreen')
        
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
        ax.set_title(f'{base} Baseline', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subsamples)
        ax.grid(axis='y', alpha=0.3)
        if i == 0:
            ax.set_ylabel('Average Value', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='upper left')
    
    fig.suptitle('Average Forecast Errors by Subsample and Baseline', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_average_errors_by_baseline.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # PLOT 3: Net Improvements (Linear)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, base in enumerate(baselines):
        offset = (i - (num_baselines - 1) / 2) * width
        vals = get_data(base, 'Avg Net Improvement (Linear)')
        ax.bar(x_pos + offset, vals, width, label=base, alpha=0.8, color=colors[base])
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Linear Net Improvement', fontsize=11, fontweight='bold')
    ax.set_title('Judgemental vs Baseline Linear Improvements by Subsample', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subsamples)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_net_improvement_linear.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # PLOT 4: Net Improvements (Quadratic)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, base in enumerate(baselines):
        offset = (i - (num_baselines - 1) / 2) * width
        vals = get_data(base, 'Avg Net Improvement (Quadratic)')
        ax.bar(x_pos + offset, vals, width, label=base, alpha=0.8, color=colors[base])
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Quadratic Net Improvement', fontsize=11, fontweight='bold')
    ax.set_title('Judgemental vs Baseline Quadratic Improvements by Subsample', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subsamples)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_net_improvement_quadratic.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # PLOT 5: Adjustment Success Rates
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, base in enumerate(baselines):
        offset = (i - (num_baselines - 1) / 2) * width
        succ = get_data(base, 'Adjustments Reducing Error (|j-r| < |b-r|)')
        fail = get_data(base, 'Adjustments Increasing Error')
        tot = succ + fail
        rate = np.divide(succ, tot, out=np.zeros_like(succ, dtype=float), where=tot!=0) * 100
        ax.bar(x_pos + offset, rate, width, label=base, alpha=0.8, color=colors[base])
    
    ax.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='50% (Random)')
    ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Adjustment Success Rates (% reducing error) by Subsample', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subsamples)
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_adjustment_success_rates.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # PLOT 6: Adjustment Direction Distribution
    fig, axes = plt.subplots(1, num_baselines, figsize=(7*num_baselines, 5), sharey=True)
    if num_baselines == 1: axes = [axes]

    for i, base in enumerate(baselines):
        ax = axes[i]
        below = get_data(base, 'Adjustments Below Baseline (j < b)')
        above = get_data(base, 'Adjustments Above Baseline (j >= b)')
        
        for idx in range(len(subsamples)):
             ax.bar(idx, below[idx], label='Below' if idx == 0 else '', alpha=0.8, color='steelblue')
             ax.bar(idx, above[idx], bottom=below[idx], label='Above' if idx == 0 else '', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
        ax.set_title(f'Adjustment Direction - {base} Baseline', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subsamples)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        if i == 0:
            ax.set_ylabel('Number of Adjustments', fontsize=11, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_adjustment_directions.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSummary statistics visualizations saved to {save_folder}")


def plot_judgemental_derivations_or_net_improvement(
    df: pd.DataFrame,
    kind: str,
    graph_folder_EDA: str | Path,
    header: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    filename_suffix: Optional[str] = None,
    show: bool = False,
    dpi: int = 180,
    y_axis_percentile: Optional[float] = None,
) -> Path | list[Path]:
    """
    Plot either:
      - kind="derivation": judgemental derivations (j - baseline) over the index
      - kind="net_improvement": net improvements (LINEAR and QUADRATIC in separate plots)

    Bars are coloured by "shock" (negative shock): r_less_<baseline> == True
      - Red   : negative shock (realised < baseline)
      - Green : otherwise
    """
    if kind not in {"derivation", "net_improvement"}:
        raise ValueError("kind must be either 'derivation' or 'net_improvement'.")

    spec = _infer_baseline_spec(df)
    baseline_label = spec["baseline_label"]
    shock_col = spec["shock_col"]

    graph_folder_EDA = Path(graph_folder_EDA)
    graph_folder_EDA.mkdir(parents=True, exist_ok=True)

    x = df.index
    x_labels = _format_quarterly_index(x)
    pos = np.arange(len(df))

    shock = df[shock_col].astype("boolean")
    colours = np.where(shock.fillna(False).to_numpy(), "red", "green")

    shock_legend = [
        Patch(facecolor="red", edgecolor="none", label=f"Negative shock: realised < {baseline_label}"),
        Patch(facecolor="green", edgecolor="none", label=f"Positive shock: realised ≥ {baseline_label}"),
    ]

    if kind == "derivation":
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.axhline(0.0, linewidth=1.0)

        y_col = spec["derivation_col"]
        y = df[y_col].to_numpy()

        ax.bar(pos, y, color=colours)
        ax.set_xticks(pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        title = header or f"Judgemental derivations vs {baseline_label}"
        ax.set_title(title)
        ax.set_ylabel(f"Derivation (judgemental - {baseline_label})")

        if y_axis_percentile is not None:
            _apply_percentile_truncation(ax, y, y_axis_percentile)

        ax.legend(handles=shock_legend, loc="best", frameon=True)

        prefix = filename_prefix or "judgemental_derivations"
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        out_path = graph_folder_EDA / f"{prefix}_vs_{baseline_label}{suffix}.png"

        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return out_path

    else:  # net_improvement
        lin_col = spec["ni_lin_col"]
        quad_col = spec["ni_quad_col"]

        y_lin = df[lin_col].to_numpy()
        y_quad = df[quad_col].to_numpy()

        out_paths = []
        prefix = filename_prefix or "net_improvement"
        suffix = f"_{filename_suffix}" if filename_suffix else ""

        # LINEAR  
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.axhline(0.0, linewidth=1.0)
        ax.bar(pos, y_lin, color=colours, label="Net improvement (linear)")
        ax.set_xticks(pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        
        title = header or f"Net improvement (linear) of judgemental forecast vs {baseline_label}"
        ax.set_title(title)
        ax.set_ylabel("Improvement (>0 is better than baseline)")

        if y_axis_percentile is not None:
            _apply_percentile_truncation(ax, y_lin, y_axis_percentile)

        metric_legend = ax.legend(loc="upper left", frameon=True)
        ax.add_artist(metric_legend)
        ax.legend(handles=shock_legend, loc="best", frameon=True)

        out_path_lin = graph_folder_EDA / f"{prefix}_linear_jdg_vs_{baseline_label}{suffix}.png"
        fig.tight_layout()
        fig.savefig(out_path_lin, dpi=dpi, bbox_inches="tight")
        out_paths.append(out_path_lin)

        if show:
            plt.show()
        else:
            plt.close(fig)

        # QUADRATIC
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.axhline(0.0, linewidth=1.0)
        ax.bar(pos, y_quad, color=colours, alpha=0.7, label="Net improvement (quadratic)")
        ax.set_xticks(pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        title = header or f"Net improvement (quadratic) of judgemental forecast vs {baseline_label}"
        ax.set_title(title)
        ax.set_ylabel("Improvement (>0 is better than baseline)")

        if y_axis_percentile is not None:
            _apply_percentile_truncation(ax, y_quad, y_axis_percentile)

        metric_legend = ax.legend(loc="upper left", frameon=True)
        ax.add_artist(metric_legend)
        ax.legend(handles=shock_legend, loc="best", frameon=True)

        out_path_quad = graph_folder_EDA / f"{prefix}_quadratic_jdg_vs_{baseline_label}.png"
        fig.tight_layout()
        fig.savefig(out_path_quad, dpi=dpi, bbox_inches="tight")
        out_paths.append(out_path_quad)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return out_paths


def plot_error_comparison(
    df: pd.DataFrame,
    error_col_jdg: str,
    error_col_benchmark: str,
    benchmark_label: str,
    graph_folder: str | Path | None = None,
    graph_folder_EDA: str | Path | None = None,
    filename_prefix: Optional[str] = None,
    filename_suffix: Optional[str] = None,
    show: bool = False,
    dpi: int = 180,
    y_axis_percentile: Optional[float] = None,
) -> Path:
    """Plot judgemental vs benchmark errors side-by-side."""
    spec = _infer_baseline_spec(df)
    shock_col = spec["shock_col"]
    
    graph_folder = graph_folder_EDA or graph_folder
    if graph_folder is None:
        raise ValueError("graph_folder is required")
    graph_folder = Path(graph_folder)
    graph_folder.mkdir(parents=True, exist_ok=True)
    
    x = df.index
    x_labels = _format_quarterly_index(x)
    pos = np.arange(len(df))
    
    y_jdg = df[error_col_jdg].to_numpy()
    y_bench = df[error_col_benchmark].to_numpy()
    
    shock = df[shock_col].astype("boolean")
    colours = np.where(shock.fillna(False).to_numpy(), "red", "green")
    
    shock_legend = [
        Patch(facecolor="red", edgecolor="none", label=f"Negative shock: realised < {benchmark_label}"),
        Patch(facecolor="green", edgecolor="none", label=f"Positive shock: realised ≥ {benchmark_label}"),
    ]
    
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axhline(0.0, linewidth=1.0)
    
    width = 0.38
    ax.bar(pos - width/2, y_jdg, width=width, color=colours, label="Judgemental error", alpha=0.9)
    ax.bar(pos + width/2, y_bench, width=width, color=colours, label=f"{benchmark_label} error", alpha=0.6)
    
    ax.set_xticks(pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    
    title = f"Judgemental vs {benchmark_label} Forecast Errors by Quarter"
    ax.set_title(title)
    ax.set_ylabel("Error (forecast - realized)")
    
    if y_axis_percentile is not None:
        combined_data = np.concatenate([y_jdg[~np.isnan(y_jdg)], y_bench[~np.isnan(y_bench)]])
        _apply_percentile_truncation(ax, combined_data, y_axis_percentile)
    
    metric_legend = ax.legend(loc="upper left", frameon=True)
    ax.add_artist(metric_legend)
    ax.legend(handles=shock_legend, loc="best", frameon=True)
    
    prefix = filename_prefix or "error_comparison"
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    out_path = graph_folder / f"{prefix}_{benchmark_label}{suffix}.png"
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return out_path


# =================================================================================================#
#                               SIGNAL DECOMPOSITION & OLS ANALYSIS
# =================================================================================================#

def transform_eval_dataframe(df: pd.DataFrame, naive_col_name: str) -> pd.DataFrame:
    """
    Transform evaluation dataframe by calculating signals relative to naive baseline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns ['realized', naive_col_name, 'ifoCast', 'judgemental']
    naive_col_name : str
        Name of the naive baseline column (e.g., 'naiveAR2', 'naiveAVERAGE_10')
    
    Returns:
    --------
    pd.DataFrame
        Transformed dataframe with signals relative to naive baseline
    """
    df_transformed = df.copy()
    
    # Calculate signals: forecast - naive baseline for the forecast columns
    df_transformed['ifoCast_signal'] = df_transformed['ifoCast'] - df_transformed[naive_col_name]
    df_transformed['judgemental_signal'] = df_transformed['judgemental'] - df_transformed[naive_col_name]
    df_transformed['error'] = df_transformed['realized'] - df_transformed['judgemental']
    
    return df_transformed


def run_signals_analysis(
    df: pd.DataFrame,
    model_name: str,
    naive_col: str,
    save_folder_graphs: str,
    save_folder_tables: str,
) -> list[dict]:
    """
    Run visualization, summary stats, and OLS regressions for the signals decomposition.
    
    Estimates 6 OLS models to test forecast efficiency and signal quality:
    - Model 1: realized ~ ifoCast_signal + judgemental_signal (no intercept)
    - Model 2: (realized - naive) ~ ifoCast_signal + judgemental_signal (no intercept)
    - Model 3: (realized - naive) ~ intercept + ifoCast_signal + judgemental_signal
    - Model 4: (realized - naive - ifoCast_signal) ~ judgemental_signal
    - Model 5: (realized - naive - ifoCast_signal) ~ intercept + judgemental_signal
    - Model 6: (realized - naive - ifoCast_signal) ~ intercept + judgemental_signal + judgemental_signal^2
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transformed evaluation dataframe with signals
    model_name : str
        Name of baseline model (e.g., "AR2")
    naive_col : str
        Name of naive baseline column
    save_folder_graphs : str
        Path to save graphs
    save_folder_tables : str
        Path to save tables
    
    Returns:
    --------
    list[dict]
        List of OLS results dictionaries
    """
    import os
    import statsmodels.api as sm
    
    print(f"\n--- Running Signals Analysis for {model_name} ---")
    
    os.makedirs(save_folder_graphs, exist_ok=True)
    os.makedirs(save_folder_tables, exist_ok=True)
    
    # 1. Visualization: Plot components as bar time series
    cols_to_plot = ['realized', naive_col, 'ifoCast_signal', 'judgemental_signal']
    colors = ['black', 'gray', 'tab:blue', 'tab:orange']
    
    fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(14, 10), sharex=True)
    if len(cols_to_plot) == 1: 
        axes = [axes]

    x_labels = [f"{ts.year}-Q{(ts.month - 1) // 3 + 1}" for ts in df.index]
    pos = np.arange(len(df))
    
    for ax, col, color in zip(axes, cols_to_plot, colors):
        if col in df.columns:
            ax.bar(pos, df[col], color=color, alpha=0.7, width=0.8)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_ylabel(col, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
    axes[-1].set_xticks(pos)
    axes[-1].set_xticklabels(x_labels, rotation=45, ha='right')
    fig.suptitle(f'Signals Decomposition - {model_name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    plot_path = os.path.join(save_folder_graphs, f"Signals_Decomposition_{model_name}.png")
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    # 2. Summary Statistics
    stats_df = df.describe()
    os.makedirs(save_folder_tables, exist_ok=True)
    stats_path = os.path.join(save_folder_tables, f"Signals_Summary_Stats_{model_name}.xlsx")
    stats_df.to_excel(stats_path)

    # 3. Covariance / Correlation
    cov_cols = [c for c in ['realized', naive_col, 'ifoCast_signal', 'judgemental_signal', 'error'] if c in df.columns]
    df_cov = df[cov_cols].dropna()

    if len(df_cov) > 1:
        cov_matrix = df_cov.cov()
        cov_path = os.path.join(save_folder_tables, f"Covariance_Matrix_{model_name}.xlsx")
        cov_matrix.to_excel(cov_path)

        corr_matrix = df_cov.corr()
        corr_path = os.path.join(save_folder_tables, f"Correlation_Matrix_{model_name}.xlsx")
        corr_matrix.to_excel(corr_path)

        # Heatmaps
        mask_cov = np.tril(np.ones_like(cov_matrix, dtype=bool), k=-1)
        fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
        sns.heatmap(cov_matrix, mask=mask_cov, annot=True, fmt=".4f", cmap="coolwarm", center=0,
                    square=True, linewidths=0.5, ax=ax_cov, cbar_kws={'label': 'Covariance'})
        ax_cov.set_title(f"Covariance Matrix – {model_name}", fontsize=13, fontweight="bold")
        fig_cov.tight_layout()
        cov_plot_path = os.path.join(save_folder_graphs, f"Covariance_Heatmap_{model_name}.png")
        fig_cov.savefig(cov_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig_cov)

        mask_corr = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, mask=mask_corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                    vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax_corr, cbar_kws={'label': 'Correlation'})
        ax_corr.set_title(f"Correlation Matrix – {model_name}", fontsize=13, fontweight="bold")
        fig_corr.tight_layout()
        corr_plot_path = os.path.join(save_folder_graphs, f"Correlation_Heatmap_{model_name}.png")
        fig_corr.savefig(corr_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig_corr)

    # 4. OLS Regressions
    results_list = []
    
    df_ols = df.copy()
    df_ols['const'] = 1.0
    df_ols['judgemental_signal_sq'] = df_ols['judgemental_signal'] ** 2

    def run_ols_and_store(y_data, x_cols, equation_label, hypothesis_str, individual_hypotheses=None):
        """Helper to run OLS and store results"""
        combined = pd.concat([y_data, df_ols[x_cols]], axis=1).dropna()
        if combined.empty or len(combined) < 3:
            print(f"  Skipping {equation_label}: insufficient data")
            return
            
        y_clean = combined.iloc[:, 0]
        X_clean = combined.iloc[:, 1:]
        
        model = sm.OLS(y_clean, X_clean)
        results = model.fit()
        
        # Joint Hypothesis Testing (F-test)
        try:
            f_test = results.f_test(hypothesis_str)
            f_val = f_test.fvalue.item() if hasattr(f_test.fvalue, "item") else float(f_test.fvalue)
            f_p = f_test.pvalue.item() if hasattr(f_test.pvalue, "item") else float(f_test.pvalue)
            reject = "Yes" if f_p < 0.05 else "No"
        except Exception as e:
            f_val = np.nan
            f_p = np.nan
            reject = f"Error"

        for term in x_cols:
            h0_val = 0.0
            if individual_hypotheses and term in individual_hypotheses:
                h0_val = individual_hypotheses[term]
            
            coef = results.params[term]
            std_err = results.bse[term]
            p_val = results.pvalues[term]
            
            res_dict = {
                "Model_Baseline": model_name,
                "Equation": equation_label,
                "Term": term,
                "Coefficient": coef,
                "Std_Error": std_err,
                "T_Stat": coef / std_err if std_err != 0 else np.nan,
                "P_Value": p_val,
                "H0_Value": h0_val,
                "Joint_Hypothesis": hypothesis_str,
                "Joint_F_Stat": f_val,
                "Joint_P_Value": f_p,
                "Reject_H0_5pct": reject
            }
            results_list.append(res_dict)

    # Model 1: realized ~ ifoCast_signal + judgemental_signal (no intercept)
    if 'ifoCast_signal' in df.columns and 'judgemental_signal' in df.columns:
        y_m1 = df['realized']
        cols_m1 = ['ifoCast_signal', 'judgemental_signal']
        hyp_m1 = "ifoCast_signal = 0, judgemental_signal = 1"
        indiv_m1 = {'ifoCast_signal': 0.0, 'judgemental_signal': 1.0}
        run_ols_and_store(y_m1, cols_m1, "1. Realized ~ Signals", hyp_m1, indiv_m1)

    # Model 2: (realized - naive) ~ ifoCast_signal + judgemental_signal (no intercept)
    if naive_col in df.columns:
        y_m2 = df['realized'] - df[naive_col]
        cols_m2 = ['ifoCast_signal', 'judgemental_signal']
        hyp_m2 = "ifoCast_signal = 1, judgemental_signal = 1"
        indiv_m2 = {'ifoCast_signal': 1.0, 'judgemental_signal': 1.0}
        run_ols_and_store(y_m2, cols_m2, "2. (Realized - Naive) ~ Signals", hyp_m2, indiv_m2)

        # Model 3: (realized - naive) ~ intercept + ifoCast_signal + judgemental_signal
        cols_m3 = ['const', 'ifoCast_signal', 'judgemental_signal']
        hyp_m3 = "const = 0, ifoCast_signal = 1, judgemental_signal = 1"
        indiv_m3 = {'const': 0.0, 'ifoCast_signal': 1.0, 'judgemental_signal': 1.0}
        run_ols_and_store(y_m2, cols_m3, "3. (Realized - Naive) ~ Const + Signals", hyp_m3, indiv_m3)

        # Model 4: (realized - naive - ifoCast_signal) ~ judgemental_signal
        y_m4 = df['realized'] - df[naive_col] - df['ifoCast_signal']
        cols_m4 = ['judgemental_signal']
        hyp_m4 = "judgemental_signal = 1"
        indiv_m4 = {'judgemental_signal': 1.0}
        run_ols_and_store(y_m4, cols_m4, "4. (Realized - Naive - ifoSignal) ~ JudgSignal", hyp_m4, indiv_m4)

        # Model 5: (realized - naive - ifoCast_signal) ~ intercept + judgemental_signal
        cols_m5 = ['const', 'judgemental_signal']
        hyp_m5 = "const = 0, judgemental_signal = 1"
        indiv_m5 = {'const': 0.0, 'judgemental_signal': 1.0}
        run_ols_and_store(y_m4, cols_m5, "5. (Realized - Naive - ifoSignal) ~ Const + JudgSignal", hyp_m5, indiv_m5)

        # Model 6: (realized - naive - ifoCast_signal) ~ intercept + judgemental_signal + judgemental_signal^2 (Efficiency test)
        cols_m6 = ['const', 'judgemental_signal', 'judgemental_signal_sq']
        hyp_m6 = "const = 0, judgemental_signal = 1, judgemental_signal_sq = 0"
        indiv_m6 = {'const': 0.0, 'judgemental_signal': 1.0, 'judgemental_signal_sq': 0.0}
        run_ols_and_store(y_m4, cols_m6, "6. Efficiency Test (Quad)", hyp_m6, indiv_m6)

    return results_list
