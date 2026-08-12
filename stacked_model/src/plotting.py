from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.interpolate import Rbf, griddata
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d


def set_plot_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
    )

    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 15.5,
        "axes.titlesize": 15.5,
        "axes.labelsize": 15.5,
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "legend.fontsize": 15.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })

def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def parity_plot(
        df: pd.DataFrame,
        observed_col: str,
        predicted_col: str,
        output_path: Path,
        title: str,
) -> None:
    set_plot_style()

    observed = df[observed_col].astype(float)
    predicted = df[predicted_col].astype(float)

    low = min(observed.min(), predicted.min())
    high = max(observed.max(), predicted.max())
    buffer = 0.05 * (high - low)

    low -= buffer
    high += buffer

    error = observed - predicted
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    fig, ax = plt.subplots(figsize=(5.5, 5.2))

    ax.scatter(
        observed,
        predicted,
        s=48,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.45,
    )

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1.4,
        color="black",
        alpha=0.8,
    )

    ax.text(
        0.05,
        0.95,
        f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nR$^2$ = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", fc="w", ec="0.75"),
    )

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    save_plot(output_path)


def correlation_heat_map_plot(
        df: pd.DataFrame,
        cols: list[str],
        output_path: Path,
        title: str = "Correlation heatmap",
) -> None:
    set_plot_style()

    # Remove obsolete / unused columns
    drop_cols = {
        "V_stripping",
        "V stripping",
        "RAW_STRIP_COL",
        "Von (s)",
    }

    cols = [
        col for col in cols
        if col in df.columns and col not in drop_cols
    ]

    if len(cols) < 2:
        raise ValueError("Need at least two valid columns for correlation heatmap.")


    label_map = {
        "Applied V": "V$_{deposition}$",
        "Von (s)": "t$_{deposition}$",
        "Voff (s)": "t$_{stripping}$",
        "Total Von (s)": "Total V$_{on}$",
        "AbsVoltage": "|V|",
        "CyclePeriod_s": "Cycle period",
        "AbsV_x_Von": "|V| x t$_{deposition}$",
        "AbsV_x_TotalVon": "|V| x total V$_{on}$",
        "PulseCount": "Cycle",
        "Total amount of deposition (ppm)": "Total\ndeposition",
        "Total amount of deposition (moles)": "Total\ndeposition",
        "Co selectivity": "Co\nselectivity",
        "Co:Ni ratio": "Co:Ni\nratio",
    }

    plot_df = df[cols].copy()

    for col in cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    corr = plot_df.corr(numeric_only=True)

    # Rename rows and columns after calculating correlation
    corr = corr.rename(index=label_map, columns=label_map)

    fig, ax = plt.subplots(figsize=(8, 6.5))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Pearson correlation coefficient"},
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    save_plot(output_path)

def parse_bo_improvement_data(
        df: pd.DataFrame,
        x_col: str = "Experiment number",
        y_col: str = "Co selectivity",
        stage_col: str = "Stage",
        initial_stage: str = "Initial",
        bo_stages: list[str] | None = None,
        initial_focus_y_min: float = 80.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse data for a focused BO improvement plot.

    Returns:
    - all_initial_df: all initial points, used for true initial best
    - baseline_display_df: initial points shown on the plot
    - display_df: plotted points
    - campaign_df: points used to calculate best-observed-so-far
    """

    if bo_stages is None:
        bo_stages = ["BO1", "BO2", "BO3"]

    work = df[[x_col, y_col, stage_col]].copy()

    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work[stage_col] = work[stage_col].astype(str).str.strip()

    work = (
        work
        .dropna(subset=[x_col, y_col])
        .sort_values(x_col)
        .copy()
    )

    all_initial_df = work[work[stage_col] == initial_stage].copy()
    bo_df = work[work[stage_col].isin(bo_stages)].copy()

    baseline_display_df = all_initial_df[
        all_initial_df[y_col] >= initial_focus_y_min
    ].copy()

    display_df = pd.concat(
        [baseline_display_df, bo_df],
        axis=0,
        ignore_index=True,
    ).sort_values(x_col).copy()

    campaign_df = pd.concat(
        [all_initial_df, bo_df],
        axis=0,
        ignore_index=True,
    ).sort_values(x_col).copy()

    campaign_df["BestSoFar"] = campaign_df[y_col].cummax()

    return all_initial_df, baseline_display_df, display_df, campaign_df


def make_piecewise_experiment_axis(
        x_values: pd.Series | np.ndarray | list[float],
        x_min: float,
        zoom_start: float = 135.0,
        pre_scale: float = 0.35,
        post_scale: float = 2.2,
        gap: float = 3.0,
) -> np.ndarray:
    """
    Compress the early experiment region and expand the BO region.

    Original experiment numbers are transformed only for display.
    Tick labels still show the true experiment numbers.
    """

    x = np.asarray(x_values, dtype=float)

    pre_width = (zoom_start - x_min) * pre_scale

    x_display = np.where(
        x < zoom_start,
        (x - x_min) * pre_scale,
        pre_width + gap + (x - zoom_start) * post_scale,
    )

    return x_display


def plot_bo_experimental_improvement(
        df: pd.DataFrame,
        output_path: Path,
        x_col: str = "Experiment number",
        y_col: str = "Co selectivity",
        stage_col: str = "Stage",
        stage_order: list[str] = None,
        initial_stage: str = "Initial",
        bo_stages: list[str] = None,
        title: str = "",
        initial_focus_y_min: float = 84.0,
        y_axis_min: float = 84.0,
        y_axis_max: float | None = 91.5,
        zoom_start: float = 135.0,
        pre_scale: float = 0.35,
        post_scale: float = 2.2,
        gap: float = 3.0,
) -> None:
    """
    Focused, publication-style BO improvement plot.

    This version uses a piecewise x-axis:
    - early experiments are compressed
    - BO region is expanded
    - tick labels still show real experiment numbers
    """

    set_plot_style()

    req = [x_col, y_col, stage_col]
    missing = [c for c in req if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if stage_order is None:
        stage_order = [initial_stage, "BO1", "BO2", "BO3"]

    if bo_stages is None:
        bo_stages = [s for s in stage_order if s != initial_stage]

    all_initial_df, baseline_display_df, display_df, campaign_df = parse_bo_improvement_data(
        df=df,
        x_col=x_col,
        y_col=y_col,
        stage_col=stage_col,
        initial_stage=initial_stage,
        bo_stages=bo_stages,
        initial_focus_y_min=initial_focus_y_min,
    )

    if display_df.empty:
        raise ValueError("No points available for focused BO improvement plot.")

    if campaign_df.empty:
        raise ValueError("No campaign data available for BO improvement plot.")


    # Piecewise x-axis transform
    x_min_real = float(campaign_df[x_col].min())

    campaign_df["x_display"] = make_piecewise_experiment_axis(
        campaign_df[x_col],
        x_min=x_min_real,
        zoom_start=zoom_start,
        pre_scale=pre_scale,
        post_scale=post_scale,
        gap=gap,
    )

    display_df["x_display"] = make_piecewise_experiment_axis(
        display_df[x_col],
        x_min=x_min_real,
        zoom_start=zoom_start,
        pre_scale=pre_scale,
        post_scale=post_scale,
        gap=gap,
    )

    bo_df = display_df[display_df[stage_col].isin(bo_stages)].copy()

    initial_display_df = display_df[
        display_df[stage_col] == initial_stage
    ].copy()


    # Colors
    initial_color = "#d9d9d9"
    bo_color = "#d84b45"
    best_line_color = "#2f2f2f"

    band_colors = {
        "BO1": "#f1d5b8",
        "BO2": "#d8ead2",
        "BO3": "#f0caca",
        "BO4": "#ddd0ee",
    }

    fig, ax = plt.subplots(figsize=(6.42, 2.48))


    # Wide stage bands using transformed x-coordinates
    # No embedded BO labels.
    min_band_width = 7.0

    for stage in bo_stages:
        sub = campaign_df[campaign_df[stage_col] == stage]

        if sub.empty:
            continue

        x0_real = float(sub[x_col].min())
        x1_real = float(sub[x_col].max())
        center_real = 0.5 * (x0_real + x1_real)

        x0 = make_piecewise_experiment_axis(
            [x0_real],
            x_min=x_min_real,
            zoom_start=zoom_start,
            pre_scale=pre_scale,
            post_scale=post_scale,
            gap=gap,
        )[0]

        x1 = make_piecewise_experiment_axis(
            [x1_real],
            x_min=x_min_real,
            zoom_start=zoom_start,
            pre_scale=pre_scale,
            post_scale=post_scale,
            gap=gap,
        )[0]

        center = make_piecewise_experiment_axis(
            [center_real],
            x_min=x_min_real,
            zoom_start=zoom_start,
            pre_scale=pre_scale,
            post_scale=post_scale,
            gap=gap,
        )[0]

        band_width = max((x1 - x0) + 2.0, min_band_width)

        ax.axvspan(
            center - band_width / 2,
            center + band_width / 2,
            color=band_colors.get(stage, "#bbbbbb"),
            alpha=0.24,
            lw=0,
            zorder=0,
        )

    # Initial displayed points
    if not initial_display_df.empty:
        ax.scatter(
            initial_display_df["x_display"],
            initial_display_df[y_col],
            s=28,
            color=initial_color,
            edgecolors="none",
            alpha=0.62,
            zorder=1,
        )

    # BO displayed points
    if not bo_df.empty:
        ax.scatter(
            bo_df["x_display"],
            bo_df[y_col],
            s=36,
            color=bo_color,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.96,
            zorder=4,
        )


    # Best observed so far
    best_line = ax.plot(
        campaign_df["x_display"],
        campaign_df["BestSoFar"],
        color=best_line_color,
        linewidth=1,
        linestyle="--",
        zorder=5,
    )[0]

    best_line.set_dash_joinstyle("round")
    best_line.set_dash_capstyle("round")
    best_line.set_solid_joinstyle("round")
    best_line.set_solid_capstyle("round")


    # Axes labels
    ax.set_xlabel("Experiment number", fontsize=15.5)
    ax.set_ylabel("Observed Co selectivity (%)", fontsize=15.5)

    if title:
        ax.set_title(title, pad=10)

    # Real experiment-number ticks, displayed on transformed axis
    real_x_min = float(display_df[x_col].min())
    real_x_max = float(display_df[x_col].max())

    pre_ticks = np.arange(
        np.ceil(real_x_min / 20) * 20,
        min(zoom_start, real_x_max) + 1,
        20,
    )

    post_ticks = np.arange(
        np.ceil(zoom_start / 5) * 5,
        real_x_max + 1,
        5,
    )

    real_ticks = np.unique(
        np.concatenate([pre_ticks, post_ticks])
    )

    tick_positions = make_piecewise_experiment_axis(
        real_ticks,
        x_min=x_min_real,
        zoom_start=zoom_start,
        pre_scale=pre_scale,
        post_scale=post_scale,
        gap=gap,
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{int(t)}" for t in real_ticks])

    x_left = make_piecewise_experiment_axis(
        [real_x_min],
        x_min=x_min_real,
        zoom_start=zoom_start,
        pre_scale=pre_scale,
        post_scale=post_scale,
        gap=gap,
    )[0] - 1.5

    x_right = make_piecewise_experiment_axis(
        [real_x_max],
        x_min=x_min_real,
        zoom_start=zoom_start,
        pre_scale=pre_scale,
        post_scale=post_scale,
        gap=gap,
    )[0] + 2.5

    if y_axis_max is None:
        y_axis_max = float(
            max(display_df[y_col].max(), all_initial_df[y_col].max())
        ) + 1.0

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_axis_min, y_axis_max)


    # No graph gridlines
    ax.grid(False)

    # Major ticks only
    ax.minorticks_off()

    # Ticks only on bottom and left
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        direction="out",
        length=4.0,
        width=0.9,
        labelsize=6,
    )

    ax.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        direction="out",
        length=4.0,
        width=0.9,
        labelsize=6,
    )

    # Spines
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("black")

    ax.xaxis.label.set_size(9.5)
    ax.yaxis.label.set_size(9.5)

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.15,
        top=0.92,
    )

    save_plot(output_path)

