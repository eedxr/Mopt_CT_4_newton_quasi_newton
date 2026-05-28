from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from core import OptimizeResult, Problem


plt.style.use("seaborn-v0_8-whitegrid")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip().lower())
    return slug.strip("_") or "item"


def save_results_table(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_plot(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric_vs_n(
    df: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
) -> None:
    _plot_grouped_metric(
        df=df,
        x_column="n",
        metric=metric,
        output_path=output_path,
        title=title,
        xlabel="dimension n",
        log_x=True,
        log_y=metric in {"iterations", "func_calls", "grad_calls", "hess_calls"},
    )


def plot_metric_vs_condition(
    df: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
) -> None:
    _plot_grouped_metric(
        df=df,
        x_column="condition_number",
        metric=metric,
        output_path=output_path,
        title=title,
        xlabel="condition number",
        log_x=True,
        log_y=metric in {"iterations", "func_calls", "grad_calls", "hess_calls"},
    )


def plot_lbfgs_memory(
    df: pd.DataFrame,
    output_path: Path,
    metric: str = "iterations",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = df.sort_values("memory")
    ax.plot(data["memory"], data[metric], marker="o", linewidth=1.8)
    ax.set_xlabel("L-BFGS memory m")
    ax.set_ylabel(metric)
    ax.set_title(f"L-BFGS memory influence: {metric}")
    save_plot(fig, output_path)


def plot_contours_with_trajectories(
    problem: Problem,
    results: Sequence[OptimizeResult],
    output_path: Path,
    title: str,
    grid_size: int = 260,
    max_path_points: int = 800,
) -> None:
    if not results or problem.bounds is None:
        return

    x_bounds, y_bounds = problem.bounds
    xs = np.linspace(x_bounds[0], x_bounds[1], grid_size)
    ys = np.linspace(y_bounds[0], y_bounds[1], grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    values = np.fromiter(
        (problem.f(point) for point in points),
        dtype=np.float64,
        count=len(points),
    )
    z = values.reshape(grid_x.shape)
    finite = z[np.isfinite(z)]

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    if finite.size:
        levels = np.unique(np.quantile(finite, np.linspace(0.02, 0.98, 34)))
        if levels.size >= 2:
            ax.contour(grid_x, grid_y, z, levels=levels, linewidths=0.75, alpha=0.75)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    same_method = len({result.method for result in results}) == 1
    for index, result in enumerate(results):
        path = _downsample_path(result.trajectory, max_path_points)
        color = colors[index % len(colors)]
        label = _trajectory_label(result, index, same_method, len(results))
        ax.plot(path[:, 0], path[:, 1], linewidth=1.65, color=color, label=label)
        ax.scatter(result.trajectory[0, 0], result.trajectory[0, 1], color=color, marker="o", s=32, zorder=4)
        ax.scatter(result.trajectory[-1, 0], result.trajectory[-1, 1], color=color, marker="x", s=48, zorder=4)

    for minimizer in problem.minimizers:
        ax.scatter(
            minimizer[0],
            minimizer[1],
            color="black",
            marker="*",
            s=95,
            zorder=5,
        )

    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, Line2D([0], [0], color="0.45", linewidth=1.0, label="level lines"))
    labels.insert(0, "level lines")
    handles.extend(
        [
            Line2D([0], [0], marker="o", linestyle="", color="0.25", label="start point"),
            Line2D([0], [0], marker="x", linestyle="", color="0.25", label="final point"),
        ]
    )
    labels.extend(["start point", "final point"])
    if problem.minimizers:
        handles.append(Line2D([0], [0], marker="*", linestyle="", color="black", markersize=10, label="known minimizer"))
        labels.append("known minimizer")
    ax.legend(handles, labels, fontsize=7)
    save_plot(fig, output_path)


def _plot_grouped_metric(
    df: pd.DataFrame,
    x_column: str,
    metric: str,
    output_path: Path,
    title: str,
    xlabel: str,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    if df.empty:
        return

    aggregate = (
        df.groupby(["method", x_column], as_index=False)[metric]
        .median(numeric_only=True)
        .sort_values(["method", x_column])
    )

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for method, part in aggregate.groupby("method"):
        valid = part[np.isfinite(part[metric])]
        if valid.empty:
            continue
        ax.plot(valid[x_column], valid[metric], marker="o", linewidth=1.7, label=str(method))

    if log_x:
        ax.set_xscale("log")
    if log_y:
        positive = aggregate[metric][aggregate[metric] > 0.0]
        if not positive.empty:
            ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=7, ncols=2)
    save_plot(fig, output_path)


def _trajectory_label(
    result: OptimizeResult,
    index: int,
    same_method: bool,
    results_count: int,
) -> str:
    if results_count == 1:
        return result.method
    if same_method:
        return f"start {index + 1}"
    return result.method


def _downsample_path(path: np.ndarray, max_points: int) -> np.ndarray:
    if len(path) <= max_points:
        return path
    indices = np.unique(np.linspace(0, len(path) - 1, max_points, dtype=np.int64))
    return path[indices]
