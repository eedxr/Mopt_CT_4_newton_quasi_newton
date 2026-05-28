from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import OptimizeResult, Problem, Vector, result_to_row, vector_to_string
from objectives import (
    generate_quadratic,
    quadratic_2d_for_trajectories,
    test_function_problems,
)
from optimizers import (
    all_comparison_optimizers,
    all_own_optimizers,
    bfgs,
    dfp,
    dogleg,
    lbfgs,
    newton_cholesky,
    newton_direction,
    nonlinear_cg_fr,
    nonlinear_cg_pr,
    quadratic_cg,
    scipy_newton_cg,
)
from plots import (
    plot_contours_with_trajectories,
    plot_lbfgs_memory,
    plot_metric_vs_condition,
    plot_metric_vs_n,
    save_results_table,
)


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TOL = 1e-8

QUADRATIC_N_VALUES = [2, 10, 50, 100]
QUADRATIC_CONDITION_VALUES = [1, 10, 100, 1000]

QUADRATIC_STARTS_2D = [
    np.array([-5.0, -5.0], dtype=np.float64),
    np.array([5.0, 5.0], dtype=np.float64),
    np.array([-5.0, 3.0], dtype=np.float64),
    np.array([3.0, -6.0], dtype=np.float64),
    np.array([0.0, 5.0], dtype=np.float64),
]

TEST_STARTS = {
    "rosenbrock": [
        np.array([-1.2, 1.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([2.0, 2.0], dtype=np.float64),
    ],
    "himmelblau": [
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([-3.0, 3.0], dtype=np.float64),
        np.array([3.0, -2.0], dtype=np.float64),
        np.array([-4.0, -4.0], dtype=np.float64),
    ],
    "ackley": [
        np.array([2.0, 2.0], dtype=np.float64),
        np.array([-2.0, 1.0], dtype=np.float64),
        np.array([0.5, -2.0], dtype=np.float64),
    ],
}


def run_quadratic_experiments() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for n in QUADRATIC_N_VALUES:
        for condition_number in QUADRATIC_CONDITION_VALUES:
            problem = generate_quadratic(
                n=n,
                condition_number=condition_number,
                seed=_quadratic_seed(n, condition_number),
            )
            x0 = np.zeros(n, dtype=np.float64)
            for optimizer in all_comparison_optimizers():
                result = optimizer(
                    problem,
                    x0=x0,
                    eps=TOL,
                    max_iter=3_000,
                    store_history=False,
                )
                rows.append(
                    result_to_row(
                        result,
                        experiment="quadratic_grid",
                        n=n,
                        condition_number=condition_number,
                        x_star=vector_to_string(problem.x_star) if problem.x_star is not None else "",
                    )
                )

    df = pd.DataFrame(rows)
    save_results_table(df, TABLES_DIR / "quadratic_grid.csv")

    for metric in ("iterations", "func_calls", "grad_calls", "hess_calls"):
        plot_metric_vs_n(
            df,
            FIGURES_DIR / "quadratic_grid" / f"{metric}_vs_n.png",
            metric=metric,
            title=f"Quadratic grid: {metric} vs dimension",
        )
        plot_metric_vs_condition(
            df,
            FIGURES_DIR / "quadratic_grid" / f"{metric}_vs_condition.png",
            metric=metric,
            title=f"Quadratic grid: {metric} vs condition number",
        )

    return df


def run_2d_trajectory_experiments() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    problem = quadratic_2d_for_trajectories()
    methods = all_own_optimizers()

    for start_index, x0 in enumerate(QUADRATIC_STARTS_2D, start=1):
        results: list[OptimizeResult] = []
        for optimizer in methods:
            result = optimizer(
                problem,
                x0=x0,
                eps=TOL,
                max_iter=600,
                store_history=True,
            )
            rows.append(
                result_to_row(
                    result,
                    experiment="quadratic_2d_trajectories",
                    start_index=start_index,
                    n=2,
                    condition_number=10,
                )
            )
            results.append(result)

        plot_contours_with_trajectories(
            problem,
            results,
            FIGURES_DIR
            / "quadratic_2d_trajectories"
            / f"start_{start_index}.png",
            title=f"2D quadratic trajectories, start {start_index}",
        )

    df = pd.DataFrame(rows)
    save_results_table(df, TABLES_DIR / "quadratic_2d_trajectories.csv")
    return df


def run_test_function_experiments() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = [
        nonlinear_cg_fr,
        nonlinear_cg_pr,
        newton_cholesky,
        newton_direction,
        dogleg,
        dfp,
        bfgs,
        lbfgs,
        scipy_newton_cg,
    ]

    for problem in test_function_problems():
        for start_index, x0 in enumerate(TEST_STARTS[problem.name], start=1):
            start_results: list[OptimizeResult] = []
            for optimizer in methods:
                result = optimizer(
                    problem,
                    x0=x0,
                    eps=TOL,
                    max_iter=8_000,
                    store_history=True,
                )
                extra = {
                    "experiment": "test_functions",
                    "start_index": start_index,
                }
                extra.update(_minimum_info(problem, result.x))
                rows.append(result_to_row(result, **extra))
                start_results.append(result)

            plot_contours_with_trajectories(
                problem,
                start_results,
                FIGURES_DIR
                / "test_functions"
                / problem.name
                / f"start_{start_index}.png",
                title=f"{problem.title}: trajectories, start {start_index}",
            )

    df = pd.DataFrame(rows)
    save_results_table(df, TABLES_DIR / "test_functions.csv")
    return df


def run_lbfgs_memory_experiment() -> pd.DataFrame:
    memory_values = [3, 5, 10, 20]
    n = 100
    condition_number = 1000
    problem = generate_quadratic(
        n=n,
        condition_number=condition_number,
        seed=_quadratic_seed(n, condition_number) + 17,
        name="lbfgs_memory_quadratic",
        title="L-BFGS memory quadratic",
    )

    rows: list[dict[str, object]] = []
    for memory in memory_values:
        result = lbfgs(
            problem,
            x0=np.zeros(n, dtype=np.float64),
            eps=TOL,
            max_iter=4_000,
            memory=memory,
            store_history=False,
        )
        rows.append(
            result_to_row(
                result,
                experiment="lbfgs_memory",
                n=n,
                condition_number=condition_number,
                memory=memory,
            )
        )

    df = pd.DataFrame(rows)
    save_results_table(df, TABLES_DIR / "lbfgs_memory.csv")
    plot_lbfgs_memory(df, FIGURES_DIR / "lbfgs_memory" / "iterations.png", metric="iterations")
    plot_lbfgs_memory(df, FIGURES_DIR / "lbfgs_memory" / "func_calls.png", metric="func_calls")
    return df


def start_experiments() -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    frames = [
        run_quadratic_experiments(),
        run_2d_trajectory_experiments(),
        run_test_function_experiments(),
        run_lbfgs_memory_experiment(),
    ]
    summary = pd.concat(frames, ignore_index=True)
    save_results_table(summary, TABLES_DIR / "summary.csv")
    return summary


def summarize_to_console(summary: pd.DataFrame) -> None:
    columns = [
        "experiment",
        "method",
        "problem",
        "n",
        "condition_number",
        "memory",
        "start_index",
        "iterations",
        "func_calls",
        "grad_calls",
        "hess_calls",
        "grad_norm_final",
        "f_final",
        "converged",
        "status",
        "nearest_minimum",
    ]
    existing_columns = [column for column in columns if column in summary.columns]
    print(summary[existing_columns].to_string(index=False))


def _minimum_info(problem: Problem, x: Vector) -> dict[str, object]:
    if not problem.minimizers:
        return {}
    distances = [float(np.linalg.norm(x - minimizer)) for minimizer in problem.minimizers]
    index = int(np.argmin(distances))
    return {
        "nearest_minimum": index + 1,
        "nearest_minimum_point": vector_to_string(problem.minimizers[index]),
        "nearest_minimum_distance": distances[index],
    }


def _quadratic_seed(n: int, condition_number: float) -> int:
    return int(100_000 + 997 * n + 31 * round(float(condition_number)))


__all__ = [
    "PROJECT_DIR",
    "RESULTS_DIR",
    "TABLES_DIR",
    "FIGURES_DIR",
    "TOL",
    "run_quadratic_experiments",
    "run_2d_trajectory_experiments",
    "run_test_function_experiments",
    "run_lbfgs_memory_experiment",
    "start_experiments",
    "summarize_to_console",
]
