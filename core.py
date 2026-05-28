from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np
import numpy.typing as npt


type Vector = npt.NDArray[np.float64]
type Matrix = npt.NDArray[np.float64]
type Bounds2D = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True, slots=True)
class Problem:
    name: str
    title: str
    f: Callable[[Vector], float]
    grad: Callable[[Vector], Vector]
    hess: Callable[[Vector], Matrix] | None = None
    x0: Vector | None = None
    bounds: Bounds2D | None = None
    x_star: Vector | None = None
    minimizers: tuple[Vector, ...] = ()
    A: Matrix | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CountedProblem:
    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.func_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0

    @property
    def name(self) -> str:
        return self.problem.name

    @property
    def title(self) -> str:
        return self.problem.title

    @property
    def A(self) -> Matrix | None:
        return self.problem.A

    def f(self, x: Vector) -> float:
        self.func_calls += 1
        return float(self.problem.f(as_vector(x)))

    def grad(self, x: Vector) -> Vector:
        self.grad_calls += 1
        return as_vector(self.problem.grad(as_vector(x)))

    def hess(self, x: Vector) -> Matrix:
        self.hess_calls += 1
        if self.problem.hess is None:
            raise ValueError(f"Hessian is not available for problem {self.problem.name}")
        return np.asarray(self.problem.hess(as_vector(x)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class OptimizeResult:
    method: str
    problem_name: str
    x0: Vector
    x: Vector
    f_value: float
    grad_norm: float
    iterations: int
    func_calls: int
    grad_calls: int
    hess_calls: int
    status: str
    trajectory: Matrix
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def as_vector(x: npt.ArrayLike) -> Vector:
    return np.asarray(x, dtype=np.float64)


def vector_to_string(x: Vector) -> str:
    return "[" + ", ".join(f"{value:.10g}" for value in np.asarray(x, dtype=float)) + "]"


def result_to_row(result: OptimizeResult, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "method": result.method,
        "problem": result.problem_name,
        "x0": vector_to_string(result.x0),
        "x_final": vector_to_string(result.x),
        "f_final": result.f_value,
        "grad_norm_final": result.grad_norm,
        "iterations": result.iterations,
        "func_calls": result.func_calls,
        "grad_calls": result.grad_calls,
        "hess_calls": result.hess_calls,
        "converged": result.converged,
        "status": result.status,
    }
    row.update(result.metadata)
    row.update(extra)
    return row
