from __future__ import annotations

from math import e, exp, pi, sqrt

import numpy as np
import numpy.typing as npt

from core import Bounds2D, Matrix, Problem, Vector, as_vector


def generate_quadratic(
    n: int,
    condition_number: float,
    seed: int | None = None,
    random_rotation: bool = True,
    x_star: npt.ArrayLike | None = None,
    c: float = 0.0,
    name: str | None = None,
    title: str | None = None,
    bounds: Bounds2D | None = None,
) -> Problem:
    if n <= 0:
        raise ValueError("n must be positive")
    if condition_number < 1.0:
        raise ValueError("condition_number must be at least 1")

    rng = np.random.default_rng(seed)
    lambdas = (
        np.array([1.0], dtype=np.float64)
        if n == 1
        else np.geomspace(1.0, float(condition_number), n, dtype=np.float64)
    )
    if random_rotation:
        raw = rng.normal(size=(n, n))
        q, _ = np.linalg.qr(raw)
        a = q.T @ np.diag(lambdas) @ q
    else:
        a = np.diag(lambdas)
    a = 0.5 * (a + a.T)

    minimizer = (
        rng.normal(size=n).astype(np.float64)
        if x_star is None
        else as_vector(x_star).astype(np.float64, copy=True)
    )
    if minimizer.shape != (n,):
        raise ValueError("x_star must have shape (n,)")

    b = a @ minimizer

    def f(x: Vector) -> float:
        point = as_vector(x)
        return float(0.5 * point @ a @ point - b @ point + c)

    def grad(x: Vector) -> Vector:
        return a @ as_vector(x) - b

    def hess(x: Vector) -> Matrix:
        del x
        return a

    readable_name = name or f"quadratic_n{n}_k{condition_number:g}"
    readable_title = title or f"Quadratic n={n}, cond={condition_number:g}"
    minimizers = (minimizer.copy(),) if n == 2 else ()
    return Problem(
        name=readable_name,
        title=readable_title,
        f=f,
        grad=grad,
        hess=hess,
        x0=np.zeros(n, dtype=np.float64),
        bounds=bounds,
        x_star=minimizer.copy(),
        minimizers=minimizers,
        A=a,
        metadata={
            "n": n,
            "condition_number": float(condition_number),
            "b": b.copy(),
            "c": float(c),
            "eigenvalues": lambdas.copy(),
        },
    )


def quadratic_2d_for_trajectories() -> Problem:
    return generate_quadratic(
        n=2,
        condition_number=10.0,
        seed=104,
        x_star=np.array([1.0, -2.0], dtype=np.float64),
        name="quadratic_2d_k10",
        title="2D quadratic, cond=10",
        bounds=((-6.0, 6.0), (-7.0, 6.0)),
    )


def rosenbrock(z: Vector) -> float:
    x, y = as_vector(z)
    return float(100.0 * (y - x**2) ** 2 + (1.0 - x) ** 2)


def grad_rosenbrock(z: Vector) -> Vector:
    x, y = as_vector(z)
    return np.array(
        [
            -400.0 * x * (y - x**2) - 2.0 * (1.0 - x),
            200.0 * (y - x**2),
        ],
        dtype=np.float64,
    )


def hess_rosenbrock(z: Vector) -> Matrix:
    x, y = as_vector(z)
    return np.array(
        [
            [1200.0 * x**2 - 400.0 * y + 2.0, -400.0 * x],
            [-400.0 * x, 200.0],
        ],
        dtype=np.float64,
    )


def himmelblau(z: Vector) -> float:
    x, y = as_vector(z)
    first = x**2 + y - 11.0
    second = x + y**2 - 7.0
    return float(first**2 + second**2)


def grad_himmelblau(z: Vector) -> Vector:
    x, y = as_vector(z)
    first = x**2 + y - 11.0
    second = x + y**2 - 7.0
    return np.array(
        [
            4.0 * x * first + 2.0 * second,
            2.0 * first + 4.0 * y * second,
        ],
        dtype=np.float64,
    )


def hess_himmelblau(z: Vector) -> Matrix:
    x, y = as_vector(z)
    return np.array(
        [
            [12.0 * x**2 + 4.0 * y - 42.0, 4.0 * (x + y)],
            [4.0 * (x + y), 12.0 * y**2 + 4.0 * x - 26.0],
        ],
        dtype=np.float64,
    )


def ackley(z: Vector) -> float:
    x, y = as_vector(z)
    radius = sqrt(0.5 * (x**2 + y**2))
    mean_cos = 0.5 * (np.cos(2.0 * pi * x) + np.cos(2.0 * pi * y))
    return float(-20.0 * exp(-0.2 * radius) - exp(mean_cos) + 20.0 + e)


def grad_ackley(z: Vector) -> Vector:
    x, y = as_vector(z)
    point = np.array([x, y], dtype=np.float64)
    radius = sqrt(0.5 * float(point @ point))

    if radius < 1e-12:
        first = np.zeros(2, dtype=np.float64)
    else:
        first = 2.0 * exp(-0.2 * radius) * point / radius

    mean_cos = 0.5 * float(np.cos(2.0 * pi * point).sum())
    second = pi * exp(mean_cos) * np.sin(2.0 * pi * point)
    return first + second


def hess_ackley(z: Vector) -> Matrix:
    return finite_difference_hessian(grad_ackley, z)


def finite_difference_hessian(
    grad: callable,
    x: npt.ArrayLike,
    step: float = 1e-5,
) -> Matrix:
    point = as_vector(x)
    n = point.size
    hessian = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        shift = np.zeros(n, dtype=np.float64)
        shift[j] = step
        hessian[:, j] = (grad(point + shift) - grad(point - shift)) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def rosenbrock_problem() -> Problem:
    minimizer = np.array([1.0, 1.0], dtype=np.float64)
    return Problem(
        name="rosenbrock",
        title="Rosenbrock",
        f=rosenbrock,
        grad=grad_rosenbrock,
        hess=hess_rosenbrock,
        x0=np.array([-1.2, 1.0], dtype=np.float64),
        bounds=((-2.0, 2.5), (-1.0, 3.0)),
        x_star=minimizer,
        minimizers=(minimizer,),
    )


def himmelblau_problem() -> Problem:
    return Problem(
        name="himmelblau",
        title="Himmelblau",
        f=himmelblau,
        grad=grad_himmelblau,
        hess=hess_himmelblau,
        x0=np.array([0.0, 0.0], dtype=np.float64),
        bounds=((-6.0, 6.0), (-6.0, 6.0)),
        minimizers=(
            np.array([3.0, 2.0], dtype=np.float64),
            np.array([-2.805118, 3.131312], dtype=np.float64),
            np.array([-3.779310, -3.283186], dtype=np.float64),
            np.array([3.584428, -1.848126], dtype=np.float64),
        ),
    )


def ackley_problem() -> Problem:
    minimizer = np.array([0.0, 0.0], dtype=np.float64)
    return Problem(
        name="ackley",
        title="Ackley",
        f=ackley,
        grad=grad_ackley,
        hess=hess_ackley,
        x0=np.array([2.0, 2.0], dtype=np.float64),
        bounds=((-5.0, 5.0), (-5.0, 5.0)),
        x_star=minimizer,
        minimizers=(minimizer,),
    )


def test_function_problems() -> list[Problem]:
    return [rosenbrock_problem(), himmelblau_problem()]
