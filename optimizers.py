from __future__ import annotations

from math import sqrt
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from core import CountedProblem, OptimizeResult, Problem, Vector, as_vector
from line_search import LineSearchResult, backtracking_line_search, strong_wolfe_line_search


Optimizer = Callable[..., OptimizeResult]


def quadratic_cg(
    problem: Problem,
    x0: npt.ArrayLike | None = None,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    store_history: bool = True,
) -> OptimizeResult:
    counted = CountedProblem(problem)
    if problem.A is None:
        return _finish(
            "quadratic_cg",
            counted,
            _start(problem, x0),
            _start(problem, x0),
            0,
            "quadratic_matrix_unavailable",
            [],
        )

    x = _start(problem, x0)
    start = x.copy()
    trajectory: list[Vector] = [x.copy()]
    a = problem.A

    g = counted.grad(x)
    if not np.all(np.isfinite(g)):
        return _finish("quadratic_cg", counted, start, x, 0, "nonfinite_gradient", trajectory, g)
    p = -g

    for k in range(max_iter + 1):
        grad_norm = float(np.linalg.norm(g))
        if grad_norm <= eps:
            return _finish("quadratic_cg", counted, start, x, k, "converged", trajectory, g)
        if k == max_iter:
            break

        ap = a @ p
        denominator = float(p @ ap)
        if denominator <= 0.0 or not np.isfinite(denominator):
            return _finish(
                "quadratic_cg",
                counted,
                start,
                x,
                k,
                "not_positive_definite_quadratic_form",
                trajectory,
                g,
            )

        alpha = -float(g @ p) / denominator
        if not np.isfinite(alpha):
            return _finish("quadratic_cg", counted, start, x, k, "nonfinite_step", trajectory, g)

        x = x + alpha * p
        if store_history:
            trajectory.append(x.copy())

        g_next = counted.grad(x)
        if not np.all(np.isfinite(g_next)):
            return _finish(
                "quadratic_cg",
                counted,
                start,
                x,
                k + 1,
                "nonfinite_gradient",
                trajectory,
                g_next,
            )

        denominator_beta = float(g @ g)
        if denominator_beta <= 0.0:
            return _finish("quadratic_cg", counted, start, x, k + 1, "converged", trajectory, g_next)
        beta = float(g_next @ g_next) / denominator_beta
        p = -g_next + beta * p
        g = g_next

    return _finish("quadratic_cg", counted, start, x, max_iter, "max_iter_reached", trajectory, g)


def nonlinear_cg_fr(
    problem: Problem,
    x0: npt.ArrayLike | None = None,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    store_history: bool = True,
) -> OptimizeResult:
    return _nonlinear_cg(
        "nonlinear_cg_fr",
        problem,
        x0=x0,
        eps=eps,
        max_iter=max_iter,
        beta_rule="fr",
        store_history=store_history,
    )


def nonlinear_cg_pr(
    problem: Problem,
    x0: npt.ArrayLike | None = None,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    store_history: bool = True,
) -> OptimizeResult:
    return _nonlinear_cg(
        "nonlinear_cg_pr",
        problem,
        x0=x0,
        eps=eps,
        max_iter=max_iter,
        beta_rule="pr",
        store_history=store_history,
    )


def newton_cholesky(
    problem: Problem,
    x0: npt.ArrayLike | None = None,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    store_history: bool = True,
) -> OptimizeResult:
    counted = CountedProblem(problem)
    x = _start(problem, x0)
    start = x.copy()
    trajectory: list[Vector] = [x.copy()]

    for k in range(max_iter):
        g = counted.grad(x)
        grad_norm = float(np.linalg.norm(g))
        if not np.isfinite(grad_norm):
            return _finish("newton_cholesky", counted, start, x, k, "nonfinite_gradient", trajectory, g)
        if grad_norm <= eps:
            return _finish("newton_cholesky", counted, start, x, k, "converged", trajectory, g)

        try:
            h = counted.hess(x)
            l = np.linalg.cholesky(h)
            y = np.linalg.solve(l, -g)
            p = np.linalg.solve(l.T, y)
        except np.linalg.LinAlgError:
            return _finish(
                "newton_cholesky",
                counted,
                start,
                x,
                k,
                "hessian_not_positive_definite",
                trajectory,
                g,
            )
        except ValueError:
            return _finish(
                "newton_cholesky",
                counted,
                start,
                x,
                k,
                "hessian_unavailable",
                trajectory,
                g,
            )

        if not np.all(np.isfinite(p)):
            return _finish("newton_cholesky", counted, start, x, k, "nonfinite_step", trajectory, g)

        x = x + p
        if store_history:
            trajectory.append(x.copy())

    return _finish("newton_cholesky", counted, start, x, max_iter, "max_iter_reached", trajectory)


def newton_direction(
    problem: Problem,
    x0: npt.ArrayLike | None = None,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    store_history: bool = True,
) -> OptimizeResult:
    counted = CountedProblem(problem)
    x = _start(problem, x0)
    start = x.copy()
    trajectory: list[Vector] = [x.copy()]
    fallback_count = 0

    for k in range(max_iter):
        g = counted.grad(x)
        grad_norm = float(np.linalg.norm(g))
        if not np.isfinite(grad_norm):
            return _finish(
                "newton_direction",
                counted,
                start,
                x,
                k,
                "nonfinite_gradient",
                trajectory,
                g,
                {"fallback_count": fallback_count},
            )
        if grad_norm <= eps:
            return _finish(
                "newton_direction",
                counted,
                start,
                x,
                k,
                "converged",
                trajectory,
                g,
                {"fallback_count": fallback_count},
            )

        p = None
        try:
            h = counted.hess(x)
            candidate = np.linalg.solve(h, -g)
            if np.all(np.isfinite(candidate)) and float(g @ candidate) < 0.0:
                p = candidate
        except (ValueError, np.linalg.LinAlgError):
            p = None

        if p is None:
            p = -g
            fallback_count += 1

        step = _quasi_newton_line_search(counted, x, p, g)
        if not step.accepted:
            return _finish(
                "newton_direction",
                counted,
                start,
                x,
                k,
                step.status,
                trajectory,
                g,
                {"fallback_count": fallback_count},
            )

        x = x + step.alpha * p
        if store_history:
            trajectory.append(x.copy())

    return _finish(
        "newton_direction",
        counted,
        start,
        x,
        max_iter,
        "max_iter_reached",
        trajectory,
        metadata={"fallback_count": fallback_count},
    )


def dogleg(
    problem: Problem,
    x0: npt.ArrayLike | None = None,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    delta0: float = 1.0,
    delta_max: float = 100.0,
    eta: float = 0.1,
    store_history: bool = True,
) -> OptimizeResult:
    counted = CountedProblem(problem)
    x = _start(problem, x0)
    start = x.copy()
    trajectory: list[Vector] = [x.copy()]
    delta = float(delta0)
    accepted_steps = 0

    if delta <= 0.0 or delta_max <= 0.0 or delta > delta_max:
        raise ValueError("trust-region radii must satisfy 0 < delta0 <= delta_max")

    for k in range(max_iter):
        g = counted.grad(x)
        grad_norm = float(np.linalg.norm(g))
        if not np.isfinite(grad_norm):
            return _finish("dogleg", counted, start, x, k, "nonfinite_gradient", trajectory, g)
        if grad_norm <= eps:
            return _finish(
                "dogleg",
                counted,
                start,
                x,
                k,
                "converged",
                trajectory,
                g,
                {"accepted_steps": accepted_steps, "final_delta": delta},
            )

        try:
            h = counted.hess(x)
        except ValueError:
            return _finish("dogleg", counted, start, x, k, "hessian_unavailable", trajectory, g)

        p = _dogleg_step(g, h, delta)
        if p is None or not np.all(np.isfinite(p)):
            return _finish("dogleg", counted, start, x, k, "dogleg_step_failed", trajectory, g)

        predicted = -float(g @ p + 0.5 * p @ h @ p)
        if predicted <= 0.0 or not np.isfinite(predicted):
            p = -delta * g / max(float(np.linalg.norm(g)), 1e-300)
            predicted = -float(g @ p + 0.5 * p @ h @ p)
            if predicted <= 0.0 or not np.isfinite(predicted):
                delta *= 0.25
                if delta < 1e-14:
                    return _finish(
                        "dogleg",
                        counted,
                        start,
                        x,
                        k,
                        "trust_region_too_small",
                        trajectory,
                        g,
                    )
                continue

        f_x = counted.f(x)
        f_new = counted.f(x + p)
        actual = f_x - f_new
        rho = actual / predicted if np.isfinite(f_new) else -np.inf

        p_norm = float(np.linalg.norm(p))
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and abs(p_norm - delta) <= 1e-8 * max(1.0, delta):
            delta = min(2.0 * delta, delta_max)

        if rho > eta and np.isfinite(f_new):
            x = x + p
            accepted_steps += 1
            if store_history:
                trajectory.append(x.copy())

        if delta < 1e-14:
            return _finish(
                "dogleg",
                counted,
                start,
                x,
                k + 1,
                "trust_region_too_small",
                trajectory,
                metadata={"accepted_steps": accepted_steps, "final_delta": delta},
            )

    return _finish(
        "dogleg",
        counted,
        start,
        x,
        max_iter,
        "max_iter_reached",
        trajectory,
        metadata={"accepted_steps": accepted_steps, "final_delta": delta},
    )


def dfp(
    problem: Problem,
    x0: npt.ArrayLike | None = None,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    store_history: bool = True,
) -> OptimizeResult:
    return _quasi_newton(
        "dfp",
        problem,
        x0=x0,
        eps=eps,
        max_iter=max_iter,
        update_rule="dfp",
        store_history=store_history,
    )


def bfgs(
    problem: Problem,
    x0: npt.ArrayLike | None = None,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    store_history: bool = True,
) -> OptimizeResult:
    return _quasi_newton(
        "bfgs",
        problem,
        x0=x0,
        eps=eps,
        max_iter=max_iter,
        update_rule="bfgs",
        store_history=store_history,
    )


def lbfgs(
    problem: Problem,
    x0: npt.ArrayLike | None = None,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    memory: int = 10,
    store_history: bool = True,
) -> OptimizeResult:
    if memory <= 0:
        raise ValueError("memory must be positive")

    counted = CountedProblem(problem)
    x = _start(problem, x0)
    start = x.copy()
    trajectory: list[Vector] = [x.copy()]
    s_history: list[Vector] = []
    y_history: list[Vector] = []
    skipped_updates = 0
    restarts = 0

    for k in range(max_iter):
        g = counted.grad(x)
        grad_norm = float(np.linalg.norm(g))
        if not np.isfinite(grad_norm):
            return _finish("lbfgs", counted, start, x, k, "nonfinite_gradient", trajectory, g)
        if grad_norm <= eps:
            return _finish(
                "lbfgs",
                counted,
                start,
                x,
                k,
                "converged",
                trajectory,
                g,
                {"memory": memory, "skipped_updates": skipped_updates, "restarts": restarts},
            )

        p = -_lbfgs_inverse_hessian_product(g, s_history, y_history)
        if float(g @ p) >= 0.0 or not np.all(np.isfinite(p)):
            p = -g
            s_history.clear()
            y_history.clear()
            restarts += 1

        step = _quasi_newton_line_search(counted, x, p, g)
        if not step.accepted:
            return _finish(
                "lbfgs",
                counted,
                start,
                x,
                k,
                step.status,
                trajectory,
                g,
                {"memory": memory, "skipped_updates": skipped_updates, "restarts": restarts},
            )

        x_next = x + step.alpha * p
        g_next = counted.grad(x_next)
        s = x_next - x
        y = g_next - g
        if _has_positive_curvature(s, y):
            s_history.append(s)
            y_history.append(y)
            if len(s_history) > memory:
                s_history.pop(0)
                y_history.pop(0)
        else:
            skipped_updates += 1

        x = x_next
        if store_history:
            trajectory.append(x.copy())

    return _finish(
        "lbfgs",
        counted,
        start,
        x,
        max_iter,
        "max_iter_reached",
        trajectory,
        metadata={"memory": memory, "skipped_updates": skipped_updates, "restarts": restarts},
    )


def scipy_newton_cg(
    problem: Problem,
    x0: npt.ArrayLike | None = None,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    store_history: bool = True,
) -> OptimizeResult:
    counted = CountedProblem(problem)
    x = _start(problem, x0)
    start = x.copy()
    trajectory: list[Vector] = [x.copy()]

    if problem.hess is None:
        return _finish(
            "scipy_newton_cg",
            counted,
            start,
            x,
            0,
            "hessian_unavailable",
            trajectory,
        )

    try:
        from scipy.optimize import minimize
    except ModuleNotFoundError:
        return _finish(
            "scipy_newton_cg",
            counted,
            start,
            x,
            0,
            "scipy_not_installed",
            trajectory,
            metadata={"scipy_status": "not_installed"},
        )

    def callback(xk: npt.NDArray[np.float64]) -> None:
        if store_history:
            trajectory.append(as_vector(xk).copy())

    result = minimize(
        fun=counted.f,
        x0=x,
        jac=counted.grad,
        hess=counted.hess,
        method="Newton-CG",
        callback=callback,
        options={"xtol": eps, "maxiter": max_iter, "disp": False},
    )

    final_x = as_vector(result.x)
    g = counted.grad(final_x)
    status = "converged" if bool(result.success) and float(np.linalg.norm(g)) <= eps else f"scipy_status_{result.status}"
    if store_history and (len(trajectory) == 0 or not np.allclose(trajectory[-1], final_x)):
        trajectory.append(final_x.copy())
    return _finish(
        "scipy_newton_cg",
        counted,
        start,
        final_x,
        int(result.nit),
        status,
        trajectory,
        g,
        {"scipy_message": str(result.message)},
    )


def all_own_optimizers() -> list[Optimizer]:
    return [
        quadratic_cg,
        nonlinear_cg_fr,
        nonlinear_cg_pr,
        newton_cholesky,
        newton_direction,
        dogleg,
        dfp,
        bfgs,
        lbfgs,
    ]


def all_comparison_optimizers() -> list[Optimizer]:
    return [*all_own_optimizers(), scipy_newton_cg]


def _nonlinear_cg(
    method: str,
    problem: Problem,
    x0: npt.ArrayLike | None,
    eps: float,
    max_iter: int,
    beta_rule: str,
    store_history: bool,
) -> OptimizeResult:
    counted = CountedProblem(problem)
    x = _start(problem, x0)
    start = x.copy()
    trajectory: list[Vector] = [x.copy()]
    g = counted.grad(x)
    p = -g
    restarts = 0

    for k in range(max_iter):
        grad_norm = float(np.linalg.norm(g))
        if not np.isfinite(grad_norm):
            return _finish(method, counted, start, x, k, "nonfinite_gradient", trajectory, g)
        if grad_norm <= eps:
            return _finish(method, counted, start, x, k, "converged", trajectory, g, {"restarts": restarts})

        if float(g @ p) >= 0.0:
            p = -g
            restarts += 1

        step = backtracking_line_search(counted, x, p, g)
        if not step.accepted:
            return _finish(method, counted, start, x, k, step.status, trajectory, g, {"restarts": restarts})

        x_next = x + step.alpha * p
        g_next = counted.grad(x_next)
        denominator = float(g @ g)
        if denominator <= 0.0:
            return _finish(method, counted, start, x_next, k + 1, "converged", trajectory, g_next)

        if beta_rule == "fr":
            beta = float(g_next @ g_next) / denominator
        elif beta_rule == "pr":
            beta = max(0.0, float(g_next @ (g_next - g)) / denominator)
        else:
            raise ValueError(f"Unknown beta_rule: {beta_rule}")

        p_next = -g_next + beta * p
        if float(g_next @ p_next) >= 0.0:
            p_next = -g_next
            restarts += 1

        x = x_next
        g = g_next
        p = p_next
        if store_history:
            trajectory.append(x.copy())

    return _finish(method, counted, start, x, max_iter, "max_iter_reached", trajectory, g, {"restarts": restarts})


def _quasi_newton(
    method: str,
    problem: Problem,
    x0: npt.ArrayLike | None,
    eps: float,
    max_iter: int,
    update_rule: str,
    store_history: bool,
) -> OptimizeResult:
    counted = CountedProblem(problem)
    x = _start(problem, x0)
    start = x.copy()
    trajectory: list[Vector] = [x.copy()]
    n = x.size
    inverse_hessian = np.eye(n, dtype=np.float64)
    skipped_updates = 0
    restarts = 0

    for k in range(max_iter):
        g = counted.grad(x)
        grad_norm = float(np.linalg.norm(g))
        if not np.isfinite(grad_norm):
            return _finish(method, counted, start, x, k, "nonfinite_gradient", trajectory, g)
        if grad_norm <= eps:
            return _finish(
                method,
                counted,
                start,
                x,
                k,
                "converged",
                trajectory,
                g,
                {"skipped_updates": skipped_updates, "restarts": restarts},
            )

        p = -inverse_hessian @ g
        if float(g @ p) >= 0.0 or not np.all(np.isfinite(p)):
            inverse_hessian = np.eye(n, dtype=np.float64)
            p = -g
            restarts += 1

        step = backtracking_line_search(counted, x, p, g)
        if not step.accepted:
            return _finish(
                method,
                counted,
                start,
                x,
                k,
                step.status,
                trajectory,
                g,
                {"skipped_updates": skipped_updates, "restarts": restarts},
            )

        x_next = x + step.alpha * p
        g_next = counted.grad(x_next)
        s = x_next - x
        y = g_next - g

        if _has_positive_curvature(s, y):
            inverse_hessian = _update_inverse_hessian(inverse_hessian, s, y, update_rule)
        else:
            skipped_updates += 1

        x = x_next
        if store_history:
            trajectory.append(x.copy())

    return _finish(
        method,
        counted,
        start,
        x,
        max_iter,
        "max_iter_reached",
        trajectory,
        metadata={"skipped_updates": skipped_updates, "restarts": restarts},
    )


def _update_inverse_hessian(
    inverse_hessian: np.ndarray,
    s: Vector,
    y: Vector,
    update_rule: str,
) -> np.ndarray:
    ys = float(y @ s)
    if update_rule == "dfp":
        by = inverse_hessian @ y
        yby = float(y @ by)
        if yby <= 1e-20 or not np.isfinite(yby):
            return inverse_hessian
        updated = inverse_hessian + np.outer(s, s) / ys - np.outer(by, by) / yby
    elif update_rule == "bfgs":
        rho = 1.0 / ys
        identity = np.eye(inverse_hessian.shape[0], dtype=np.float64)
        left = identity - rho * np.outer(s, y)
        right = identity - rho * np.outer(y, s)
        updated = left @ inverse_hessian @ right + rho * np.outer(s, s)
    else:
        raise ValueError(f"Unknown update_rule: {update_rule}")
    return 0.5 * (updated + updated.T)


def _quasi_newton_line_search(
    counted: CountedProblem,
    x: Vector,
    p: Vector,
    g: Vector,
) -> LineSearchResult:
    step = strong_wolfe_line_search(counted, x, p, g)
    if step.accepted:
        return step
    fallback = backtracking_line_search(counted, x, p, g)
    if fallback.accepted:
        return fallback
    return step


def _lbfgs_inverse_hessian_product(
    g: Vector,
    s_history: list[Vector],
    y_history: list[Vector],
) -> Vector:
    if not s_history:
        return g.copy()

    q = g.copy()
    alphas: list[float] = []
    rhos = [1.0 / float(y @ s) for s, y in zip(s_history, y_history)]

    for s, y, rho in reversed(list(zip(s_history, y_history, rhos))):
        alpha = rho * float(s @ q)
        alphas.append(alpha)
        q = q - alpha * y

    s_last = s_history[-1]
    y_last = y_history[-1]
    gamma = float(s_last @ y_last) / max(float(y_last @ y_last), 1e-300)
    r = gamma * q

    for s, y, rho, alpha in zip(s_history, y_history, rhos, reversed(alphas)):
        beta = rho * float(y @ r)
        r = r + s * (alpha - beta)

    return r


def _dogleg_step(g: Vector, h: np.ndarray, delta: float) -> Vector | None:
    try:
        p_b = np.linalg.solve(h, -g)
        if float(g @ p_b) >= 0.0:
            p_b = None
    except np.linalg.LinAlgError:
        p_b = None

    if p_b is not None and float(np.linalg.norm(p_b)) <= delta:
        return p_b

    gg = float(g @ g)
    ghg = float(g @ h @ g)
    if gg <= 0.0:
        return np.zeros_like(g)

    if ghg > 0.0 and np.isfinite(ghg):
        p_u = -(gg / ghg) * g
    else:
        p_u = -delta * g / sqrt(gg)

    p_u_norm = float(np.linalg.norm(p_u))
    if p_u_norm >= delta or p_b is None:
        return delta * p_u / max(p_u_norm, 1e-300)

    d = p_b - p_u
    a = float(d @ d)
    b = 2.0 * float(p_u @ d)
    c = float(p_u @ p_u) - delta**2
    discriminant = max(0.0, b * b - 4.0 * a * c)
    tau = (-b + sqrt(discriminant)) / (2.0 * a)
    tau = min(1.0, max(0.0, tau))
    return p_u + tau * d


def _has_positive_curvature(s: Vector, y: Vector) -> bool:
    ys = float(y @ s)
    scale = max(1.0, float(np.linalg.norm(s) * np.linalg.norm(y)))
    return np.isfinite(ys) and ys > 1e-12 * scale


def _start(problem: Problem, x0: npt.ArrayLike | None) -> Vector:
    if x0 is not None:
        return as_vector(x0).astype(np.float64, copy=True)
    if problem.x0 is None:
        raise ValueError(f"x0 is required for problem {problem.name}")
    return as_vector(problem.x0).astype(np.float64, copy=True)


def _finish(
    method: str,
    counted: CountedProblem,
    start: Vector,
    x: Vector,
    iterations: int,
    status: str,
    trajectory: list[Vector],
    g: Vector | None = None,
    metadata: dict[str, Any] | None = None,
) -> OptimizeResult:
    if g is None:
        try:
            g = counted.grad(x)
        except Exception:
            g = np.full_like(x, np.nan)
    grad_norm = float(np.linalg.norm(g)) if np.all(np.isfinite(g)) else float("nan")
    try:
        f_value = counted.f(x)
    except Exception:
        f_value = float("nan")

    if not trajectory:
        trajectory = [start.copy(), x.copy()]
    elif not np.allclose(trajectory[-1], x, rtol=0.0, atol=0.0):
        trajectory = [*trajectory, x.copy()]

    return OptimizeResult(
        method=method,
        problem_name=counted.problem.name,
        x0=start.copy(),
        x=x.copy(),
        f_value=float(f_value),
        grad_norm=grad_norm,
        iterations=int(iterations),
        func_calls=counted.func_calls,
        grad_calls=counted.grad_calls,
        hess_calls=counted.hess_calls,
        status=status,
        trajectory=np.asarray(trajectory, dtype=np.float64),
        metadata=dict(metadata or {}),
    )
