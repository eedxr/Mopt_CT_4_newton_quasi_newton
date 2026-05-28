from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core import CountedProblem, Vector


@dataclass(frozen=True, slots=True)
class LineSearchResult:
    alpha: float
    accepted: bool
    status: str


def backtracking_line_search(
    problem: CountedProblem,
    x: Vector,
    p: Vector,
    g: Vector,
    alpha0: float = 1.0,
    rho: float = 0.5,
    c1: float = 1e-4,
    max_backtracks: int = 80,
    min_alpha: float = 1e-16,
) -> LineSearchResult:
    if alpha0 <= 0.0:
        raise ValueError("alpha0 must be positive")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be in (0, 1)")
    if not 0.0 < c1 < 1.0:
        raise ValueError("c1 must be in (0, 1)")

    alpha = float(alpha0)
    fx = problem.f(x)
    directional_derivative = float(g @ p)

    if not np.isfinite(fx):
        return LineSearchResult(alpha=0.0, accepted=False, status="nonfinite_value")
    if directional_derivative >= 0.0:
        return LineSearchResult(
            alpha=0.0,
            accepted=False,
            status="not_descent_direction",
        )

    for _ in range(max_backtracks + 1):
        x_new = x + alpha * p
        f_new = problem.f(x_new)
        if np.isfinite(f_new) and f_new <= fx + c1 * alpha * directional_derivative:
            return LineSearchResult(alpha=alpha, accepted=True, status="accepted")

        alpha *= rho
        if alpha < min_alpha:
            break

    return LineSearchResult(alpha=0.0, accepted=False, status="line_search_failed")


def strong_wolfe_line_search(
    problem: CountedProblem,
    x: Vector,
    p: Vector,
    g: Vector,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    growth: float = 2.0,
    shrink: float = 0.5,
    alpha_max: float = 1e6,
    max_iter: int = 50,
    max_zoom_iter: int = 60,
    min_interval: float = 1e-14,
) -> LineSearchResult:
    if not 0.0 < c1 < c2 < 1.0:
        raise ValueError("strong Wolfe constants must satisfy 0 < c1 < c2 < 1")
    if growth <= 1.0:
        raise ValueError("growth must be greater than 1")
    if not 0.0 < shrink < 1.0:
        raise ValueError("shrink must be in (0, 1)")

    f_x = problem.f(x)
    derphi0 = float(g @ p)
    if not np.isfinite(f_x):
        return LineSearchResult(alpha=0.0, accepted=False, status="nonfinite_value")
    if derphi0 >= 0.0:
        return LineSearchResult(alpha=0.0, accepted=False, status="not_descent_direction")

    def phi(alpha: float) -> float:
        value = problem.f(x + alpha * p)
        return float(value) if np.isfinite(value) else float("inf")

    def derphi(alpha: float) -> float:
        return float(problem.grad(x + alpha * p) @ p)

    def zoom(alpha_lo: float, alpha_hi: float, phi_lo: float) -> LineSearchResult:
        alpha_mid = 0.5 * (alpha_lo + alpha_hi)
        for _ in range(max_zoom_iter):
            alpha_mid = 0.5 * (alpha_lo + alpha_hi)
            phi_mid = phi(alpha_mid)

            if phi_mid > f_x + c1 * alpha_mid * derphi0 or phi_mid >= phi_lo:
                alpha_hi = alpha_mid
            else:
                derphi_mid = derphi(alpha_mid)
                if abs(derphi_mid) <= c2 * abs(derphi0):
                    return LineSearchResult(alpha=alpha_mid, accepted=True, status="accepted")
                if derphi_mid * (alpha_hi - alpha_lo) >= 0.0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_mid
                phi_lo = phi_mid

            if abs(alpha_hi - alpha_lo) <= min_interval:
                break

        return LineSearchResult(
            alpha=0.0,
            accepted=False,
            status="strong_wolfe_zoom_failed",
        )

    alpha_prev = 0.0
    phi_prev = f_x
    alpha = min(float(alpha0), alpha_max)

    for iteration in range(max_iter):
        phi_alpha = phi(alpha)
        if phi_alpha > f_x + c1 * alpha * derphi0 or (iteration > 0 and phi_alpha >= phi_prev):
            return zoom(alpha_prev, alpha, phi_prev)

        derphi_alpha = derphi(alpha)
        if abs(derphi_alpha) <= c2 * abs(derphi0):
            return LineSearchResult(alpha=alpha, accepted=True, status="accepted")
        if derphi_alpha >= 0.0:
            return zoom(alpha, alpha_prev, phi_alpha)

        alpha_prev = alpha
        phi_prev = phi_alpha
        next_alpha = min(alpha * growth, alpha_max)
        if next_alpha <= alpha:
            break
        alpha = next_alpha

    alpha = min(alpha, alpha_max)
    for _ in range(max_iter):
        phi_alpha = phi(alpha)
        if np.isfinite(phi_alpha) and phi_alpha <= f_x + c1 * alpha * derphi0:
            return LineSearchResult(
                alpha=alpha,
                accepted=False,
                status="strong_wolfe_fallback_armijo_only",
            )
        alpha *= shrink

    return LineSearchResult(alpha=0.0, accepted=False, status="strong_wolfe_failed")
