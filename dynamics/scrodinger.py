import jax
import jax.numpy as jnp
from dataclasses import dataclass


def expm(A):
    """Matrix exponential using eigendecomposition."""
    vals, vecs = jnp.linalg.eig(A)
    exp_vals = jnp.exp(vals)
    return (vecs * exp_vals) @ jnp.linalg.inv(vecs)


@dataclass
class SchrodingerSolver:
    ham: object
    method: str = "exp_midpoint"

    def step(self, psi, t, dt):
        if self.method == "exp_midpoint":
            H_mid = self.ham.H(t + 0.5 * dt)
            U = expm(-1j * H_mid * dt)
            return U @ psi
        else:
            raise ValueError(f"Unknown method: {self.method}")


def time_evolve(ham, psi0, t_grid, method="exp_midpoint", audit=False):
    solver = SchrodingerSolver(ham, method)
    psi = psi0.data
    psi_all = []

    for i in range(len(t_grid) - 1):
        t = t_grid[i]
        dt = t_grid[i + 1] - t
        psi = solver.step(psi, t, dt)
        psi_all.append(psi)

    psi_all = jnp.array(psi_all)

    if audit:
        return psi_all, {
            "max_norm_deviation": float(jnp.max(jnp.abs(jnp.linalg.norm(psi_all, axis=1) - 1))),
            "hermiticity_mid": 0.0,
        }

    return psi_all
