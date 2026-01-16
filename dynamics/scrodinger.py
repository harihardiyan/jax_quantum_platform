import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from ..core.states import StateVector
from ..core.hamiltonian import HamiltonianBase
from ..diagnostics.audit_numerics import check_norm, check_hermitian


def time_evolve(
    ham: HamiltonianBase,
    psi0: StateVector,
    t_grid: jnp.ndarray,
    method: str = "exp_midpoint",
    audit: bool = False,
):
    """
    Time evolution of a pure state under a (possibly time-dependent) Hamiltonian H(t).

    Parameters
    ----------
    ham : HamiltonianBase
        Hamiltonian object providing H(t).
    psi0 : StateVector
        Initial state |psi(0)>.
    t_grid : jnp.ndarray
        1D array of time points (monotonically increasing).
    method : str
        Time-stepping method. Currently supports:
        - "exp_midpoint": midpoint rule with matrix exponential.
    audit : bool
        If True, returns additional numerical diagnostics.

    Returns
    -------
    psi_all : jnp.ndarray
        Array of shape (Nt, dim) containing the state at each time in t_grid.
    audit_log : dict (optional)
        If audit=True, returns a dictionary with norm and hermiticity diagnostics.
    """
    if psi0.hilbert is not ham.hilbert:
        raise ValueError("Hilbert space of state and Hamiltonian do not match.")

    def step(carry, t_next):
        t_prev, psi_prev = carry
        dt = t_next - t_prev

        if method == "exp_midpoint":
            t_mid = 0.5 * (t_prev + t_next)
            H_mid = ham.H(t_mid)
            U = expm(-1j * H_mid * dt)
            psi_next = U @ psi_prev
        else:
            raise NotImplementedError(f"Method '{method}' is not implemented.")

        return (t_next, psi_next), psi_next

    t0 = t_grid[0]
    carry0 = (t0, psi0.data)

    (_, _), psi_rest = jax.lax.scan(step, carry0, t_grid[1:])
    psi_all = jnp.vstack([psi0.data[None, :], psi_rest])

    if not audit:
        return psi_all

    # Norm audit over the trajectory
    norms = jax.vmap(lambda psi: check_norm(psi)[0])(psi_all)

    # Hermiticity audit at mid-time
    t_mid_global = 0.5 * (t_grid[0] + t_grid[-1])
    H_mid_global = ham.H(t_mid_global)
    herm_diff, _ = check_hermitian(H_mid_global)

    audit_log = {
        "norms": norms,
        "max_norm_deviation": float(jnp.max(jnp.abs(norms - 1.0))),
        "hermiticity_mid": float(herm_diff),
    }

    return psi_all, audit_log
