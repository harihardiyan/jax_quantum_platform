import jax
import jax.numpy as jnp

from ..core.hamiltonian import HamiltonianBase
from ..diagnostics.audit_numerics import check_trace_rho  # you will add this later


def lindblad_rhs(
    rho: jnp.ndarray,
    H: jnp.ndarray,
    jump_ops: jnp.ndarray,
):
    """
    Right-hand side of the Lindblad master equation:

        dρ/dt = -i [H, ρ] + sum_k (L_k ρ L_k† - 1/2 {L_k† L_k, ρ})

    Parameters
    ----------
    rho : jnp.ndarray
        Density matrix of shape (dim, dim).
    H : jnp.ndarray
        Hamiltonian matrix of shape (dim, dim).
    jump_ops : jnp.ndarray
        Array of jump operators of shape (n_jump, dim, dim).

    Returns
    -------
    drho_dt : jnp.ndarray
        Time derivative of the density matrix.
    """
    comm = H @ rho - rho @ H
    unitary_part = -1j * comm

    def lindblad_term(L):
        L_rho = L @ rho
        L_dag = jnp.conj(L.T)
        return L_rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)

    if jump_ops.ndim == 2:
        jump_ops = jump_ops[None, :, :]

    lindblad_sum = jax.vmap(lindblad_term)(jump_ops).sum(axis=0)
    return unitary_part + lindblad_sum


def lindblad_time_evolve(
    ham: HamiltonianBase,
    rho0: jnp.ndarray,
    t_grid: jnp.ndarray,
    jump_ops: jnp.ndarray,
    method: str = "rk4",
    audit: bool = False,
):
    """
    Time evolution of a density matrix under a Lindblad master equation.

    Parameters
    ----------
    ham : HamiltonianBase
        Hamiltonian object providing H(t).
    rho0 : jnp.ndarray
        Initial density matrix of shape (dim, dim).
    t_grid : jnp.ndarray
        1D array of time points.
    jump_ops : jnp.ndarray
        Array of jump operators of shape (n_jump, dim, dim) or (dim, dim).
    method : str
        Integration method. Currently supports:
        - "rk4": explicit 4th-order Runge-Kutta.
    audit : bool
        If True, returns diagnostics (e.g., trace preservation).

    Returns
    -------
    rho_all : jnp.ndarray
        Array of shape (Nt, dim, dim) with density matrices at each time.
    audit_log : dict (optional)
        If audit=True, returns diagnostics.
    """
    dim = ham.hilbert.dim
    if rho0.shape != (dim, dim):
        raise ValueError("rho0 shape does not match Hilbert space dimension.")

    def step(carry, t_next):
        t_prev, rho_prev = carry
        dt = t_next - t_prev

        if method == "rk4":
            def rhs(t, rho):
                Ht = ham.H(t)
                return lindblad_rhs(rho, Ht, jump_ops)

            k1 = rhs(t_prev, rho_prev)
            k2 = rhs(t_prev + 0.5 * dt, rho_prev + 0.5 * dt * k1)
            k3 = rhs(t_prev + 0.5 * dt, rho_prev + 0.5 * dt * k2)
            k4 = rhs(t_prev + dt, rho_prev + dt * k3)
            rho_next = rho_prev + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise NotImplementedError(f"Method '{method}' is not implemented.")

        return (t_next, rho_next), rho_next

    t0 = t_grid[0]
    carry0 = (t0, rho0)

    (_, _), rho_rest = jax.lax.scan(step, carry0, t_grid[1:])
    rho_all = jnp.concatenate([rho0[None, :, :], rho_rest], axis=0)

    if not audit:
        return rho_all

    # Simple audit: trace preservation over time
    traces = jax.vmap(lambda r: jnp.trace(r).real)(rho_all)
    max_trace_dev = float(jnp.max(jnp.abs(traces - 1.0)))

    audit_log = {
        "traces": traces,
        "max_trace_deviation": max_trace_dev,
    }

    return rho_all, audit_log
