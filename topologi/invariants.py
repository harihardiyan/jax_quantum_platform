import jax.numpy as jnp
from .berry import berry_phase


def chern_number(F12: jnp.ndarray, dk1: float, dk2: float) -> float:
    """
    Compute the Chern number using discretized Berry curvature:

        C = (1 / 2π) * sum_k F12(k) * dk1 * dk2

    Parameters
    ----------
    F12 : jnp.ndarray
        Berry curvature values on a grid, shape (Nk1, Nk2).
    dk1, dk2 : float
        Grid spacings.

    Returns
    -------
    float
        Chern number (not necessarily integer due to discretization).
    """
    return float(jnp.sum(F12) * dk1 * dk2 / (2 * jnp.pi))


def z2_invariant(wilson_phases: jnp.ndarray) -> int:
    """
    Compute a simple Z2 invariant from Wilson loop phases.

    This is a placeholder for more sophisticated methods (Fu-Kane, parity eigenvalues).
    For now, we use:
        Z2 = (# of crossings of π mod 2)

    Parameters
    ----------
    wilson_phases : jnp.ndarray
        Array of Wilson loop phases.

    Returns
    -------
    int
        Z2 invariant (0 or 1).
    """
    crossings = jnp.sum(jnp.isclose(jnp.abs(wilson_phases), jnp.pi, atol=1e-2))
    return int(crossings % 2)


def winding_number(phases: jnp.ndarray) -> float:
    """
    Compute the winding number of a 1D Berry phase curve.

    Parameters
    ----------
    phases : jnp.ndarray
        Array of phases along a loop.

    Returns
    -------
    float
        Winding number.
    """
    diffs = jnp.diff(phases)
    total = jnp.sum(jnp.unwrap(diffs))
    return float(total / (2 * jnp.pi))
