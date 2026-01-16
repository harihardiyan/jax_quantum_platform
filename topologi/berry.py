import jax
import jax.numpy as jnp


def berry_connection(psi_k: jnp.ndarray, dpsi_k: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Berry connection A(k) = Im <psi(k) | d/dk psi(k)>.

    Parameters
    ----------
    psi_k : jnp.ndarray
        Normalized eigenvector at momentum k, shape (dim,).
    dpsi_k : jnp.ndarray
        Derivative of eigenvector w.r.t. k, shape (dim,).

    Returns
    -------
    float
        Berry connection A(k).
    """
    inner = jnp.vdot(psi_k, dpsi_k)
    return jnp.imag(inner)


def berry_phase(psi_k_list: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Berry phase along a closed loop using the discretized formula:

        gamma = -Im log prod_k <psi_k | psi_{k+1}>

    Parameters
    ----------
    psi_k_list : jnp.ndarray
        Array of eigenvectors along a loop, shape (Nk, dim).

    Returns
    -------
    float
        Berry phase in radians.
    """
    overlaps = jax.vmap(lambda a, b: jnp.vdot(a, b))(
        psi_k_list[:-1], psi_k_list[1:]
    )
    # Close the loop
    overlaps = jnp.append(overlaps, jnp.vdot(psi_k_list[-1], psi_k_list[0]))

    phase = -jnp.angle(jnp.prod(overlaps))
    return phase


def berry_curvature(F12: jnp.ndarray) -> jnp.ndarray:
    """
    Identity function for clarity: Berry curvature is often computed
    directly from Wilson loops or finite differences.

    Parameters
    ----------
    F12 : jnp.ndarray
        Berry curvature value(s).

    Returns
    -------
    jnp.ndarray
        Same as input.
    """
    return F12
