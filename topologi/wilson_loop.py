import jax
import jax.numpy as jnp


def overlap(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the overlap <u|v> for multi-band states.

    Parameters
    ----------
    u, v : jnp.ndarray
        Arrays of shape (n_bands, dim).

    Returns
    -------
    jnp.ndarray
        Overlap matrix of shape (n_bands, n_bands).
    """
    return u @ jnp.conj(v.T)


def wilson_line(U: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Wilson line operator W = U_N ... U_2 U_1.

    Parameters
    ----------
    U : jnp.ndarray
        Array of overlap matrices of shape (N, n_bands, n_bands).

    Returns
    -------
    jnp.ndarray
        Wilson line matrix of shape (n_bands, n_bands).
    """
    def step(carry, Ui):
        return Ui @ carry, None

    W, _ = jax.lax.scan(step, jnp.eye(U.shape[1], dtype=jnp.complex128), U)
    return W


def wilson_loop(eigvecs: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Wilson loop eigenvalues for a closed loop in momentum space.

    Parameters
    ----------
    eigvecs : jnp.ndarray
        Array of eigenvectors along the loop, shape (Nk, n_bands, dim).

    Returns
    -------
    phases : jnp.ndarray
        Wilson loop phases (Wannier centers) in radians.
    """
    Nk = eigvecs.shape[0]

    # Compute overlap matrices U_k = <u_k | u_{k+1}>
    U_list = []
    for k in range(Nk - 1):
        U_list.append(overlap(eigvecs[k], eigvecs[k + 1]))
    # Close the loop
    U_list.append(overlap(eigvecs[-1], eigvecs[0]))

    U_array = jnp.array(U_list)

    # Wilson loop operator
    W = wilson_line(U_array)

    # Eigenvalues of W = exp(i * 2π * x)
    evals, _ = jnp.linalg.eig(W)
    phases = jnp.angle(evals)

    return phases
