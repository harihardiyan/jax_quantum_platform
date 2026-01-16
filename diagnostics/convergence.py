import jax.numpy as jnp


def check_hermitian(H: jnp.ndarray, tol: float = 1e-10):
    """
    Check hermiticity of a matrix H.

    Parameters
    ----------
    H : jnp.ndarray
        Matrix of shape (dim, dim).
    tol : float
        Numerical tolerance.

    Returns
    -------
    diff : float
        Maximum absolute deviation from hermiticity.
    ok : bool
        True if H is Hermitian within tolerance.
    """
    diff = jnp.max(jnp.abs(H - jnp.conj(H.T)))
    return diff, bool(diff < tol)


def check_norm(psi: jnp.ndarray, tol: float = 1e-8):
    """
    Check normalization of a state vector.

    Parameters
    ----------
    psi : jnp.ndarray
        State vector of shape (dim,).
    tol : float
        Numerical tolerance.

    Returns
    -------
    norm_val : float
        Norm of the state.
    ok : bool
        True if |psi| ≈ 1.
    """
    n = jnp.vdot(psi, psi).real
    return float(n), bool(jnp.abs(n - 1.0) < tol)


def check_trace_rho(rho: jnp.ndarray, tol: float = 1e-8):
    """
    Check trace of a density matrix.

    Parameters
    ----------
    rho : jnp.ndarray
        Density matrix of shape (dim, dim).
    tol : float
        Numerical tolerance.

    Returns
    -------
    tr : float
        Trace of rho.
    ok : bool
        True if Tr(rho) ≈ 1.
    """
    tr = jnp.trace(rho).real
    return float(tr), bool(jnp.abs(tr - 1.0) < tol)


def check_positive_semidefinite(rho: jnp.ndarray, tol: float = 1e-8):
    """
    Check if a density matrix is positive semidefinite.

    Parameters
    ----------
    rho : jnp.ndarray
        Density matrix.
    tol : float
        Numerical tolerance.

    Returns
    -------
    min_eig : float
        Minimum eigenvalue.
    ok : bool
        True if all eigenvalues >= -tol.
    """
    evals = jnp.linalg.eigvalsh(rho)
    min_eig = float(jnp.min(evals))
    return min_eig, bool(min_eig >= -tol)
