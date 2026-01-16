import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from ..core.hamiltonian import HamiltonianBase
from ..diagnostics.audit_numerics import check_hermitian


def compute_floquet_operator(
    ham: HamiltonianBase,
    T: float,
    n_steps: int = 200,
):
    """
    Compute the Floquet operator U(T) for a time-periodic Hamiltonian H(t+T) = H(t).

    The evolution over one period is approximated by a time-ordered product
    using a simple midpoint rule.

    Parameters
    ----------
    ham : HamiltonianBase
        Time-periodic Hamiltonian.
    T : float
        Period of the drive.
    n_steps : int
        Number of time steps used to approximate the time-ordered exponential.

    Returns
    -------
    U_F : jnp.ndarray
        Floquet operator U(T) of shape (dim, dim).
    """
    dt = T / n_steps
    dim = ham.hilbert.dim
    U = jnp.eye(dim, dtype=jnp.complex128)

    def step(carry, k):
        U_prev = carry
        t_mid = (k + 0.5) * dt
        H_mid = ham.H(t_mid)
        U_step = expm(-1j * H_mid * dt)
        U_next = U_step @ U_prev
        return U_next, None

    U_final, _ = jax.lax.scan(step, U, jnp.arange(n_steps))
    return U_final


def floquet_spectrum(
    ham: HamiltonianBase,
    T: float,
    n_steps: int = 200,
):
    """
    Compute the Floquet operator and its quasi-energies.

    Parameters
    ----------
    ham : HamiltonianBase
        Time-periodic Hamiltonian.
    T : float
        Period of the drive.
    n_steps : int
        Number of time steps for the Floquet operator.

    Returns
    -------
    U_F : jnp.ndarray
        Floquet operator U(T).
    quasi_energies : jnp.ndarray
        Quasi-energies in the range (-pi/T, pi/T].
    """
    U_F = compute_floquet_operator(ham, T, n_steps)
    evals, _ = jnp.linalg.eig(U_F)

    # Eigenvalues of U_F are exp(-i * eps * T)
    phases = jnp.angle(evals)
    quasi_energies = -phases / T
    return U_F, quasi_energies


def floquet_audit(ham: HamiltonianBase, T: float, n_steps: int = 200):
    """
    Basic audit for Floquet computation:
    - Check unitarity of U_F via U_F† U_F ≈ I.
    - Check hermiticity of H(t) at a few sample times.
    """
    U_F = compute_floquet_operator(ham, T, n_steps)
    dim = ham.hilbert.dim

    # Unitarity check
    UU = jnp.conj(U_F.T) @ U_F
    dev_unitarity = jnp.max(jnp.abs(UU - jnp.eye(dim, dtype=jnp.complex128)))

    # Hermiticity checks at sample times
    ts = jnp.linspace(0.0, T, 5)
    herm_diffs = []
    for t in ts:
        Ht = ham.H(t)
        diff, _ = check_hermitian(Ht)
        herm_diffs.append(diff)
    herm_diffs = jnp.array(herm_diffs)

    return {
        "unitarity_deviation": float(dev_unitarity),
        "max_hermiticity_deviation": float(jnp.max(herm_diffs)),
    }
