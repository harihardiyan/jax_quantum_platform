import jax.numpy as jnp
from jax import vmap
from ..core.hamiltonian import HamiltonianBase


def expectation_value(psi: jnp.ndarray, H: jnp.ndarray) -> jnp.ndarray:
    """
    Compute <psi|H|psi> for a pure state.

    Parameters
    ----------
    psi : jnp.ndarray
        State vector of shape (dim,).
    H : jnp.ndarray
        Hamiltonian matrix of shape (dim, dim).

    Returns
    -------
    float
        Expectation value.
    """
    return jnp.real(jnp.conj(psi) @ (H @ psi))


def vqe_cost(ham: HamiltonianBase, ansatz, params: jnp.ndarray) -> jnp.ndarray:
    """
    VQE cost function: E(theta) = <psi(theta)|H|psi(theta)>.

    Parameters
    ----------
    ham : HamiltonianBase
        Hamiltonian object.
    ansatz : Ansatz
        Variational ansatz object.
    params : jnp.ndarray
        Parameter vector.

    Returns
    -------
    float
        Energy expectation value.
    """
    psi = ansatz(params)
    H = ham.H(0.0)  # static Hamiltonian
    return expectation_value(psi, H)


def fidelity(psi: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """
    Fidelity between two pure states.

    Returns
    -------
    float
        |<psi|phi>|^2
    """
    return jnp.abs(jnp.vdot(psi, phi))**2
