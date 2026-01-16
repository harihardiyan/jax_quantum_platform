import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable


@dataclass
class Ansatz:
    """
    Generic variational ansatz wrapper.

    Parameters
    ----------
    fn : Callable
        A function mapping parameters -> state vector (jnp.ndarray of shape (dim,)).
        Must be pure JAX and compatible with jit/vmap.
    n_params : int
        Number of trainable parameters.
    """
    fn: Callable
    n_params: int

    def __call__(self, params: jnp.ndarray) -> jnp.ndarray:
        if params.shape != (self.n_params,):
            raise ValueError(
                f"Expected parameter shape {(self.n_params,)}, got {params.shape}"
            )
        return self.fn(params)


# ============================================================
# Example built-in ansatz
# ============================================================

def single_qubit_rotation_ansatz():
    """
    A simple single-qubit ansatz:
        |psi(theta)> = Rz(theta2) Rx(theta1) |0>

    Returns
    -------
    Ansatz
        An ansatz object with 2 parameters.
    """
    def fn(params):
        theta_x, theta_z = params
        Rx = jnp.array([
            [jnp.cos(theta_x/2), -1j*jnp.sin(theta_x/2)],
            [-1j*jnp.sin(theta_x/2), jnp.cos(theta_x/2)]
        ], dtype=jnp.complex128)

        Rz = jnp.array([
            [jnp.exp(-1j*theta_z/2), 0],
            [0, jnp.exp(1j*theta_z/2)]
        ], dtype=jnp.complex128)

        psi0 = jnp.array([1.0 + 0j, 0.0 + 0j])
        return Rz @ (Rx @ psi0)

    return Ansatz(fn=fn, n_params=2)
