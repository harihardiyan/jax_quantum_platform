from dataclasses import dataclass
import jax.numpy as jnp
from .hilbert import HilbertSpace

@dataclass
class HamiltonianBase:
    """
    Abstraksi Hamiltonian generik H(t).
    Semua turunan harus mengimplementasikan H(t, params).
    """
    hilbert: HilbertSpace

    def H(self, t, params=None) -> jnp.ndarray:
        """
        Return matrix H(t) berukuran (dim, dim).
        Harus pure JAX (tanpa side-effect).
        """
        raise NotImplementedError


@dataclass
class StaticHamiltonian(HamiltonianBase):
    """
    Hamiltonian statis: H(t) = H0.
    """
    H0: jnp.ndarray

    def __post_init__(self):
        if self.H0.shape != (self.hilbert.dim, self.hilbert.dim):
            raise ValueError(
                f"H0 shape {self.H0.shape} tidak cocok dengan dim Hilbert {self.hilbert.dim}"
            )

    def H(self, t, params=None) -> jnp.ndarray:
        return self.H0
