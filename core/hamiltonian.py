import jax.numpy as jnp
from dataclasses import dataclass
from .hilbert import HilbertSpace

@dataclass
class StaticHamiltonian:
    hilbert: HilbertSpace
    H0: jnp.ndarray

    def H(self, t, params=None):
        return self.H0
