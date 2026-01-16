import jax.numpy as jnp
from dataclasses import dataclass
from .hilbert import HilbertSpace

@dataclass
class StateVector:
    hilbert: HilbertSpace
    data: jnp.ndarray

    def __post_init__(self):
        if self.data.shape != (self.hilbert.dim,):
            raise ValueError("State dimension mismatch")

    def normalize(self):
        n = jnp.linalg.norm(self.data)
        return StateVector(self.hilbert, self.data / n)

    def prob(self, idx: int):
        return jnp.abs(self.data[idx])**2
