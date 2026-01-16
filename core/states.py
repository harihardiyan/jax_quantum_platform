import jax.numpy as jnp
from dataclasses import dataclass
from .hilbert import HilbertSpace


@dataclass
class StateVector:
    """
    Pure state vector living in a Hilbert space.

    Parameters
    ----------
    hilbert : HilbertSpace
        The Hilbert space the state belongs to.
    data : jnp.ndarray
        Complex vector of shape (dim,).
    """
    hilbert: HilbertSpace
    data: jnp.ndarray

    def __post_init__(self):
        if self.data.shape != (self.hilbert.dim,):
            raise ValueError(
                f"State dimension mismatch: expected {(self.hilbert.dim,)}, got {self.data.shape}"
            )

    def normalize(self):
        """Return a normalized copy of the state."""
        n = jnp.linalg.norm(self.data)
        return StateVector(self.hilbert, self.data / n)

    def inner(self, other):
        """Compute inner product <self|other>."""
        return jnp.vdot(self.data, other.data)

    def prob(self, basis_index: int):
        """Probability of measuring |basis_index>."""
        return jnp.abs(self.data[basis_index])**2

    def as_array(self):
        """Return raw JAX array."""
        return self.data
