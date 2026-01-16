from dataclasses import dataclass

@dataclass(frozen=True)
class HilbertSpace:
    """
    Abstract Hilbert space representation.
    """
    dim: int
    label: str = "generic"

    def __repr__(self):
        return f"HilbertSpace(dim={self.dim}, label={self.label})"
