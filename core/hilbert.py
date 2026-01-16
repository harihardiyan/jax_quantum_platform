from dataclasses import dataclass

@dataclass(frozen=True)
class HilbertSpace:
    """
    Representasi ruang Hilbert abstrak.
    Untuk awal: hanya dimensi dan label.
    """
    dim: int
    label: str = "generic"

    def __repr__(self):
        return f"HilbertSpace(dim={self.dim}, label='{self.label}')"
