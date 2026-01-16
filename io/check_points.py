import json
import jax.numpy as jnp
from typing import Dict, Any


def save_checkpoint(params: jnp.ndarray, step: int, filename: str):
    """
    Save optimizer or variational parameters to a checkpoint file.

    Parameters
    ----------
    params : jnp.ndarray
        Parameter vector.
    step : int
        Current optimization step.
    filename : str
        Output filename.
    """
    data = {
        "step": step,
        "params": params.tolist(),
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def load_checkpoint(filename: str) -> Dict[str, Any]:
    """
    Load parameters and step index from a checkpoint file.

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    dict
        Dictionary containing:
        - "step": int
        - "params": jnp.ndarray
    """
    with open(filename, "r") as f:
        data = json.load(f)

    return {
        "step": data["step"],
        "params": jnp.array(data["params"]),
    }
