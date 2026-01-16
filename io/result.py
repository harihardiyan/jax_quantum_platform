import json
import jax.numpy as jnp
from typing import Any, Dict


def save_results(data: Dict[str, Any], filename: str):
    """
    Save simulation results to a JSON file.

    Parameters
    ----------
    data : dict
        Dictionary containing results (energies, states, diagnostics, etc.).
    filename : str
        Output filename.
    """
    serializable = {}
    for key, value in data.items():
        if isinstance(value, jnp.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value

    with open(filename, "w") as f:
        json.dump(serializable, f, indent=4)


def load_results(filename: str) -> Dict[str, Any]:
    """
    Load simulation results from a JSON file.

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    dict
        Loaded results.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    # Convert lists back to jnp arrays when appropriate
    out = {}
    for key, value in data.items():
        if isinstance(value, list):
            out[key] = jnp.array(value)
        else:
            out[key] = value

    return out
