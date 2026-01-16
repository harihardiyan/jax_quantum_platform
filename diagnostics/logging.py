import json
from datetime import datetime


def save_log(data: dict, filename: str):
    """
    Save a diagnostic log dictionary to a JSON file.

    Parameters
    ----------
    data : dict
        Dictionary containing diagnostic information.
    filename : str
        Output filename (e.g., 'log.json').
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def timestamp():
    """
    Return a human-readable timestamp string.

    Returns
    -------
    str
        Timestamp in ISO format.
    """
    return datetime.now().isoformat()


def attach_timestamp(data: dict):
    """
    Add a timestamp field to a diagnostic dictionary.

    Parameters
    ----------
    data : dict
        Diagnostic dictionary.

    Returns
    -------
    dict
        Updated dictionary with timestamp.
    """
    out = dict(data)
    out["timestamp"] = timestamp()
    return out
