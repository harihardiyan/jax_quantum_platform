import json
from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    """
    Configuration object for quantum simulation experiments.

    This class stores all relevant parameters needed to reproduce
    a simulation, such as Hamiltonian parameters, time grids,
    variational settings, etc.

    The configuration can be saved to and loaded from JSON files.
    """
    name: str
    parameters: Dict[str, Any]

    def to_json(self, filename: str):
        """
        Save configuration to a JSON file.

        Parameters
        ----------
        filename : str
            Output JSON filename.
        """
        with open(filename, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def from_json(filename: str):
        """
        Load configuration from a JSON file.

        Parameters
        ----------
        filename : str
            Input JSON filename.

        Returns
        -------
        ExperimentConfig
        """
        with open(filename, "r") as f:
            data = json.load(f)
        return ExperimentConfig(name=data["name"], parameters=data["parameters"])
