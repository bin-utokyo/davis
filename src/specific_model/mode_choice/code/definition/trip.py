import os
import sys

from typing import Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from los import Los

__all__ = ["Trip"]


@dataclass
class Trip:
    trip_id: int
    dep_time: datetime
    arr_time: datetime
    o_zone: int
    d_zone: int
    model: "ModeChoiceModel"
    mode: Optional[int] = None

    def get_los(self, los_dict: dict[tuple[int, int], Los]) -> Optional[Los]:
        key = (self.o_zone, self.d_zone)
        return los_dict.get(key)

    def calculate_log_likelihood(self, los: Los, params: np.ndarray) -> float:
        """
        Calculate the log likelihood of the chosen mode given the Level of Service data and model parameters.

        Args:
            los (Los): Level of Service data for the trip.
            params (np.ndarray): Model parameters corresponding to los.attribute_names.

        Returns:
            float: The log likelihood of the chosen mode.
        """
        if self.mode is None:
            raise ValueError("Chosen mode is not specified for the trip.")
        if not self.model.isValid(los, params):
            raise ValueError("Invalid Level of Service data or parameters.")

        probabilities = self.model.calculate_mode_probability(los, params)
        return float(np.log(np.clip(probabilities[int(self.mode)], 1e-10, None)) if int(self.mode) in probabilities and probabilities[int(self.mode)] > 0 else 0)
    
    def choose_mode(self, los: Los, params: np.ndarray) -> int:
        """
        Choose the mode stochastically for a given level of service (LOS) and parameters.

        Args:
            los (Los): The level of service object containing relevant attributes.
            params (np.ndarray): The model parameters to use for the choice.

        Returns:
            int: The ID of the chosen mode.
        """
        if not self.model.isValid(los, params):
            raise ValueError("Invalid Level of Service data or parameters.")
        return self.model.choose_mode(los, params)


# 遅延インポート
from abc_mc import ModeChoiceModel
