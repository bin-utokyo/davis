import numpy as np

from ..definition import Los
from ..abc import ModeChoiceModel

__all__ = ["MNL"]


class MNL(ModeChoiceModel):
    def calculate_mode_probability(self, los: "Los", params: np.ndarray) -> dict[int, float]:
        """
        Calculate mode choice probabilities using Multinomial Logit model.

        Args:
            los (Los): Level of Service data for the trip.
            params (np.ndarray): Model parameters corresponding to los.attribute_names.

        Returns:
            dict[int, float]: Dictionary of mode choice probabilities.
        """
        if not self.isValid(los, params):
            raise ValueError("Parameter vector length must match attribute names length.")

        utilities = np.full(len(los.availability), -np.inf, dtype=np.float32)
        for i, mode in enumerate(los.availability.keys()):
            available = los.availability[int(mode)]
            attr_values = los.attributes[int(mode)]
            if available:
                utilities[i] = np.dot(params, attr_values)

        utilities = np.array(utilities)
        exp_utilities = np.exp(utilities - np.max(utilities))
        probabilities = exp_utilities / np.sum(exp_utilities)

        probabilities_dict = {mode: prob for mode, prob in zip(los.availability.keys(), probabilities)}

        return probabilities_dict
    

    def isValid(self, los: Los, params: np.ndarray) -> bool:
        return len(params) == len(los.attribute_names)