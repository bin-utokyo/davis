from abc import ABC, abstractmethod

import numpy as np


__all__ = ["ModeChoiceModel"]


class ModeChoiceModel(ABC):
  @abstractmethod
  def calculate_mode_probability(self, los: "Los", params: np.ndarray) -> dict[int, float]:
    """
    Calculate the mode choice probabilities for a given level of service (LOS) and parameters.

    Args:
        los (Los): The level of service object containing relevant attributes.
        params (np.ndarray): The model parameters to use for the calculation.

    Returns:
        dict[int, float]: A dictionary mapping mode IDs to their respective probabilities.
    """
    pass

  @abstractmethod
  def isValid(self, los: "Los", params: np.ndarray) -> bool:
    """
    Validate the model parameters for a given level of service (LOS).

    Args:
        los (Los): The level of service object containing relevant attributes.
        params (np.ndarray): The model parameters to validate.

    Returns:
        bool: True if the parameters are valid, False otherwise.
    """
    pass


# 遅延インポート
from .definition import Los
