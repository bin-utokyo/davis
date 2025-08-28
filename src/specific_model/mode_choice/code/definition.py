from typing import Any, Optional, Hashable
from dataclasses import dataclass
from datetime import datetime
import re

import numpy as np

__all__ = ["Los", "Trip"]


@dataclass
class Los:
    o_zone: int
    d_zone: int
    availability: dict[int, bool]  # key: mode, value: availability
    attribute_names: list[str]  # list of attribute names
    attributes: dict[int, list[float]]  # key: mode, value: list of attribute values

    @staticmethod
    def from_dict(data: dict[Hashable, Any]) -> "Los":
        o_zone = int(data["OZone"])
        d_zone = int(data["DZone"])

        # keyのうち{mode番号}Availableの形式のものを抽出
        pattern = r"^(\d+)Available$"
        availability = {
            int(m.group(1)): bool(data[k])
            for k in data.keys()
            if (m := re.match(pattern, str(k)))
        }
        # keyのうち{mode番号}{属性名}の形式になっているものを抽出
        pattern_attr = r"^(\d+)(.*)$"
        attribute_names = set([m.group(2) for k in data.keys() if (m := re.match(pattern_attr, str(k))) and not re.match(pattern, str(k))])
        ## sort
        attribute_names = sorted(list(attribute_names))
        attributes = {int(mode): [0.0] * len(attribute_names) for mode in availability.keys()}
        for mode in attributes.keys():
            for i, att_name in enumerate(attribute_names):
                if (value := data.get(f"{mode}{att_name}", None)) is not None:
                    attributes[int(mode)][i] = float(value)

        return Los(
            o_zone=o_zone,
            d_zone=d_zone,
            availability=availability,
            attributes=attributes,
            attribute_names=attribute_names
        )

@dataclass
class Trip:
    trip_id: int
    dep_time: datetime
    arr_time: datetime
    o_zone: int
    d_zone: int
    mode: Optional[int] = None

    def get_los(self, los_dict: dict[tuple[int, int], Los]) -> Optional[Los]:
        key = (self.o_zone, self.d_zone)
        return los_dict.get(key)
    
    def calculate_mode_probability(self, los: Los, params: np.ndarray) -> dict[int, float]:
        """
        Calculate mode choice probabilities using Multinomial Logit model.

        Args:
            los (Los): Level of Service data for the trip.
            params (np.ndarray): Model parameters corresponding to los.attribute_names.

        Returns:
            dict[int, float]: Dictionary of mode choice probabilities.
        """
        if len(params) != len(los.attribute_names):
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

        probabilities = self.calculate_mode_probability(los, params)
        return float(np.log(np.clip(probabilities[int(self.mode)], 1e-10, None)) if int(self.mode) in probabilities and probabilities[int(self.mode)] > 0 else 0)
