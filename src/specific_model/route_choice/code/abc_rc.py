from abc import ABC, abstractmethod

import numpy as np


class RouteChoiceModel(ABC):
    @abstractmethod
    def calculate_transition_probability(self, link_transition: "LinkTransition", params: np.ndarray) -> dict[int, float]:
        """Calculate the transition probability for a given link transition (current link, next link, destination node).

        Args:
            link_transition (LinkTransition): The link transition object.
            params (np.ndarray): Model parameters.

        Returns:
            dict[int, float]: A dictionary mapping link IDs to their transition probabilities.
        """
        pass
    
    @abstractmethod
    def get_param_size(self) -> int:
        """Get the size of the model parameters.

        Returns:
            int: The size of the model parameters.
        """
        pass

    @abstractmethod
    def is_valid(self, params: np.ndarray) -> bool:
        """Check if the model parameters are valid.

        Args:
            params (np.ndarray): Model parameters.

        Returns:
            bool: True if the parameters are valid, False otherwise.
        """
        pass

    def choose_transition(self, link_transition: "LinkTransition", params: np.ndarray) -> int:
        """Choose the next link to transition to based on link_transition (current link, destination node).

        Args:
            link_transition (LinkTransition): The current link transition object.
            params (np.ndarray): Model parameters.

        Returns:
            int: The ID of the next link to transition to.
        """
        probabilities = self.calculate_transition_probability(link_transition, params)
        
        rnd = np.random.rand()
        cumulative_prob = 0.0
        for link_id, prob in probabilities.items():
            cumulative_prob += prob
            if rnd < cumulative_prob:
                return link_id
        raise ValueError("No valid transition found.")


from definition import LinkTransition