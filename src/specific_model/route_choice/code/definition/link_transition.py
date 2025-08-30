import os
import sys
from dataclasses import dataclass

from typing import Optional
import numpy as np
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from network import Network

__all__ = ["LinkTransition"]

@dataclass
class LinkTransition:
    trip_id: int
    link_id: int
    next_link_id: Optional[int]
    destination_node_id: int
    down_link_ids: list[int]
    model: "RouteChoiceModel"

    def calculate_log_likelihood(self, params: np.ndarray) -> float:
        """
        Calculate the log likelihood of the link transition given the network and model parameters.

        Args:
            network (Network): The network object containing link and node information.
            params (np.ndarray): Model parameters corresponding to link attributes.

        Returns:
            float: The log likelihood of the link transition.
        """
        if not self.model.is_valid(params):
            raise ValueError("Invalid link transition or parameters.")
        if self.next_link_id is None:
            raise ValueError("Next link ID is not specified for the link transition.")

        probabilities = self.model.calculate_transition_probability(self, params)
        return float(np.log(np.clip(probabilities[int(self.next_link_id)], 1e-10, None)) if probabilities[int(self.next_link_id)] > 0 else 0)

    def choose_next_link(self, params: np.ndarray) -> int:
        """
        Choose the next link in the route based on the transition probabilities.

        Args:
            params (np.ndarray): Model parameters corresponding to link attributes.

        Returns:
            int: The ID of the next link to transition to.
        """
        if not self.model.is_valid(params):
            raise ValueError("Invalid model parameters.")

        return self.model.choose_transition(self, params)
    

    @staticmethod
    def from_dict(data: dict, network: "Network", model: "RouteChoiceModel") -> Optional["LinkTransition"]:
        """Create a LinkTransition instance from a dictionary.

        Args:
            data (dict): The input dictionary.
            network (Network): The network object.
            model (RouteChoiceModel): The route choice model.

        Returns:
            Optional[LinkTransition]: The created LinkTransition instance.
        """
        if data["LinkID"] not in network.link_id2idx or data["DestinationNodeID"] not in network.node_id2idx:
            return None
        
        row_idx = network.link_id2idx[data["LinkID"]]
        row = network.link_adj_matrix.getrow(row_idx)
        down_link_idxs = row.indices  # 非ゼロ要素の列インデックス
        down_link_ids = [network.link_list[i] for i in down_link_idxs]

        if "NextLinkID" in data:
            if data["NextLinkID"] not in network.link_id2idx:
                return None
            if data["NextLinkID"] not in down_link_ids:
                return None

        return LinkTransition(
            trip_id=data["TripID"],
            link_id=data["LinkID"],
            next_link_id=data.get("NextLinkID"),
            destination_node_id=data["DestinationNodeID"],
            down_link_ids=down_link_ids,
            model=model
        )


# 遅延インポート
from abc_rc import RouteChoiceModel