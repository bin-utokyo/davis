import os
from logging import getLogger, StreamHandler, Formatter

import numpy as np

# logger
loglevel = os.environ.get("LOGLEVEL", "WARNING").upper()
log_format = "[%(asctime)s] %(levelname)s:%(filename)s %(lineno)d:%(message)s"
logger = getLogger(__name__)
formatter = Formatter(log_format)
handler = StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(loglevel)

__all__ = ["Viterbi"]

class Viterbi:
    def __init__(self, num_states: int):
        """
        Initialize the Viterbi algorithm with the number of states.
        
        Args:
            num_states (int): Number of states in the Markov model.
        """
        self.num_states = num_states

        self.T: dict[int, tuple[float, list[int], float]] = dict()  # {start_state: (total_path_prob, path, path_prob)}

    def initialize(self, start_prob: list | np.ndarray) -> None:
        """
        Initialize the Viterbi algorithm with the starting probabilities.
        
        Args:
            start_prob (list | np.ndarray): Starting probabilities for each state (num_states).

        Raises:
            Exception: If the shape of start_prob does not match the number of states.
        """
        # start_prob: (num_states)
        if len(start_prob) != self.num_states:
            raise Exception("start_prob shape error")
        self.T = {i: (start_prob[i], [i], start_prob[i]) for i in range(self.num_states)}

    def forward(self, transition_prob: list | np.ndarray, emission_prob: list | np.ndarray) -> None:
        """
        Perform the forward step of the Viterbi algorithm using transition and emission probabilities.
        
        Args:
            transition_prob (list | np.ndarray): Transition probabilities between states (num_states, num_states).
            emission_prob (list | np.ndarray): Emission probabilities for each state (num_states).

        Raises:
            Exception: If the shape of transition_prob or emission_prob does not match the number of states.
        """
        # transition_prob: (num_states, num_states)
        # emission_prob: (num_states)
        if len(transition_prob) != self.num_states or len(transition_prob[0]) != self.num_states:
            raise Exception("transition_prob shape error")
        if len(emission_prob) != self.num_states:
            raise Exception("emission_prob shape error")

        T = dict()
        for j in range(self.num_states):  # next state
            total_path_prob = 0.0
            argmax = [j]
            arg = 0.0
            for i in range(self.num_states):  # current state
                if self.T[i][1] is None:
                    raise Exception("T[i][1] is None")
                p_ij = transition_prob[i][j] * emission_prob[j]
                if p_ij < 0.0:
                    raise Exception("Negative probability encountered")
                total_path_prob_tmp = self.T[i][0] * p_ij
                path_prob_tmp = self.T[i][2] * p_ij
                total_path_prob += total_path_prob_tmp
                # save most probable path
                if path_prob_tmp > arg:
                    argmax = self.T[i][1] + [j]
                    arg = path_prob_tmp
            T[j] = (total_path_prob, argmax, arg)

        self.T = T

    def get_path(self) -> tuple[list[int], float]:
        """
        Get the most probable path and its probability from the Viterbi algorithm.
        
        Returns:
            tuple[list[int], float]: Most probable path as a list of state indices and its probability.

        Raises:
            Exception: If no path is found.
        """
        # return: (path, path_prob)
        argmax = None
        arg = 0.0
        for i in range(self.num_states):  # final state
            if self.T[i][2] > arg:
                argmax = self.T[i][1]
                arg = self.T[i][2]
        if argmax is None:
            logger.warning("No path found")
            return [], 0.0
        return argmax, arg