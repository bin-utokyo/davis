import sys
from pathlib import Path
from abc import ABC, abstractmethod

from typing import Any, TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from scipy.optimize import minimize

sys.path.append(str(Path(__file__).resolve().parent.parent))
from abc_rc import RouteChoiceModel
from definition import BLE, BLENetwork, LinkTransition
from algorithm import BLEHybridMapmatching

class EMABC(ABC):
    @abstractmethod
    def expectation_step(self, data: Any) -> Any:
        """
        E-step of EM algorithm. q(t+1) = argmax_q ELBO(q, \\theta)

        Args:
            data (Any): Observation data to calculate ELBO.

        Returns:
            Any: Any return values.
        """
        pass

    @abstractmethod
    def maximization_step(self, data: Any) -> Any:
        """
        M-step of EM algorithm. \\theta(t+1) = argmax_\\theta ELBO(q, \\theta)

        Args:
            data (Any): Sampled hidden variables from E-step.

        Returns:
            Any: Any return values.
        """
        pass

    @abstractmethod
    def fit(self, data: Any, max_iterations: int = 100, tol: float = 1e-4, *args: Any, **kwargs: Any) -> Any:
        """
        Fit the model to the data using the EM algorithm.

        Args:
            data (Any): Observation data to fit the model.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-4.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Any: Any return values.
        """
        pass


class EMBLERouteChoice(EMABC):
    def __init__(self, network: BLENetwork, model: RouteChoiceModel, mapmatching: BLEHybridMapmatching):
        """
        Initialize the EMRouteChoice with a route choice model.

        Args:
            network (BLENetwork): The BLE network.
            model (RouteChoiceModel): The route choice model to be used in the EM algorithm.
        """
        self.network = network
        self.model = model
        self.mapmatching = mapmatching

        self.param_history = [np.zeros(self.model.get_param_size(), dtype=np.float32)]
        self.transition_list: list[LinkTransition] = []

    def expectation_step(self, data: Any) -> pd.DataFrame:
        """
        E-step of EM algorithm for BLE route choice model. Performs map matching to compute expected sufficient statistics.

        Args:
            data (Any): The BLE data for map matching.

        Returns:
            pd.DataFrame: DataFrame of expected sufficient statistics for each trip.
        """
        if not isinstance(data, BLE):
            raise ValueError("Input data must be an instance of BLE.")
        result = self.mapmatching.match(data)
        return result

    def maximization_step(self, data: Any) -> None:
        """
        M-step of EM algorithm for BLE route choice model. Updates model parameters based on expected sufficient statistics.

        Args:
            data (Any): List of link transitions from E-step.
        """
        if not isinstance(data, list) or not all(isinstance(t, LinkTransition) for t in data):
            raise ValueError("Input data must be a list of LinkTransition instances.")
        
        compute_minus_ll = lambda params: self.compute_minus_ll(params, data)
        res = minimize(compute_minus_ll, self.param_history[-1], method="Nelder-Mead")
        self.param_history.append(res.x)

    def fit(self, data: Any, max_iterations: int = 100, tol: float = 1e-4, *args: Any, **kwargs: Any) -> Any:
        """
        Fit the BLE route choice model to the data using the EM algorithm.

        Args:
            data (Any): Observation data to fit the model. BLE instance.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-4.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Any: String representation of the parameter history.
        """
        # initialize parameter history
        self.param_history = [np.zeros(self.model.get_param_size(), dtype=np.float32)]
        param_change = float("inf")
        transition_list = []

        for iteration in range(max_iterations):
            print(f"EM Iteration {iteration + 1}")
            print("  E-step start")
            mapmatching_result = self.expectation_step(data)

            transition_list = [LinkTransition.from_dict(row, self.network, self.model) for row in mapmatching_result.to_dict(orient="records")]
            # Remove None values
            transition_list = [t for t in transition_list if t is not None]
            self.transition_list = transition_list
            print(f"  M-step start with {len(transition_list)} link transitions")
            self.maximization_step(transition_list)

            param_change = np.linalg.norm(self.param_history[-1] - self.param_history[-2])
            if param_change < tol:
                print("EM algorithm converged.")
                break

            # Update mapmatching model with new parameters
            self.mapmatching.set_model(self.model, self.param_history[-1])
            print("----------------")

        if param_change >= tol:
            print("EM algorithm did not converge within the maximum number of iterations.")

        print("----------------")
        if len(transition_list) > 0 and all([t is not None for t in transition_list]):
            result_str = self.get_result_string(self.network, self.model, cast(list[LinkTransition], transition_list), self.param_history)
            print(result_str)
            return result_str
        return ""


    @staticmethod
    def compute_minus_ll(params: np.ndarray, transition_list: list[LinkTransition]) -> float:
        """
        Compute the negative log-likelihood for the given parameters and link transitions.

        Args:
            params (np.ndarray): Model parameters.
            transition_list (list[LinkTransition]): List of link transitions.
        Returns:
            float: The negative log-likelihood.
        """
        ll = 0.0
        for transition in transition_list:
            ll += transition.calculate_log_likelihood(params)
        return -ll
    
    @staticmethod
    def compute_hessian(params: np.ndarray, transition_list: list[LinkTransition]) -> np.ndarray:
        """
        Compute the Hessian matrix of the negative log-likelihood using numerical differentiation.

        Args:
            params (np.ndarray): Model parameters.
            transition_list (list[LinkTransition]): List of link transitions.

        Returns:
            np.ndarray: The Hessian matrix.
        """
        h = 10 ** -4  # 数値微分用の微小量
        n = len(params)
        res = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                params_ijp = params.copy()
                params_ijp[i] += h
                params_ijp[j] += h

                params_ijm = params.copy()
                params_ijm[i] += h
                params_ijm[j] -= h

                params_jim = params.copy()
                params_jim[i] -= h
                params_jim[j] += h

                params_jjm = params.copy()
                params_jjm[i] -= h
                params_jjm[j] -= h

                f_ijp = -EMBLERouteChoice.compute_minus_ll(params_ijp, transition_list)
                f_ijm = -EMBLERouteChoice.compute_minus_ll(params_ijm, transition_list)
                f_jim = -EMBLERouteChoice.compute_minus_ll(params_jim, transition_list)
                f_jjm = -EMBLERouteChoice.compute_minus_ll(params_jjm, transition_list)

                res[i, j] = (f_ijp - f_ijm - f_jim + f_jjm) / (4 * h * h)
        return res
    
    @staticmethod
    def get_result_string(network: BLENetwork, model: RouteChoiceModel, transition_list: list[LinkTransition], param_history: list[np.ndarray]) -> str:
        """
        Get a string representation of the parameter history.

        Args:
            param_history (list[np.ndarray]): List of parameter arrays.

        Returns:
            str: String representation of the parameter history.
        """
        t_val = param_history[-1] / np.sqrt(-np.diag(np.linalg.pinv(EMBLERouteChoice.compute_hessian(param_history[-1], transition_list))))
        LL0 = -EMBLERouteChoice.compute_minus_ll(np.zeros(model.get_param_size()), transition_list)
        LL = -EMBLERouteChoice.compute_minus_ll(param_history[-1], transition_list)
        rho2 = 1 - LL / LL0
        adj_rho2 = 1 - (LL - len(param_history[-1])) / LL0
        aic = -2 * LL + 2 * len(param_history[-1])

        result_str = f"""
        sample number = {len(transition_list)}
            variables = [{', '.join(map(str, network.f_name))}]
            parameter = [{', '.join(map(str, param_history[-1]))}]
                t value = [{', '.join(map(str, t_val))}]
                    L0 = {LL0}
                    LL = {LL}
                    rho2 = {rho2}
        adjusted rho2 = {adj_rho2}
                    AIC = {aic}
                discount = {model.get_beta(param_history[-1]) if hasattr(model, 'get_beta') else 'N/A'}
        """
        return result_str

# 遅延インポート
if TYPE_CHECKING:
    from mapmatching import HybridMapmatching

