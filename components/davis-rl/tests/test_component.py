from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from davis_rl.__main__ import link_probabilities, prepare


class RecursiveLogitTest(unittest.TestCase):
    def request(self, network: Path, observations: Path) -> dict:
        return {
            "inputs": {
                "network": {
                    "resolved": {"path": str(network), "media_type": "text/csv"}
                },
                "observations": {
                    "resolved": {
                        "path": str(observations),
                        "media_type": "text/csv",
                    }
                },
            },
            "config": {
                "network_roles": {
                    "link_id": "link",
                    "from_node": "from",
                    "to_node": "to",
                },
                "observation_roles": {
                    "trip_id": "trip",
                    "step": "step",
                    "link_id": "link",
                    "destination": "destination",
                },
                "terms": [{"parameter": "beta", "column": "cost"}],
                "parameters": {"beta": {"initial": -1.0}},
            },
        }

    def test_outgoing_probabilities_sum_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            network = root / "network.csv"
            observations = root / "observations.csv"
            network.write_text(
                "link,from,to,cost\na,O,D,1\nb,O,D,1\n", encoding="utf-8"
            )
            observations.write_text(
                "trip,step,link,destination\n1,1,a,D\n", encoding="utf-8"
            )
            data = prepare(self.request(network, observations))

            probabilities = link_probabilities(np.array([-1.0]), "D", data)

            np.testing.assert_allclose(probabilities, [0.5, 0.5])

    def test_observed_path_must_be_connected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            network = root / "network.csv"
            observations = root / "observations.csv"
            network.write_text(
                "link,from,to,cost\na,O,A,1\nb,B,D,1\n", encoding="utf-8"
            )
            observations.write_text(
                "trip,step,link,destination\n1,1,a,D\n1,2,b,D\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "disconnected"):
                prepare(self.request(network, observations))

    def test_rejects_a_cycle_without_a_finite_value_function(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            network = root / "network.csv"
            observations = root / "observations.csv"
            network.write_text(
                "link,from,to,cost\noa,O,A,0\nao,A,O,0\nad,A,D,0\n",
                encoding="utf-8",
            )
            observations.write_text(
                "trip,step,link,destination\n1,1,oa,D\n1,2,ad,D\n",
                encoding="utf-8",
            )
            data = prepare(self.request(network, observations))

            with self.assertRaisesRegex(ValueError, "no unique solution"):
                link_probabilities(np.array([0.0]), "D", data)

    def test_trip_stops_when_it_first_reaches_its_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            network = root / "network.csv"
            observations = root / "observations.csv"
            network.write_text(
                "link,from,to,cost\nod,O,D,1\ndx,D,X,1\nxd,X,D,1\n",
                encoding="utf-8",
            )
            observations.write_text(
                "trip,step,link,destination\n1,1,od,D\n1,2,dx,D\n1,3,xd,D\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "before its final link"):
                prepare(self.request(network, observations))


if __name__ == "__main__":
    unittest.main()
