from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from davis_nl.__main__ import null_log_likelihood, prepare, probabilities


class NestedLogitTest(unittest.TestCase):
    def request(self, path: Path) -> dict:
        return {
            "inputs": {
                "choice_data": {
                    "source": {},
                    "resolved": {"path": str(path), "media_type": "text/csv"},
                }
            },
            "config": {
                "roles": {
                    "case_id": "case",
                    "alternative_id": "alternative",
                    "chosen": "chosen",
                },
                "terms": [{"parameter": "constant", "constant": 1}],
                "nests": [
                    {
                        "name": "motorized",
                        "alternatives": ["train", "car"],
                        "scale_mu_d": {"fixed": 1.0},
                    },
                    {
                        "name": "active",
                        "alternatives": ["walk"],
                        "scale_mu_d": {"fixed": 1.0},
                    },
                ],
            },
        }

    def test_probabilities_sum_to_one_for_each_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "choice.csv"
            path.write_text(
                "case,alternative,chosen\n1,train,1\n1,car,0\n1,walk,0\n2,train,0\n2,car,1\n2,walk,0\n",
                encoding="utf-8",
            )
            data = prepare(self.request(path))

            predicted = probabilities(np.array([0.0]), data)

            for indices in data.groups:
                self.assertAlmostEqual(float(np.sum(predicted[indices])), 1.0)
            np.testing.assert_allclose(predicted, np.full(6, 1.0 / 3.0))
            self.assertAlmostEqual(null_log_likelihood(data), -2.0 * np.log(3.0))

    def test_nests_must_partition_alternatives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "choice.csv"
            path.write_text(
                "case,alternative,chosen\n1,train,1\n1,car,0\n1,walk,0\n",
                encoding="utf-8",
            )
            request = self.request(path)
            request["config"]["nests"][1]["alternatives"] = ["train", "walk"]

            with self.assertRaisesRegex(ValueError, "multiple nests"):
                prepare(request)

    def test_pdf_upper_scale_mu_d_formula(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "choice.csv"
            path.write_text(
                "case,alternative,chosen,x\n1,train,1,1\n1,car,0,0\n1,walk,0,0\n",
                encoding="utf-8",
            )
            request = self.request(path)
            request["config"]["terms"] = [{"parameter": "beta", "column": "x"}]
            request["config"]["nests"][0]["scale_mu_d"] = {"fixed": 0.5}
            current = probabilities(np.array([1.0]), prepare(request))

            inclusive_value = np.log(np.exp(1.0) + 1.0)
            nest_probability = np.exp(0.5 * inclusive_value) / (
                np.exp(0.5 * inclusive_value) + 1.0
            )
            expected_train = nest_probability * np.exp(1.0) / (np.exp(1.0) + 1.0)
            self.assertAlmostEqual(float(current[0]), float(expected_train))


if __name__ == "__main__":
    unittest.main()
