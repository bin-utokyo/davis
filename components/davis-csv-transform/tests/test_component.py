from __future__ import annotations

import unittest

from davis_csv_transform.__main__ import apply_calculations


class CalculationTest(unittest.TestCase):
    def test_creates_linear_combination_and_supports_chaining(self) -> None:
        rows = [{"time": "10", "cost": "2"}, {"time": "5", "cost": "3"}]
        fields = apply_calculations(
            ["time", "cost"],
            rows,
            {
                "calculations": [
                    {
                        "output": "generalized_cost",
                        "operation": "linear_combination",
                        "terms": [
                            {"column": "time"},
                            {"column": "cost", "coefficient": 2},
                        ],
                    },
                    {
                        "output": "centered_cost",
                        "operation": "linear_combination",
                        "terms": [
                            {"column": "generalized_cost"},
                            {"constant": -10},
                        ],
                    },
                ]
            },
        )

        self.assertEqual(fields, ["time", "cost", "generalized_cost", "centered_cost"])
        self.assertEqual(rows[0]["generalized_cost"], "14")
        self.assertEqual(rows[0]["centered_cost"], "4")


if __name__ == "__main__":
    unittest.main()
