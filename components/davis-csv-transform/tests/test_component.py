from __future__ import annotations

import unittest

from davis_csv_transform.__main__ import apply_calculations, join_tables


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


class JoinTest(unittest.TestCase):
    def test_many_to_one_join_preserves_keys_and_imports_renamed_columns(self) -> None:
        fields, rows, summary = join_tables(
            ["person_id", "zone_id"],
            [
                {"person_id": "001", "zone_id": "01"},
                {"person_id": "002", "zone_id": "02"},
            ],
            ["zone", "population", "area"],
            [
                {"zone": "01", "population": "1000", "area": "4"},
                {"zone": "02", "population": "2000", "area": "5"},
            ],
            {
                "input": "zones",
                "left_on": "zone_id",
                "right_on": "zone",
                "columns": {"zone_population": "population", "zone_area": "area"},
            },
        )

        self.assertEqual(
            fields, ["person_id", "zone_id", "zone_population", "zone_area"]
        )
        self.assertEqual(rows[0]["person_id"], "001")
        self.assertEqual(rows[0]["zone_population"], "1000")
        self.assertEqual(summary["unmatched_rows"], 0)

    def test_duplicate_lookup_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            join_tables(
                ["id"],
                [{"id": "1"}],
                ["id", "value"],
                [{"id": "1", "value": "a"}, {"id": "1", "value": "b"}],
                {
                    "input": "lookup",
                    "left_on": "id",
                    "right_on": "id",
                    "columns": {"value": "value"},
                },
            )

    def test_unmatched_rows_require_explicit_permission(self) -> None:
        specification = {
            "input": "lookup",
            "left_on": "id",
            "right_on": "id",
            "columns": {"value": "value"},
        }
        with self.assertRaisesRegex(ValueError, "no row"):
            join_tables(
                ["id"],
                [{"id": "missing"}],
                ["id", "value"],
                [{"id": "present", "value": "a"}],
                specification,
            )

        specification["allow_unmatched"] = True
        fields, rows, summary = join_tables(
            ["id"],
            [{"id": "missing"}],
            ["id", "value"],
            [{"id": "present", "value": "a"}],
            specification,
        )
        self.assertEqual(fields, ["id", "value"])
        self.assertEqual(rows, [{"id": "missing", "value": ""}])
        self.assertEqual(summary["unmatched_rows"], 1)


if __name__ == "__main__":
    unittest.main()
