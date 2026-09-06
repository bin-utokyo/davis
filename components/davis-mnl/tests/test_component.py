from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from davis_mnl.__main__ import prepare, read_csv


class CsvInputTest(unittest.TestCase):
    def test_auto_reads_cp932_and_preserves_leading_zero_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "choice.csv"
            path.write_bytes("人物ID,手段\n001,徒歩\n".encode("cp932"))

            frame = read_csv(path, None)

            self.assertEqual(frame.columns.tolist(), ["人物ID", "手段"])
            self.assertEqual(frame.loc[0, "人物ID"], "001")
            self.assertEqual(frame.loc[0, "手段"], "徒歩")

    def test_parquet_engine_is_available_and_preserves_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "choice.parquet"
            pd.DataFrame(
                {"person_id": ["001", "002"], "travel_time": [10.0, 20.0]}
            ).to_parquet(path, index=False)

            frame = pd.read_parquet(path)

            self.assertEqual(frame["person_id"].tolist(), ["001", "002"])
            self.assertEqual(frame["travel_time"].tolist(), [10.0, 20.0])

    def test_composite_case_key_and_chosen_alternative_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "choice.csv"
            pd.DataFrame(
                {
                    "person": ["1", "1", "1", "1", "2", "2"],
                    "year": ["2020", "2020", "2021", "2021", "2020", "2020"],
                    "alternative": ["a", "b", "a", "b", "a", "b"],
                    "target": ["b", "b", "a", "a", "b", "b"],
                    "cost": [1, 2, 2, 1, 1, 3],
                }
            ).to_csv(path, index=False)
            request = {
                "inputs": {
                    "choice_data": {
                        "resolved": {"path": str(path), "media_type": "text/csv"},
                        "source": {"kind": "local", "path": str(path)},
                    }
                },
                "config": {
                    "roles": {
                        "case_id": ["person", "year"],
                        "alternative_id": "alternative",
                        "chosen_alternative": "target",
                    },
                    "terms": [{"parameter": "beta_cost", "column": "cost"}],
                    "estimation": {"development_case_limit": 2},
                },
            }

            prepared = prepare(request)

            self.assertEqual(prepared.case_columns, ["person", "year"])
            self.assertEqual(len(prepared.groups), 2)
            self.assertEqual(len(prepared.frame), 4)
            self.assertEqual(int(prepared.chosen.sum()), 2)


if __name__ == "__main__":
    unittest.main()
