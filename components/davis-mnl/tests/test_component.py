from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from davis_mnl.__main__ import read_csv


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


if __name__ == "__main__":
    unittest.main()
