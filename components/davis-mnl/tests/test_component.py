from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
