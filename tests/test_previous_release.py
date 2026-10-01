"""Reading the last release (previous_release.py)."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from previous_release import previous_release_files


class PreviousReleaseTests(unittest.TestCase):
    def release(self, directory, content=b"a,b\n1,2\n", listed=b"a,b\n1,2\n"):
        Path(directory, "etf_daily.csv").write_bytes(content)
        manifest = {"files": {"etf_daily.csv": {"sha256": hashlib.sha256(listed).hexdigest()}}}
        Path(directory, "release_manifest.json").write_text(json.dumps(manifest))

    def test_listed_files_are_read_from_the_local_copy(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PREVIOUS_RELEASE_DIR": directory}):
            self.release(directory)
            files = previous_release_files(["etf_daily.csv", "etf_snapshots.csv"])
        self.assertEqual(files, {"etf_daily.csv": b"a,b\n1,2\n"})

    def test_a_file_that_disagrees_with_its_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PREVIOUS_RELEASE_DIR": directory}):
            self.release(directory, content=b"changed\n")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                previous_release_files(["etf_daily.csv"])


if __name__ == "__main__":
    unittest.main()
