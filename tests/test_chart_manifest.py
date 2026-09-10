"""Producer contract for Chart Library's release manifest."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from chart_manifest import CHART_INPUT_FILES, write_chart_input_manifest


class ChartManifestTests(unittest.TestCase):
    def test_manifest_binds_every_input_and_updates_after_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in CHART_INPUT_FILES:
                (output/name).write_bytes(name.encode())
            manifest = write_chart_input_manifest(output, '2026-09-08')
            self.assertEqual(set(manifest['files']), set(CHART_INPUT_FILES))
            self.assertEqual(manifest['report_date'], '2026-09-08')
            self.assertEqual(json.loads((output/'chart_input_manifest.json').read_text()), manifest)
            for name in CHART_INPUT_FILES:
                self.assertEqual(manifest['files'][name], hashlib.sha256(name.encode()).hexdigest())
            (output/CHART_INPUT_FILES[0]).write_bytes(b'new release')
            updated = write_chart_input_manifest(output, '2026-09-09')
            self.assertNotEqual(updated['files'][CHART_INPUT_FILES[0]], manifest['files'][CHART_INPUT_FILES[0]])
            self.assertFalse((output/'chart_input_manifest.json.tmp').exists())

    def test_missing_input_does_not_replace_previous_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'chart_input_manifest.json'
            target.write_text('previous manifest')
            with self.assertRaises(FileNotFoundError):
                write_chart_input_manifest(directory, '2026-09-08')
            self.assertEqual(target.read_text(), 'previous manifest')
