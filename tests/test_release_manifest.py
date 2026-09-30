"""Producer contract for the shared release manifest."""
import json
import tempfile
import unittest
from pathlib import Path
from release_manifest import write_release_manifest


class ReleaseManifestTests(unittest.TestCase):
    def test_release_manifest_records_every_csv_and_release_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'alpha.csv').write_text('a,b\n1,2\n')
            (output / 'beta.csv.gz').write_bytes(b'compressed-placeholder')
            manifest = write_release_manifest(output, '2026-09-09')
            self.assertEqual(manifest['schema_version'], 1)
            self.assertEqual(manifest['release_id'], '2026-09-09')
            self.assertEqual(manifest['report_date'], '2026-09-09')
            self.assertEqual(set(manifest['files']), {'alpha.csv', 'beta.csv.gz'})
            self.assertEqual(
                json.loads((output / 'release_manifest.json').read_text()), manifest
            )

    def test_an_explicit_file_list_excludes_leftovers_and_requires_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'alpha.csv').write_text('a\n1\n')
            (output / 'retired.csv').write_text('old\n')
            manifest = write_release_manifest(output, '2026-09-09', ['alpha.csv'])
            self.assertEqual(set(manifest['files']), {'alpha.csv'})
            with self.assertRaisesRegex(FileNotFoundError, 'missing.csv'):
                write_release_manifest(output, '2026-09-09', ['alpha.csv', 'missing.csv'])


if __name__ == '__main__':
    unittest.main()
