"""Bind the chart consumer's inputs to one Report Library release."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

RELEASE_MANIFEST_NAME = 'release_manifest.json'
RELEASE_MANIFEST_VERSION = 1


def write_release_manifest(output_dir, report_date):
    """Write the immutable metadata contract shared by downstream consumers.

    The manifest is created only after every CSV export has completed.  Each file
    record is intentionally small and JSON-friendly so browser and Python consumers
    can validate the same release without importing the producer's code.
    """
    output = Path(output_dir)
    files = {}
    for path in sorted(output.glob('*.csv*')):
        if path.name == RELEASE_MANIFEST_NAME:
            continue
        files[path.name] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size,
        }

    release_date = str(report_date)[:10]
    manifest = {
        'schema_version': RELEASE_MANIFEST_VERSION,
        'release_id': release_date,
        'report_date': release_date,
        'generated_at': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        'files': files,
    }
    target = output / RELEASE_MANIFEST_NAME
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(target)
    return manifest
