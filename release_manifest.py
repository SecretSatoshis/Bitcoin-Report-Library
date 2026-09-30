"""Write release_manifest.json: the report date plus the hash and size of every published file."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

RELEASE_MANIFEST_NAME = 'release_manifest.json'
RELEASE_MANIFEST_VERSION = 1


def write_release_manifest(output_dir, report_date, filenames=None):
    """Write the immutable metadata contract shared by downstream consumers.

    The manifest is created only after every CSV export has completed. `filenames` lists
    exactly the files this run published (main.py passes it), so a leftover file in the
    directory is never mistaken for part of the release; without it, every CSV present is
    listed. Each record is small and JSON-friendly so browser and Python consumers can
    validate the same release without importing the producer's code.
    """
    output = Path(output_dir)
    if filenames is None:
        paths = [p for p in output.glob('*.csv*') if p.name != RELEASE_MANIFEST_NAME]
    else:
        paths = [output / name for name in filenames]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Release files were not written: {', '.join(missing)}")
    files = {}
    for path in sorted(paths):
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
