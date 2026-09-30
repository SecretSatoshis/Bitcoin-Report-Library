"""Write release_manifest.json: the report date plus the hash and size of every published file."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

RELEASE_MANIFEST_NAME = 'release_manifest.json'
RELEASE_MANIFEST_VERSION = 1


def write_release_manifest(output_dir, report_date, filenames=None, model_parameters=None):
    """Write the manifest after every file is written, and return it.

    `filenames` lists the files this run published, so a leftover file is never listed;
    without it, every CSV in the directory is. `model_parameters` holds the fitted model
    coefficients. The file is replaced atomically.
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
    if model_parameters:
        manifest['model_parameters'] = model_parameters
    target = output / RELEASE_MANIFEST_NAME
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(target)
    return manifest
