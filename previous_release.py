"""Files from the last published release, each checked against that release's manifest.

Some outputs carry state from release to release (the ETF snapshots and history) or fall back to
the last published copy when a source fails (the annual series, the ETF tables). The release
workflow copies the committed release to PREVIOUS_RELEASE_DIR before rebuilding, so the files
are read locally; without it they are downloaded from the published site.
"""
import hashlib
import json
import os
from pathlib import Path

import requests

from data_definitions import API_TIMEOUT, RELEASE_BASE_URL

MANIFEST = "release_manifest.json"


def _read(name: str) -> bytes:
    directory = os.environ.get("PREVIOUS_RELEASE_DIR")
    if directory:
        return (Path(directory) / name).read_bytes()
    response = requests.get(f"{RELEASE_BASE_URL}/{name}", timeout=API_TIMEOUT * 4)
    response.raise_for_status()
    return response.content


def previous_release_files(names) -> dict[str, bytes]:
    """The named files of the last release that it lists; a file it does not list is left out.

    Raises when the release cannot be read or a file does not match its manifest.
    """
    records = json.loads(_read(MANIFEST))["files"]
    files = {}
    for name in names:
        if name not in records:
            continue
        content = _read(name)
        if hashlib.sha256(content).hexdigest() != records[name]["sha256"]:
            raise RuntimeError(f"the previous {name} does not match its manifest")
        files[name] = content
    return files
