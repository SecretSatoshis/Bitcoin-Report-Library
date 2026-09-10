"""Bind the chart consumer's inputs to one Report Library release."""
import hashlib
import json
from pathlib import Path

CHART_INPUT_FILES = (
    'master_metrics_data.csv.gz', 'drawdown_data.csv', 'cycle_low_data.csv',
    'halving_data.csv', 'report_ohlc_summary.csv',
)


def write_chart_input_manifest(output_dir, report_date):
    output = Path(output_dir)
    manifest = {
        'version': 1,
        'report_date': str(report_date)[:10],
        'files': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                  for name in CHART_INPUT_FILES},
    }
    target = output / 'chart_input_manifest.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    temporary.replace(target)
    return manifest
