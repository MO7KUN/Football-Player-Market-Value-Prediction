"""Download public Transfermarkt-derived CSVs with provenance and SHA256 hashes."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE = 'https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data'
TABLES = ['players', 'appearances', 'player_valuations', 'transfers']

def download(out=Path('data/raw'), refresh=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out.parent / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for table in TABLES:
        expected = manifest.get(table, {}).get('sha256')
        path = out / f'{table}.csv.gz'
        url = f'{BASE}/{path.name}'
        if refresh or not path.exists():
            print(f'Downloading {table}', flush=True)
            temp = path.with_suffix('.part')
            with requests.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with temp.open('wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            temp.replace(path)
            manifest[table] = {'url': url, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat()}
        manifest.setdefault(table, {'url': url, 'downloaded_at_utc': 'unknown (pre-existing file)'})
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if expected and digest != expected and not refresh:
            raise ValueError(f'{table}: snapshot changed. Use --refresh deliberately to update the study; otherwise obtain the original hash-matched file.')
        manifest[table].update(bytes=path.stat().st_size, sha256=digest)
        print(f'{table}: {path.stat().st_size:,} bytes', flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('data/raw'))
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    download(args.out, args.refresh)
