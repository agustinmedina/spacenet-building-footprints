"""Select reproducible random RGB/label pairs across the entire Vegas catalog."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from download_sample import BASE, PREFIX, NS, download

ROOT = Path(__file__).resolve().parents[1]


def catalog():
    records, token = [], None
    while True:
        args = {'list-type': 2, 'prefix': PREFIX+'PS-RGB/', 'max-keys': 1000}
        if token:
            args['continuation-token'] = token
        with urllib.request.urlopen(BASE+'?'+urllib.parse.urlencode(args), timeout=60) as response:
            tree = ET.fromstring(response.read())
        for item in tree.findall('s:Contents', NS):
            key = item.findtext('s:Key', namespaces=NS)
            if key.endswith('.tif'):
                records.append({'image': key, 'image_bytes': int(item.findtext('s:Size', namespaces=NS)),
                                'label': key.replace('/PS-RGB/', '/geojson_buildings/').replace('_PS-RGB_', '_geojson_buildings_').replace('.tif', '.geojson')})
        token = tree.findtext('s:NextContinuationToken', namespaces=NS)
        if not token:
            return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    out = ROOT/'data/raw/vegas_experiment'
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'manifest.json'
    frozen = ROOT/'data/manifests/vegas_experiment.json'
    if not manifest_path.exists() and frozen.exists():
        manifest_path.write_text(frozen.read_text())
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        available = catalog()
        # Hash ordering is stable across Python versions and independent of S3 order.
        selected = sorted(available, key=lambda r: hashlib.sha256(('42:'+r['image']).encode()).hexdigest())[:200]
        assert len(selected) == 200
        manifest = {'seed': 42, 'selection': 'SHA256(42:key), first 200 of complete RGB catalog', 'catalog_count': len(available), 'records': selected}
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    records = manifest['records']
    print(f"Catalog: {manifest['catalog_count']} images; selected: {len(records)}; RGB: {sum(r['image_bytes'] for r in records)/1024**2:.1f} MiB", flush=True)
    if not args.download:
        return

    def pair(record):
        for field in ('image', 'label'):
            key = record[field]
            path = out/key.removeprefix(PREFIX)
            if path.exists():
                if field == 'image' and path.stat().st_size == record['image_bytes']:
                    continue
                if field == 'label':
                    try:
                        if json.loads(path.read_text()).get('type') == 'FeatureCollection':
                            continue
                    except (ValueError, OSError):
                        pass
            for attempt in range(3):
                try:
                    download(key, path)
                    if field == 'image':
                        assert path.stat().st_size == record['image_bytes']
                    else:
                        assert json.loads(path.read_text()).get('type') == 'FeatureCollection'
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt+1)
        return record['image']

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, _ in enumerate(pool.map(pair, records), 1):
            if index % 20 == 0:
                print(f'{index}/{len(records)} pairs checked', flush=True)


if __name__ == '__main__':
    main()
