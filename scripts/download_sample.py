"""Download paired SpaceNet 2 Vegas RGB tiles and building labels anonymously."""
import argparse
import json
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = 'https://spacenet-dataset.s3.amazonaws.com/'
PREFIX = 'spacenet/SN2_buildings/train/AOI_2_Vegas/'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def download(key, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    with urllib.request.urlopen(BASE + urllib.parse.quote(key), timeout=120) as response:
        with temporary.open('wb') as target:
            shutil.copyfileobj(response, target)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--download', action='store_true', help='Download; otherwise only list selected pairs')
    args = parser.parse_args()
    if not 1 <= args.count <= 1000:
        parser.error('--count must be between 1 and 1000')
    query = urllib.parse.urlencode({'list-type': 2, 'prefix': PREFIX + 'PS-RGB/', 'max-keys': args.count})
    with urllib.request.urlopen(BASE + '?' + query, timeout=60) as response:
        root = ET.fromstring(response.read())
    output = Path(__file__).resolve().parents[1] / 'data/raw/AOI_2_Vegas'
    records = []
    for item in root.findall('s:Contents', NS):
        key = item.findtext('s:Key', namespaces=NS)
        if not key.endswith('.tif'):
            continue
        label = key.replace('/PS-RGB/', '/geojson_buildings/').replace('_PS-RGB_', '_geojson_buildings_').replace('.tif', '.geojson')
        print(Path(key).name, flush=True)
        for source in (key, label):
            if args.download:
                download(source, output / source.removeprefix(PREFIX))
        if args.download:
            annotation = json.loads((output / label.removeprefix(PREFIX)).read_text())
            if annotation.get('type') != 'FeatureCollection':
                raise ValueError(f'Invalid GeoJSON: {label}')
        records.append({'image': key, 'label': label, 'image_bytes': int(item.findtext('s:Size', namespaces=NS))})
    if len(records) != args.count:
        raise RuntimeError(f'Expected {args.count} pairs, found {len(records)}')
    if args.download:
        (output / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'{len(records)} pairs; RGB total: {sum(r["image_bytes"] for r in records) / 1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
