"""Prepare experiment 02 and enforce a 250-metre gap between train and validation."""
import json
from pathlib import Path
import zipfile

import numpy as np
from osgeo import gdal, osr
from shapely.geometry import Polygon, mapping
from PIL import Image, ImageDraw
from prepare_coco import prepare

ROOT = Path(__file__).resolve().parents[1]


def spatial_partition(footprints, fraction=0.8, gap=250):
    """Choose a north-south dividing line by tile positions only, never labels."""
    cut = float(np.quantile([g.centroid.x for g in footprints.values()], fraction))
    train = {i for i, g in footprints.items() if g.bounds[2] <= cut-gap/2}
    val = {i for i, g in footprints.items() if g.bounds[0] >= cut+gap/2}
    excluded = set(footprints)-train-val
    assert train and val and not train & val
    minimum = min(footprints[t].distance(footprints[v]) for t in train for v in val)
    assert minimum >= gap-1e-6, 'Spatial leakage: gap too small'
    return train, val, excluded, {'cut_easting_m': cut, 'required_gap_m': gap, 'minimum_train_val_distance_m': minimum}


def main():
    raw = ROOT/'data/raw/vegas_experiment'
    out = ROOT/'data/processed/vegas_experiment'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((raw/'manifest.json').read_text())
    paths, excluded_coverage = [], []
    for item in manifest['records']:
        path = raw/'PS-RGB'/Path(item['image']).name
        ds = gdal.Open(str(path))
        array = ds.ReadAsArray()
        complete = np.any(array != 0, axis=0)
        masks_valid = all(np.all(ds.GetRasterBand(i).GetMaskBand().ReadAsArray() != 0) for i in range(1, 4))
        if complete.all() and masks_valid:
            paths.append(path)
        else:
            excluded_coverage.append({'source': path.name, 'zero_rgb_percent': float((~complete).mean()*100), 'gdal_masks_all_valid': bool(masks_valid)})
    print(f'Full coverage: {len(paths)}; excluded for missing coverage: {len(excluded_coverage)}', flush=True)
    prepare(ROOT, raw=raw, out=out, paths=paths, preview_name='experiment_coco_preview.jpg')
    coco = json.loads((out/'instances_all.json').read_text())
    reports = json.loads((out/'preparation_report.json').read_text())
    images = {im['id']: im for im in coco['images']}
    utm = osr.SpatialReference()
    utm.ImportFromEPSG(32611)
    utm.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    footprints = {}
    for r in reports:
        source = osr.SpatialReference()
        source.ImportFromWkt(r['crs_wkt'])
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(source, utm)
        gt = r['crop_geotransform']
        im = images[r['image_id']]
        corners = []
        for x, y in [(0,0), (im['width'],0), (im['width'],im['height']), (0,im['height'])]:
            xx, yy, _ = transform.TransformPoint(gt[0]+gt[1]*x+gt[2]*y, gt[3]+gt[4]*x+gt[5]*y)
            corners.append((xx, yy))
        footprints[im['id']] = Polygon(corners)
    train, val, buffer_ids, stats = spatial_partition(footprints)
    assert len(train) >= 80 and len(val) >= 15, 'Insufficient spatially separated data'
    counts = {}
    for split, ids in [('train', train), ('val', val)]:
        subset = {'info': {'description': 'SpaceNet 2 Vegas experiment 02; spatial holdout'}, 'licenses': [], 'categories': coco['categories'],
                  'images': [im for im in coco['images'] if im['id'] in ids],
                  'annotations': [a for a in coco['annotations'] if a['image_id'] in ids]}
        assert subset['annotations'], f'No buildings in {split}'
        (out/f'instances_{split}.json').write_text(json.dumps(subset)+'\n')
        positive = {a['image_id'] for a in subset['annotations']}
        counts[split] = {'images': len(ids), 'buildings': len(subset['annotations']), 'negative_images': len(ids-positive)}
    summary = {**stats, 'projection': 'EPSG:32611', 'sampled_pairs': len(manifest['records']), 'counts': counts,
               'excluded_coverage': excluded_coverage, 'excluded_spatial_buffer_image_ids': sorted(buffer_ids),
               'train_image_ids': sorted(train), 'val_image_ids': sorted(val),
               'note': 'Within-Vegas spatial validation, not an independent city or final test set. Zero RGB treated as missing coverage for this dataset.'}
    (out/'split_report.json').write_text(json.dumps(summary, indent=2)+'\n')
    features = []
    for i, geom in footprints.items():
        split = 'train' if i in train else 'val' if i in val else 'buffer'
        features.append({'type': 'Feature', 'properties': {'image_id': i, 'file_name': images[i]['file_name'], 'split': split}, 'geometry': mapping(geom)})
    # Projected GeoJSON explicitly declares its CRS for inspection in QGIS.
    (out/'split_footprints_utm.geojson').write_text(json.dumps({'type':'FeatureCollection', 'crs': {'type':'name','properties':{'name':'EPSG:32611'}}, 'features':features}))
    minx = min(g.bounds[0] for g in footprints.values()); maxx = max(g.bounds[2] for g in footprints.values())
    miny = min(g.bounds[1] for g in footprints.values()); maxy = max(g.bounds[3] for g in footprints.values())
    canvas = Image.new('RGB', (1000,850), 'white'); draw=ImageDraw.Draw(canvas)
    scale = min(900/(maxx-minx), 700/(maxy-miny))
    def pixel(x,y): return (50+(x-minx)*scale, 800-(y-miny)*scale)
    colors={'train':'#3178b5','val':'#e77c25','buffer':'#bbbbbb'}
    for feature in features:
        i=feature['properties']['image_id']; split=feature['properties']['split']
        draw.polygon([pixel(x,y) for x,y in footprints[i].exterior.coords], fill=colors[split], outline='#333333')
    draw.text((30,20), f'Train (blue): {len(train)} | Validation (orange): {len(val)} | Buffer (gray): {len(buffer_ids)}', fill='black')
    draw.text((30,45), f'Minimum train-validation distance: {stats["minimum_train_val_distance_m"]:.0f} m | UTM 11N', fill='black')
    canvas.save(ROOT/'outputs/experiment_spatial_split.png')
    bundle=ROOT/'outputs/spacenet_experiment_02.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for filename in ['instances_train.json','instances_val.json','preparation_report.json','split_report.json','split_footprints_utm.geojson']:
            archive.write(out/filename, filename)
        archive.write(raw/'manifest.json','source_manifest.json')
        for i in sorted(train | val):
            name=images[i]['file_name']; archive.write(out/'images'/name,'images/'+name)
        archive.writestr('SOURCE.txt','SpaceNet 2: https://spacenet.ai/spacenet-buildings-dataset-v2/\nCC BY-SA 4.0. Van Etten et al. (2018), arXiv:1807.01232.\nDerived RGB PNG and COCO instance masks; see preparation_report.json.\n')
    with zipfile.ZipFile(bundle) as archive: assert archive.testzip() is None
    print(json.dumps({'counts':counts,'minimum_distance_m':stats['minimum_train_val_distance_m'],'zip_mib':bundle.stat().st_size/1024**2}, indent=2), flush=True)


if __name__ == '__main__':
    main()
