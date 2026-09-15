"""Prepare the local sample using GDAL, NumPy and Pillow (available in QGIS).

From the QGIS Python console, use runpy.run_path with run_name='__main__'.
"""
import json
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr
from PIL import Image, ImageDraw


def encode_rle(mask):
    """COCO uncompressed RLE, column-major, beginning with a zero run."""
    flat = mask.astype(np.uint8).ravel(order='F')
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, edges, flat.size]).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {'size': list(mask.shape), 'counts': counts}


def decode_rle(rle):
    values = np.arange(len(rle['counts'])) % 2
    return np.repeat(values, rle['counts']).reshape(rle['size'], order='F')


def prepare(root, raw=None, out=None, paths=None, preview_name='coco_preview.jpg'):
    raw = raw or root / 'data/raw/AOI_2_Vegas'
    out = out or root / 'data/processed/vegas_sample'
    for folder in ('images', 'coverage'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    (root / 'outputs').mkdir(exist_ok=True)
    coco = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'building'}]}
    report, previews = [], []
    annotation_id = 1
    paths = sorted(paths if paths is not None else (raw / 'PS-RGB').glob('*.tif'))
    for image_id, path in enumerate(paths, 1):
        ds = gdal.Open(str(path))
        array = ds.ReadAsArray()
        assert array.shape[0] == 3
        # Dataset-specific heuristic, NOT a NoData declaration in the source.
        coverage = np.any(array != 0, axis=0)
        rows, cols = np.where(coverage)
        if not len(rows):
            raise ValueError(f'Empty RGB image: {path}')
        x0, y0, x1, y1 = int(cols.min()), int(rows.min()), int(cols.max()+1), int(rows.max()+1)
        crop = array[:, y0:y1, x0:x1]
        valid = coverage[y0:y1, x0:x1]
        height, width = valid.shape
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        stretch = []
        for b in range(3):
            lo, hi = np.percentile(crop[b][valid], [2, 98])
            stretch.append([float(lo), float(hi)])
            rgb[:, :, b] = (np.clip((crop[b].astype(float)-lo) / max(hi-lo, 1), 0, 1)*255).astype(np.uint8)
        rgb[~valid] = 0
        name = path.stem + '.png'
        Image.fromarray(rgb).save(out / 'images' / name)
        Image.fromarray(valid.astype(np.uint8)*255).save(out / 'coverage' / name)
        gt = ds.GetGeoTransform()
        cropped_gt = (gt[0]+x0*gt[1]+y0*gt[2], gt[1], gt[2], gt[3]+x0*gt[4]+y0*gt[5], gt[4], gt[5])
        # Keep the transform for converting future predictions back to GIS.
        coco['images'].append({'id': image_id, 'file_name': name, 'width': width, 'height': height})
        label = raw / 'geojson_buildings' / path.name.replace('_PS-RGB_', '_geojson_buildings_').replace('.tif', '.geojson')
        vectors = ogr.Open(str(label))
        source_layer = vectors.GetLayer(0)
        source_crs = source_layer.GetSpatialRef().Clone()
        target_crs = osr.SpatialReference()
        target_crs.ImportFromWkt(ds.GetProjection())
        for crs in (source_crs, target_crs):
            crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(source_crs, target_crs)
        skipped, empty, count = 0, 0, 0
        overlay = rgb.copy()
        for feature in source_layer:
            original = feature.GetGeometryRef()
            if original is None or original.GetGeometryName() not in ('POLYGON', 'MULTIPOLYGON'):
                skipped += 1
                continue
            geometry = original.Clone()
            if geometry.Transform(transform) != 0:
                raise ValueError('Coordinate transformation failed')
            if not geometry.IsValid():
                geometry = geometry.MakeValid()
            vector = ogr.GetDriverByName('Memory').CreateDataSource('')
            layer = vector.CreateLayer('building', target_crs, ogr.wkbUnknown)
            item = ogr.Feature(layer.GetLayerDefn())
            item.SetGeometry(geometry)
            layer.CreateFeature(item)
            raster = gdal.GetDriverByName('MEM').Create('', width, height, 1, gdal.GDT_Byte)
            raster.SetGeoTransform(cropped_gt)
            raster.SetProjection(ds.GetProjection())
            raster.GetRasterBand(1).Fill(0)
            if gdal.RasterizeLayer(raster, [1], layer, burn_values=[1]) != 0:
                raise ValueError('Rasterization failed')
            mask = (raster.ReadAsArray() > 0) & valid
            yy, xx = np.where(mask)
            if not len(xx):
                empty += 1
                continue
            rle = encode_rle(mask)
            assert np.array_equal(decode_rle(rle), mask), 'RLE round-trip failed'
            bbox = [int(xx.min()), int(yy.min()), int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)]
            assert bbox[0]+bbox[2] <= width and bbox[1]+bbox[3] <= height
            coco['annotations'].append({'id': annotation_id, 'image_id': image_id, 'category_id': 1, 'segmentation': rle, 'bbox': bbox, 'area': int(mask.sum()), 'iscrowd': 0})
            annotation_id += 1
            count += 1
            # Draw the encoded mask, not the source polygon: checks actual COCO output.
            decoded = decode_rle(rle).astype(bool)
            interior = decoded.copy()
            interior[1:] &= decoded[:-1]
            interior[:-1] &= decoded[1:]
            interior[:, 1:] &= decoded[:, :-1]
            interior[:, :-1] &= decoded[:, 1:]
            overlay[decoded & ~interior] = [255, 55, 35]
        report.append({'image_id': image_id, 'source': path.relative_to(root).as_posix(), 'original_shape': list(array.shape), 'source_nodata': [ds.GetRasterBand(i).GetNoDataValue() for i in (1,2,3)], 'zero_rgb_percent': float((~coverage).mean()*100), 'crop_xyxy': [x0,y0,x1,y1], 'crop_geotransform': list(cropped_gt), 'crs_wkt': ds.GetProjection(), 'percentile_2_98_per_band': stretch, 'coverage_file': 'coverage/'+name, 'buildings': count, 'skipped_nonpolygons': skipped, 'empty_after_clipping': empty, 'eligible_for_smoke_training': bool(coverage.all())})
        tile = Image.fromarray(overlay)
        tile.thumbnail((390, 340))
        panel = Image.new('RGB', (410, 390), 'white')
        panel.paste(tile, (10, 40))
        draw = ImageDraw.Draw(panel)
        draw.text((10, 8), f'{path.stem.split("_")[-1]} | {count} buildings | coverage {coverage.mean():.1%}', fill='black')
        if len(previews) < 12:
            previews.append(panel)
        if image_id % 20 == 0:
            print(f'Prepared {image_id}/{len(paths)}', flush=True)
    if not report:
        raise ValueError('No input images')
    eligible = {r['image_id'] for r in report if r['eligible_for_smoke_training']}
    smoke = {**coco, 'images': [i for i in coco['images'] if i['id'] in eligible], 'annotations': [a for a in coco['annotations'] if a['image_id'] in eligible]}
    for name, content in [('instances_all.json', coco), ('instances_train_smoke.json', smoke), ('preparation_report.json', report)]:
        (out / name).write_text(json.dumps(content, indent=2)+'\n', encoding='utf-8')
    sheet = Image.new('RGB', (410*2, 390*((len(previews)+1)//2)), '#dddddd')
    for i, panel in enumerate(previews):
        sheet.paste(panel, ((i%2)*410, (i//2)*390))
    sheet.save(root / 'outputs' / preview_name)
    print(json.dumps({'images': len(coco['images']), 'instances': len(coco['annotations']), 'smoke_images': len(smoke['images']), 'smoke_instances': len(smoke['annotations']), 'skipped_nonpolygons': sum(r['skipped_nonpolygons'] for r in report), 'empty_after_clipping': sum(r['empty_after_clipping'] for r in report), 'output': str(out)}, indent=2))


if __name__ == '__main__':
    prepare(Path(__file__).resolve().parents[1])
