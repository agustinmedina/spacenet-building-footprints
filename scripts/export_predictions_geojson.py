"""Export COCO prediction masks to geographic polygons, preserving instances."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
from osgeo import gdal, ogr, osr
from pycocotools import mask as mask_util


def export(predictions_path, report_path, destination, threshold=0.5):
    gdal.UseExceptions()
    reports = {r['image_id']:r for r in json.loads(report_path.read_text())}
    predictions = json.loads(predictions_path.read_text())
    wgs = osr.SpatialReference(); wgs.ImportFromEPSG(4326)
    utm = osr.SpatialReference(); utm.ImportFromEPSG(32611)
    for srs in (wgs,utm): srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    features=[]; ids=set(); max_area_error=0.
    for index,pred in enumerate(predictions):
        if pred['score'] < threshold:
            continue
        assert pred['category_id']==1
        report=reports[pred['image_id']]
        mask=mask_util.decode(pred['segmentation']).astype(np.uint8)
        if mask.ndim!=2 or not mask.any():
            raise ValueError(f'Invalid or empty instance mask: {index}')
        height,width=mask.shape
        x0,y0,x1,y1=report['crop_xyxy']
        assert (height,width)==(y1-y0,x1-x0)
        gt=report['crop_geotransform']
        source_srs=osr.SpatialReference(); source_srs.ImportFromWkt(report['crs_wkt'])
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        raster=gdal.GetDriverByName('MEM').Create('',width,height,1,gdal.GDT_Byte)
        raster.SetGeoTransform(gt); raster.SetProjection(source_srs.ExportToWkt())
        band=raster.GetRasterBand(1); band.WriteArray(mask)
        vector=ogr.GetDriverByName('Memory').CreateDataSource('')
        layer=vector.CreateLayer('instance',source_srs,ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn('value',ogr.OFTInteger))
        # The same binary band masks the background; holes and disconnected parts survive.
        gdal.Polygonize(band,band,layer,0,[],callback=None)
        geometry=ogr.Geometry(ogr.wkbMultiPolygon)
        for part in layer:
            geometry.AddGeometry(part.GetGeometryRef())
        assert not geometry.IsEmpty() and geometry.IsValid()
        expected_area=int(mask.sum())*abs(gt[1]*gt[5]-gt[2]*gt[4])
        relative_error=abs(geometry.GetArea()-expected_area)/expected_area
        assert relative_error < 1e-5
        max_area_error=max(max_area_error,relative_error)
        # Rasterize the exported shape back onto the source pixel grid.
        check=gdal.GetDriverByName('MEM').Create('',width,height,1,gdal.GDT_Byte)
        check.SetGeoTransform(gt); check.SetProjection(source_srs.ExportToWkt())
        check.GetRasterBand(1).Fill(0)
        layer.ResetReading()
        gdal.RasterizeLayer(check,[1],layer,burn_values=[1])
        assert np.array_equal(check.ReadAsArray(),mask), f'Round-trip differs: {index}'
        area_geometry=geometry.Clone()
        assert area_geometry.Transform(osr.CoordinateTransformation(source_srs,utm))==0
        area_m2=area_geometry.GetArea()
        geographic=geometry.Clone()
        assert geographic.Transform(osr.CoordinateTransformation(source_srs,wgs))==0
        xmin,xmax,ymin,ymax=geographic.GetEnvelope()
        assert -116<xmin<=xmax<-114 and 35<ymin<=ymax<37, 'Coordinates outside Vegas'
        instance_id=f'vegas02_{pred["image_id"]}_{index}'
        ids.add(pred['image_id'])
        features.append({'type':'Feature','id':instance_id,
                         'properties':{'instance_id':instance_id,'image_id':pred['image_id'],'source_image':Path(report['source'].replace('\\','/')).name,
                                       'class':'building','confidence':float(pred['score']),'area_m2':round(area_m2,3),'mask_area_px':int(mask.sum()),
                                       'experiment':'02','split':'validation','score_threshold':threshold},
                         'geometry':json.loads(geographic.ExportToJson())})
    destination.parent.mkdir(parents=True,exist_ok=True)
    document={'type':'FeatureCollection','name':'SpaceNet Vegas — predicted buildings',
              'description':'Model predictions on spatial validation imagery; WGS84 longitude/latitude; CC BY-SA 4.0 derived SpaceNet data. Unreviewed predictions, not authoritative mapping.',
              'features':features}
    destination.write_text(json.dumps(document,ensure_ascii=False)+'\n',encoding='utf-8')
    loaded=ogr.Open(str(destination)); exported=loaded.GetLayer(0)
    assert exported.GetFeatureCount()==len(features)
    assert all(f.GetGeometryRef().IsValid() for f in exported)
    validation={'features':len(features),'images_with_predictions':len(ids),'score_threshold':threshold,'crs':'EPSG:4326, longitude/latitude',
                'round_trip_masks_checked':len(features),'max_relative_area_error':max_area_error,
                'geometry_validity':'all valid','simplification':'none','cross_tile_deduplication':'not performed',
                'note':'These are validation-set predictions, not a separate test on new client imagery.'}
    destination.with_suffix('.verification.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,default=ROOT/'results/experiment02/coco_instances_results.json')
    parser.add_argument('--report',type=Path,default=ROOT/'results/experiment02/preparation_report.json')
    parser.add_argument('--output',type=Path,default=ROOT/'portfolio/exports/buildings_vegas_validation.geojson')
    parser.add_argument('--threshold',type=float,default=0.5)
    args=parser.parse_args()
    if not 0<=args.threshold<=1: parser.error('threshold must be between 0 and 1')
    export(args.predictions,args.report,args.output,args.threshold)
