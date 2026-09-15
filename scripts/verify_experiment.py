"""Validate spatial separation, COCO masks, PNGs and notebook contract locally."""
import ast
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from shapely.geometry import box, shape
from prepare_coco import encode_rle, decode_rle
from prepare_experiment import spatial_partition

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Edge cases: empty mask, solid mask, holes, disconnected parts and F-order.
    rng = np.random.default_rng(7)
    masks = [np.zeros((3,7),dtype=bool),np.ones((3,7),dtype=bool),rng.random((11,9))>0.5]
    hole=np.ones((9,9),dtype=bool); hole[2:7,2:7]=0; masks.append(hole)
    for mask in masks:
        assert np.array_equal(decode_rle(encode_rle(mask)),mask)
    footprints={i:box(i*200,0,i*200+100,100) for i in range(20)}
    train,val,excluded,stats=spatial_partition(footprints)
    assert train.isdisjoint(val) and train|val|excluded == set(footprints)
    assert stats['minimum_train_val_distance_m'] >= 250
    out=ROOT/'data/processed/vegas_experiment'
    summary=json.loads((out/'split_report.json').read_text())
    spatial=json.loads((out/'split_footprints_utm.geojson').read_text())
    regions={f['properties']['image_id']:shape(f['geometry']) for f in spatial['features']}
    ids={}; image_files={}; annotations_total=0
    for split in ('train','val'):
        coco=json.loads((out/f'instances_{split}.json').read_text())
        images={im['id']:im for im in coco['images']}
        assert len(images)==len(coco['images'])
        ids[split]=set(images)
        image_files[split]={im['file_name'] for im in images.values()}
        assert ids[split]==set(summary[f'{split}_image_ids'])
        assert len({a['id'] for a in coco['annotations']})==len(coco['annotations'])
        for im in images.values():
            with Image.open(out/'images'/im['file_name']) as png:
                assert png.mode=='RGB' and png.size==(im['width'],im['height'])
        for ann in coco['annotations']:
            im=images[ann['image_id']]
            mask=decode_rle(ann['segmentation'])
            assert mask.shape==(im['height'],im['width'])
            assert int(mask.sum())==ann['area']>0
            yy,xx=np.where(mask)
            assert ann['bbox']==[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]
        assert len(coco['annotations'])==summary['counts'][split]['buildings']
        annotations_total+=len(coco['annotations'])
    assert ids['train'].isdisjoint(ids['val'])
    assert image_files['train'].isdisjoint(image_files['val'])
    distances=[regions[t].distance(regions[v]) for t in ids['train'] for v in ids['val']]
    assert min(distances)>=250-1e-6
    assert abs(min(distances)-summary['minimum_train_val_distance_m'])<1e-6
    with zipfile.ZipFile(ROOT/'outputs/spacenet_experiment_02.zip') as archive:
        assert archive.testzip() is None
        names=archive.namelist(); assert len(names)==len(set(names))
        assert {n.removeprefix('images/') for n in names if n.startswith('images/')}==image_files['train']|image_files['val']
    notebook=json.loads((ROOT/'notebooks/02_mask_rcnn_spatial_validation.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code': ast.parse(cell['source'])
    # Exercise the actual notebook matching function: duplicate predictions, missed
    # buildings, no predictions, no labels and the exact threshold boundary.
    source=next(c['source'] for c in notebook['cells'] if c['cell_type']=='code' and 'tp = fp = fn = 0' in c['source'])
    function=next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='count_matches')
    namespace={}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<notebook matching>','exec'),namespace)
    match=namespace['count_matches']
    assert match(np.array([[1.,0.],[0.9,0.],[0.,0.]]))==(1,2,1)
    assert match(np.zeros((0,2)))==(0,0,2)
    assert match(np.zeros((3,0)))==(0,3,0)
    assert match(np.array([[0.5]]))==(1,0,0)
    result={'status':'passed','images':{k:len(v) for k,v in ids.items()},'masks_checked':annotations_total,'minimum_distance_m':min(distances),
            'checks':['RLE edge cases','all PNGs and masks','COCO boxes and IDs','spatial gap','ZIP integrity and membership','notebook syntax','one-to-one matching edge cases'],
            'not_executed':['Detectron2 GPU training','COCO evaluator in Colab']}
    (ROOT/'outputs/experiment_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
