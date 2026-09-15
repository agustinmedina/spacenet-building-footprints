"""Audit experiment 02 predictions against the fixed validation set."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mu


def size_group(area):
    return 'small' if area < 32**2 else 'medium' if area < 96**2 else 'large'


def main():
    source=Path(sys.argv[1])
    data=ROOT/'data/processed/vegas_experiment'
    out=ROOT/'outputs/experiment02_review'
    out.mkdir(parents=True,exist_ok=True)
    predictions=json.loads(source.read_text())
    (out/'coco_instances_results.json').write_text(json.dumps(predictions))
    coco=json.loads((data/'instances_val.json').read_text())
    images={i['id']:i for i in coco['images']}
    by_gt=defaultdict(list); by_pred=defaultdict(list)
    for a in coco['annotations']: by_gt[a['image_id']].append(a)
    for p in predictions:
        assert p['image_id'] in images and p['category_id']==1
        by_pred[p['image_id']].append(p)
    totals=Counter(); fp_types=Counter(); fn_types=Counter()
    sizes=defaultdict(Counter); matched_ious=[]; per_image=[]; visual=[]
    for image_id, im in images.items():
        gt=by_gt[image_id]
        preds=sorted(by_pred[image_id],key=lambda p:p['score'],reverse=True)
        gr=[mu.frPyObjects(a['segmentation'],im['height'],im['width']) for a in gt]
        pr=[p['segmentation'] for p in preds]
        ious=mu.iou(pr,gr,[0]*len(gt)) if pr and gr else np.zeros((len(pr),len(gr)))
        n=sum(p['score']>=0.5 for p in preds)
        matched=set(); tp_preds=set(); fp_cases=[]
        for k in range(n):
            candidates=[j for j in range(len(gt)) if j not in matched and ious[k,j]>=0.5]
            if candidates:
                j=max(candidates,key=lambda j:ious[k,j]); matched.add(j); tp_preds.add(k)
                matched_ious.append(float(ious[k,j]))
            else:
                best=float(ious[k].max()) if gt else 0.
                kind='duplicate_or_matching_conflict' if best>=0.5 else 'overlap_below_iou_0.5' if best>=0.1 else 'little_or_no_gt_overlap'
                fp_types[kind]+=1
                fp_cases.append({'prediction_index':k,'score':preds[k]['score'],'best_mask_iou':best,'type':kind})
        fn_cases=[]
        for j,a in enumerate(gt):
            group=size_group(a['area']); sizes[group]['total']+=1
            if j in matched:
                sizes[group]['tp']+=1
                continue
            sizes[group]['fn']+=1
            high=float(ious[:n,j].max()) if n else 0.
            low=float(ious[n:,j].max()) if len(preds)>n else 0.
            kind='matching_conflict' if high>=0.5 else 'recoverable_below_score_0.5' if low>=0.5 else 'overlap_below_iou_0.5' if high>=0.1 else 'little_or_no_prediction_overlap'
            fn_types[kind]+=1
            fn_cases.append({'annotation_id':a['id'],'size':group,'area':a['area'],'best_iou_high_score':high,'best_iou_low_score':low,'type':kind})
        row={'image_id':image_id,'file_name':im['file_name'],'tp':len(matched),'fp':n-len(matched),'fn':len(gt)-len(matched),'fp_cases':fp_cases,'fn_cases':fn_cases}
        per_image.append(row)
        totals.update({k:row[k] for k in ('tp','fp','fn')})
        visual.append((row,im,gr,pr[:n],matched,tp_preds))
    assert dict(totals)=={'tp':545,'fp':214,'fn':196}, 'Audit does not reproduce submitted metrics'
    summary={'counts':dict(totals),'false_positive_types':dict(fp_types),'false_negative_types':dict(fn_types),
             'ground_truth_by_size':{k:{**dict(v),'recall':v['tp']/v['total']} for k,v in sizes.items()},
             'matched_mask_iou':{'median':float(np.median(matched_ious)),'fraction_at_least_0.75':float(np.mean(np.array(matched_ious)>=0.75))},
             'interpretation':'IoU-based error bins are diagnostic proxies, not verified semantic causes. Small/medium/large use mask areas <1024 / <9216 / >=9216 px. Low-score analysis only sees exported predictions (score >=0.05, max 100/image).',
             'per_image':per_image}
    (out/'error_analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Three panels per difficult tile: image, ground truth, predictions.
    selected=sorted(visual,key=lambda x:x[0]['fp']+x[0]['fn'],reverse=True)[:6]
    sheet=Image.new('RGB',(990,370*len(selected)),'white')
    draw=ImageDraw.Draw(sheet)
    for rowidx,(row,im,gr,pr,matched,tp_preds) in enumerate(selected):
        rgb=np.array(Image.open(data/'images'/im['file_name']))
        panels=[rgb.copy(),rgb.copy(),rgb.copy()]
        for col,rles,correct in [(1,gr,matched),(2,pr,tp_preds)]:
            for k,rle in enumerate(rles):
                mask=mu.decode(rle).astype(bool)
                color=np.array([0,210,80] if k in correct else [0,200,255] if col==1 else [255,40,40])
                panels[col][mask]=(panels[col][mask]*0.6+color*0.4).astype(np.uint8)
        titles=['Original',f'GT: cyan = missed ({row["fn"]})',f'Pred: red = FP ({row["fp"]})']
        for col,panel in enumerate(panels):
            image=Image.fromarray(panel); image.thumbnail((320,320))
            sheet.paste(image,(col*330,rowidx*370+45))
            draw.text((col*330+3,rowidx*370+8),im['file_name'].split('_')[-1],fill='black')
            draw.text((col*330+3,rowidx*370+25),titles[col],fill='black')
    sheet.save(out/'difficult_cases.jpg')
    print(json.dumps({k:v for k,v in summary.items() if k!='per_image'},indent=2))


if __name__=='__main__':main()
