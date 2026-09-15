"""Build experiment 02, reusing the environment setup from the completed smoke run."""
import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
previous = json.loads((ROOT/'notebooks/01_mask_rcnn_spacenet_colab.ipynb').read_text(encoding='utf-8'))
cells = []


def add(kind, source):
    source = textwrap.dedent(source).strip()+'\n'
    cell = {'cell_type':kind,'metadata':{},'source':source}
    if kind == 'code':
        ast.parse(source)
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


def md(source): add('markdown', source)
def code(source): add('code', source)


md('''
# Experimento 02: entrenar y validar por zonas

Usá este notebook nuevo y `spacenet_experiment_02.zip`. El `.ipynb` se abre como
notebook; el `.zip` se elige **en el selector de la celda 3**, sin editar rutas.

**Objetivo:** entrenar con más datos y medir sobre imágenes de una zona reservada.
La muestra parte de 200 pares de Las Vegas; se excluyen cobertura incompleta y una
franja espacial. Los conteos exactos aparecen en la celda 3 y en `split_report.json`.
Hay al menos 250 m entre imágenes de entrenamiento y validación.

Activá una GPU y ejecutá las celdas en orden. Son **1.500 iteraciones** con 2 imágenes
por lote, validación cada 500 y descarga de resultados al final. El tiempo depende
de la GPU y Colab puede interrumpir la sesión. Se puede guardar en Drive en la celda 5.

Empezamos desde pesos COCO, no desde el modelo anterior de 8 imágenes. La validación
es dentro de Las Vegas; no mide generalización a otras ciudades. No hay test final
independiente. Este experimento aún no se ejecutó en Colab.
''')
# Preserve the known working GPU checks and installation code.
for cell in previous['cells'][1:5]:
    source = cell['source']
    add(cell['cell_type'], ''.join(source) if isinstance(source,list) else source)

md('''
## 3. Subir el ZIP y verificar datos

Ejecutá la celda, pulsá **Seleccionar archivos** y elegí `spacenet_experiment_02.zip`
de la carpeta `SpaceNet/outputs`. No pegues la ruta de Windows en el código.
''')
code('''
import io, zipfile, hashlib
import numpy as np
from PIL import Image
from google.colab import files
from pycocotools import mask as mask_util
uploaded = files.upload()
zips = [(name, data) for name, data in uploaded.items() if name.lower().endswith('.zip')]
if len(zips) != 1:
    raise ValueError('Volvé a ejecutar esta celda y elegí solamente spacenet_experiment_02.zip, no el archivo .ipynb.')
zip_name, zip_bytes = zips[0]
dataset_sha256 = hashlib.sha256(zip_bytes).hexdigest()
data_root = Path('/content/spacenet_experiment_02')
data_root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
    required = {'instances_train.json','instances_val.json','split_report.json','preparation_report.json'}
    if not required <= set(archive.namelist()):
        raise ValueError('Este ZIP no es el del experimento 02. Elegí spacenet_experiment_02.zip.')
    assert archive.testzip() is None, 'ZIP corrupto'
    for member in archive.infolist():
        assert (data_root/member.filename).resolve().is_relative_to(data_root.resolve())
    archive.extractall(data_root)
del uploaded, zips, zip_bytes
split_report = json.loads((data_root/'split_report.json').read_text())
assert split_report['minimum_train_val_distance_m'] >= 250-1e-6
assert not set(split_report['train_image_ids']) & set(split_report['val_image_ids'])
datasets = {}
for split in ('train','val'):
    dataset = json.loads((data_root/f'instances_{split}.json').read_text())
    images = {im['id']: im for im in dataset['images']}
    assert set(images) == set(split_report[f'{split}_image_ids'])
    assert len(images) == split_report['counts'][split]['images']
    for im in images.values():
        with Image.open(data_root/'images'/im['file_name']) as picture:
            assert picture.size == (im['width'],im['height']) and picture.mode == 'RGB'
    for ann in dataset['annotations']:
        im = images[ann['image_id']]
        rle = mask_util.frPyObjects(ann['segmentation'], im['height'], im['width'])
        assert int(mask_util.area(rle)) == ann['area']
        assert np.array_equal(mask_util.toBbox(rle), ann['bbox'])
    datasets[split] = dataset
    print(split, '| imágenes:',len(images),'| edificios:',len(dataset['annotations']))
print('Separación mínima (m):', round(split_report['minimum_train_val_distance_m'],1))
print('Datos y máscaras verificados.')
''')
md('## 4. Registrar entrenamiento y validación')
code('''
import cv2
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
for split in ('train','val'):
    name = f'vegas02_{split}'
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name); MetadataCatalog.remove(name)
    register_coco_instances(name, {}, str(data_root/f'instances_{split}.json'), str(data_root/'images'))
train_records = DatasetCatalog.get('vegas02_train')
val_records = DatasetCatalog.get('vegas02_val')
assert not {r['file_name'] for r in train_records} & {r['file_name'] for r in val_records}
print('Imágenes negativas conservadas:',sum(not r['annotations'] for r in train_records))
fig, axes = plt.subplots(1,2,figsize=(12,6))
for ax, records, split in zip(axes,[train_records,val_records],['train','val']):
    record = next(r for r in records if r['annotations'])
    rgb = cv2.imread(record['file_name'])[:,:,::-1]
    ax.imshow(Visualizer(rgb, metadata=MetadataCatalog.get(f'vegas02_{split}')).draw_dataset_dict(record).get_image())
    ax.set_title(f'Etiquetas: {split}'); ax.axis('off')
plt.show()
''')
md('''
## 5. Configurar y elegir dónde guardar

Con `SAVE_TO_DRIVE = False`, los archivos se guardan temporalmente en Colab.
Podés cambiarlo a `True` para guardar checkpoints en tu Google Drive; Colab pedirá
conectar tu cuenta. Para reanudar tras una interrupción, usá la misma carpeta y
poné `RESUME = True`. Volvé a cargar el ZIP si la sesión se reinició.

El ritmo equivale aproximadamente a `1500 × 2 / cantidad_de_imágenes_train` pasadas
por los datos. No garantiza convergencia ni precisión. Primero evaluaremos el resultado.
''')
code('''
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.env import seed_all_rng
SAVE_TO_DRIVE = False
RESUME = False
RUN_NAME = 'vegas02_run1'
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    run_dir = Path('/content/drive/MyDrive/SpaceNet')/RUN_NAME
else:
    run_dir = Path('/content')/RUN_NAME
run_dir.mkdir(parents=True, exist_ok=True)
if (run_dir/'last_checkpoint').exists() and not RESUME:
    raise ValueError('La carpeta ya contiene un entrenamiento. Cambiá RUN_NAME para uno nuevo, o RESUME=True para continuar.')
if RESUME and not (run_dir/'last_checkpoint').exists():
    raise ValueError('No hay checkpoint para reanudar en esta carpeta.')
manifest_file = run_dir/'dataset_sha256.txt'
if RESUME:
    assert manifest_file.read_text().strip() == dataset_sha256, 'El dataset cambió; iniciá otra corrida.'
manifest_file.write_text(dataset_sha256)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.MODEL.DEVICE = 'cuda'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # bajo para construir la curva precision-recall
cfg.DATASETS.TRAIN = ('vegas02_train',)
cfg.DATASETS.TEST = ('vegas02_val',)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.INPUT.MIN_SIZE_TRAIN = (650,)
cfg.INPUT.MAX_SIZE_TRAIN = 650
cfg.INPUT.MIN_SIZE_TEST = 650
cfg.INPUT.MAX_SIZE_TEST = 650
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.STEPS = (1000,1300)
cfg.SOLVER.CHECKPOINT_PERIOD = 250
cfg.TEST.EVAL_PERIOD = 500
cfg.TEST.DETECTIONS_PER_IMAGE = 100  # COCO estándar; puede limitar escenas muy densas
cfg.SEED = 42
seed_all_rng(cfg.SEED)
cfg.OUTPUT_DIR = str(run_dir)
if RESUME:
    assert (run_dir/'config.yaml').read_text() == cfg.dump(), 'La configuración cambió; usá un RUN_NAME nuevo.'
(run_dir/'config.yaml').write_text(cfg.dump())
(run_dir/'environment.txt').write_text(environment)
(run_dir/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, tasks=('bbox','segm'), distributed=False,
                             output_dir=output_folder or str(Path(cfg.OUTPUT_DIR)/'validation'), use_fast_impl=False)
mapper = DatasetMapper(cfg,is_train=True)
for record in train_records[:3] + [r for r in train_records if not r['annotations']][:1]:
    assert 'instances' in mapper(record)
print('Pasadas aproximadas:',round(cfg.SOLVER.MAX_ITER*cfg.SOLVER.IMS_PER_BATCH/len(train_records),1))
print('Configuración lista. Checkpoint cada 250 iteraciones; validación cada 500.')
''')
md('''
## 6. Entrenar

Esperá hasta que termine. Si aparece `CUDA out of memory`, reiniciá la sesión,
cambiá `IMS_PER_BATCH` a 1 y usá un nuevo `RUN_NAME`; se reducirá a la mitad el
número de imágenes vistas a igual cantidad de iteraciones. No cambies parámetros
a ciegas para forzar un resultado favorable en validación.
''')
code('''
trainer = Trainer(cfg)
trainer.resume_or_load(resume=RESUME)
trainer.train()
assert (run_dir/'model_final.pth').exists()
print('Entrenamiento terminado.')
''')
md('''
## 7. Evaluar el modelo final en toda la validación

`segm/AP` resume varios umbrales IoU; `segm/AP50` usa IoU=0,5. Detectron2 los
presenta en escala 0–100. No son el porcentaje de confianza que aparece en cada
detección. El límite estándar es 100 detecciones por imagen.

Fuente: [evaluadores de Detectron2](https://detectron2.readthedocs.io/en/latest/modules/evaluation.html).
''')
code('''
import gc
del trainer
gc.collect(); torch.cuda.empty_cache()
eval_cfg = cfg.clone()
eval_cfg.MODEL.WEIGHTS = str(run_dir/'model_final.pth')
predictor = DefaultPredictor(eval_cfg)
evaluator = COCOEvaluator('vegas02_val', tasks=('bbox','segm'), distributed=False,
                         output_dir=str(run_dir/'final_validation'), use_fast_impl=False)
loader = build_detection_test_loader(eval_cfg,'vegas02_val')
results = inference_on_dataset(predictor.model, loader, evaluator)
(run_dir/'validation_metrics.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results,indent=2))
''')
md('''
## 8. Contar aciertos, falsos positivos y edificios omitidos

Usamos confianza ≥0,5 e IoU de máscara ≥0,5, con correspondencia uno a uno.
Las predicciones se procesan de mayor a menor confianza; una detección duplicada
cuenta como falso positivo. Este F1 de máscaras es una métrica propia, **no el
scorer oficial de polígonos de SpaceNet**. No buscamos el mejor umbral en esta corrida.
''')
code('''
from pycocotools.coco import COCO
from collections import defaultdict
ground_truth = COCO(str(data_root/'instances_val.json'))
predictions = json.loads((run_dir/'final_validation/coco_instances_results.json').read_text())
grouped = defaultdict(list)
for p in predictions:
    if p['score'] >= 0.5:
        grouped[p['image_id']].append(p)
def count_matches(ious, threshold=0.5):
    matched = set()
    for row in ious:
        candidates = [j for j in range(ious.shape[1]) if j not in matched and row[j] >= threshold]
        if candidates:
            matched.add(max(candidates,key=lambda j:row[j]))
    tp = len(matched)
    return tp, ious.shape[0]-tp, ious.shape[1]-tp
tp = fp = fn = 0
per_image = []
for image_id in ground_truth.getImgIds():
    labels = ground_truth.loadAnns(ground_truth.getAnnIds(imgIds=[image_id]))
    gt_rles = [ground_truth.annToRLE(a) for a in labels]
    preds = sorted(grouped[image_id],key=lambda p:p['score'],reverse=True)
    ious = mask_util.iou([p['segmentation'] for p in preds],gt_rles,[0]*len(gt_rles)) if preds and gt_rles else np.zeros((len(preds),len(gt_rles)))
    image_tp, image_fp, image_fn = count_matches(ious)
    tp += image_tp; fp += image_fp; fn += image_fn
    per_image.append({'image_id':image_id,'tp':image_tp,'fp':image_fp,'fn':image_fn})
scores = {'score_threshold':0.5,'mask_iou_threshold':0.5,'tp':tp,'fp':fp,'fn':fn,
          'precision':tp/(tp+fp) if tp+fp else 0., 'recall':tp/(tp+fn) if tp+fn else 0.,
          'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., 'per_image':per_image}
(run_dir/'building_metrics.json').write_text(json.dumps(scores,indent=2))
print({k:v for k,v in scores.items() if k!='per_image'})
''')
md('''
## 9. Revisar predicciones de validación y curvas

Se muestran las primeras cuatro imágenes de validación en orden de nombre, sin
seleccionarlas por el resultado. Solo dibujamos máscaras, con confianza ≥0,5,
para evitar que las etiquetas de texto tapen los edificios.
''')
code('''
metrics = [json.loads(line) for line in (run_dir/'metrics.json').read_text().splitlines()]
losses = [m for m in metrics if 'total_loss' in m]
assert losses and all(np.isfinite(m['total_loss']) for m in losses)
fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].plot([m['iteration'] for m in losses],[m['total_loss'] for m in losses]); axes[0].set_title('Pérdida de entrenamiento')
evaluations = [m for m in metrics if 'segm/AP50' in m]
axes[1].plot([m['iteration'] for m in evaluations],[m['segm/AP50'] for m in evaluations],marker='o'); axes[1].set_title('AP50 máscaras — validación')
for ax in axes: ax.set_xlabel('Iteración'); ax.grid()
fig.tight_layout(); fig.savefig(run_dir/'learning_curves.png'); plt.show()
examples = sorted(val_records,key=lambda r:r['file_name'])[:4]
fig, axes = plt.subplots(len(examples),2,figsize=(12,6*len(examples)), squeeze=False)
for row, record in enumerate(examples):
    bgr = cv2.imread(record['file_name']); rgb = bgr[:,:,::-1]
    gt_masks = [mask_util.decode(a['segmentation']) for a in record['annotations']]
    gt_vis = Visualizer(rgb).overlay_instances(masks=gt_masks if gt_masks else None,alpha=0.4).get_image()
    instances = predictor(bgr)['instances'].to('cpu')
    instances = instances[instances.scores >= 0.5]
    pred_vis = Visualizer(rgb).overlay_instances(masks=instances.pred_masks.numpy() if len(instances) else None,alpha=0.4).get_image()
    axes[row,0].imshow(gt_vis); axes[row,0].set_title(f'Etiquetas — {Path(record["file_name"]).stem.split("_")[-1]}')
    axes[row,1].imshow(pred_vis); axes[row,1].set_title(f'Validación — {len(instances)} predicciones')
    for ax in axes[row]: ax.axis('off')
fig.tight_layout(); fig.savefig(run_dir/'validation_comparison.png'); plt.show()
''')
md('''
## 10. Descargar resultados

Primero se descarga un ZIP pequeño con métricas y figuras para compartir. Después,
la celda siguiente descarga el modelo final con su configuración. Los checkpoints
intermedios no se incluyen para evitar duplicar varios archivos grandes.
''')
code('''
import shutil
for name in ('split_report.json','preparation_report.json','source_manifest.json'):
    shutil.copy2(data_root/name,run_dir/name)
report_zip = Path('/content/spacenet_experiment_02_report.zip')
with zipfile.ZipFile(report_zip,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in ('validation_metrics.json','building_metrics.json','learning_curves.png','validation_comparison.png','metrics.json','config.yaml','environment.txt','pip-freeze.txt','split_report.json','dataset_sha256.txt'):
        archive.write(run_dir/name,name)
files.download(str(report_zip))
''')
code('''
model_zip = Path('/content/spacenet_experiment_02_model.zip')
with zipfile.ZipFile(model_zip,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in ('model_final.pth','config.yaml','environment.txt','pip-freeze.txt','preparation_report.json','source_manifest.json','split_report.json','dataset_sha256.txt'):
        archive.write(run_dir/name,name)
files.download(str(model_zip))
''')
md('''
Compartí `validation_comparison.png`, `learning_curves.png` y `building_metrics.json`
para decidir el próximo cambio. No aumentes iteraciones o ajustes umbrales sin revisar
qué errores predominan. La validación puede usarse para desarrollo; una evaluación
final necesitará otra zona sin usar para elegir parámetros.

Datos: [SpaceNet 2](https://spacenet.ai/spacenet-buildings-dataset-v2/), CC BY-SA 4.0.
Van Etten et al. (2018), arXiv:1807.01232. Se conserva la atribución en el ZIP.
''')
for index, cell in enumerate(cells): cell['id']=f'exp02-{index:02d}'
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'02_mask_rcnn_spatial_validation.ipynb'},'accelerator':'GPU'},'cells':cells}
path=ROOT/'notebooks/02_mask_rcnn_spatial_validation.ipynb'
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Notebook written and all {sum(c["cell_type"]=="code" for c in cells)} code cells syntax-checked: {path}')
