"""Build the teaching notebook and upload archive from the prepared sample."""
import ast
import hashlib
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(source).strip()+'\n'})


def code(source):
    source = textwrap.dedent(source).strip()+'\n'
    ast.parse(source)
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': source, 'execution_count': None, 'outputs': []})


md('''
# SpaceNet → primera prueba de Mask R-CNN

**Objetivo:** comprobar carga de máscaras, entrenamiento e inferencia con 8 imágenes
y 166 edificios. No es una evaluación de precisión: las predicciones se harán sobre
imágenes de entrenamiento. No hay un conjunto de validación independiente.

1. En Colab, seleccioná **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**.
2. Ejecutá las celdas de arriba hacia abajo con el botón ▶.
3. Cuando se solicite, subí `spacenet_colab_sample.zip` (no el notebook).
4. Descargá los resultados antes de cerrar la sesión: `/content` es temporal.

Se usan los PNG preparados, sin volver a normalizar. Las imágenes con relleno negro
quedan fuera del entrenamiento. El GeoTIFF original no se modifica.

La instalación compila Detectron2 para el PyTorch de esta sesión. Puede tardar varios
minutos. Este notebook fue revisado localmente, pero todavía no ejecutado en una GPU Colab.
''')
md('## 1. Comprobar GPU y entorno')
code('''
import os, sys, json, subprocess
from pathlib import Path
import torch, torchvision
from torch.utils.cpp_extension import CUDA_HOME
print('Python:', sys.version)
print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__)
print('CUDA de PyTorch:', torch.version.cuda, '| CUDA_HOME:', CUDA_HOME)
assert torch.cuda.is_available(), 'Activá una GPU y volvé a ejecutar esta celda.'
assert CUDA_HOME is not None, 'Falta el compilador CUDA. Revisá el entorno GPU.'
print('GPU:', torch.cuda.get_device_name(0))
subprocess.run(['nvidia-smi'], check=True)
# Comprueba que torchvision tiene sus operaciones CUDA compatibles con torch.
torchvision.ops.nms(torch.tensor([[0.,0.,10.,10.]], device='cuda'), torch.tensor([1.], device='cuda'), 0.5)
''')
md('''
## 2. Instalar Detectron2

Se mantiene el PyTorch de Colab y se compila Detectron2 en ese entorno. Si la
instalación falla, detenete y compartí el error de esta celda; no sigas con training.

Fuentes: [instalación oficial](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)
y [máscaras RLE y datasets](https://github.com/facebookresearch/detectron2/blob/main/docs/tutorials/datasets.md).
''')
code('''
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ninja', 'setuptools', 'wheel', 'pycocotools'], check=True)
build_env = dict(os.environ, MAX_JOBS='2')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
                'git+https://github.com/facebookresearch/detectron2.git'], env=build_env, check=True)
import detectron2
from detectron2.utils.collect_env import collect_env_info
environment = collect_env_info()
print(environment)
''')
md('## 3. Subir y validar el ZIP de datos')
code('''
import zipfile
import numpy as np
from PIL import Image
from google.colab import files
from pycocotools import mask as mask_util
uploaded = files.upload()
assert 'spacenet_colab_sample.zip' in uploaded, 'Elegí spacenet_colab_sample.zip.'
data_root = Path('/content/spacenet_sample')
data_root.mkdir(exist_ok=True)
with zipfile.ZipFile('spacenet_colab_sample.zip') as archive:
    assert archive.testzip() is None, 'ZIP corrupto'
    for member in archive.infolist():
        target = (data_root / member.filename).resolve()
        assert target.is_relative_to(data_root.resolve()), 'Ruta inválida dentro del ZIP'
    archive.extractall(data_root)
del uploaded
annotation_path = data_root / 'instances_train_smoke.json'
raw = json.loads(annotation_path.read_text())
assert len(raw['images']) == 8 and len(raw['annotations']) == 166
image_by_id = {im['id']: im for im in raw['images']}
for im in raw['images']:
    with Image.open(data_root / 'images' / im['file_name']) as picture:
        assert picture.size == (im['width'], im['height']) and picture.mode == 'RGB'
for ann in raw['annotations']:
    im = image_by_id[ann['image_id']]
    rle = ann['segmentation']
    compressed = mask_util.frPyObjects(rle, im['height'], im['width'])
    mask = mask_util.decode(compressed)
    assert mask.shape == (im['height'], im['width'])
    assert int(mask.sum()) == ann['area']
    assert np.array_equal(mask_util.toBbox(compressed), ann['bbox'])
print('OK: 8 imágenes y 166 máscaras verificadas con pycocotools.')
''')
md('## 4. Registrar el dataset y ver las etiquetas')
code('''
import cv2
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
dataset_name = 'vegas_smoke'
if dataset_name in DatasetCatalog.list():
    DatasetCatalog.remove(dataset_name)
    MetadataCatalog.remove(dataset_name)
register_coco_instances(dataset_name, {}, str(annotation_path), str(data_root / 'images'))
records = DatasetCatalog.get(dataset_name)
metadata = MetadataCatalog.get(dataset_name)
assert len(records) == 8 and sum(len(d['annotations']) for d in records) == 166
example = next(d for d in records if len(d['annotations']) >= 15)
bgr = cv2.imread(example['file_name'])
vis = Visualizer(bgr[:, :, ::-1], metadata=metadata).draw_dataset_dict(example)
plt.figure(figsize=(9,9)); plt.imshow(vis.get_image()); plt.axis('off'); plt.show()
print('Estas son etiquetas reales; todavía no son predicciones.')
''')
md('''
## 5. Configurar una prueba corta

100 iteraciones, una imagen por lote y una clase (`building`). Partimos de pesos
COCO; la cabeza de clasificación se adapta a una sola clase. Es esperable que el
registro avise que no carga algunos pesos de la cabeza por diferencias de tamaño.
Una iteración es una actualización del modelo, no una época.
''')
code('''
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.MODEL.DEVICE = 'cuda'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.DATASETS.TRAIN = (dataset_name,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.INPUT.MIN_SIZE_TRAIN = (512,)
cfg.INPUT.MAX_SIZE_TRAIN = 650
cfg.INPUT.MIN_SIZE_TEST = 512
cfg.INPUT.MAX_SIZE_TEST = 650
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 100
cfg.SOLVER.WARMUP_ITERS = 10
cfg.SOLVER.STEPS = ()
cfg.SOLVER.CHECKPOINT_PERIOD = 100
cfg.TEST.EVAL_PERIOD = 0
cfg.SEED = 42
cfg.OUTPUT_DIR = '/content/spacenet_run'
Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True)
Path(cfg.OUTPUT_DIR, 'config.yaml').write_text(cfg.dump())
Path(cfg.OUTPUT_DIR, 'environment.txt').write_text(environment)
Path(cfg.OUTPUT_DIR, 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
# Verifica el mapper tanto con edificios como con la imagen negativa.
mapper = DatasetMapper(cfg, is_train=True)
for record in records:
    mapped = mapper(record)
    assert 'instances' in mapped
    if record['annotations']:
        assert mapped['instances'].has('gt_masks')
print('Mapper OK. Configuración lista para 100 iteraciones.')
''')
md('''
## 6. Entrenar

Esperá a que termine. La pérdida (`total_loss`) debe ser finita; una pérdida baja
en esta muestra no demuestra que el modelo generalice. Si se agota la memoria GPU,
reiniciá la sesión y usá 384 para `MIN_SIZE_TRAIN` y 512 para `MAX_SIZE_TRAIN`.
''')
code('''
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
assert Path(cfg.OUTPUT_DIR, 'model_final.pth').exists()
print('Entrenamiento terminado; checkpoint guardado.')
''')
md('## 7. Ver pérdidas y predicciones sobre una imagen de entrenamiento')
code('''
metrics = [json.loads(line) for line in Path(cfg.OUTPUT_DIR, 'metrics.json').read_text().splitlines()]
losses = [m for m in metrics if 'total_loss' in m]
assert losses and all(np.isfinite(m['total_loss']) for m in losses)
plt.figure(figsize=(8,3))
plt.plot([m['iteration'] for m in losses], [m['total_loss'] for m in losses])
plt.xlabel('Iteración'); plt.ylabel('Pérdida de entrenamiento'); plt.grid(); plt.show()
del trainer
import gc
gc.collect(); torch.cuda.empty_cache()
inference_cfg = cfg.clone()
inference_cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR, 'model_final.pth'))
inference_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
predictor = DefaultPredictor(inference_cfg)
instances = predictor(bgr)['instances'].to('cpu')
predicted = Visualizer(bgr[:, :, ::-1], metadata=metadata).draw_instance_predictions(instances)
fig, axes = plt.subplots(1, 2, figsize=(14,7))
axes[0].imshow(vis.get_image()); axes[0].set_title('Etiquetas')
axes[1].imshow(predicted.get_image()); axes[1].set_title('Predicción — imagen de entrenamiento')
for ax in axes: ax.axis('off')
fig.tight_layout(); fig.savefig(Path(cfg.OUTPUT_DIR, 'comparison.png')); plt.show()
np.savez_compressed(Path(cfg.OUTPUT_DIR, 'prediction_masks.npz'),
                    masks=instances.pred_masks.numpy(), scores=instances.scores.numpy(),
                    boxes=instances.pred_boxes.tensor.numpy(), image_id=example['image_id'])
print(f'{len(instances)} predicciones con umbral 0.3. Cero predicciones también es posible en esta prueba corta.')
''')
md('## 8. Descargar el resultado')
code(r'''
import shutil
shutil.copy2(data_root / 'preparation_report.json', Path(cfg.OUTPUT_DIR, 'preparation_report.json'))
Path(cfg.OUTPUT_DIR, 'run_notes.txt').write_text(
    'Prueba de funcionamiento: 8 imágenes, 166 edificios, sin validación independiente.\n'
    'comparison.png muestra una imagen de entrenamiento. No reportar precisión con esta prueba.\n'
    'Las máscaras predichas están en píxeles del PNG; ver preparation_report.json para georreferencia.\n')
archive_path = shutil.make_archive('/content/spacenet_smoke_results', 'zip', cfg.OUTPUT_DIR)
files.download(archive_path)
''')
md('''
## Después de esta prueba

Si carga los datos, termina las 100 iteraciones y genera predicciones, el circuito
básico funciona. El siguiente experimento requiere más imágenes y separación espacial
entre entrenamiento y validación. Recién entonces mediremos precisión y recall por edificio.

Dataset: [SpaceNet 2](https://spacenet.ai/spacenet-buildings-dataset-v2/), CC BY-SA 4.0.
Van Etten, A., Lindenbaum, D., & Bacastow, T.M. (2018), SpaceNet: A Remote Sensing
Dataset and Challenge Series, arXiv:1807.01232.
''')

notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}, 'colab': {'name': '01_mask_rcnn_spacenet_colab.ipynb'}, 'accelerator': 'GPU'}, 'cells': cells}
for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i:02d}'
path = ROOT / 'notebooks/01_mask_rcnn_spacenet_colab.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
sample = ROOT / 'data/processed/vegas_sample'
annotations = json.loads((sample / 'instances_train_smoke.json').read_text())
bundle = ROOT / 'outputs/spacenet_colab_sample.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ['instances_train_smoke.json', 'preparation_report.json']:
        archive.write(sample / name, name)
    for im in annotations['images']:
        archive.write(sample / 'images' / im['file_name'], 'images/'+im['file_name'])
    archive.writestr('SOURCE.txt', 'SpaceNet 2: https://spacenet.ai/spacenet-buildings-dataset-v2/\nCC BY-SA 4.0. See original dataset license and attribution.\nVan Etten et al. (2018), arXiv:1807.01232.\nRGB PNGs and COCO masks derived from AOI 2 Vegas; processing documented in preparation_report.json.\n')
with zipfile.ZipFile(bundle) as archive:
    assert archive.testzip() is None
print(f'Notebook: {path}\nZIP: {bundle}\nSize: {bundle.stat().st_size/1024**2:.2f} MiB\nSHA256: {hashlib.sha256(bundle.read_bytes()).hexdigest()}')
