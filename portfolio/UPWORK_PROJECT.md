# Satellite Building Footprint Extraction with Mask R-CNN

**Role:** Geospatial machine learning developer

## Short description

Built a satellite-to-GeoJSON pipeline using Mask R-CNN and SpaceNet 2. Prepared imagery and labels, trained on 146 tiles, and validated on 32 geographically separated tiles. Achieved 71.8% precision, 73.5% recall and 72.7% mask F1 at confidence/IoU >=0.5. Exported 759 predicted building instances with confidence and area attributes; verified geometry and mask-to-polygon consistency. Includes Colab notebooks, QGIS examples and error analysis. Public-data prototype; small buildings remain a limitation.

## Detailed case study

Built a reproducible pipeline that converts satellite imagery into georeferenced
building polygons. Using the public SpaceNet 2 Las Vegas dataset, I checked image
coverage and label alignment, converted annotations to COCO instance masks,
fine-tuned Mask R-CNN, and evaluated predictions on a geographically separated
validation area. I exported the predicted masks to GeoJSON and checked the
conversion against the original pixel grid.

The model achieved **71.8% precision, 73.5% recall and 72.7% mask F1** on 32 held-out
tiles containing 741 labeled buildings, using confidence and mask IoU thresholds
of 0.5. Mask AP50 was 75.78 and mask AP was 46.36 (0–100 scale).

Deliverables include reproducible Colab notebooks, a documented spatial split,
error analysis, and GIS-ready GeoJSON with confidence scores and building areas.
Error analysis identified small buildings as the main weakness and informed the
next experiment instead of increasing training time without evidence.

**Skills:** Python · GIS · Computer Vision · Instance Segmentation · GDAL

**Scope:** Independent portfolio prototype using public labeled data. Results are
specific to this Las Vegas validation sample; outputs require review before use.
This is not a client commission, a production deployment, or an independent test
in another city. Mask F1 is not the official SpaceNet polygon scorer.

**Data credit:** SpaceNet 2 / SpaceNet Partners, CC BY-SA 4.0.
https://spacenet.ai/spacenet-buildings-dataset-v2/
Van Etten, Lindenbaum & Bacastow (2018), arXiv:1807.01232.
