# GIS demonstration

Open `SpaceNet_portfolio.qgz` in QGIS after downloading the repository. It references
four local sample GeoTIFFs in `data/` and the 759 predicted instances in `exports/`.
Paths are relative; keep those folders next to the project file.

The saved view is the first validation example, img5416. Select another raster and
use **Zoom to Layer** to view the other examples. The full prediction layer covers
31 validation tiles with detections; only four imagery tiles are bundled here.

`figures/portfolio_cover.png` is the presentation image. Cyan contours are predictions,
not manually verified building footprints. False positives are retained.

Coordinates: WGS84 longitude/latitude. Area attribute: UTM 11N square metres.
No polygon simplification or cross-tile deduplication was applied.

See the [main README](../README.md) for metrics, limitations and reproduction steps.
Data: SpaceNet Partners, CC BY-SA 4.0; see [attribution](../DATA_ATTRIBUTION.md).
