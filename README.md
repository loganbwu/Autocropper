# Autocropper

Review and crop CR3 RAW files using AI detection, then write Lightroom-compatible XMP sidecar files.

## Features

- **Web review UI**: Browse processed CR3 files side-by-side (original vs crop), accept or reject each with a keypress
- **Classic crop mode**: GDINO detects the person bounding box, SAM 2.1 refines it to a pixel-accurate silhouette, a configurable margin is applied
- **ML crop mode**: Case-based reasoning — match the query photo against a training dataset of previously-cropped photos using SAM 2.1 mask shape and ViTPose face position similarity (k-NN)
- **XMP compatibility**: Writes Lightroom-compatible XMP sidecar files; preserves all existing metadata (ratings, colour corrections, keywords) and only updates crop and keyword tags

## Installation

Requires [rye](https://rye.astral.sh/).

```bash
rye sync
```

## Usage

### Web review UI

```bash
rye run autocropper
```

Opens a browser-based review interface. Select a folder of CR3 files, then step through each photo using keyboard shortcuts:

| Key | Action |
|-----|--------|
| `1` | Keep original (no crop) |
| `2` | Apply crop |

Before pressing `2`, the crop panel is draggable — click and drag to pan the crop window and fine-tune the framing. The adjusted position is saved to the XMP sidecar.

The header controls adjust the prefetch buffer size and crop margin in real time. The buffer indicator shows how many crops are ready (e.g. `4 / 10`); the second number is editable. Accepted crops are written as XMP sidecars immediately; photos already reviewed (XMP with `HasCrop=True`) are skipped on subsequent runs.

#### ML crop mode (web UI)

Upload a training dataset (`.pkl` file) using the **dataset** file picker in the header. Once loaded, toggle the **ML** button to switch between classic and ML crop modes. Switching modes restarts processing from the first unreviewed photo.

The buffer indicator shows **green** slots for ML crops and **blue** for classic margin crops. The crop panel label shows **Apply ML crop** or **Apply margin crop** accordingly.

### Batch CLI

```bash
# Process all CR3 files in the default folder (~/Desktop/Test)
rye run autocrop

# Process a specific folder
rye run autocrop /path/to/photos

# Use ML crop mode with a training dataset
rye run autocrop --dataset data/training.pkl /path/to/photos

# Include all detected people (default: main subject only)
rye run autocrop --all-people

# Re-crop files that already have crops
rye run autocrop --force
```

When `--dataset` is supplied, ML crop prediction is attempted for each image; photos where ML fails (no person, no face, insufficient neighbours) fall back to classic GDINO+SAM crop automatically.

## XMP Keywords

Every accepted crop writes the following keywords into the XMP sidecar:

| Keyword | When written |
|---------|-------------|
| `AutoCropper` | Always — marks every photo touched by this tool |
| `AutoCropper_ML` | When the k-NN ML prediction was used |
| `AutoCropper_Margin` | When the classic GDINO+SAM geometric crop was used |

Existing keywords in the XMP are preserved; new ones are merged in without duplication.

## ML Crop Mode

ML mode predicts crops using case-based reasoning (k-NN) over a training dataset of already-cropped photos.

### How it works

For each new photo:

1. **Person detection** — Grounding DINO locates the person; if multiple people are found, the largest bounding box is used
2. **Segmentation** — SAM 2.1 hiera-tiny produces a pixel-accurate binary mask of the person
3. **Pose estimation** — ViTPose extracts face keypoints (nose, eyes, ears); their centroid within the mask is recorded
4. **k-NN matching** — The query mask and face centroid are compared against all training records with a matching aspect ratio (within ±0.05)
5. **Prediction** — The crop centre and margin are averaged from the 10 nearest neighbours and converted to pixel coordinates

The similarity metric between two photos is:

```
distance = alpha × (1 − mask_IoU) + (1 − alpha) × face_centroid_distance
```

where `alpha` is stored in the dataset (default 0.5). If no face keypoints are detected, only mask IoU is used and all aspect-ratio-matching records are eligible.

### Building a training dataset

Collect a folder of already-cropped images. Both CR3 files (crop read from XMP sidecar) and raster images (JPEG, PNG, TIFF — full frame treated as the crop) are accepted.

```bash
rye run build-training-dataset ~/path/to/cropped/photos output.pkl
```

The script runs Grounding DINO + SAM 2.1 + ViTPose on each image and saves a `TrainingDataset` to `output.pkl`. Images where no person or face is detected are skipped.

### Optimising the alpha weight

```bash
rye run optimize-weights output.pkl
```

Runs leave-one-out cross-validation over `alpha ∈ [0.0, 0.1, …, 1.0]` and reports the value with the lowest mean crop-centre prediction error. The dataset can then be reloaded with the updated alpha.

### Training dataset notes

- **Gzip compression** — datasets are saved and loaded in gzip format automatically; older uncompressed `.pkl` files are still readable.
- **Mirror augmentation** — each record is horizontally mirrored at load time, doubling the effective dataset size without requiring extra source images.

## How Classic Crop Works

1. Grounding DINO detects the person bounding box
2. SAM 2.1 hiera-tiny refines the box to a pixel-accurate silhouette mask
3. The mask bounding box is used as the person envelope
4. A configurable margin is added (default 20%)
5. The crop is expanded so the subject fits within the Instagram 5:4 safe zone
6. Aspect ratio is enforced (original image ratio is preserved)
7. Zoom is capped so the crop covers at least 50% of the image width

## Diagnostic Tools

### diagnostic-knn

```bash
rye run diagnostic-knn <input_folder> <dataset.pkl> <output_folder> [--n N] [--cols C]
```

For each image, produces a composite JPEG showing:

- **Left panel** — query image with blue/red mask overlay, green face keypoints, yellow predicted crop box, and a cyan→yellow convergence trace showing how the predicted crop centre shifts as more neighbours are included
- **Right grid** — the *n* nearest training neighbours, each showing the source photo with blue/red overlay, a green face centroid dot, and a red crop centre dot; falls back to a blue silhouette when no source image is available

### diagnostic-masks

```bash
rye run diagnostic-masks <input_folder> <output_folder> [--n N] [--yolo-padding P]
```

Produces a five-panel JPEG per image comparing segmentation strategies (useful for evaluating detector quality):

| Panel | Strategy |
|-------|----------|
| 1 | GDINO bbox → SAM 2.1 (current pipeline) |
| 2 | YOLOv8-pose tight bbox → SAM 2.1 |
| 3 | YOLOv8-pose + symmetric padding → SAM 2.1 |
| 4 | YOLOv8-pose + keypoints union bbox → SAM 2.1 |
| 5 | YOLOv8-seg mask directly (no SAM) |

Overlay colours: blue = person mask, red = background, yellow = SAM prompt box, green = face keypoints. Per-component and full-pipeline timing is printed at the end.

## Configuration

Key constants in `src/autocropper/main.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `GDINO_MODEL` | `IDEA-Research/grounding-dino-tiny` | Person detection model |
| `SAM2_MODEL` | `facebook/sam2.1-hiera-tiny` | Segmentation model |
| `CONFIDENCE` | `0.3` | Detection confidence threshold |
| `MARGIN_RATIO` | `0.20` | Margin around detected subject (20%) |
| `INSTAGRAM_RATIO` | `5/4` | Safe-zone aspect ratio |
| `MAX_ZOOM` | `0.5` | Minimum crop width as fraction of image width |

Key constants in `src/autocropper/ml_crop.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_INFERENCE_SIZE` | `800` | Longest edge fed to models (px) |
| `MASK_SIZE` | `1024` | Stored mask resolution (px) |
| `KNN_COMPARE_SIZE` | `128` | Downsampled mask resolution for k-NN comparison |
| `AR_EPSILON` | `0.05` | Aspect ratio tolerance for candidate matching |
| `DEFAULT_N_NEIGHBORS` | `10` | Number of neighbours used for prediction |
| `FACE_KP_THRESHOLD` | `0.3` | Minimum ViTPose keypoint confidence |

## Requirements

- Python 3.11+
- rye
- torch, transformers, torchvision, accelerate
- timm, pillow, rawpy, flask, tqdm, scipy
- ultralytics (for `diagnostic-masks` only)
