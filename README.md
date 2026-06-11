# Autocropper

Review and crop CR3 RAW files using AI detection, then write Lightroom-compatible XMP sidecar files.

## Features

- **Web review UI**: Browse processed CR3 files side-by-side (original vs crop), accept or reject each with a keypress
- **Classic crop mode**: Geometric crop — detect person with Grounding DINO, refine with ViTPose keypoints, add configurable margin
- **ML crop mode**: Case-based reasoning — segment person with MobileSAM, match against a training dataset of previously-cropped photos using mask shape and face position similarity
- **XMP compatibility**: Writes Lightroom-compatible XMP sidecar files; preserves all existing metadata (ratings, colour corrections, keywords) and only updates crop tags

## Installation

Requires [rye](https://rye.astral.sh/).

```bash
rye sync
```

## Usage

### Web review UI

```bash
rye run autocrop-web
```

Opens a browser-based review interface. Select a folder of CR3 files, then step through each photo using keyboard shortcuts:

| Key | Action |
|-----|--------|
| `1` | Keep original (no crop) |
| `2` | Apply crop |

The header controls adjust the prefetch buffer size and crop margin in real time. Accepted crops are written as XMP sidecars immediately; photos already reviewed (XMP with `HasCrop=True`) are skipped on subsequent runs.

#### ML crop mode

Upload a training dataset (`.pkl` file) using the **dataset** file picker in the header. Once loaded, toggle the **ML** button to switch between classic and ML crop modes. Switching modes restarts processing from the first unreviewed photo.

The buffer indicator dots are **green** for ML crops and **blue** for classic margin crops. The crop panel label shows **Apply ML crop** or **Apply margin crop** accordingly.

### Batch CLI

```bash
# Process all CR3 files in the default folder (~/Desktop/Test)
rye run autocrop

# Process a specific folder
rye run autocrop /path/to/photos

# Include all detected people (default: main subject only)
rye run autocrop --all-people

# Re-crop files that already have crops
rye run autocrop --force
```

## ML Crop Mode

ML mode predicts crops using case-based reasoning (k-NN) over a training dataset of already-cropped photos.

### How it works

For each new photo:

1. **Person detection** — Grounding DINO locates the person; if multiple people are found, the largest bounding box is used
2. **Segmentation** — MobileSAM produces a pixel-accurate binary mask of the person
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

The script runs Grounding DINO + MobileSAM + ViTPose on each image and saves a `TrainingDataset` to `output.pkl`. Images where no person or face is detected are skipped.

### Optimising the alpha weight

```bash
rye run optimize-weights output.pkl
```

Runs leave-one-out cross-validation over `alpha ∈ [0.0, 0.1, …, 1.0]` and reports the value with the lowest mean crop-centre prediction error. The dataset can then be reloaded with the updated alpha.

## How Classic Crop Works

1. Grounding DINO detects person bounding boxes
2. ViTPose refines with pose keypoints
3. Keypoints and boxes are merged into a single bounding box
4. A configurable margin is added (default 20%)
5. The crop is expanded so the subject fits within the Instagram 5:4 safe zone
6. Aspect ratio is enforced (original image ratio is preserved)
7. Zoom is capped so the crop covers at least 50% of the image width

## Configuration

Key constants in `src/autocropper/main.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `GDINO_MODEL` | `IDEA-Research/grounding-dino-tiny` | Person detection model |
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

## XMP Metadata

- **If XMP exists**: Only the five crop tags (`HasCrop`, `CropLeft`, `CropTop`, `CropRight`, `CropBottom`) are updated; all other metadata is preserved
- **If XMP does not exist**: A new sidecar is created with crop data only
- Files with `HasCrop=True` are skipped by default in both the web UI and batch CLI

## Requirements

- Python 3.8+
- rye
- torch, transformers, torchvision
- mobile-sam (`pip install git+https://github.com/ChaoningZhang/MobileSAM.git`)
- timm (MobileSAM dependency)
- pillow, rawpy, flask, tqdm, scipy, accelerate
