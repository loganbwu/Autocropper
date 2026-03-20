# Autocropper

Auto-crop CR3 RAW files using AI pose detection and write Lightroom-compatible XMP crop metadata.

## Features

- **AI-Powered Detection**: Uses RT-DETR for person detection and ViTPose for keypoint estimation (via HuggingFace)
- **Smart Cropping**: Automatically calculates optimal crop based on detected keypoints and bounding boxes
- **Aspect Ratio Preservation**: Maintains original image aspect ratio in crops
- **XMP Metadata Preservation**: Reads existing XMP sidecar files and only updates cropping tags, preserving all other metadata (e.g., colour corrections, ratings, keywords)
- **Lightroom Compatible**: Generates XMP files that work seamlessly with Adobe Lightroom

## Installation

Requires [rye](https://rye.astral.sh/) and [exiftool](https://exiftool.org/).

```bash
rye sync
```

The `autocrop` command is then available via:

```bash
rye run autocrop
```

Or install globally:

```bash
rye install .
```

## Usage

```bash
# Process all CR3 files in the default location (~/Desktop/Test)
autocrop

# Process CR3 files in a specific directory
autocrop /path/to/photos

# Crop to include all detected people instead of just the main subject
autocrop --all-people
autocrop -a /path/to/photos

# Force re-crop even if files already have crops
autocrop --force
autocrop -f /path/to/photos
```

## How It Works

1. Extracts preview JPEG from CR3 files using exiftool
2. Runs RT-DETR person detection to locate subjects
3. Runs ViTPose pose estimation to refine keypoint locations
4. Calculates a merged bounding box around all detected people/keypoints
5. Adds a configurable margin (default 10%) around the detection
6. Enforces original aspect ratio while keeping all subjects in frame
7. Writes or updates XMP sidecar file with crop metadata

## Smart Skip & XMP Preservation

### Skip Existing Crops (Default Behaviour)

By default, files that already have crop data in their XMP sidecar are skipped. This prevents accidentally overwriting manual crops made in Lightroom. Use `--force` to override.

### XMP Metadata Preservation

- **If XMP exists**: Parses the existing file and only updates the 5 crop-related tags (`HasCrop`, `CropLeft`, `CropTop`, `CropRight`, `CropBottom`)
- **If XMP doesn't exist**: Creates a new XMP file with crop data
- All other metadata (temperature, exposure, keywords, ratings, etc.) is left untouched

## Configuration

Edit `src/autocropper/main.py` to adjust:

- `DETECTOR_MODEL`: HuggingFace model ID for person detection (default: `PekingU/rtdetr_r50vd_coco_o365`)
- `POSE_MODEL`: HuggingFace model ID for pose estimation (default: `usyd-community/vitpose-base-simple`)
- `CONFIDENCE`: Person detection confidence threshold (default: `0.3`)
- `KEYPOINT_SCORE`: Minimum keypoint confidence to include (default: `0.3`)
- `MARGIN_RATIO`: Margin around detected subjects (default: `0.10` = 10%)
- `DEFAULT_ROOT`: Default directory to process (default: `~/Desktop/Test`)

## Requirements

- Python 3.8+
- exiftool
- torch
- transformers
- scipy
- pillow
- tqdm
