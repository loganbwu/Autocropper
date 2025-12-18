# Autocropper

Auto-crop CR3 RAW files using YOLO pose detection and write Lightroom-compatible XMP crop metadata.

## Features

- **AI-Powered Detection**: Uses YOLO11 pose detection to identify people in images
- **Smart Cropping**: Automatically calculates optimal crop based on detected keypoints and bounding boxes
- **Aspect Ratio Preservation**: Maintains original image aspect ratio in crops
- **XMP Metadata Preservation**: Reads existing XMP sidecar files and only updates cropping tags, preserving all other metadata (e.g., color corrections, ratings, keywords)
- **Lightroom Compatible**: Generates XMP files that work seamlessly with Adobe Lightroom

## Usage

```bash
# Process all CR3 files in default location (~/Pictures)
python auto_crop_cr3.py

# Process CR3 files in a specific directory
python auto_crop_cr3.py /path/to/photos
```

## How It Works

1. Extracts preview JPEG from CR3 files using exiftool
2. Runs YOLO pose detection to identify people and keypoints
3. Calculates a merged bounding box around all detected people/keypoints
4. Adds configurable margin (default 30%) around the detection
5. Enforces original aspect ratio while maintaining all subjects in frame
6. Writes or updates XMP sidecar file with crop metadata

## XMP Preservation (Added: 2025-12-19)

The script now intelligently handles existing XMP files:
- **If XMP exists**: Parses the existing file and only updates the 5 crop-related tags (`HasCrop`, `CropLeft`, `CropTop`, `CropRight`, `CropBottom`)
- **If XMP doesn't exist**: Creates a new XMP file with crop data
- **All other metadata preserved**: Temperature, exposure, contrast, keywords, ratings, etc. remain untouched

This ensures you can run auto-cropping on photos that have already been edited in Lightroom without losing your adjustments.

## Configuration

Edit `auto_crop_cr3.py` to adjust:
- `MODEL_NAME`: YOLO model to use (default: `yolo11n-pose.pt`)
- `CONFIDENCE`: Detection confidence threshold (default: 0.1)
- `MARGIN_RATIO`: Margin around detected subjects (default: 0.30 = 30%)
- `DEFAULT_ROOT`: Default directory to process (default: ~/Pictures)

## Testing

Run the test suite to verify XMP preservation functionality:

```bash
python test_xmp_preservation.py
```

## Requirements

- Python 3.12+
- exiftool
- OpenCV (cv2)
- ultralytics (YOLO)
- numpy

## Notes

- Only processes files with `.cr3` extension (case-insensitive)
- Skips images where no people are detected
- XMP files are written in the same directory as the CR3 files
- Uses XMP format compatible with Adobe Camera Raw/Lightroom
