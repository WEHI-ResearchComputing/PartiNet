---
sidebar_position: 2
---

# 2. Detect

PartiNet uses a modified version of an adaptive YOLO architecture called *DynamicDet* (Lin et al., 2023) to identify particles in cryo-EM micrographs. This stage provides highly accurate particle detection with customizable confidence and overlap thresholds.


## Quick Start

```shell title="Local Installation"
partinet detect \
    --weight /path/to/downloaded/model_weights.pt \
    --source /data/partinet_picking/denoised \
    --device 0,1,2,3 \
    --project /data/partinet_picking
```
```shell title="Apptainer/Singularity"
apptainer exec --nv --no-home \
    -B /data oras://ghcr.io/wehi-researchcomputing/partinet:main-singularity partinet detect \
    --weight /path/to/downloaded/model_weights.pt \
    --source /data/partinet_picking/denoised \
    --device 0,1,2,3 \
    --project /data/partinet_picking
```

```shell title="Docker"
docker run --gpus all -v /data:/data \
    ghcr.io/wehi-researchcomputing/partinet:main partinet detect \
    --weight /path/to/downloaded/model_weights.pt \
    --source /data/partinet_picking/denoised \
    --device 0,1,2,3 \
    --project /data/partinet_picking
```

## Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--weight` | Path to pre-trained model weights file | `/data/models/partinet_yolov7_weights.pt` |
| `--source` | Path to directory containing micrographs to process | `/data/partinet_picking/denoised` |
| `--project` | Path to project directory where outputs will be saved | `/data/partinet_picking` |

### Customizable Parameters

| Parameter | Type | Description | Default | Notes |
|-----------|------|-------------|---------|--------|
| `--conf-thres` | float | Confidence threshold for detection (0.0-1.0) | `0.1` | Lower = more detections, higher = fewer but more confident |
| `--iou-thres` | float | IoU threshold for non-maximum suppression (0.0-1.0) | `0.2` | Higher = more aggressive overlap removal |
| `--device` | string | GPU devices to use (comma-separated) | `0,1,2,3` | Use `0` for single GPU, `0,1` for two GPUs, etc. Leave empty for CPU only |


## Parameter Configuration Guide

### Confidence Threshold (`--conf-thres`)

The confidence threshold determines the minimum confidence score required for a detection to be considered valid. Even with denoised micrographs, overall confidence will be low due to low SNR. It is not recommended to increase this above 0.5. We recommend setting this threshold quite low `0.0-0.3` during this stage, and then filtering during STAR file generation.

**Recommended starting values:**
- **New datasets, small proteins, low defocus**: `0.0-0.2`
- **Large proteins/complexes, higher defocus**: `0.2-0.5`

### IoU Threshold (`--iou-thres`)

The Intersection over Union (IoU) threshold controls how aggressively overlapping detections are removed. The higher the value, the more that overlapping particles for removed

**Recommended starting value**: `0.2`


## Model Weights

### Pre-trained Weights
PartiNet model weights are available on [HuggingFace](https://huggingface.co/MihinP/PartiNet).
Weights can be downloaded through the browser or through CLI via Git LFS

```shell
# Verify Git LFS is installed
git lfs --help
mkdir PartiNet_weights
cd PartiNet_weights
git clone git@hf.co:MihinP/PartiNet
```

You will see two `.pt` files available: `denoised_micrographs.pt` and `raw_micrographs.pt`.

### Custom Model Weights

If you have [trained your own PartiNet model](../../docs/category/training) you may use custom weights:

```shell
partinet detect \
    --backbone-detector yolov7-w6 \
    --weight /path/to/your/custom_weights.pt \
    --source /data/partinet_picking/denoised \
    --conf-thres 0.4 \
    --iou-thres 0.2 \
    --device 0,1 \
    --project /data/partinet_picking
```

## Input Requirements

### Supported Formats

**Standard workflow (recommended):**

- **PNG files** (`.png`) from the Denoise stage — use with `denoised_micrographs.pt`

**Skip-denoise workflow:**

- **MRC files** (`.mrc`) motion-corrected micrographs — use with `raw_micrographs.pt`
- Raw MRC inputs are per-micrograph min–max normalised to 8-bit before inference (not the same as EMAN2 or CryoSegNet display)

### Skip denoise (raw MRC)

If you want to pick directly on motion-corrected micrographs without denoising:

```shell
partinet detect \
    --weight /path/to/raw_micrographs.pt \
    --source /data/partinet_picking/motion_corrected \
    --project /data/partinet_picking \
    --device 0
```

Use `raw_micrographs.pt` only with raw MRC inputs. For denoised PNG, use `denoised_micrographs.pt`.

When generating a STAR file after skip-denoise, set `--images` to the **same folder** used as detect `--source` (e.g. `motion_corrected/`, not `denoised/`).

### Directory Structure

Your source directory should contain micrographs:

```
denoised/
├── micrograph_001.png
├── micrograph_002.png
├── micrograph_003.png
└── ...
```

## Output

### Directory Structure

After detection, your project directory will contain:

```
partinet_picking/
├── motion_corrected/          # 📁 Your input micrographs
│   ├── micrograph1.mrc
│   ├── micrograph2.mrc
│   └── ...
├── denoised/                  # 🧹 Created by denoise stage
│   ├── micrograph1.png
│   ├── micrograph2.png
│   └── ...
├── exp/                       # 🎯 Created by detect stage
│   ├── labels/               # 📋 Detection coordinates
│   │   ├── micrograph1.txt
│   │   ├── micrograph2.txt
│   │   └── ...
│   ├── micrograph1.png       # 🖼️ Micrographs with detections drawn
│   ├── micrograph2.png
│   └── ...
└── partinet_detect.log
```

### Log File Output

The `partinet_detect.log` file provides detailed processing information:

```log
2025-09-10 16:08:07 | INFO | Using devices: ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
2025-09-10 16:08:11 | INFO | Model loaded on cuda:0
2025-09-10 16:08:15 | INFO | Model loaded on cuda:1
2025-09-10 16:08:17 | INFO | Model loaded on cuda:2
2025-09-10 16:08:20 | INFO | Model loaded on cuda:3
2025-09-10 16:08:21 | INFO | [cuda:0] Processing 000000633063778279389_FoilHole_30903560_Data_30900510_30900512_20201031_010810_fractions_patch_aligned.png
2025-09-10 16:08:21 | INFO | [cuda:1] Processing 000005251007454303721_FoilHole_30909875_Data_30900510_30900512_20201101_040227_fractions_patch_aligned.png
2025-09-10 16:08:21 | INFO | [cuda:2] Processing 000012049843629116854_FoilHole_30902824_Data_30900510_30900512_20201030_230632_fractions_patch_aligned.png
2025-09-10 16:08:22 | INFO | [cuda:3] Processing 000018235333323367981_FoilHole_30905009_Data_30900510_30900512_20201031_082713_fractions_patch_aligned.png
2025-09-10 16:08:24 | INFO | [cuda:1] 000005251007454303721_FoilHole_30909875_Data_30900510_30900512_20201101_040227_fractions_patch_aligned.png: 94 particles, 3516.2 ms
2025-09-10 16:08:24 | INFO | [cuda:1] Processing 000022392129605454366_FoilHole_30903097_Data_30900534_30900536_20201031_002940_fractions_patch_aligned.png
2025-09-10 16:08:25 | INFO | [cuda:2] 000012049843629116854_FoilHole_30902824_Data_30900510_30900512_20201030_230632_fractions_patch_aligned.png: 272 particles, 3604.4 ms
```


### Output File Formats

#### Detection Coordinates (`labels/*.txt`)

Each txt file corresponds to a single micrograph. Each line represents one detection in YOLO format:
```
class_id x_center y_center width height confidence
0 0.5234 0.3456 0.0234 0.0198 0.8567
```

Where coordinates are normalized (0.0-1.0) relative to micrograph dimensions.

## What's Next
- [STAR](/docs/stages/star)

Congratulations! You have picked particles from your micrographs, you can now move to the final stage with [STAR](/docs/stages/star).

## References
Lin, Z., Wang, Y., Zhang, J., & Chu, X. (2023). DynamicDet: A Unified Dynamic Architecture for Object Detection (arXiv:2304.05552). arXiv. https://doi.org/10.48550/arXiv.2304.05552


## See Also

- [Denoise](/docs/stages/denoise)
- [STAR](/docs/stages/star)

