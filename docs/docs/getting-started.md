---
sidebar_position: 3
---

# Getting Started

This guide walks you through your first PartiNet analysis using the three-stage pipeline. We'll process cryo-EM micrographs from start to finish.

## Prerequisites

Before starting, ensure you have:
- PartiNet installed (see [Installation](installation.md))
- Motion-corrected micrographs in a source directory
- A project directory where outputs will be saved
- GPU access for optimal performance

## Directory Structure

PartiNet expects and creates the following directory structure:

```
project_directory/
├── motion_corrected/          # 📁 Your input micrographs
│   ├── micrograph1.mrc
│   ├── micrograph2.mrc
│   └── ...
├── denoised/                  # 🧹 Created by denoise stage
│   ├── micrograph1.mrc
│   ├── micrograph2.mrc
│   └── ...
├── exp/                       # 🎯 Created by detect stage
│   ├── labels/               # 📋 Detection coordinates
│   │   ├── micrograph1.txt
│   │   ├── micrograph2.txt
│   │   └── ...
│   ├── micrograph1.png    # 🖼️ Micrographs with detections drawn
│   ├── micrograph2.
│   └── ...
└── partinet_particles.star   # ⭐ Final STAR file (created by star stage)
```

**Pipeline Flow:**
1. **Input** → `motion_corrected/` (your micrographs)
2. **Stage 1** → `denoised/` (cleaned micrographs)
3. **Stage 2** → `exp*/` (detections + visualizations)
4. **Stage 3** → `*.star` (final particle coordinates)

## Stage 1: Denoise

The first stage removes noise from your micrographs and improves signal-to-noise ratios:

<div class="container-tabs">

```shell title="Local Installation"
partinet denoise \
    --source /data/my_project/motion_corrected \
    --project /data/my_project
```

</div>

**What this does:**
- Reads micrographs from `motion_corrected/` directory
- Applies denoising algorithms
- Saves cleaned micrographs to `denoised/` directory in your project folder

## Stage 2: Detect

The detection stage identifies particles in your denoised micrographs:

<div class="container-tabs">

```shell title="Local Installation"
partinet detect \
    --weight /path/to/downloaded/model_weights.pt \
    --source /data/partinet_picking/denoised \
    --device 0,1,2,3 \
    --project /data/partinet_picking
```

</div>

**What this creates:**
- `exp/` directory in your project folder
- `exp/labels/` directory containing detection coordinates for each micrograph
- Micrographs with detection boxes drawn on top (saved in `exp/`)

**Key parameters:**
- `--backbone-detector`: Neural network architecture to use
- `--weight`: Path to trained model weights
- `--conf-thres`: Confidence threshold for detections (0.0 = accept all)
- `--iou-thres`: Intersection over Union threshold for filtering overlapping detections
- `--device`: GPU devices to use (0,1,2,3 = use 4 GPUs)

## Stage 3: Star

The final stage converts detections to STAR format and applies confidence filtering:

<div class="container-tabs">

```shell title="Local Installation"
partinet star \
    --labels /data/my_project/exp/labels \
    --images /data/my_project/denoised \
    --output /data/my_project/partinet_particles.star \
    --conf 0.1
```

</div>

**What this does:**
- Reads detection labels from `exp/labels/`
- Filters particles based on confidence threshold (0.1 in this example)
- Creates a STAR file ready for further processing in RELION or other software

## Output Files

After running all three stages, you'll have:

1. **Denoised micrographs** (`denoised/`) - Cleaned input for particle detection
2. **Detection visualizations** (`exp/*.mrc`) - Micrographs with particle boxes drawn
3. **Detection coordinates** (`exp/labels/*.txt`) - Raw detection data
4. **STAR file** (`*.star`) - Final particle coordinates ready for downstream processing


## Next Steps

- Learn more about individual stages: [Denoise](stages/denoise.md), [Detect](stages/detect.md), [STAR](stages/star.md)

## Troubleshooting

If you encounter issues:
- Ensure all paths exist and are accessible
- Check GPU availability with `nvidia-smi`
- Verify container mounting with `-B` flags includes all necessary paths