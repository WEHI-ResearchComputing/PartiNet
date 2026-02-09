# PartiNet – Documentation

**Version:** 0.2  
**Container Location:** `/stornext/Projects/cryoEM/cryoEM_data/lab_shakeel/perera.m/PartiNet/PartiNet_v0.2.sif`  
**Model Weights:** `/stornext/Projects/cryoEM/cryoEM_data/lab_shakeel/perera.m/PartiNet/denoised_micrographs.pt`

---

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Directory Structure](#directory-structure)
  - [Stage 1: Denoise](#stage-1-denoise)
  - [Stage 2: Detect](#stage-2-detect)
  - [Stage 3: Star](#stage-3-star)
  - [Output Files](#output-files)
- [Detailed Stage Documentation](#detailed-stage-documentation)
  - [Denoise Stage](#denoise-stage)
  - [Detect Stage](#detect-stage)
  - [Star Stage](#star-stage)
- [Training](#training)
  - [Split Training Data](#split-training-data)
  - [Train Dual Detectors (Step 1)](#train-dual-detectors-step-1)
  - [Train Adaptive Router (Step 2)](#train-adaptive-router-step-2)
  - [Training Output Reference](#training-output-reference)

---

## Introduction

PartiNet is a powerful command-line tool for particle picking on cryo-EM micrographs. It provides a comprehensive three-stage pipeline designed to clean, identify, and prepare particles from experimental data for subsequent processing.

### The Three-Stage Pipeline

**Stage 1: Denoise** – Removes noise and artifacts from raw data using fast heuristic denoising algorithms, improving signal-to-noise ratios.

**Stage 2: Detect** – Identifies and locates individual particles within cleaned data using a dynamic adaptive architecture.

**Stage 3: Star** – Prepares particle data for further processing and provides reports on particle populations in your dataset.

### Key Features

- **Fast picking** – Leverages state-of-the-art dynamic deep learning models
- **Accurate picking** – Accurately identifies proteins and filters junk
- **Overcome orientation bias** – Identifies rare views of proteins
- **Multi-species identification** – Handles heterogeneous samples without prior box size estimation
- **Batch processing** – Efficient parallel processing capabilities

---

## Getting Started

This guide walks you through your first PartiNet analysis using the three-stage pipeline.

### Prerequisites

Before starting, ensure you have:
- Motion-corrected micrographs in a source directory
- A project directory where outputs will be saved
- GPU access for optimal performance
- Access to the PartiNet container and model weights

**Load the Apptainer module:**
```shell
module load apptainer
```

**Set up environment variables (recommended):**
```shell
export PARTINET_SIF="/stornext/Projects/cryoEM/cryoEM_data/lab_shakeel/perera.m/PartiNet/PartiNet_v0.2.sif"
export PARTINET_WEIGHTS="/stornext/Projects/cryoEM/cryoEM_data/lab_shakeel/perera.m/PartiNet/denoised_micrographs.pt"
```

### Directory Structure

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
│   ├── micrograph2.png
│   └── ...
└── partinet_particles.star   # ⭐ Final STAR file (created by star stage)
```

**Pipeline Flow:**
1. **Input** → `motion_corrected/` (your micrographs)
2. **Stage 1** → `denoised/` (cleaned micrographs)
3. **Stage 2** → `exp*/` (detections + visualizations)
4. **Stage 3** → `*.star` (final particle coordinates)

### Stage 1: Denoise

The first stage removes noise from your micrographs and improves signal-to-noise ratios:

```shell
module load apptainer

apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet denoise \
    --source /path/to/my_project/motion_corrected \
    --project /path/to/my_project
```

**What this does:**
- Reads micrographs from `motion_corrected/` directory
- Applies denoising algorithms
- Saves cleaned micrographs to `denoised/` directory in your project folder

### Stage 2: Detect

The detection stage identifies particles in your denoised micrographs:

```shell
apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet detect \
    --weight $PARTINET_WEIGHTS \
    --source /path/to/my_project/denoised \
    --device 0,1,2,3 \
    --project /path/to/my_project
```

**What this creates:**
- `exp/` directory in your project folder
- `exp/labels/` directory containing detection coordinates for each micrograph
- Micrographs with detection boxes drawn on top (saved in `exp/`)

**Key parameters:**
- `--weight`: Path to trained model weights
- `--conf-thres`: Confidence threshold for detections (0.0 = accept all, default: 0.1)
- `--iou-thres`: Intersection over Union threshold for filtering overlapping detections (default: 0.2)
- `--device`: GPU devices to use (0,1,2,3 = use 4 GPUs)

### Stage 3: Star

The final stage converts detections to STAR format and applies confidence filtering:

```shell
apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet star \
    --labels /path/to/my_project/exp/labels \
    --images /path/to/my_project/denoised \
    --output /path/to/my_project/partinet_particles.star \
    --conf 0.1
```

**What this does:**
- Reads detection labels from `exp/labels/`
- Filters particles based on confidence threshold (0.1 in this example)
- Creates a STAR file ready for further processing in RELION or other software

### Output Files

After running all three stages, you'll have:

1. **Denoised micrographs** (`denoised/`) – Cleaned input for particle detection
2. **Detection visualizations** (`exp/*.png`) – Micrographs with particle boxes drawn
3. **Detection coordinates** (`exp/labels/*.txt`) – Raw detection data
4. **STAR file** (`*.star`) – Final particle coordinates ready for downstream processing

---

## Detailed Stage Documentation

### Denoise Stage

Denoising can vastly improve particle picking by helping to increase signal to noise in low-dose micrographs. PartiNet implements a modified heuristic Wiener filter denoiser based on the method from CryoSegNet (Gyawali et al., 2024). PartiNet's implementation introduces multiprocessing, allowing for high-throughput denoising of large datasets.

#### Parameters

**Required Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--source` | Directory containing motion-corrected micrographs in MRC format | `/data/my_project/motion_corrected` |
| `--project` | Parent project directory where all outputs will be saved | `/data/my_project` |

**Optional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_workers` | int | max available CPUs | Number of CPU workers for processing |
| `--img_format` | string | `png` | Output format for denoised images (`png`, `jpg`, `mrc`) |

#### Input Requirements

Your motion-corrected micrographs should ideally meet these criteria:

- **Format**: single-slice MRC files from RELION or CryoSPARC
- **Motion correction**: Total full frame motion should be **less than 100 pixels**
- **CTF estimation**: CTF fit resolution should be **less than 10 Angstroms**
- **Convergence**: Motion correction and CTF estimation should have converged appropriately

#### Setup Instructions

**1. Create Project Directory:**
```shell
mkdir my_project
cd my_project
mkdir motion_corrected
```

**2. Transfer Motion-Corrected Micrographs:**

From CryoSPARC:
```shell
# Using symbolic links (faster, saves space)
ln -s /path/to/cryosparc/project/job_number/motioncorrected/*_fractions_patch_aligned.mrc motion_corrected/
```

From RELION:
```shell
# Link motion-corrected micrographs
ln -s /path/to/relion/project/MotionCorr/job_number/Micrographs/*.mrc motion_corrected/
```

**3. Run Denoising:**
```shell
module load apptainer

apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet denoise \
    --source /path/to/my_project/motion_corrected \
    --project /path/to/my_project
```

#### Output

After denoising, your project directory will contain:

```
my_project/
├── motion_corrected/
│   └── [original MRC files]
├── denoised/
│   ├── micrograph_001_fractions_patch_aligned.png
│   ├── micrograph_002_fractions_patch_aligned.png
│   └── ...
└── partinet_denoise.log
```

#### Advanced Usage

**Custom CPU Configuration:**

PartiNet automatically optimizes CPU usage. To manually set workers:
```shell
apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet denoise \
    --source /path/to/motion_corrected \
    --project /path/to/project \
    --num_workers 16
```

**Different Output Formats:**

By default PartiNet outputs denoised images in `png` format. To use MRC format:
```shell
apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet denoise \
    --source /path/to/motion_corrected \
    --project /path/to/project \
    --img_format mrc
```

---

### Detect Stage

PartiNet uses a modified version of an adaptive YOLO architecture called *DynamicDet* (Lin et al., 2023) to identify particles in cryo-EM micrographs. This stage provides highly accurate particle detection with customizable confidence and overlap thresholds.

#### Quick Start

```shell
module load apptainer

apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet detect \
    --weight $PARTINET_WEIGHTS \
    --source /path/to/my_project/denoised \
    --device 0,1,2,3 \
    --project /path/to/my_project
```

#### Parameters

**Required Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--weight` | Path to pre-trained model weights file | `$PARTINET_WEIGHTS` |
| `--source` | Path to directory containing micrographs to process | `/path/to/my_project/denoised` |
| `--project` | Path to project directory where outputs will be saved | `/path/to/my_project` |

**Customizable Parameters:**

| Parameter | Type | Description | Default | Notes |
|-----------|------|-------------|---------|--------|
| `--conf-thres` | float | Confidence threshold for detection (0.0-1.0) | `0.1` | Lower = more detections, higher = fewer but more confident |
| `--iou-thres` | float | IoU threshold for non-maximum suppression (0.0-1.0) | `0.2` | Higher = more aggressive overlap removal |
| `--device` | string | GPU devices to use (comma-separated) | `0,1,2,3` | Use `0` for single GPU, `0,1` for two GPUs, etc. |

#### Parameter Configuration Guide

**Confidence Threshold (`--conf-thres`):**

The confidence threshold determines the minimum confidence score required for a detection to be considered valid. It is recommended to set this low (0.0-0.3) during detection and then filter during STAR generation.

- **0.0**: Accept all detections (maximum recall)
- **0.1**: Balanced approach (recommended starting point)
- **0.3**: Higher precision, may miss some particles

**IoU Threshold (`--iou-thres`):**

Recommended starting value: `0.2`

Controls how aggressively overlapping detections are filtered. Higher values allow more overlapping boxes.

#### Output

After detection, your project directory will contain:

```
my_project/
├── denoised/
├── exp/
│   ├── labels/
│   │   ├── micrograph_001.txt
│   │   ├── micrograph_002.txt
│   │   └── ...
│   ├── micrograph_001.png
│   ├── micrograph_002.png
│   └── ...
└── partinet_detect.log
```

The `labels/` directory contains YOLO-format text files with detection coordinates.

---

### Star Stage

The `partinet star` command is the final step in your particle picking pipeline. It converts the particle coordinates detected in the previous stage into a standardized STAR file format that can be used with downstream cryo-EM processing software like RELION, cryoSPARC, or other reconstruction programs.

#### Quick Start

```shell
module load apptainer

apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet star \
    --labels /path/to/my_project/exp/labels \
    --images /path/to/my_project/denoised \
    --output /path/to/my_project/partinet_particles.star \
    --conf 0.1
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `--labels` | Path | Yes | Directory containing the particle coordinate files (`.txt` format) from the detection stage |
| `--images` | Path | Yes | Directory containing the denoised micrographs corresponding to the labels |
| `--output` | Path | Yes | Output path for the generated STAR file |
| `--conf` | Float | Yes | Confidence threshold for filtering particle coordinates (0.0-1.0) |

#### Input Requirements

At this stage the project directory should contain:
- `motion_corrected/` – Original micrographs
- `denoised/` – Denoised micrographs
- `exp/labels/` – Detection coordinate files
- `partinet_detect.log` – Detection stage log
- `partinet_denoise.log` – Denoise stage log

#### Confidence Threshold

Typical range: 0.1-0.3. Choose based on dataset quality and downstream needs.

- **0.1**: Inclusive picking (more particles, some false positives)
- **0.2**: Balanced approach
- **0.3**: Conservative picking (fewer false positives, may miss some particles)

#### Output

The command generates a STAR file (`partinet_particles.star`) containing:
- Particle coordinates (X, Y positions)
- Confidence scores
- Micrograph names
- Image dimensions

This STAR file can be directly imported into RELION or cryoSPARC for particle extraction and further processing.

---

## Training

### Split Training Data

The split command organizes your annotated particle data into training and validation sets, preparing it for PartiNet model training. This step can either convert STAR files from manual picking sessions directly to YOLO format, or split existing YOLO labels into organized train/val directories.

#### Quick Start

```shell
module load apptainer

apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet split \
    --star /path/to/my_project/particles.star \
    --images /path/to/my_project/denoised \
    --output /path/to/my_project/training_data
```

#### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--star` | Path to input STAR file from picking | `/path/to/my_project/particles.star` |
| `--images` | Directory containing the micrograph images | `/path/to/my_project/denoised` |
| `--output` | Output directory for organized train/val data | `/path/to/my_project/training_data` |

#### Output

After splitting, training data will be organized as:

```
training_data/
├── images/
│   ├── train/
│   │   ├── micrograph_001.png
│   │   └── ...
│   └── val/
│       ├── micrograph_050.png
│       └── ...
├── labels/
│   ├── train/
│   │   ├── micrograph_001.txt
│   │   └── ...
│   └── val/
│       ├── micrograph_050.txt
│       └── ...
├── train.txt
├── val.txt
└── cryo_training.yaml
```

---

### Train Dual Detectors (Step 1)

PartiNet's architecture requires a two-step training regime. Step 1 trains the dual detectors.

#### Quick Start

```shell
module load apptainer

apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet train step1 \
    --weight $PARTINET_WEIGHTS \
    --data /path/to/cryo_training.yaml \
    --project /path/to/training_output_step1
```

#### Parameters

**Required:**
- `--weight`: Path to pre-trained weights
- `--data`: Path to training configuration YAML file
- `--project`: Output directory for training results

**Optional:**
- `--workers`: Number of data loading workers (default: 8)
- `--device`: GPU devices to use (default: 0,1,2,3)
- `--batch`: Batch size (default: 16)
- `--epochs`: Number of training epochs (default: 300)

#### Example with Custom Parameters

```shell
apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet train step1 \
    --weight $PARTINET_WEIGHTS \
    --data /path/to/cryo_training.yaml \
    --project /path/to/training_step1 \
    --batch 32 \
    --epochs 200 \
    --device 0,1
```

---

### Train Adaptive Router (Step 2)

Step 2 trains the adaptive router. The `--weight` parameter must point to a Step 1 checkpoint file (e.g., `last.pt`).

#### Quick Start

```shell
module load apptainer

apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet train step2 \
    --weight /path/to/training_step1/exp/weights/last.pt \
    --data /path/to/cryo_training.yaml \
    --project /path/to/training_output_step2 \
    --epochs 10
```

#### Parameters

**Required:**
- `--weight`: Path to Step 1 checkpoint file (typically `last.pt`)
- `--data`: Path to training configuration YAML file
- `--project`: Output directory for training results

**Optional:**
- `--epochs`: Number of training epochs (default: 10, recommended for Step 2)
- `--workers`: Number of data loading workers
- `--device`: GPU devices to use
- `--batch`: Batch size

---

### Training Output Reference

This section describes the output generated during PartiNet training (both Step 1 and Step 2).

#### Directory Structure

A completed training run produces an `exp*/` folder with:

```
exp/
├── cfg.yaml              # Model configuration
├── hyp.yaml              # Hyperparameters used
├── opt.yaml              # Training options
├── LR.png                # Learning rate schedule
├── results.png           # Training metrics plots
├── results.txt           # Training metrics (text format)
├── confusion_matrix.png  # Confusion matrix
├── F1_curve.png          # F1 score curve
├── P_curve.png           # Precision curve
├── R_curve.png           # Recall curve
├── PR_curve.png          # Precision-Recall curve
├── train_batch*.jpg      # Training batch visualizations
├── test_batch*_labels.jpg    # Validation labels
├── test_batch*_pred.jpg      # Validation predictions
├── events.out.tfevents.*     # TensorBoard logs
└── weights/
    ├── best.pt           # Best model checkpoint
    ├── last.pt           # Last epoch checkpoint
    └── epoch_*.pt        # Periodic checkpoints
```

#### Monitoring Training

**Using TensorBoard:**

```shell
module load apptainer

apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    tensorboard --logdir /path/to/your_project_folder
```

Then open `http://localhost:6006` in your web browser.

#### Resuming Training

To resume training from a checkpoint, use the `last.pt` file:

```shell
apptainer exec --nv --no-home \
    -B /vast,/stornext \
    $PARTINET_SIF \
    partinet train step1 \
    --weight /path/to/training_step1/exp/weights/last.pt \
    --data /path/to/cryo_training.yaml \
    --project /path/to/training_step1_resumed
```

---

## References

- Bepler, T. et al. (2020). Topaz-Denoise: general deep denoising models for cryoEM and cryoET. *Nature Communications*.
- Wagner, T. & Raunser, S. (2020). JANNI: Neural Network Filtering for CryoEM Data. *Communications Biology*.
- Gyawali, P. K. et al. (2024). CryoSegNet: automatic instance segmentation of macromolecules from cryo-EM micrographs.
- Lin, T. et al. (2023). DynamicDet: A Unified Dynamic Architecture for Object Detection. *CVPR 2023*.

---

**Document Version:** 1.0  
**Last Updated:** October 2025
