---
sidebar_position: 1
---

# 1. Split Training Data

The split command organizes your annotated particle data into training and validation sets, preparing it for PartiNet model training. This step can either convert STAR files from manual picking sessions directly to YOLO format, or split existing YOLO labels into organized train/val directories.

## Quick Start

<div class="container-tabs">

```shell title="Apptainer/Singularity"
apptainer exec --nv --no-home \
    -B /data oras://ghcr.io/wehi-researchcomputing/partinet:main-singularity partinet split \
    --star /data/partinet_picking/particles.star \
    --images /data/partinet_picking/denoised \
    --output /data/partinet_picking/training_data
```

```shell title="Docker"
docker run --gpus all -v /data:/data \
    ghcr.io/wehi-researchcomputing/partinet:main partinet split \
    --star /data/partinet_picking/particles.star \
    --images /data/partinet_picking/denoised \
    --output /data/partinet_picking/training_data
```

```shell title="Local Installation"
partinet split \
    --star /data/partinet_picking/particles.star \
    --images /data/partinet_picking/denoised \
    --output /data/partinet_picking/training_data
```

</div>

## Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--star` | Path to input STAR file from picking, or labels directory if using `--split-only` | `/data/partinet_picking/particles.star` |
| `--images` | Directory containing the micrograph images | `/data/partinet_picking/denoised` |
| `--output` | Output directory for organized train/val data | `/data/partinet_picking/training_data` |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--class-id` | int | `0` | Class ID to assign to all particles (for multi-class training) |
| `--test-size` | float | `0.25` | Proportion of dataset to use for validation (0.0-1.0) |
| `--split-only` | flag | `False` | Skip STAR conversion and only split existing YOLO labels |

## Input Requirements

### STAR Files from Manual Picking

If converting from STAR format, your STAR file should contain:

- **Required columns**: `_rlnMicrographName`, `_rlnCoordinateX`, `_rlnCoordinateY`, `_rlnDiameter`
- **Format**: Standard RELION STAR file format
- **Source**: Picking in RELION

If you are importing coordinates from CryoSPARC, we recommend using the `pyem` Python library (https://github.com/asarnow/pyem) to convert your `particles.cs` coordinates to STAR format. 

**For STAR conversion:**
```
partinet_picking/
├── particles.star
└── denoised/
    ├── micrograph_001.png
    ├── micrograph_002.png
    └── ...
```

### YOLO Labels (Split-Only Mode)
If you already have YOLO format labels and only need to organize them:

```shell title="Local Installation"
partinet split \
    --star /data/partinet_picking/labels \
    --images /data/partinet_picking/denoised \
    --output /data/partinet_picking/training_data \
    --split-only
```

If using `--split-only` with existing YOLO labels:

- **Format**: Standard YOLO format text files (`.txt`)
- **Content**: `class x_center y_center width height` (normalized 0-1)
- **Naming**: Label files must match image filenames (e.g., `img001.txt` for `img001.png`)

**For split-only mode:**
```
partinet_picking/
├── labels/
│   ├── micrograph_001.txt
│   ├── micrograph_002.txt
│   └── ...
└── denoised/
    ├── micrograph_001.png
    ├── micrograph_002.png
    └── ...
```



### Custom Train/Val Split Ratio

Adjust the validation set size (default is 25%):

```shell title="Local Installation"
# 80/20 train/val split
partinet split \
    --star /data/partinet_picking/particles.star \
    --images /data/partinet_picking/denoised \
    --output /data/partinet_picking/training_data \
    --test-size 0.2

# 90/10 train/val split (for larger datasets)
partinet split \
    --star /data/partinet_picking/particles.star \
    --images /data/partinet_picking/denoised \
    --output /data/partinet_picking/training_data \
    --test-size 0.1
```

## Output

### Directory Structure

After splitting, your output directory will be organized for YOLO training:

```
training_data/
├── images/
│   ├── train/
│   │   ├── micrograph_001.png
│   │   ├── micrograph_003.png
│   │   └── ...
│   └── val/
│       ├── micrograph_002.png
│       ├── micrograph_004.png
│       └── ...
├── labels/
│   ├── train/
│   │   ├── micrograph_001.txt
│   │   ├── micrograph_003.txt
│   │   └── ...
│   └── val/
│       ├── micrograph_002.txt
│       ├── micrograph_004.txt
│       └── ...
├── train.txt
├── val.txt
└── cryo_training.yaml
```

### Configuration Files

#### train.txt and val.txt

These files list the absolute paths to all training and validation images:

```text title="train.txt"
/data/partinet_picking/training_data/images/train/micrograph_001.png
/data/partinet_picking/training_data/images/train/micrograph_003.png
...
```

#### cryo_training.yaml

The YAML configuration file is ready for YOLO training:

```yaml title="cryo_training.yaml"
train: /data/partinet_picking/training_data/train.txt
val: /data/partinet_picking/training_data/val.txt

# number of classes
nc: 1

# class names
names: [ 'particle' ]
```

### YOLO Label Format

Each label file contains one line per particle:

```text title="micrograph_001.txt"
0 0.523456 0.678912 0.045123 0.045123
0 0.234567 0.456789 0.043210 0.043210
0 0.789012 0.123456 0.046789 0.046789
```

Format: `class x_center y_center width height` (all values normalized 0-1)

### Console Output

The command provides progress information:

```log
Converting STAR file to YOLO format and splitting for training...
============================================================
Step 1: Converting STAR file to YOLO format
============================================================
Parsing STAR file: /data/partinet_picking/particles.star
Found 15847 particles
Found 126 unique micrographs
Processed micrograph_001: 125 particles
Processed micrograph_002: 118 particles
...
Conversion complete! Labels saved to /data/partinet_picking/training_data/temp_labels

============================================================
Step 2: Splitting data into train/val sets
============================================================
Dataset split complete!
Training samples: 95
Validation samples: 31
Configuration saved to: /data/partinet_picking/training_data/cryo_training.yaml

All done! Training data ready in /data/partinet_picking/training_data
```

## Advanced Usage

### Multi-Class Training

If you have multiple particle types, assign different class IDs:

```shell
partinet split \
    --star /data/partinet_picking/large_particles.star \
    --images /data/partinet_picking/denoised \
    --output /data/partinet_picking/training_data \
    --class-id 0

partinet split \
    --star /data/partinet_picking/small_particles.star \
    --images /data/partinet_picking/denoised \
    --output /data/partinet_picking/training_data \
    --class-id 1 \
```

**Note**: You'll need to manually update `cryo_training.yaml` to reflect multiple classes:

```yaml
nc: 2
names: [ 'large_particle', 'small_particle' ]
```

### Handling Different Image Formats

The script automatically detects and handles multiple image formats:

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- TIFF (`.tif`, `.tiff`)
- MRC (`.mrc`)

No additional configuration needed - just ensure your images are in the specified directory.


### Quality Control

Before splitting:

1. **Verify STAR file**: Ensure all coordinates are within image boundaries
2. **Check image quality**: Remove corrupted or low-quality micrographs
3. **Balance dataset**: Aim for representative distribution of particle orientations
4. **Minimum samples**: Have at least 50-100 micrographs for meaningful training

### Validation Strategy

- **Small datasets**: Use higher validation proportion (20-25%) to ensure robust evaluation
- **Large datasets**: Use smaller validation proportion (10-15%) to maximize training data

## Troubleshooting

### Missing Image Files

**Error**: `Warning: Image file not found for micrograph_001. Skipping.`

**Solution**: Ensure image files match the names in your STAR file or label directory. Check for:
- Correct file extensions
- Case sensitivity in filenames
- Complete path to images directory

### STAR File Parsing Issues

**Error**: `Found 0 particles`

**Solution**: Verify your STAR file format:
- Contains `data_` and `loop_` sections
- Has required columns: `_rlnMicrographName`, `_rlnCoordinateX`, `_rlnCoordinateY`, `_rlnDiameter`
- Uses space-delimited format

## What's Next

- [1. Train PartiNet to identify particles](training/train1.md)
- [2. Train PartiNet to rout micrographs](training/train2.md)