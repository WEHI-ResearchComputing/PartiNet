# PartiNet 🔬

PartiNet is a three-stage pipeline for automated particle picking in cryo-EM micrographs, combining advanced denoising with state-of-the-art deep learning detection.


## Features

- 🧹 Advanced denoising for improved signal-to-noise ratio
- 🎯 Deep learning-based particle detection
- ⚡ Multi-GPU support for faster processing
- 🔄 Seamless integration with RELION workflows
- 📊 Confidence-based particle filtering
- 🖼️ Visual detection validation

## Prerequisites

Before starting, ensure you have:
- Motion-corrected micrographs
- GPU access (recommended)
- PartiNet installation (see Installation section)

## Installation

```bash
git clone git@github.com:WEHI-ResearchComputing/PartiNet.git
cd PartiNet
pip install .
```

Alternatively, use our containers:

```bash
# Docker
docker run ghcr.io/wehi-researchcomputing/partinet:latest

# Singularity/Apptainer
singularity run oras://ghcr.io/wehi-researchcomputing/partinet:latest
```

## Directory Structure

```
project_directory/
├── motion_corrected/          # 📁 Input micrographs
├── denoised/                  # 🧹 Denoised outputs
├── exp/                       # 🎯 Detection results
│   ├── labels/               # 📋 Coordinates
│   └── ...                   # 🖼️ Visualizations
└── partinet_particles.star    # ⭐ Final output
```

## Pipeline Stages

### 1. Denoise
```bash
partinet denoise \
  --source /data/my_project/motion_corrected \
  --project /data/my_project
```

### 2. Detect
```bash
partinet detect \
  --weight /path/to/model_weights.pt \
  --source /data/partinet_picking/denoised \
  --device 0,1,2,3 \
  --project /data/partinet_picking
```

### 3. Generate STAR File
```bash
partinet star \
  --labels /data/my_project/exp/labels \
  --images /data/my_project/denoised \
  --output /data/my_project/partinet_particles.star \
  --conf 0.1
```

## Key Parameters

### Detection
- `--backbone-detector`: Choice of neural network architecture
- `--weight`: Path to model weights
- `--conf-thres`: Detection confidence threshold
- `--iou-thres`: Overlap filtering threshold
- `--device`: GPU device selection

### STAR Generation
- `--conf`: Confidence threshold for particle filtering
- `--output`: Path for final STAR file

## Output Files

1. **Denoised Micrographs** (`denoised/*.mrc`)
   - Cleaned micrographs with improved SNR

2. **Detection Results** (`exp/`)
   - `labels/*.txt`: Particle coordinates
   - `*.png`: Visualization overlays

3. **STAR File** (`partinet_particles.star`)
   - Ready for RELION processing

## Advanced Usage

For detailed information about specific commands:

```bash
partinet --help
partinet <command> --help
```

Available commands:
- `denoise`: Clean input micrographs
- `detect`: Identify particles
- `star`: Generate STAR files
- `train`: Train custom models (step1/step2)
- `test`: Evaluate model performance

## Troubleshooting

- **GPU Issues**
  - Verify GPU availability: `nvidia-smi`
  - Check CUDA installation
  - Ensure proper device selection

- **Path Issues**
  - Verify directory permissions
  - Check mount points in container setups
  - Ensure absolute paths are used

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Citation

If you use PartiNet in your research, please cite:
```
Citation information will be added upon publication
```

## Support

For issues and questions:
- Open an [Issue](https://github.com/WEHI-ResearchComputing/PartiNet/issues)
- Check existing [Discussions](https://github.com/WEHI-ResearchComputing/PartiNet/discussions)
