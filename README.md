# PartiNet 🔬

PartiNet is a three-stage pipeline for automated particle picking in cryo-EM micrographs, combining advanced denoising with state-of-the-art deep learning detection.

Installation and Usage can be found at our [Documentation](https://wehi-researchcomputing.github.io/PartiNet/) 

Use our pretrained model at [Model Weights](https://huggingface.co/MihinP/PartiNet)


## Features

- 🧹 Heuristic denoising for improved signal-to-noise ratio
- 🎯 Dynamic deep learning particle detection
- ⚡ Multi-GPU support for faster processing
- 🔄 Seamless integration with cryoSPARC and RELION workflows
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
```

Alternatively, use our containers:
```
# Singularity/Apptainer
apptainer exec --nv --no-home \
    -B /path/to/your/data:/data oras://ghcr.io/wehi-researchcomputing/partinet:main-singularity \
    partinet --help
```

```
# Docker
docker pull ghcr.io/wehi-researchcomputing/partinet:main
docker run --gpus all -v /path/to/your/data:/data \
    ghcr.io/wehi-researchcomputing/partinet:main partinet --help
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

## Output Files

1. **Denoised Micrographs** (`denoised/*.png`)
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
- `train`: Train custom models


## Troubleshooting

- **GPU Issues**
  - Verify GPU availability: `nvidia-smi`
  - Check CUDA installation
  - Ensure proper device selection
  - AMD and Intel GPUs are currently untested

- **Path Issues**
  - Verify directory permissions
  - Check mount points in container setups
  - Ensure absolute paths are used

## Contributing

We welcome contributions! Please raise issues or initiate pull requests on this repo.

## License

This project is licensed under the terms of MIT license.

## Citation

If you use PartiNet in your research, please cite:
```
Citation information will be added upon publication
```

## Support

For issues and questions:
- Open an [Issue](https://github.com/WEHI-ResearchComputing/PartiNet/issues)
- Check existing [Discussions](https://github.com/WEHI-ResearchComputing/PartiNet/discussions)
