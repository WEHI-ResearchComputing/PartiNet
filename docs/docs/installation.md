---
sidebar_position: 2
---

# Installation

PartiNet can be installed using several methods. Choose the option that best fits your environment and requirements.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Git (for source installation)

_Please note that AMD/Intel GPUs have not been tested, but may still used with PartiNet_

## Method 1: Install from Source (Recommended)

This method gives you the latest version and full control over the installation:

```shell
# Create new python environment
conda create -n partinet python=3.9
conda activate partinet
# or using venv
python -m venv partinet-env
source partinet-env/bin/activate

# Install PartiNet
git clone git@github.com:WEHI-ResearchComputing/PartiNet.git
cd PartiNet
pip install .
```

## Method 2: Apptainer/Singularity Container

For users who prefer containerized environments or have limited system permissions:

**Option A: Pull and store locally**
```shell
apptainer pull partinet.sif oras://ghcr.io/wehi-researchcomputing/partinet:main-singularity
apptainer exec --nv --no-home -B /vast partinet.sif partinet --help
```

**Option B: Run directly from registry**
```shell
apptainer exec --nv --no-home \
    -B /vast oras://ghcr.io/wehi-researchcomputing/partinet:main-singularity \
    partinet --help
```

**Container options explained:**
- `--nv`: Enables NVIDIA GPU support
- `--no-home`: Prevents mounting your home directory
- `-B /vast`: Mounts the `/vast` directory (adjust path as needed for your data directory)

## Method 3: Docker Container

For Docker users:

```shell
docker pull ghcr.io/wehi-researchcomputing/partinet:main
docker run --gpus all -v /path/to/your/data:/data \
    ghcr.io/wehi-researchcomputing/partinet:main partinet --help
```

**Docker options explained:**
- `--gpus all`: Enables GPU support (requires nvidia-docker)
- `-v /path/to/your/data:/data`: Mounts your data directory


## Verification

After installation, verify that PartiNet is working correctly:

```shell
partinet --help
```

You should see version information and available commands.

## GPU Support

PartiNet is designed to leverage GPU acceleration for optimal performance. Ensure you have:

- NVIDIA GPU with CUDA compute capability 3.5+ (e.g., NVIDIA A30, A100, H100)
- CUDA drivers installed
- For containers: nvidia-docker (Docker) or `--nv` flag (Apptainer)

_AMD and Intel GPUs have not been tested and may not support full PartiNet functionality_


## Model Weights
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

## Next Steps

Once installed, proceed to [Getting Started](getting-started.md) to run your first PartiNet analysis.