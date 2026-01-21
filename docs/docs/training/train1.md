---
sidebar_position: 2
---

# 2. Train Dual Detectors
PartiNet's architecture requires a two-step training regime. The first step is to train PartiNet's dual detectors to identify particles in a micrograph.

![train workflow](./assets/supp_16_partinet-training.svg)

## Quick Start

<div class="container-tabs">
```shell title="Apptainer/Singularity"
apptainer exec --nv --no-home \
    -B /data oras://ghcr.io/wehi-researchcomputing/partinet:main-singularity partinet train step1 \
    --weight /data/partinet_publicweights.pt \
    --data /data/cryo_training.yaml \
    --project /data/partinet_trainstep1
```
```shell title="Docker"
docker run --gpus all -v /data:/data \
    ghcr.io/wehi-researchcomputing/partinet:main partinet train step1 \
    --weight /data/partinet_publicweights.pt \
    --data /data/cryo_training.yaml \
    --project /data/partinet_trainstep1
```
```shell title="Local Installation"
partinet train step1 \
    --weight /data/partinet_publicweights.pt \
    --data /data/cryo_training.yaml \
    --project /data/partinet_trainstep1
```

</div>

## Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--weight` | Path to the pre-trained weights file. We recommend starting with the supplied public weights (`partinet_publicweights.pt`) for faster convergence and better performance. If you have a large training dataset (>1000 annotated micrographs), you may try training from scratch by providing `''` to this flag. |
| `--data` | Path to the YAML configuration file containing your training dataset information (see [Data Preparation](training/split.md) for format details). |
| `--project` | Output directory where training results, checkpoints, and logs will be saved. |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--workers` | 8 | Number of data loading workers. Adjust based on your CPU cores and I/O performance. |
| `--device` | None | Specify GPU device (e.g., `0`, `1`, or `0,1` for multiple GPUs). If not specified, uses all available GPUs. |
| `--batch` | 16 | Training batch size. If you encounter out-of-memory (OOM) errors, reduce this value (try 8 or 4). |
| `--epochs` | 100 | Number of training epochs. More epochs may improve performance but increase training time. Overfitting may occur with too many epochs of training. It is important to [monitor validation metrics](training/training-output.md) during training |

:::tip Training Recommendations
- **Starting weights**: We strongly recommend using the supplied pre-trained weights as your starting point. This provides better initialization and typically results in faster training and improved final performance.
- **Training from scratch**: Only consider training without pre-trained weights if you have a substantial training dataset (≥1000 annotated micrographs).
- **Memory issues**: If you encounter OOM errors, reduce the `--batch` parameter progressively (16 → 8 → 4) until training succeeds.
:::

## Training Output

See [Training Output Reference](training/training-output.md) for details about the files generated during training, monitoring progress, and resuming interrupted runs.

**To resume this step from a checkpoint:**

<div class="container-tabs">
```shell title="Apptainer/Singularity"
apptainer exec --nv --no-home \
    -B /data oras://ghcr.io/wehi-researchcomputing/partinet:main-singularity partinet train step1 \
    --weight /data/partinet_trainstep1/exp3/weights/last.pt \
    --data /data/cryo_training.yaml \
    --project /data/partinet_trainstep1
```
```shell title="Docker"
docker run --gpus all -v /data:/data \
    ghcr.io/wehi-researchcomputing/partinet:main partinet train step1 \
    --weight /data/partinet_trainstep1/exp3/weights/last.pt \
    --data /data/cryo_training.yaml \
    --project /data/partinet_trainstep1
```
```shell title="Local Installation"
partinet train step1 \
    --weight /data/partinet_trainstep1/exp3/weights/last.pt \
    --data /data/cryo_training.yaml \
    --project /data/partinet_trainstep1
```

</div>