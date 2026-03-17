---
sidebar_position: 3
---

# 3. STAR

The `partinet star` command is the final step in your particle picking pipeline. It converts the particle coordinates detected in the previous stage into a standardized STAR file format that can be used with downstream cryo-EM processing software like RELION, cryoSPARC, or other reconstruction programs.

## Quick Start

```shell title="Local Installation"
partinet star \
    --labels /data/partinet_picking/exp/labels \
    --images /data/partinet_picking/denoised \
    --output /data/partinet_picking/output.star \
    --conf 0.1
```

```shell title="Apptainer/Singularity"
apptainer exec --nv --no-home \
    -B /data oras://ghcr.io/wehi-researchcomputing/partinet:main-singularity partinet star \
    --labels /data/partinet_picking/exp/labels \
    --images /data/partinet_picking/denoised \
    --output /data/partinet_picking/output.star \
    --conf 0.1
```

```shell title="Docker"
docker run --gpus all -v /data:/data \
    ghcr.io/wehi-researchcomputing/partinet:main partinet star \
    --labels /data/partinet_picking/exp/labels \
    --images /data/partinet_picking/denoised \
    --output /data/partinet_picking/output.star \
    --conf 0.1
```


## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `--labels` | Path | Yes | Directory containing the particle coordinate files (`.txt` format) from the detection stage |
| `--images` | Path | Yes | Directory containing the denoised micrographs corresponding to the labels |
| `--output` | Path | Yes | Output path for the generated STAR file |
| `--conf` | Float | Yes | Confidence threshold for filtering particle coordinates (0.0-1.0) |

## Input Requirements

At this stage of the pipeline, your directory structure should look like:
```
partinet_picking/
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
│   ├── micrograph1.mrc       # 🖼️ Micrographs with detections drawn
│   ├── micrograph2.mrc
│   └── ...
└── partinet_detect.log
└── partinet_denoise.log
```


## Confidence Threshold

The `--conf` parameter controls the quality filtering of detected particles:

- **Typical range**: 0.1-0.3 for most datasets

Choose your confidence threshold based on:
- Dataset quality and signal-to-noise ratio
- Downstream processing requirements
- Balance between particle quantity and quality

## CryoSPARC Output

The command generates a STAR file compatible with CryoSPARC containing:
- Particle coordinates (X, Y positions)
- Corresponding micrograph paths

## RELION output

Use `--relion` and `--relion-project-dir` to generate RELION-compatible STAR outputs under `<relion_project>/partinet`:

- Micrograph manifest: `<relion_project>/partinet/pick.star`
- Per-micrograph coordinates: `<relion_project>/partinet/movies/<micrograph_basename>.star`

Example:

```bash
partinet star \
  --labels /data/partinet_picking/exp/labels \
  --images /data/partinet_picking/denoised \
  --output /data/partinet_picking/output.star \
  --conf 0.2 \
  --relion \
  --relion-project-dir /data/relion/EMPIAR-10089 \
  --mrc-prefix MotionCorr/job003/movies
```


## Next Steps

After generating your STAR file, you can:
- Import it into RELION/CryoSPARC for 2D classification and 3D reconstruction
- Perform additional particle filtering or sorting based on your specific needs

:::warning

If you are using PartiNet v1.0.0, the Denoise flips micrographs in the y-axis due to matrix transpose operations. If you denoised micrographs in PartiNet ensure that when you import and extract particle coordinates that you toggle `Flip in y` in CryoSPARC prior to particle extraction. This has been fixed in v1.0.1+. You can check your version with `partinet --help`

:::




