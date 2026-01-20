---
sidebar_position: 1
---

# Introduction

PartiNet is a powerful command-line tool for particle picking on cryo-EM micrographs. It provides a comprehensive three-stage pipeline designed to clean, identify, and prepare particles from experimental data for subsequent processing.

## The Three-Stage Pipeline

PartiNet processes data through three sequential stages, each building on the output of the previous stage:

### 1. Denoise
The first stage removes noise and artifacts from your raw data using fast heuristic denoising algorithms. This stage improves signal-to-noise ratios and prepares micrographs for accurate particle detection.

### 2. Detect  
The detection stage identifies and locates individual particles within your cleaned data. Using a dynamic adaptive architecture, it quickly and accurately identifies particles within micrographs.

### 3. Star
The final stage prepares particle data for further processing and provides reports on particle populations in your dataset.

## Key Features

- **Fast picking** - Leverages state-of-the-art dynamic deep learning models for accurate particle processing
- **Accurate picking** - PartiNet accurately identifies proteins in your micrographs and filters junk prior to further processing
- **Overcome orientation bias** - PartiNet identifies rare views of proteins in your dataset
- **Multi-species identification** - PartiNet can identify and pick heterogeneous samples without requiring prior estimation of box sizes
- **Batch processing** - Process multiple files efficiently with parallel processing capabilities

## Use Cases

PartiNet is ideal for:
- Identifying rare views
- Picking on heterogeneous datasets
- Reporting on particle populations

## Next Steps

- **New to PartiNet?** Start with [Installation](installation.md) to get up and running
- **Ready to begin?** Follow our [Getting Started](getting-started.md) guide for your first analysis
- **Need specific details?** Check the individual stage documentation: [Denoise](stages/denoise.md), [Detect](stages/detect.md), [Star](stages/star.md)

<!-- ## Getting Help

If you encounter issues or need assistance:
- Check the [Troubleshooting](reference/troubleshooting.md) guide
- Review the complete [CLI Reference](reference/cli-reference.md)
- Look at our [Examples](examples/) for common use cases -->