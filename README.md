# PartiNet
PartiNet is a particle-picking pipeline for cryo-EM micrographs. It provides denoising, adaptive detection, and STAR file generation for downstream processing.

# Links
- Documentation: https://mihinp.github.io/partinet_documentation/
- Model weights (Hugging Face): https://huggingface.co/MihinP/PartiNet

# Getting started (quick)
1. Clone the repository

```powershell
git clone git@github.com:WEHI-ResearchComputing/PartiNet.git
cd PartiNet
```

2. Create a Python virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
```

3. Install requirements

```powershell
pip install -r requirements.txt
# or editable install for development
pip install -e .
```

4. Download model weights (see Hugging Face README)

```powershell
# If you have git-lfs and access via HTTPS/SSH
git lfs install
git clone https://huggingface.co/MihinP/PartiNet
# or use the huggingface_hub python client
python -m pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="MihinP/PartiNet", filename="best.pt", repo_type="model")
PY
```

# Quick usage examples

- Denoise images

```powershell
partinet denoise --source /data/raw_micrographs --project /data/partinet_project
```

- Detect particles

```powershell
partinet detect --weight /path/to/best.pt --source /data/partinet_project/denoised --project /data/partinet_project
```

- Generate STAR files

```powershell
partinet star --project /data/partinet_project --output /data/partinet_project/exp/particles.star
```

# Containerized usage

- Docker

```powershell
docker run --gpus all -v /data:/data ghcr.io/wehi-researchcomputing/partinet:main partinet detect --weight /path/to/best.pt --source /data/denoised --project /data/partinet_project
```

- Apptainer / Singularity

```powershell
apptainer exec --nv --no-home -B /data oras://ghcr.io/wehi-researchcomputing/partinet:main-singularity partinet detect --weight /path/to/best.pt --source /data/denoised --project /data/partinet_project
```

# Development notes
- Tests and CI: see `.github/workflows/` for CI pipelines.
- Contributing: open issues and PRs on the main repo. Use the documentation site for user-facing docs and developer notes.


# Support
- For questions or issues, open an issue in the main repo: https://github.com/WEHI-ResearchComputing/PartiNet/issues
