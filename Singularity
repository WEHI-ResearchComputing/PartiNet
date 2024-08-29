bootstrap: docker
from: python:3.9.19-slim-bookworm

%files
  pyproject.toml /opt/PartiNet/pyproject.toml
  partinet /opt/PartiNet/partinet

%post
  python -m pip install --no-cache-dir /opt/PartiNet

%labels
  AUTHORS Mihin Perera, Edward Yang, Julie Iskander
  MAINTAINERS Mihin Perera, Edward Yang, Julie Iskander
  VERSION v0.0.01
