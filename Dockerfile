FROM python:3.9.19-slim-bookworm

COPY pyproject.toml /opt/PartiNet/pyproject.toml
COPY partinet /opt/PartiNet/partinet

RUN python -m pip install --no-cache-dir --no-cache /opt/PartiNet

LABEL AUTHORS Mihin Perera, Edward Yang, Julie Iskander
LABEL MAINTAINERS Mihin Perera, Edward Yang, Julie Iskander
LABEL VERSION v0.0.01
