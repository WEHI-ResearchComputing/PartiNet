FROM python:3.9.19-slim-bookworm

COPY pyproject.toml /opt/PartiNet/pyproject.toml
COPY partinet /opt/PartiNet/partinet

RUN apt-get update && apt-get install libglib2.0-0 -y && rm -rf /var/cache/apt/archives /var/lib/apt/lists/* && apt-get clean
RUN python -m pip install --no-cache-dir --no-cache /opt/PartiNet

LABEL AUTHORS Mihin Perera, Edward Yang, Julie Iskander
LABEL MAINTAINERS Mihin Perera, Edward Yang, Julie Iskander
LABEL VERSION v0.0.01
