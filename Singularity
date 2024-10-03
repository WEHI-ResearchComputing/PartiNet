bootstrap: docker
from: python:3.9.19-slim-bookworm

%files
  . /opt/PartiNet

%post
  # install system dependencies
  apt-get update
  apt-get install libglib2.0-0 -y

  # cleanup apt package index
  rm -rf /var/cache/apt/archives /var/lib/apt/lists/*
  apt-get clean

  python -m pip install --no-cache-dir /opt/PartiNet

%labels
  AUTHORS Mihin Perera, Edward Yang, Julie Iskander
  MAINTAINERS Mihin Perera, Edward Yang, Julie Iskander
  VERSION v0.0.01
