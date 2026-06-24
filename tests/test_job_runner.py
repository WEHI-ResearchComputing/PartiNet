import importlib.util
import os
from unittest import mock

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_JOB_RUNNER_PATH = os.path.join(_ROOT, "partinet", "gui", "job_runner.py")
_spec = importlib.util.spec_from_file_location("partinet_job_runner", _JOB_RUNNER_PATH)
_job_runner = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_job_runner)

JobSpec = _job_runner.JobSpec
SlurmOptions = _job_runner.SlurmOptions
_write_slurm_script = _job_runner._write_slurm_script
load_slurm_defaults = _job_runner.load_slurm_defaults
stream_job = _job_runner.stream_job


def test_load_slurm_defaults_empty_without_config(monkeypatch):
    monkeypatch.delenv("PARTINET_SLURM_CONFIG", raising=False)
    opts = load_slurm_defaults()
    assert opts.partition == ""
    assert opts.partinet_cmd == "partinet"


def test_write_slurm_script_has_no_site_defaults(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    spec = JobSpec(
        stage="denoise",
        command=["partinet", "denoise", "--source", "/data/mrc", "--project", str(project)],
        project_dir=str(project),
        log_path=str(project / "partinet_denoise.log"),
        slurm=SlurmOptions(partinet_cmd="partinet"),
    )
    script = _write_slurm_script(spec, spec.slurm)
    text = open(script, encoding="utf-8").read()
    assert "#SBATCH --partition=" not in text
    assert "partinet denoise" in text
    assert str(project) in text


def test_stream_job_local_subprocess(tmp_path):
    log_path = tmp_path / "out.log"
    with mock.patch.object(_job_runner.subprocess, "Popen") as popen:
        proc = mock.Mock()
        proc.stdout = iter(["line1\n"])
        proc.poll.side_effect = [None, 0]
        proc.returncode = 0
        popen.return_value = proc
        spec = JobSpec(
            stage="star",
            command=["partinet", "star", "--labels", "a", "--images", "b", "--output", str(log_path), "--conf", "0.1"],
            project_dir=str(tmp_path),
            log_path=str(log_path),
            mode="local",
        )
        chunks = list(stream_job(spec))
    assert any("Running:" in c for c in chunks)
    assert chunks[-1].endswith("--- Complete ---")
