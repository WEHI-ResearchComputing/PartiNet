import os
from unittest import mock

import pytest

from partinet.gui import job_runner

JobSpec = job_runner.JobSpec
SlurmOptions = job_runner.SlurmOptions
_write_slurm_script = job_runner._write_slurm_script
load_slurm_defaults = job_runner.load_slurm_defaults
resolve_slurm_options = job_runner.resolve_slurm_options
stream_job = job_runner.stream_job


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
    script = _write_slurm_script(spec, resolve_slurm_options("denoise", spec.slurm))
    text = open(script, encoding="utf-8").read()
    assert "#SBATCH --partition=" not in text
    assert "#SBATCH --cpus-per-task=32" in text
    assert "#SBATCH --mem=100G" in text
    assert "partinet denoise" in text
    assert str(project) in text


def test_stage_slurm_defaults(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for stage, cpus, mem, gpus in [
        ("denoise", "32", "100G", ""),
        ("detect", "32", "100G", "4"),
        ("star", "16", "64G", ""),
    ]:
        spec = JobSpec(
            stage=stage,
            command=["partinet", stage],
            project_dir=str(project),
            log_path=str(project / f"{stage}.log"),
            slurm=SlurmOptions(partinet_cmd="partinet"),
        )
        slurm = resolve_slurm_options(stage, spec.slurm)
        assert slurm.cpus == cpus
        assert slurm.mem == mem
        assert slurm.gpus == gpus


def test_stream_job_local_subprocess(tmp_path):
    log_path = tmp_path / "out.log"
    with mock.patch.object(job_runner.subprocess, "Popen") as popen:
        proc = mock.Mock()
        stdout = mock.Mock()
        stdout.readline.side_effect = ["line1\n", ""]
        stdout.read.return_value = ""
        proc.stdout = stdout
        proc.poll.return_value = 0
        proc.returncode = 0
        proc.pid = 1234
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


def test_stream_job_local_cancel(tmp_path):
    log_path = tmp_path / "out.log"
    with mock.patch.object(job_runner.subprocess, "Popen") as popen, mock.patch.object(
        job_runner, "is_cancelled", side_effect=[False, True]
    ):
        proc = mock.Mock()
        proc.stdout = mock.Mock(readline=mock.Mock(return_value=""))
        proc.poll.return_value = None
        proc.pid = 5678
        popen.return_value = proc
        spec = JobSpec(
            stage="denoise",
            command=["partinet", "denoise", "--source", "/data", "--project", str(tmp_path)],
            project_dir=str(tmp_path),
            log_path=str(log_path),
            mode="local",
        )
        chunks = list(stream_job(spec))
    assert chunks[-1].endswith("--- CANCELLED ---\n")
    proc.terminate.assert_called()
