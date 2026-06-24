import os
from unittest import mock

import pytest

from partinet.gui import job_registry
from partinet.gui.job_runner import JobSpec

register_job = job_registry.register_job
list_jobs = job_registry.list_jobs
cancel_job = job_registry.cancel_job
attach_local_pid = job_registry.attach_local_pid
_manifest_path = job_registry._manifest_path


def _spec_for(project):
    return JobSpec(
        stage="denoise",
        command=["partinet", "denoise", "--source", "/data", "--project", project],
        project_dir=project,
        log_path=os.path.join(project, "partinet_denoise.log"),
        mode="local",
    )


def test_register_and_list_active_job(tmp_path):
    project = str(tmp_path / "project")
    os.makedirs(project)
    rec = register_job(_spec_for(project), "local")
    jobs = list_jobs(project)
    assert len(jobs) == 1
    assert jobs[0].job_key == rec.job_key
    assert jobs[0].stage == "denoise"


def test_manifest_round_trip(tmp_path):
    project = str(tmp_path / "project")
    os.makedirs(project)
    rec = register_job(_spec_for(project), "slurm")
    path = _manifest_path(project)
    assert os.path.isfile(path)
    jobs = list_jobs(project)
    assert jobs[0].job_key == rec.job_key


def test_cancel_local_proc(tmp_path):
    project = str(tmp_path / "project")
    os.makedirs(project)
    rec = register_job(_spec_for(project), "local")
    proc = mock.Mock()
    proc.poll.return_value = None
    proc.pid = 4242
    attach_local_pid(rec.job_key, proc)
    msg = cancel_job(rec.job_key, project)
    assert "cancellation" in msg.lower()
    proc.terminate.assert_called_once()


def test_cancel_slurm_calls_scancel(tmp_path):
    project = str(tmp_path / "project")
    os.makedirs(project)
    rec = register_job(_spec_for(project), "slurm")
    job_registry.attach_slurm_id(rec.job_key, "999001", "/tmp/job.sh")
    with mock.patch.object(job_registry.subprocess, "run") as run:
        cancel_job(rec.job_key, project)
    assert any(call.args[0][:2] == ["scancel", "999001"] for call in run.call_args_list)


def test_list_jobs_empty_without_project(tmp_path):
    assert list_jobs("") == []
    assert list_jobs(str(tmp_path / "missing")) == []


def test_read_job_log(tmp_path):
    project = str(tmp_path / "project")
    os.makedirs(project)
    log_path = os.path.join(project, "partinet_denoise.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("denoise started\n")
    rec = register_job(_spec_for(project), "local")
    records = job_registry._load_manifest(project)
    records[rec.job_key].log_path = log_path
    job_registry._save_manifest(project, records)
    text = job_registry.read_job_log(project, rec.job_key)
    assert "denoise started" in text
