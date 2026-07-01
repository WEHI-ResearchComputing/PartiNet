"""Track and cancel GUI-submitted PartiNet jobs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from partinet.gui.job_runner import JobSpec


_ACTIVE = {"running", "pending"}


@dataclass
class JobRecord:
    job_key: str
    stage: str
    mode: str
    project_dir: str
    log_path: str
    script_path: str = ""
    slurm_job_id: str = ""
    pid: Optional[int] = None
    status: str = "running"
    submitted_at: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "JobRecord":
        pid = data.get("pid")
        return cls(
            job_key=str(data["job_key"]),
            stage=str(data["stage"]),
            mode=str(data["mode"]),
            project_dir=str(data["project_dir"]),
            log_path=str(data["log_path"]),
            script_path=str(data.get("script_path") or ""),
            slurm_job_id=str(data.get("slurm_job_id") or ""),
            pid=int(pid) if pid not in (None, "") else None,
            status=str(data.get("status") or "running"),
            submitted_at=str(data.get("submitted_at") or ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def display_id(self) -> str:
        if self.mode == "slurm" and self.slurm_job_id:
            return self.slurm_job_id
        if self.pid is not None:
            return str(self.pid)
        return self.job_key[:8]


@dataclass
class _RegistryState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel_flags: Dict[str, bool] = field(default_factory=dict)
    local_procs: Dict[str, subprocess.Popen] = field(default_factory=dict)
    project_by_key: Dict[str, str] = field(default_factory=dict)


_STATE = _RegistryState()


def _jobs_dir(project_dir: str) -> str:
    path = os.path.join(project_dir, ".partinet_jobs")
    os.makedirs(path, exist_ok=True)
    return path


def _manifest_path(project_dir: str) -> str:
    return os.path.join(_jobs_dir(project_dir), "manifest.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pid_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _slurm_active(job_id: str) -> bool:
    if not job_id:
        return False
    try:
        proc = subprocess.run(
            ["squeue", "-h", "-j", job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(proc.stdout.strip())
    except FileNotFoundError:
        return False


def _load_manifest(project_dir: str) -> Dict[str, JobRecord]:
    path = _manifest_path(project_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, dict):
        return {}
    return {key: JobRecord.from_dict(val) for key, val in jobs.items() if isinstance(val, dict)}


def _save_manifest(project_dir: str, records: Dict[str, JobRecord]) -> None:
    path = _manifest_path(project_dir)
    payload = {"jobs": {key: rec.to_dict() for key, rec in records.items()}}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def _reconcile_record(record: JobRecord) -> JobRecord:
    if record.status not in _ACTIVE:
        return record
    if record.mode == "slurm":
        if record.slurm_job_id and _slurm_active(record.slurm_job_id):
            return record
        if record.slurm_job_id:
            record.status = "completed"
        return record
    if record.pid is None:
        return record
    if _pid_alive(record.pid):
        return record
    record.status = "completed"
    return record


def register_job(spec: "JobSpec", mode: str, script_path: str = "") -> JobRecord:
    job_key = uuid.uuid4().hex
    record = JobRecord(
        job_key=job_key,
        stage=spec.stage,
        mode=mode,
        project_dir=spec.project_dir,
        log_path=spec.log_path,
        script_path=script_path,
        status="running",
        submitted_at=_utc_now(),
    )
    with _STATE.lock:
        _STATE.cancel_flags[job_key] = False
        _STATE.project_by_key[job_key] = spec.project_dir
        records = _load_manifest(spec.project_dir)
        records[job_key] = record
        _save_manifest(spec.project_dir, records)
    return record


def attach_local_pid(job_key: str, proc: subprocess.Popen) -> None:
    with _STATE.lock:
        _STATE.local_procs[job_key] = proc
        project_dir = _STATE.project_by_key.get(job_key)
        if not project_dir:
            return
        records = _load_manifest(project_dir)
        if job_key in records:
            records[job_key].pid = proc.pid
            _save_manifest(project_dir, records)


def attach_slurm_id(job_key: str, slurm_job_id: str, script_path: str = "") -> None:
    with _STATE.lock:
        project_dir = _STATE.project_by_key.get(job_key)
        if not project_dir:
            return
        records = _load_manifest(project_dir)
        if job_key not in records:
            return
        records[job_key].slurm_job_id = slurm_job_id
        if script_path:
            records[job_key].script_path = script_path
        _save_manifest(project_dir, records)


def is_cancelled(job_key: str) -> bool:
    with _STATE.lock:
        return _STATE.cancel_flags.get(job_key, False)


def complete_job(job_key: str, status: str) -> None:
    with _STATE.lock:
        _STATE.cancel_flags.pop(job_key, None)
        _STATE.local_procs.pop(job_key, None)
        project_dir = _STATE.project_by_key.pop(job_key, None)
        if not project_dir:
            return
        records = _load_manifest(project_dir)
        if job_key in records:
            records[job_key].status = status
            _save_manifest(project_dir, records)


def list_jobs(project_dir: str) -> List[JobRecord]:
    p = (project_dir or "").strip()
    if not p or not os.path.isdir(p):
        return []
    with _STATE.lock:
        records = _load_manifest(p)
        updated: Dict[str, JobRecord] = {}
        active: List[JobRecord] = []
        for key, rec in records.items():
            rec = _reconcile_record(rec)
            updated[key] = rec
            if rec.status in _ACTIVE:
                active.append(rec)
        _save_manifest(p, updated)
    active.sort(key=lambda r: r.submitted_at, reverse=True)
    return active


def cancel_job(job_key: str, project_dir: str = "") -> str:
    if not job_key:
        return "No job selected."
    slurm_id = ""
    proc = None
    local_pid = None
    with _STATE.lock:
        _STATE.cancel_flags[job_key] = True
        proc = _STATE.local_procs.get(job_key)
        resolved_project = (project_dir or "").strip() or _STATE.project_by_key.get(job_key, "")
        if resolved_project:
            records = _load_manifest(resolved_project)
            rec = records.get(job_key)
            if rec:
                slurm_id = rec.slurm_job_id
                local_pid = rec.pid
                rec.status = "cancelled"
                _save_manifest(resolved_project, records)

    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    elif local_pid is not None:
        try:
            os.kill(local_pid, signal.SIGTERM)
        except OSError:
            pass

    if slurm_id:
        try:
            subprocess.run(["scancel", slurm_id], capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return f"Cancelled locally; scancel not found for Slurm job {slurm_id}."

    return "Job cancellation requested."


def get_job_record(project_dir: str, job_key: str) -> Optional[JobRecord]:
    p = (project_dir or "").strip()
    if not p or not job_key:
        return None
    records = _load_manifest(p)
    rec = records.get(job_key)
    if rec is None:
        return None
    return _reconcile_record(rec)


def _read_tail(path: str, max_bytes: int = 512_000) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        return fh.read().decode("utf-8", errors="replace")


def read_job_log(project_dir: str, job_key: str) -> str:
    if not job_key:
        return "Select a job to view its log."
    rec = get_job_record(project_dir, job_key)
    if rec is None:
        return "Job not found."
    parts = [f"**{rec.stage}** · {rec.mode} · {rec.display_id()} · {rec.status}"]
    if rec.log_path:
        chunk = _read_tail(rec.log_path)
        if chunk:
            parts.append(f"\n--- {rec.log_path} ---\n{chunk.rstrip()}")
    if rec.script_path:
        slurm_out = rec.script_path.replace(".sh", "_slurm.out")
        chunk = _read_tail(slurm_out)
        if chunk:
            parts.append(f"\n--- {slurm_out} ---\n{chunk.rstrip()}")
    if len(parts) == 1:
        parts.append("\nNo log output yet.")
    return "\n".join(parts)


def jobs_markdown(project_dir: str) -> str:
    if not (project_dir or "").strip():
        return "Set a project directory to view jobs."
    jobs = list_jobs(project_dir)
    if not jobs:
        return "No running jobs for this project."
    lines = ["| Stage | Mode | ID | Status | Started |", "| --- | --- | --- | --- | --- |"]
    for job in jobs:
        lines.append(
            f"| {job.stage} | {job.mode} | {job.display_id()} | {job.status} | {job.submitted_at} |"
        )
    return "\n".join(lines)


def jobs_dropdown_update(project_dir: str):
    import gradio as gr

    jobs = list_jobs(project_dir)
    choice_list = [(f"{j.stage} · {j.mode} · {j.display_id()}", j.job_key) for j in jobs]
    return gr.update(
        choices=choice_list,
        value=jobs[0].job_key if jobs else None,
    )
