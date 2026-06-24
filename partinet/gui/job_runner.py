"""Local and Slurm job execution for the PartiNet GUI."""

from __future__ import annotations

import logging
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Iterator, List, Optional

import yaml

POLL_INTERVAL = 1.0
_SLURM_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}


@dataclass
class SlurmOptions:
    partition: str = ""
    account: str = ""
    time_limit: str = ""
    cpus: str = ""
    gpus: str = ""
    mem: str = ""
    extra_sbatch: str = ""
    preamble: str = ""
    partinet_cmd: str = "partinet"


@dataclass
class JobSpec:
    stage: str
    command: List[str]
    project_dir: str
    log_path: str
    mode: str = "local"
    slurm: Optional[SlurmOptions] = None
    run_fn: Optional[Callable[[], None]] = None
    loggers: List[str] = field(default_factory=list)


def load_slurm_defaults() -> SlurmOptions:
    """Load optional user-owned YAML defaults from PARTINET_SLURM_CONFIG."""
    path = os.environ.get("PARTINET_SLURM_CONFIG", "")
    if not path or not os.path.isfile(path):
        return SlurmOptions()
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        return SlurmOptions()
    fields = {f.name for f in SlurmOptions.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return SlurmOptions(**{k: str(v) for k, v in data.items() if k in fields and v is not None})


def merge_slurm_options(base: SlurmOptions, override: SlurmOptions) -> SlurmOptions:
    """GUI fields override config-file defaults when non-empty."""
    merged = {}
    for name in SlurmOptions.__dataclass_fields__:  # type: ignore[attr-defined]
        gui_val = getattr(override, name, "")
        cfg_val = getattr(base, name, "")
        merged[name] = gui_val if str(gui_val).strip() else cfg_val
    return SlurmOptions(**merged)


class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        self.q.put(self.format(record) + "\n")


def _tail_file(path: str, offset: int) -> tuple[str, int]:
    if not os.path.exists(path):
        return "", offset
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        fh.seek(offset)
        chunk = fh.read()
        return chunk, fh.tell()


def _stream_local_callable(spec: JobSpec) -> Iterator[str]:
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    log_q: queue.Queue = queue.Queue()
    handler = _QueueHandler(log_q)
    handler.setFormatter(fmt)

    for name in spec.loggers:
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)

    result = {"done": False, "error": None}

    def _target():
        try:
            if spec.run_fn is None:
                raise RuntimeError("Local job missing run_fn")
            spec.run_fn()
        except Exception:
            result["error"] = traceback.format_exc()
        finally:
            result["done"] = True

    threading.Thread(target=_target, daemon=True).start()
    output = ""
    offset = 0
    while not result["done"]:
        time.sleep(POLL_INTERVAL)
        while not log_q.empty():
            output += log_q.get_nowait()
        chunk, offset = _tail_file(spec.log_path, offset)
        if chunk:
            output += chunk
        yield output

    while not log_q.empty():
        output += log_q.get_nowait()
    chunk, offset = _tail_file(spec.log_path, offset)
    if chunk:
        output += chunk

    for name in spec.loggers:
        logging.getLogger(name).removeHandler(handler)

    if result["error"]:
        output += f"\n--- FAILED ---\n{result['error']}"
    else:
        output += "\n--- Complete ---"
    yield output


def _stream_local_subprocess(spec: JobSpec) -> Iterator[str]:
    slurm = spec.slurm or SlurmOptions()
    cmd = list(spec.command)
    if slurm.partinet_cmd and slurm.partinet_cmd != "partinet":
        cmd = [slurm.partinet_cmd] + cmd[1:]

    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    output = f"Running: {shlex.join(cmd)}\n"
    yield output
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if line:
            output += line
            yield output
        elif proc.poll() is not None:
            break
        else:
            time.sleep(0.05)
    remaining = proc.stdout.read()
    if remaining:
        output += remaining
    if proc.returncode != 0:
        output += f"\n--- FAILED --- (exit code {proc.returncode})\n"
    else:
        output += "\n--- Complete ---"
    yield output


def _jobs_dir(project_dir: str) -> str:
    path = os.path.join(project_dir, ".partinet_jobs")
    os.makedirs(path, exist_ok=True)
    return path


def _write_slurm_script(spec: JobSpec, slurm: SlurmOptions) -> str:
    jobs_dir = _jobs_dir(spec.project_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = os.path.join(jobs_dir, f"{spec.stage}_{stamp}.sh")
    slurm_out = os.path.join(jobs_dir, f"{spec.stage}_{stamp}_slurm.out")

    lines = ["#!/bin/bash", f"#SBATCH --job-name=partinet_{spec.stage}", f"#SBATCH --output={slurm_out}"]
    if slurm.partition.strip():
        lines.append(f"#SBATCH --partition={slurm.partition.strip()}")
    if slurm.account.strip():
        lines.append(f"#SBATCH --account={slurm.account.strip()}")
    if slurm.time_limit.strip():
        lines.append(f"#SBATCH --time={slurm.time_limit.strip()}")
    if slurm.cpus.strip():
        lines.append(f"#SBATCH --cpus-per-task={slurm.cpus.strip()}")
    if slurm.gpus.strip():
        lines.append(f"#SBATCH --gres=gpu:{slurm.gpus.strip()}")
    if slurm.mem.strip():
        lines.append(f"#SBATCH --mem={slurm.mem.strip()}")
    for extra in slurm.extra_sbatch.splitlines():
        extra = extra.strip()
        if extra:
            lines.append(extra if extra.startswith("#SBATCH") else f"#SBATCH {extra}")

    lines.append("set -euo pipefail")
    if slurm.preamble.strip():
        lines.append(slurm.preamble.strip())

    cmd = list(spec.command)
    if slurm.partinet_cmd and slurm.partinet_cmd != "partinet":
        cmd[0] = slurm.partinet_cmd
    lines.append(f"cd {shlex.quote(spec.project_dir)}")
    lines.append(shlex.join(cmd))

    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o750)
    return script_path


def _slurm_state(job_id: str) -> str:
    try:
        proc = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            capture_output=True,
            text=True,
            check=False,
        )
        state = proc.stdout.strip().splitlines()
        if state and state[0]:
            return state[0].strip()
    except FileNotFoundError:
        pass
    try:
        proc = subprocess.run(
            ["sacct", "-j", job_id, "-n", "-X", "-o", "State"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in proc.stdout.splitlines():
            token = line.strip().split()[0] if line.strip() else ""
            if token:
                return token
    except FileNotFoundError:
        pass
    return "UNKNOWN"


def _slurm_active(job_id: str) -> bool:
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


def _stream_slurm(spec: JobSpec) -> Iterator[str]:
    slurm = merge_slurm_options(load_slurm_defaults(), spec.slurm or SlurmOptions())
    script_path = _write_slurm_script(spec, slurm)
    slurm_out = script_path.replace(".sh", "_slurm.out")
    output = f"Submitting Slurm job\nScript: {script_path}\n"

    try:
        proc = subprocess.run(
            ["sbatch", "--parsable", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        yield output + "\n--- FAILED ---\nsbatch not found on PATH\n"
        return
    except subprocess.CalledProcessError as exc:
        yield output + f"\n--- FAILED ---\n{exc.stderr or exc.stdout}\n"
        return

    job_id = proc.stdout.strip().split(";")[0].strip()
    output += f"Job ID: {job_id}\n"
    yield output

    log_offset = 0
    slurm_offset = 0
    while True:
        time.sleep(POLL_INTERVAL)
        chunk, log_offset = _tail_file(spec.log_path, log_offset)
        if chunk:
            output += chunk
        chunk, slurm_offset = _tail_file(slurm_out, slurm_offset)
        if chunk:
            output += chunk
        if chunk:
            yield output

        if _slurm_active(job_id):
            continue

        state = _slurm_state(job_id)
        if state in _SLURM_TERMINAL:
            chunk, log_offset = _tail_file(spec.log_path, log_offset)
            if chunk:
                output += chunk
            chunk, slurm_offset = _tail_file(slurm_out, slurm_offset)
            if chunk:
                output += chunk
            if state == "COMPLETED":
                output += "\n--- Complete ---"
            else:
                output += f"\n--- FAILED --- (Slurm state: {state})\n"
            yield output
            return


def stream_job(spec: JobSpec) -> Iterator[str]:
    """Run a job locally or via Slurm and yield growing log text."""
    mode = (spec.mode or "local").strip().lower()
    if mode == "slurm":
        yield from _stream_slurm(spec)
        return
    if spec.run_fn is not None:
        yield from _stream_local_callable(spec)
        return
    yield from _stream_local_subprocess(spec)
