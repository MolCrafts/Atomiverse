"""Actual hardware, inputs and build identity; unavailable facts stay unknown."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                              check=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_info(root: Path, manifest: Path | None = None, exclude: tuple[Path, ...] = ()) -> dict:
    if manifest is not None:
        data = json.loads(manifest.read_text())
        hashes = data["source_files"]
        for name, expected in hashes.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != expected:
                raise ValueError(f"source snapshot does not match its manifest: {name}")
        if digest(hashes) != data["tree_sha256"]:
            raise ValueError("invalid source manifest digest")
        return {**data, "snapshot_verified": True, "build_verified": False}
    status = command(["git", "status", "--porcelain", "--untracked-files=normal"], root)
    diff = command(["git", "diff", "HEAD", "--binary"], root)
    files = command(["git", "ls-files", "--cached", "--others", "--exclude-standard"], root)
    commit = command(["git", "rev-parse", "HEAD"], root)
    if files is None or commit is None:
        raise ValueError("source identity unavailable; use a verified --source-manifest for a transported snapshot")
    # Includes new method/framework files; an old .so never inherits this identity.
    hashes = {name: file_hash(root / name) for name in (files or "").splitlines()
              if (root / name).is_file() and not any(
                  (root / name).resolve().is_relative_to(path.resolve()) for path in exclude)}
    return {"repository": "MolCrafts/Atomiverse",
            "commit": commit,
            "dirty": status != "", "status": status, "diff_sha256": digest(diff),
            "tree_sha256": digest(hashes), "source_files": hashes,
            "build_verified": False}


def hardware_info() -> dict:
    system = platform.system()
    cpu = platform.processor() or None
    physical = memory = None
    if system == "Darwin":
        cpu = command(["sysctl", "-n", "machdep.cpu.brand_string"])
        physical = command(["sysctl", "-n", "hw.physicalcpu"])
        memory = command(["sysctl", "-n", "hw.memsize"])
    elif Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in
                    Path("/proc/cpuinfo").read_text().splitlines() if "model name" in line), cpu)
        memory = next((int(line.split()[1]) * 1024 for line in
                       Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")), None)
        topology = {p.read_text().strip() for p in Path("/sys/devices/system/cpu").glob(
            "cpu[0-9]*/topology/thread_siblings_list")}
        physical = len(topology) or None
    gpus = command(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
                    "--format=csv,noheader,nounits"])
    return {"os": platform.platform(), "architecture": platform.machine(),
            "cpu_model": cpu, "physical_cores": int(physical) if physical else None,
            "logical_cpus": os.cpu_count(), "host_memory_bytes": int(memory) if memory else None,
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "gpus": gpus.splitlines() if gpus else [], "gpu_query_available": gpus is not None,
            "scheduler": {key: os.environ[key] for key in (
                "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
                "CUDA_VISIBLE_DEVICES") if key in os.environ},
            "python": platform.python_version()}
