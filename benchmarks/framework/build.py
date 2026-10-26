"""CMake build receipts tie an observed native library to unchanged source inputs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .execution import write_json
from .provenance import command, file_hash, source_info


def attest_build(root: Path, build: Path, out: Path, *, jobs: int = 2,
                 source_manifest: Path | None = None) -> dict:
    root, build, out = root.resolve(), build.resolve(), out.resolve()
    cache_path = build / "CMakeCache.txt"
    cache = {}
    for line in cache_path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and ":" in key and not key.startswith(("#", "//")):
            cache[key.split(":", 1)[0]] = value
    if Path(cache.get("CMAKE_HOME_DIRECTORY", "/invalid")).resolve() != root:
        raise ValueError("CMake build directory belongs to a different source tree")
    before = source_info(root, source_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".build.log")
    with log.open("w") as stream:
        subprocess.run(["cmake", "--build", str(build), "--target", "_atomiverse", "--parallel", str(jobs)],
                       cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
    after = source_info(root, source_manifest)
    if before["tree_sha256"] != after["tree_sha256"]:
        raise ValueError("source changed during the attested build")
    binaries = list(build.glob("_atomiverse*.so")) + list(build.glob("_atomiverse*.pyd"))
    if len(binaries) != 1:
        raise ValueError("expected exactly one native extension in the build directory")
    partner = Path(cache.get("ATV_MOLPY_PATH", "/invalid"))
    receipt = {"schema": "atomiverse.build-receipt/1", "source_tree_sha256": before["tree_sha256"],
               "commit": before["commit"], "dirty": before["dirty"],
               "binary_sha256": file_hash(binaries[0]), "binary_path": str(binaries[0]),
               "cmake_cache_sha256": file_hash(cache_path), "build_log_sha256": file_hash(log),
               "cmake_settings": {key: value for key, value in cache.items() if key.startswith(("ATV_", "CMAKE_CXX_", "CMAKE_CUDA_", "Rust_TOOLCHAIN"))},
               "molpy_source_commit": command(["git", "rev-parse", "HEAD"], partner),
               "molpy_cargo_lock_sha256": file_hash(partner / "Cargo.lock") if (partner / "Cargo.lock").is_file() else None}
    write_json(out, receipt)
    return receipt


def verify_receipt(path: Path, source: dict) -> dict:
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != "atomiverse.build-receipt/1":
        raise ValueError("unsupported build receipt")
    if receipt.get("source_tree_sha256") != source["tree_sha256"] or receipt.get("commit") != source["commit"]:
        raise ValueError("build receipt does not match current source")
    binary = receipt.get("binary_sha256", "")
    if len(binary) != 64 or any(c not in "0123456789abcdef" for c in binary):
        raise ValueError("build receipt has no valid native binary fingerprint")
    return receipt
