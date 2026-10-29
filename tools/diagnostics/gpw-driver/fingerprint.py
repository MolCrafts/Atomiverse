#!/usr/bin/env python3
"""Fingerprint the local source scope used by the opt-in GPW checks."""
import hashlib
import json
from pathlib import Path
import sys

paths = set()
for name in ("include/atomiverse", "src", "tests/cpu/gpw", "tests/cpu/planewave",
             "tests/cuda/gpw", "tests/cuda/planewave", "cmake"):
    paths.update(p for p in Path(name).rglob("*") if p.is_file())
for name in ("gpw-driver", "gpw-residency", "pbe-family", "cuda-pw-public",
             "pbe-periodic-oracles", "pw-analytic-stress"):
    paths.update(p for p in (Path("tools/diagnostics") / name).glob("*") if p.is_file())
for name in ("CMakeLists.txt", "tests/CMakeLists.txt", "examples/nc-extension/pins.h",
             "tests/cpu/scf/test_xc_functional.cpp", "tests/cpu/planewave/test_xc_builder.cpp",
             "tools/diagnostics/pw-analytic-stress/qe-2026-10-28/pins.h",
             "tools/diagnostics/pw-analytic-stress/qe-2026-10-28/SHA256SUMS.json",
             "tools/diagnostics/pw-analytic-stress/qe-k8-2026-10-28/SHA256SUMS.json",
             "tools/diagnostics/pw-analytic-stress/rawrefs/manifest.json",
             "tools/diagnostics/pw-analytic-stress/qe-gamma-pbe-fixed-2026-10-28/pins.h",
             "tools/diagnostics/pw-analytic-stress/qe-gamma-pbe-fixed-2026-10-28/reference.json",
             "tools/diagnostics/pw-analytic-stress/gaussian-rawrefs/manifest.json",
             "tools/diagnostics/pw-analytic-stress/fixed-k-rawrefs/manifest.json",
             "tools/diagnostics/pw-analytic-stress/empty-beta-rawrefs/manifest.json",
             "tools/diagnostics/pw-analytic-stress/qe-gamma-rks-fd-2026-10-28/pins.h",
             "tools/diagnostics/pw-analytic-stress/qe-gamma-rks-fd-2026-10-28/reference.json",
             "tools/diagnostics/pw-analytic-stress/qe-gamma-uks-fixed-2026-10-28/pins.h",
             "tools/diagnostics/pw-analytic-stress/qe-gamma-uks-fixed-2026-10-28/references.json",
             "tools/diagnostics/pw-analytic-stress/qe-gamma-uks-fixed-2026-10-28/portable-files.json",
             "tools/diagnostics/pw-analytic-stress/qe-gamma-uks-fixed-2026-10-28/reference-family-audit.json"):
    paths.add(Path(name))
Path(sys.argv[1]).write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(paths)}, indent=2) + "\n")
