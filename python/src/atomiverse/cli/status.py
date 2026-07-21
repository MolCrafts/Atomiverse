"""``python -m atomiverse status`` — modules / fingerprint / build dirs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

def _package_status_path() -> Path:
    import atomiverse

    return Path(atomiverse.__file__).resolve().parent / "_install_state.json"


def load_install_state(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load build or install state JSON.

    Prefers explicit *path*, then package ``_install_state.json``, then
    ``<build_dir>/atomiverse_cli_state.json`` via the active-build pointer.
    Never uses a home-directory cache.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(path)
    else:
        candidates.append(_package_status_path())
        try:
            from atomiverse.cli.build import (
                read_active_build_dir,
                state_file_in_build,
            )

            candidates.append(state_file_in_build(read_active_build_dir()))
        except Exception:
            pass
    for p in candidates:
        if not p.is_file():
            continue
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def format_status(state: Optional[Dict[str, Any]]) -> str:
    lines = ["atomiverse status"]
    if state is None:
        lines.append("  modules: (none — run `atomiverse config …` first)")
        lines.append("  fingerprint: (none)")
        lines.append(
            "  hint: python -m atomiverse config md --mpi --src /path/to/Atomiverse"
        )
        return "\n".join(lines)

    modules = state.get("modules") or []
    if isinstance(modules, list) and modules:
        lines.append("  modules: " + ", ".join(modules))
    else:
        lines.append("  modules: (none)")

    fp = state.get("fingerprint") or ""
    lines.append(f"  fingerprint: {fp if fp else '(none)'}")

    for key in ("source_root", "build_dir"):
        if state.get(key):
            lines.append(f"  {key}: {state[key]}")

    for flag in ("configured", "compiled", "installed"):
        if flag in state:
            lines.append(f"  {flag}: {state[flag]}")

    defines = state.get("defines") or {}
    if isinstance(defines, dict) and defines:
        lines.append("  defines:")
        for key in sorted(defines):
            lines.append(f"    {key}={defines[key]}")
    return "\n".join(lines)


def run_status(*, path: Optional[Path] = None, stream=None) -> int:
    out = stream if stream is not None else sys.stdout
    state = load_install_state(path)
    print(format_status(state), file=out)
    return 0
