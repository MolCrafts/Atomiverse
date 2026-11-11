"""``python -m atomiverse status`` — installed extension + active build."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _installed_build_info_path() -> Path:
    """``_build_info.json`` that CMake ships inside the native wheel."""
    import atomiverse

    return Path(atomiverse.__file__).resolve().parent / "_build_info.json"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_installed_build() -> Optional[Dict[str, Any]]:
    """Defines the installed extension was compiled with, or None."""
    return _read_json(_installed_build_info_path())


def load_active_build() -> Optional[Dict[str, Any]]:
    """State of the active ``config``/``compile`` build tree, or None."""
    from atomiverse.cli.build import read_active_build_dir, state_file_in_build
    from atomiverse.cli.config import ConfigError

    try:
        return _read_json(state_file_in_build(read_active_build_dir()))
    except ConfigError:
        return None


def format_status(
    installed: Optional[Dict[str, Any]],
    active: Optional[Dict[str, Any]],
    *,
    extension: bool,
    load_error: Optional[BaseException] = None,
) -> str:
    from atomiverse.cli.config import enabled_modules_from_defines

    lines = ["atomiverse status"]
    if load_error is not None:
        lines.append(f"  extension: BROKEN — failed to load: {load_error}")
    elif not extension:
        lines.append("  extension: not installed (pure-Python package)")
    elif installed is None:
        lines.append("  extension: installed (no build info — not from a native wheel)")
    else:
        defines = installed.get("defines") or {}
        modules = enabled_modules_from_defines(defines)
        lines.append("  extension: installed")
        lines.append("  modules:   " + (", ".join(modules) or "(none)"))
        for key in ("ATV_USE_MPI", "ATV_REAL_STORAGE", "ATV_REAL_WORK", "ATV_ID_SIZE"):
            if key in defines:
                lines.append(f"    {key}={defines[key]}")

    if active is None:
        lines.append("  build:     (none — run `atomiverse config …` first)")
        lines.append(
            "  hint: python -m atomiverse config md --src /path/to/Atomiverse"
        )
        return "\n".join(lines)
    lines.append(f"  build:     {active.get('build_dir', '?')}")
    lines.append(f"    source:  {active.get('source_root', '?')}")
    lines.append("    modules: " + (", ".join(active.get("modules") or []) or "(none)"))
    lines.append(f"    wheel:   {active.get('wheel') or '(not compiled)'}")
    lines.append(f"    molpy:   {active.get('molpy_wheel') or '(not compiled)'}")
    lines.append(f"    installed: {bool(active.get('installed'))}")
    return "\n".join(lines)


def run_status(*, stream=None) -> int:
    from atomiverse._binding import extension_available, extension_load_error

    out = stream if stream is not None else sys.stdout
    print(
        format_status(
            load_installed_build(),
            load_active_build(),
            extension=extension_available(),
            load_error=extension_load_error(),
        ),
        file=out,
    )
    return 0
