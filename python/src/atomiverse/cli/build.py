"""Native build pipeline aligned with CMake: config → compile → install.

Streams **raw** cmake output to the calling terminal (no capture).

Build tree defaults to ``<source_root>/build`` (normal cmake layout).
Full state lives in ``<build_dir>/atomiverse_cli_state.json``.
A tiny pointer file next to the installed package records which build_dir
is active for subsequent ``compile`` / ``install`` / ``status`` calls.
There is **no** home-directory cache.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from atomiverse.cli.config import (
    ConfigError,
    InstallConfig,
    define_fingerprint,
    enabled_modules_from_defines,
    resolve_defines,
)
from atomiverse.cli.source import SourceSpec

# Filename for full state inside the cmake binary dir.
_STATE_NAME = "atomiverse_cli_state.json"
# Pointer in the installed pure package: {"build_dir": "…"}.
_ACTIVE_POINTER_NAME = "_active_build.json"


def default_build_dir(source_root: Path) -> Path:
    """Default cmake ``-B`` directory: ``<src>/build``."""
    return source_root.resolve() / "build"


def state_file_in_build(build_dir: Path) -> Path:
    """Full CLI state path inside the build tree."""
    return Path(build_dir).resolve() / _STATE_NAME


def active_build_pointer_path() -> Path:
    """Small pointer file beside the installed ``atomiverse`` package."""
    import atomiverse

    return Path(atomiverse.__file__).resolve().parent / _ACTIVE_POINTER_NAME


def write_active_build_pointer(build_dir: Path) -> Path:
    """Record which build tree subsequent compile/install should use."""
    path = active_build_pointer_path()
    path.write_text(
        json.dumps({"build_dir": str(Path(build_dir).resolve())}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


def read_active_build_dir() -> Path:
    """Resolve the active build dir from the package pointer."""
    path = active_build_pointer_path()
    if not path.is_file():
        raise ConfigError(
            "no active build — run `python -m atomiverse config … --src …` first"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        bdir = Path(data["build_dir"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ConfigError(f"unreadable active-build pointer {path}: {exc}") from exc
    return bdir.resolve()


def state_path() -> Path:
    """Path of the active full build state (inside the build tree)."""
    return state_file_in_build(read_active_build_dir())


# Back-compat alias
def install_state_path() -> Path:
    return state_path()


@dataclass
class BuildState:
    """Persisted between ``config`` / ``compile`` / ``install``."""

    version: str
    source_root: str  # Atomiverse repo root (contains CMakeLists.txt)
    python_dir: str  # python/ package dir
    build_dir: str
    defines: Dict[str, str] = field(default_factory=dict)
    modules: List[str] = field(default_factory=list)
    fingerprint: str = ""
    configured: bool = False
    compiled: bool = False
    installed: bool = False

    def save(self, path: Optional[Path] = None) -> Path:
        bdir = Path(self.build_dir)
        p = path if path is not None else state_file_in_build(bdir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Keep package pointer so compile/install find this tree.
        try:
            write_active_build_pointer(bdir)
        except OSError as exc:
            print(
                f"warning: could not write active-build pointer: {exc}",
                file=sys.stderr,
            )
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BuildState":
        if path is not None:
            p = Path(path)
        else:
            p = state_file_in_build(read_active_build_dir())
        if not p.is_file():
            raise ConfigError(
                "no build state — run `python -m atomiverse config …` first "
                f"(looked for {p})"
            )
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"unreadable build state {p}: {exc}") from exc
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def repo_root_from_source(source: SourceSpec) -> Path:
    """Locate Atomiverse CMake root from a resolved :class:`SourceSpec`.

    *source.location* is the ``python/`` package directory for local trees.
    """
    loc = Path(source.location).resolve()
    if not source.is_url:
        # python/ → parent is repo root
        if (loc / "CMakeLists.txt").is_file():
            return loc
        if (loc.parent / "CMakeLists.txt").is_file():
            return loc.parent
        if loc.name == "python" and (loc.parent / "CMakeLists.txt").is_file():
            return loc.parent
        raise ConfigError(
            f"cannot find CMakeLists.txt for source {loc} "
            "(pass --src to an Atomiverse checkout)"
        )
    raise ConfigError(
        "remote --url source is not yet supported for the staged cmake "
        "pipeline; use --src /path/to/Atomiverse for config/compile/install"
    )


def build_cmake_configure_command(
    source_root: Path,
    build_dir: Path,
    defines: Mapping[str, str],
    *,
    python: Optional[str] = None,
) -> List[str]:
    """Assemble ``cmake -S … -B … -D…`` (no execution)."""
    if defines.get("ATV_BUILD_PYTHON", "").upper() not in ("ON", "TRUE", "1"):
        raise ConfigError("config requires ATV_BUILD_PYTHON=ON")

    exe = python or sys.executable
    cmd: List[str] = [
        "cmake",
        "-S",
        str(source_root),
        "-B",
        str(build_dir),
        f"-DPython_EXECUTABLE={exe}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    for key in sorted(defines):
        cmd.append(f"-D{key}={defines[key]}")
    return cmd


def build_cmake_compile_command(
    build_dir: Path,
    *,
    jobs: Optional[int] = None,
    target: str = "_atomiverse",
) -> List[str]:
    cmd: List[str] = ["cmake", "--build", str(build_dir), "--target", target]
    if jobs is not None and jobs > 0:
        cmd.extend(["-j", str(jobs)])
    else:
        cmd.append("-j")
    return cmd


def build_cmake_install_command(
    build_dir: Path,
    prefix: Path,
) -> List[str]:
    """``cmake --install`` into *prefix* (typically a site-packages dir)."""
    return [
        "cmake",
        "--install",
        str(build_dir),
        "--prefix",
        str(prefix),
    ]


def site_packages_prefix() -> Path:
    """Directory that should receive ``atomiverse/_atomiverse*.so``.

    CMake installs to ``${PREFIX}/atomiverse/``, so PREFIX is site-packages.
    """
    import site

    # Prefer the purelib for the running interpreter.
    try:
        from sysconfig import get_paths

        pure = get_paths().get("purelib")
        if pure:
            return Path(pure)
    except Exception:
        pass
    paths = site.getsitepackages()
    if not paths:
        raise ConfigError("cannot determine site-packages for cmake --install")
    return Path(paths[0])


def _run_streaming(cmd: Sequence[str], *, dry_run: bool = False) -> int:
    """Run *cmd* with inherited stdio (raw cmake/pip output)."""
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        print("(dry-run: not executed)", flush=True)
        return 0
    proc = subprocess.run(list(cmd), check=False)
    return int(proc.returncode)


def run_config(
    cfg: InstallConfig,
    source: SourceSpec,
    *,
    build_dir: Optional[Path] = None,
    dry_run: bool = False,
    python: Optional[str] = None,
) -> int:
    """CMake configure stage (``cmake -S -B -D…``)."""
    defines = resolve_defines(cfg)
    defines["ATV_BUILD_PYTHON"] = "ON"
    fp = define_fingerprint(defines)
    modules = enabled_modules_from_defines(defines)

    source_root = repo_root_from_source(source)
    python_dir = Path(source.location).resolve()
    bdir = (
        build_dir.expanduser().resolve()
        if build_dir is not None
        else default_build_dir(source_root)
    )

    cmd = build_cmake_configure_command(
        source_root, bdir, defines, python=python
    )

    print("atomiverse config", flush=True)
    print(f"  version:     {source.version}", flush=True)
    print(f"  source_root: {source_root}", flush=True)
    print(f"  build_dir:   {bdir}", flush=True)
    print(f"  modules:     {', '.join(modules)}", flush=True)
    print(f"  fingerprint: {fp}", flush=True)
    print("  cmake.define:", flush=True)
    for k in sorted(defines):
        print(f"    {k}={defines[k]}", flush=True)

    if not dry_run:
        bdir.mkdir(parents=True, exist_ok=True)

    code = _run_streaming(cmd, dry_run=dry_run)
    if code != 0:
        print(f"cmake configure failed with exit code {code}", file=sys.stderr)
        return code

    state = BuildState(
        version=source.version,
        source_root=str(source_root),
        python_dir=str(python_dir),
        build_dir=str(bdir),
        defines=dict(sorted(defines.items())),
        modules=modules,
        fingerprint=fp,
        configured=not dry_run,
        compiled=False,
        installed=False,
    )
    if not dry_run:
        state_file = state.save()
        print(f"  state: {state_file}", flush=True)
        print(f"  active: {active_build_pointer_path()}", flush=True)
    print("config complete" if not dry_run else "config dry-run complete", flush=True)
    return 0


def run_compile(
    *,
    jobs: Optional[int] = None,
    dry_run: bool = False,
    state_file: Optional[Path] = None,
) -> int:
    """CMake build stage (``cmake --build``)."""
    state = BuildState.load(state_file)
    if not state.configured and not dry_run:
        # Allow compile if build dir already has cache from a prior config
        cache = Path(state.build_dir) / "CMakeCache.txt"
        if not cache.is_file():
            raise ConfigError(
                "build tree is not configured — run `atomiverse config …` first"
            )

    cmd = build_cmake_compile_command(Path(state.build_dir), jobs=jobs)
    print("atomiverse compile", flush=True)
    print(f"  build_dir: {state.build_dir}", flush=True)
    print(f"  modules:   {', '.join(state.modules)}", flush=True)

    code = _run_streaming(cmd, dry_run=dry_run)
    if code != 0:
        print(f"cmake --build failed with exit code {code}", file=sys.stderr)
        return code

    if not dry_run:
        state.compiled = True
        state.save(state_file)
    print(
        "compile complete" if not dry_run else "compile dry-run complete",
        flush=True,
    )
    return 0


def run_install_stage(
    *,
    dry_run: bool = False,
    state_file: Optional[Path] = None,
    prefix: Optional[Path] = None,
) -> int:
    """CMake install stage (``cmake --install`` into site-packages)."""
    state = BuildState.load(state_file)
    if not state.compiled and not dry_run:
        # Soft check: require build dir
        if not Path(state.build_dir).is_dir():
            raise ConfigError(
                "nothing to install — run `atomiverse compile` first"
            )

    pref = prefix if prefix is not None else site_packages_prefix()
    cmd = build_cmake_install_command(Path(state.build_dir), pref)

    print("atomiverse install", flush=True)
    print(f"  build_dir: {state.build_dir}", flush=True)
    print(f"  prefix:    {pref}", flush=True)
    print(f"  modules:   {', '.join(state.modules)}", flush=True)

    code = _run_streaming(cmd, dry_run=dry_run)
    if code != 0:
        print(f"cmake --install failed with exit code {code}", file=sys.stderr)
        return code

    if not dry_run:
        state.installed = True
        state.save(state_file)
        # Also write a compact status view next to the package when possible
        try:
            _write_status_sidecar(state)
        except OSError as exc:
            print(f"warning: could not write status sidecar: {exc}", file=sys.stderr)

    print(
        "install complete" if not dry_run else "install dry-run complete",
        flush=True,
    )
    return 0


def _write_status_sidecar(state: BuildState) -> None:
    """Write status-friendly JSON used by ``atomiverse status``."""
    import atomiverse

    path = Path(atomiverse.__file__).resolve().parent / "_install_state.json"
    payload = {
        "modules": state.modules,
        "defines": state.defines,
        "fingerprint": state.fingerprint,
        "build_dir": state.build_dir,
        "source_root": state.source_root,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def dry_run_defines(cfg: InstallConfig) -> Dict[str, str]:
    """Resolve defines for unit tests without invoking cmake."""
    defines = resolve_defines(cfg)
    defines["ATV_BUILD_PYTHON"] = "ON"
    return defines


# ── legacy names kept for test imports during migration ─────────────────────


def write_install_state(defines: Mapping[str, str]) -> Path:
    """Write status sidecar only (tests / status)."""
    import atomiverse

    path = Path(atomiverse.__file__).resolve().parent / "_install_state.json"
    payload = {
        "modules": enabled_modules_from_defines(defines),
        "defines": dict(sorted(defines.items())),
        "fingerprint": define_fingerprint(defines),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
