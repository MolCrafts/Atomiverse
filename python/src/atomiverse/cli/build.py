"""Native build pipeline: config → compile → install, owned by pip.

``config`` resolves product modules into CMake defines and records them.
``compile`` runs ``pip wheel`` on the checkout's ``python/`` project with
``wheel.cmake=true`` — scikit-build-core drives CMake in a persistent build
tree, so a repeat compile is incremental — and on the checkout's
``<ATV_MOLPY_PATH>/molpy``, leaving two wheels in ``<build_dir>/dist``.
``install`` hands both to ``pip install``, so the extension is a
RECORD-tracked file of the distribution: ``pip uninstall`` removes it and an
upgrade replaces it.

molpy and the native bridge are built from the same local molpy checkout.
ATV_MOLPY_PATH selects that repository; its default is the sibling ../molpy.
A Frame capsule from a separately built molpy fails the capsule-name check.

Raw pip / cmake output streams to the terminal (no capture). State lives in
``<build_dir>/atomiverse_cli_state.json``; the active build dir is recorded
per environment in ``<sys.prefix>/share/atomiverse/active_build.json``,
never inside the installed package and never in the home directory.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
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
from atomiverse.cli.source import SourceSpec, molpy_source_root

_STATE_NAME = "atomiverse_cli_state.json"

# Build-time packages `pip wheel --no-build-isolation` needs in this env.
_NATIVE_REQUIRES = {
    "scikit_build_core": "scikit-build-core>=0.10",
    "nanobind": "nanobind>=2.4",
    "maturin": "maturin>=1.0,<2.0",
}


def default_build_dir(source_root: Path) -> Path:
    """Default build directory: ``<src>/build-python``."""
    return source_root.resolve() / "build-python"


def state_file_in_build(build_dir: Path) -> Path:
    """Full CLI state path inside the build tree."""
    return Path(build_dir).resolve() / _STATE_NAME


def active_build_pointer_path() -> Path:
    """Per-environment pointer to the active build directory."""
    return Path(sys.prefix) / "share" / "atomiverse" / "active_build.json"


def read_active_build_dir() -> Path:
    """Resolve the active build dir from the environment pointer."""
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


@dataclass
class BuildState:
    """Persisted between ``config`` / ``compile`` / ``install``."""

    version: str
    source_root: str  # Atomiverse repo root (contains CMakeLists.txt)
    python_dir: str  # python/ project pip builds
    build_dir: str
    defines: Dict[str, str] = field(default_factory=dict)
    modules: List[str] = field(default_factory=list)
    fingerprint: str = ""
    wheel: str = ""  # native wheel produced by compile
    molpy_wheel: str = ""  # molpy wheel built from the local molpy checkout by compile
    installed: bool = False

    def save(self, path: Optional[Path] = None) -> Path:
        p = path if path is not None else state_file_in_build(Path(self.build_dir))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BuildState":
        p = Path(path) if path is not None else state_file_in_build(
            read_active_build_dir()
        )
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


def build_wheel_command(
    python_dir: Path,
    build_dir: Path,
    defines: Mapping[str, str],
    *,
    python: Optional[str] = None,
) -> List[str]:
    """Assemble the ``pip wheel`` command for the native wheel (no execution).

    Every define is passed as ``-C cmake.define.KEY=VALUE``, overriding the
    pyproject defaults, so the wheel holds exactly the requested modules.
    """
    if defines.get("ATV_BUILD_PYTHON", "").upper() not in ("ON", "TRUE", "1"):
        raise ConfigError("native wheel requires ATV_BUILD_PYTHON=ON")
    cmd: List[str] = [
        python or sys.executable,
        "-m",
        "pip",
        "wheel",
        str(python_dir),
        "--verbose",  # stream the backend's CMake output instead of a spinner
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(build_dir / "dist"),
        "-C",
        "wheel.cmake=true",
        "-C",
        f"build-dir={build_dir / 'cmake'}",
        "-C",
        "cmake.build-type=Release",
    ]
    for key in sorted(defines):
        cmd.extend(["-C", f"cmake.define.{key}={defines[key]}"])
    return cmd


def build_molpy_wheel_command(
    source_root: Path, build_dir: Path, *, python: Optional[str] = None, molpy_root: Optional[Path] = None
) -> List[str]:
    """Assemble the ``pip wheel`` command for molpy from the local molpy checkout.

    Run with ``CARGO_TARGET_DIR=<build_dir>/molpy-target`` (see
    ``molpy_target_dir``) so a repeat compile reuses the cargo build.
    """
    return [
        python or sys.executable,
        "-m",
        "pip",
        "wheel",
        str((molpy_root if molpy_root is not None else molpy_source_root(source_root)) / "molpy"),
        "--verbose",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(build_dir / "dist"),
    ]


def molpy_target_dir(build_dir: Path) -> Path:
    """Persistent cargo target directory of the molpy wheel build."""
    return build_dir / "molpy-target"


def build_install_command(
    wheels: Sequence[Path], *, python: Optional[str] = None
) -> List[str]:
    """``pip install`` of the native and molpy wheels.

    ``--force-reinstall`` because the pure and native wheels share a
    version, and the local molpy wheel may share the installed one's;
    ``--no-deps`` because the pure install already resolved them.
    """
    return [
        python or sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--force-reinstall",
        *(str(w) for w in wheels),
    ]


def _require_native_build_deps() -> None:
    missing = [
        req
        for mod, req in _NATIVE_REQUIRES.items()
        if importlib.util.find_spec(mod) is None
    ]
    if missing:
        raise ConfigError(
            "native build needs "
            + ", ".join(missing)
            + " in this environment — run `pip install 'atomiverse[native]'`"
        )


def build_env(jobs: Optional[int] = None) -> Dict[str, str]:
    """Environment of the two ``pip wheel`` builds.

    maturin's PEP 517 hook execs the ``maturin`` executable from PATH, and an
    environment's scripts directory is on PATH only when it is activated —
    ``.venv/bin/python -m atomiverse compile`` is not. Putting this
    interpreter's scripts directory first makes the build use the tools the
    native extra installed beside it.
    """
    env = dict(os.environ)
    scripts = sysconfig.get_path("scripts")
    env["PATH"] = os.pathsep.join(p for p in (scripts, env.get("PATH", "")) if p)
    if jobs is not None and jobs > 0:
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    return env


def _run_streaming(
    cmd: Sequence[str],
    *,
    dry_run: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Run *cmd* with inherited stdio (raw pip/cmake output)."""
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        print("(dry-run: not executed)", flush=True)
        return 0
    proc = subprocess.run(list(cmd), check=False, env=env)
    return int(proc.returncode)


def run_config(
    cfg: InstallConfig,
    source: SourceSpec,
    *,
    build_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    """Resolve defines and record the build (no compilation)."""
    defines = resolve_defines(cfg)
    defines["ATV_MOLPY_PATH"] = str(molpy_source_root(source.root))
    fp = define_fingerprint(defines)
    modules = enabled_modules_from_defines(defines)
    bdir = (
        build_dir.expanduser().resolve()
        if build_dir is not None
        else default_build_dir(source.root)
    )

    print("atomiverse config", flush=True)
    print(f"  version:     {source.version}", flush=True)
    print(f"  source_root: {source.root}", flush=True)
    print(f"  build_dir:   {bdir}", flush=True)
    print(f"  modules:     {', '.join(modules)}", flush=True)
    print(f"  fingerprint: {fp}", flush=True)
    print("  cmake.define:", flush=True)
    for k in sorted(defines):
        print(f"    {k}={defines[k]}", flush=True)

    if dry_run:
        print("compile will run:", flush=True)
        print(
            "+ " + " ".join(build_wheel_command(source.python_dir, bdir, defines)),
            flush=True,
        )
        print("+ " + " ".join(build_molpy_wheel_command(source.root, bdir)), flush=True)
        print("config dry-run complete", flush=True)
        return 0

    state = BuildState(
        version=source.version,
        source_root=str(source.root),
        python_dir=str(source.python_dir),
        build_dir=str(bdir),
        defines=dict(sorted(defines.items())),
        modules=modules,
        fingerprint=fp,
    )
    print(f"  state:  {state.save()}", flush=True)
    pointer = active_build_pointer_path()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps({"build_dir": str(bdir)}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  active: {pointer}", flush=True)
    print("config complete", flush=True)
    return 0


def run_compile(
    *,
    jobs: Optional[int] = None,
    dry_run: bool = False,
    state_file: Optional[Path] = None,
) -> int:
    """Build the native wheel (``pip wheel`` → scikit-build-core → CMake)."""
    state = BuildState.load(state_file)
    bdir = Path(state.build_dir)
    if "ATV_MOLPY_PATH" not in state.defines:
        raise ConfigError("build state has no molpy source; run atomiverse config again")
    cmd = build_wheel_command(Path(state.python_dir), bdir, state.defines)
    env = build_env(jobs)

    print("atomiverse compile", flush=True)
    print(f"  build_dir: {bdir}", flush=True)
    print(f"  modules:   {', '.join(state.modules)}", flush=True)
    if not dry_run:
        _require_native_build_deps()
        # One wheel in dist/ at a time, so install never picks a stale one.
        shutil.rmtree(bdir / "dist", ignore_errors=True)

    code = _run_streaming(cmd, dry_run=dry_run, env=env)
    if code != 0:
        print(f"pip wheel failed with exit code {code}", file=sys.stderr)
        return code

    molpy_env = dict(env)
    molpy_env["CARGO_TARGET_DIR"] = str(molpy_target_dir(bdir))
    if jobs is not None and jobs > 0:
        molpy_env["CARGO_BUILD_JOBS"] = str(jobs)
    molpy_cmd = build_molpy_wheel_command(
        Path(state.source_root), bdir, molpy_root=Path(state.defines["ATV_MOLPY_PATH"])
    )
    code = _run_streaming(molpy_cmd, dry_run=dry_run, env=molpy_env)
    if code != 0:
        print(f"pip wheel of molpy failed with exit code {code}", file=sys.stderr)
        return code
    if dry_run:
        print("compile dry-run complete", flush=True)
        return 0

    state.wheel = str(_one_wheel(bdir / "dist", "atomiverse-*.whl"))
    state.molpy_wheel = str(_one_wheel(bdir / "dist", "molcrafts_molpy-*.whl"))
    state.installed = False
    state.save(state_file)
    print(f"  wheel:     {state.wheel}", flush=True)
    print(f"  molpy:     {state.molpy_wheel}", flush=True)
    print("compile complete", flush=True)
    return 0


def _one_wheel(dist: Path, pattern: str) -> Path:
    wheels = sorted(dist.glob(pattern))
    if len(wheels) != 1:
        raise ConfigError(f"expected one {pattern} in {dist}, found {len(wheels)}")
    return wheels[0]


def run_install_stage(
    *,
    dry_run: bool = False,
    state_file: Optional[Path] = None,
) -> int:
    """Install the native and molpy wheels into this environment with pip."""
    state = BuildState.load(state_file)
    wheels = [state.wheel, state.molpy_wheel]
    if not all(w and Path(w).is_file() for w in wheels):
        if not dry_run:
            raise ConfigError("no compiled wheels — run `atomiverse compile` first")
    cmd = build_install_command(
        [Path(state.wheel or "<wheel>"), Path(state.molpy_wheel or "<molpy-wheel>")]
    )

    print("atomiverse install", flush=True)
    print(f"  wheel:   {state.wheel or '(not compiled yet)'}", flush=True)
    print(f"  molpy:   {state.molpy_wheel or '(not compiled yet)'}", flush=True)
    print(f"  modules: {', '.join(state.modules)}", flush=True)

    code = _run_streaming(cmd, dry_run=dry_run)
    if code != 0:
        print(f"pip install failed with exit code {code}", file=sys.stderr)
        return code
    if dry_run:
        print("install dry-run complete", flush=True)
        return 0
    state.installed = True
    state.save(state_file)
    print("install complete", flush=True)
    return 0
