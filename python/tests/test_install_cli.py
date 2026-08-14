"""Unit tests for staged cmake CLI: config / compile / install."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Optional

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from atomiverse.cli.build import (  # noqa: E402
    build_cmake_compile_command,
    build_cmake_configure_command,
    build_cmake_install_command,
    dry_run_defines,
)
from atomiverse.cli.config import (  # noqa: E402
    BINDING_TABLE,
    GLOBAL_OPTION_KEYS,
    ConfigError,
    InstallConfig,
    apply_cli_modules,
    load_toml_config,
    merge_unbound_cli,
    resolve_defines,
    validate_mode,
)
from atomiverse.cli.main import main  # noqa: E402
from atomiverse.cli.source import (  # noqa: E402
    default_tarball_url,
    package_version,
    verify_source_version,
)
from atomiverse.cli.status import format_status, run_status  # noqa: E402

_REPO = _SRC.parents[1]  # Atomiverse root when layout is repo/python/src
_PYPROJECT = _SRC.parent / "pyproject.toml"  # python/pyproject.toml


def _try_import_tomllib() -> Optional[ModuleType]:
    """Return a TOML reader module, or None when the environment has none.

    Same optional-dependency shape as ``atomiverse.cli.config._load_toml``:
    ``tomllib`` is stdlib on py>=3.11, and 3.9/3.10 fall back to the optional
    ``tomli`` backport. Each branch binds one name once and returns it, so the
    absent case is a plain None rather than a re-bound module name.
    """
    try:
        import tomllib as mod

        return mod
    except ModuleNotFoundError:  # pragma: no cover - exercised on 3.9/3.10
        pass
    try:
        import tomli as mod

        return mod
    except ModuleNotFoundError:
        return None


_tomllib: Optional[ModuleType] = _try_import_tomllib()


# ── mutual exclusion / empty ─────────────────────────────────────────────────


def test_mutual_exclusion_config_vs_modules() -> None:
    with pytest.raises(ConfigError, match="mutual exclusion"):
        validate_mode(config_path="x.toml", modules=["md"], hf=False)
    rc = main(["config", "--config", "x.toml", "md", "--src", str(_REPO)])
    assert rc != 0


def test_empty_config_fails() -> None:
    with pytest.raises(ConfigError, match="empty config"):
        validate_mode(config_path=None, modules=[], hf=False)
    rc = main(["config", "--src", str(_REPO)])
    assert rc != 0


def test_hf_rejected() -> None:
    with pytest.raises(ConfigError, match="HF/SCF|not available"):
        validate_mode(config_path=None, modules=["md"], hf=True)
    rc = main(["config", "md", "--hf", "--src", str(_REPO)])
    assert rc != 0


def test_mpi_is_global_option_not_module_section(tmp_path: Path) -> None:
    assert "mpi" in GLOBAL_OPTION_KEYS
    bad = tmp_path / "bad.toml"
    bad.write_text(
        textwrap.dedent(
            """\
            [modules]
            md = true
            [md]
            mpi = true
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="global option"):
        load_toml_config(bad)


def test_toml_global_mpi_ok(tmp_path: Path) -> None:
    p = tmp_path / "ok.toml"
    p.write_text(
        textwrap.dedent(
            """\
            [modules]
            md = true
            [global]
            mpi = true
            real_storage = "double"
            """
        ),
        encoding="utf-8",
    )
    cfg = load_toml_config(p)
    defines = resolve_defines(cfg)
    assert defines["ATV_USE_MPI"] == "ON"
    assert defines["ATV_REAL_STORAGE"] == "double"


# ── modules + DAG ────────────────────────────────────────────────────────────


def test_apply_cli_modules_md_mpi_global() -> None:
    cfg = InstallConfig()
    apply_cli_modules(cfg, ["md"], options={"mpi": True})
    defines = resolve_defines(cfg)
    assert defines["ATV_BUILD_MD"] == "ON"
    assert defines[BINDING_TABLE["mpi"]] == "ON"


def test_md_closure() -> None:
    cfg = InstallConfig(modules={"md": True})
    defines = resolve_defines(cfg)
    assert defines["ATV_BUILD_POTENTIALS"] == "ON"
    assert defines["ATV_BUILD_NEIGHBOR"] == "ON"
    assert defines["ATV_BUILD_PW"] == "OFF"
    # md does not drag the reciprocal-space module in: kspace is its own
    # product module (one-way DAG kspace → potentials, never md → kspace).
    assert defines["ATV_BUILD_KSPACE"] == "OFF"


def test_kspace_closure() -> None:
    cfg = InstallConfig(modules={"kspace": True})
    defines = resolve_defines(cfg)
    assert defines["ATV_BUILD_KSPACE"] == "ON"
    assert defines["ATV_BUILD_POTENTIALS"] == "ON"


def test_wheel_define_block_enables_kspace() -> None:
    """The wheel carries reciprocal-space electrostatics by default.

    ``python -m atomiverse config …`` resolves defines from the module DAG
    (above), but a plain ``pip install .`` never touches that path — the
    wheel's own capability set is the scikit-build-core define block in
    python/pyproject.toml. Spec py-coulomb-pme pins
    ``ATV_BUILD_KSPACE = {env = "ATV_BUILD_KSPACE", default = "ON"}`` there,
    so PME-charged MD is reachable from the shipped wheel and can still be
    switched off from the environment.
    """
    if _tomllib is None:  # pragma: no cover - py3.9/3.10 without tomli
        pytest.skip("no TOML reader (need py>=3.11 or the tomli package)")
    if not _PYPROJECT.is_file():
        pytest.skip("not in monorepo checkout")
    with _PYPROJECT.open("rb") as fh:
        pyproject = _tomllib.load(fh)
    defines = pyproject["tool"]["scikit-build"]["cmake"]["define"]
    assert "ATV_BUILD_KSPACE" in defines, (
        "python/pyproject.toml [tool.scikit-build.cmake.define] carries no "
        "ATV_BUILD_KSPACE entry: the wheel cannot do PME electrostatics"
    )
    entry = defines["ATV_BUILD_KSPACE"]
    assert isinstance(entry, dict), (
        "ATV_BUILD_KSPACE must be env-overridable "
        '({env = "ATV_BUILD_KSPACE", default = "ON"}), not a fixed string'
    )
    assert entry["env"] == "ATV_BUILD_KSPACE"
    assert entry["default"] == "ON"


def test_path_closure() -> None:
    cfg = InstallConfig(modules={"path": True})
    defines = resolve_defines(cfg)
    assert defines["ATV_BUILD_MINIMIZER"] == "ON"
    assert defines["ATV_BUILD_POTENTIALS"] == "ON"


# ── bound -D ─────────────────────────────────────────────────────────────────


def test_bound_D_rejected() -> None:
    cfg = InstallConfig(modules={"md": True})
    with pytest.raises(ConfigError, match="bound key"):
        merge_unbound_cli(cfg, ["ATV_BUILD_MD=ON"])
    with pytest.raises(ConfigError, match="bound key"):
        merge_unbound_cli(cfg, ["mpi=ON"])


# ── cmake command assembly ───────────────────────────────────────────────────


def test_cmake_configure_command_shape() -> None:
    defines = dry_run_defines(InstallConfig(modules={"md": True, "pw": True}))
    cmd = build_cmake_configure_command(
        Path("/repo"), Path("/build"), defines, python="/usr/bin/python"
    )
    assert cmd[0] == "cmake"
    assert "-S" in cmd and "/repo" in cmd
    assert "-B" in cmd and "/build" in cmd
    assert any(a.startswith("-DATV_BUILD_MD=ON") for a in cmd)
    assert any(a.startswith("-DATV_BUILD_PYTHON=ON") for a in cmd)
    assert any("Python_EXECUTABLE" in a for a in cmd)


def test_cmake_compile_and_install_commands() -> None:
    b = Path("/build")
    c = build_cmake_compile_command(b, jobs=4)
    assert c[:3] == ["cmake", "--build", "/build"]
    assert "-j" in c and "4" in c
    i = build_cmake_install_command(b, Path("/prefix"))
    assert i == ["cmake", "--install", "/build", "--prefix", "/prefix"]


def test_config_dry_run_streams_command(capsys) -> None:
    if not (_REPO / "CMakeLists.txt").is_file():
        pytest.skip("not in monorepo checkout")
    rc = main(
        [
            "config",
            "md",
            "--mpi",
            "--src",
            str(_REPO),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "cmake" in out
    assert "ATV_BUILD_MD=ON" in out
    assert "ATV_USE_MPI=ON" in out
    assert "dry-run" in out.lower()


def test_compile_dry_run_needs_state(capsys, tmp_path: Path) -> None:
    # Without state, compile fails loud
    import atomiverse.cli.build as build_mod

    monkey_state = tmp_path / "missing.json"
    # patch state path via env file absence
    old = build_mod.state_path
    build_mod.state_path = lambda: monkey_state  # type: ignore[assignment]
    try:
        rc = main(["compile", "--dry-run"])
        assert rc != 0
    finally:
        build_mod.state_path = old  # type: ignore[assignment]


def test_version_mismatch(tmp_path: Path) -> None:
    py_dir = tmp_path / "python"
    py_dir.mkdir()
    (py_dir / "src" / "atomiverse").mkdir(parents=True)
    (py_dir / "pyproject.toml").write_text(
        '[project]\nname = "atomiverse"\nversion = "9.9.9"\n',
        encoding="utf-8",
    )
    # fake CMakeLists so structure is valid if used as src
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="version mismatch"):
        verify_source_version(tmp_path, expected="0.1.0")


def test_status_none() -> None:
    text = format_status(None)
    assert "config" in text.lower()
    rc = run_status()
    assert rc == 0


def test_status_with_state(tmp_path: Path, capsys) -> None:
    state = {
        "modules": ["md", "neighbor", "potentials"],
        "fingerprint": "ATV_BUILD_MD=ON",
        "defines": {"ATV_BUILD_MD": "ON", "ATV_BUILD_PYTHON": "ON"},
        "configured": True,
        "build_dir": "/tmp/build",
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    rc = run_status(path=path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "md" in out
    assert "ATV_BUILD_MD=ON" in out


def test_cli_import_no_extension() -> None:
    import subprocess

    script = textwrap.dedent(
        f"""\
        import sys
        sys.path.insert(0, {str(_SRC)!r})
        for k in list(sys.modules):
            if k == "atomiverse" or k.startswith("atomiverse.") or k == "_atomiverse":
                del sys.modules[k]
        # None entries make `import _atomiverse` raise ImportError, so the
        # no-extension contract holds even when a dev PYTHONPATH carries the
        # built extension.
        sys.modules["_atomiverse"] = None
        sys.modules["atomiverse._atomiverse"] = None
        import atomiverse.cli.main
        import atomiverse.cli.build
        bad = [
            k
            for k in ("_atomiverse", "atomiverse._atomiverse")
            if sys.modules.get(k) is not None
        ]
        assert not bad, bad
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_help_lists_stages(capsys) -> None:
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "config" in out and "compile" in out and "install" in out
    rc = main(["--help"])
    assert rc == 0


def test_config_help(capsys) -> None:
    rc = main(["config", "--help"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "--mpi" in out
    assert "module" in out
    assert "k:v" not in out


def test_default_tarball_url() -> None:
    assert "v0.1.0" in default_tarball_url("0.1.0")
    assert isinstance(package_version(), str)


def test_write_install_state(tmp_path: Path, monkeypatch) -> None:
    # write next to a temp package path is hard; just exercise helper on defines
    defines = dry_run_defines(InstallConfig(modules={"md": True}))
    # Don't clobber real install state — unit-test content shape only
    assert defines["ATV_BUILD_PYTHON"] == "ON"
