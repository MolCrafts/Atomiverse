"""Unit tests for the native-build CLI: config / compile / install."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from atomiverse.cli.build import (  # noqa: E402
    BuildState,
    build_install_command,
    build_molpy_wheel_command,
    build_wheel_command,
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
    package_version,
    resolve_source,
)
from atomiverse.cli.status import format_status, run_status  # noqa: E402


# ── mutual exclusion / empty ─────────────────────────────────────────────────


def test_mutual_exclusion_config_vs_modules() -> None:
    with pytest.raises(ConfigError, match="mutual exclusion"):
        validate_mode(config_path="x.toml", modules=["md"], hf=False)
    rc = main(["config", "--config", "x.toml", "md", "--src", "/no/such/checkout"])
    assert rc != 0


def test_empty_config_fails() -> None:
    with pytest.raises(ConfigError, match="empty config"):
        validate_mode(config_path=None, modules=[], hf=False)
    rc = main(["config", "--src", "/no/such/checkout"])
    assert rc != 0


def test_hf_is_a_normal_module() -> None:
    """``hf`` configures Gaussian SCF like any other product module.

    Spec py-scf-single-point ships the ``atomiverse.scf`` bindings, so the
    old hard rejection is gone: ``hf`` maps to ``ATV_BUILD_HF`` in the same
    table as every other module (mirrors test_kspace_closure).
    """
    cfg = InstallConfig(modules={"hf": True})
    defines = resolve_defines(cfg)
    assert defines["ATV_BUILD_HF"] == "ON"


def test_hf_flag_no_longer_rejected() -> None:
    """``--hf`` beside a module passes validation (no ConfigError left)."""
    validate_mode(config_path=None, modules=["md"], hf=True)


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


@pytest.mark.parametrize("value", ['"false"', '"off"', "0", "1", "[]", "{}"])
def test_toml_module_requires_boolean(tmp_path: Path, value: str) -> None:
    config = tmp_path / "invalid.toml"
    config.write_text(f"[modules]\nmd = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="md.*boolean"):
        load_toml_config(config)


@pytest.mark.parametrize("enabled", [True, False])
def test_toml_module_boolean(tmp_path: Path, enabled: bool) -> None:
    config = tmp_path / "valid.toml"
    config.write_text(
        f"[modules]\nmd = {str(enabled).lower()}\n", encoding="utf-8"
    )
    assert load_toml_config(config).modules["md"] is enabled


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


# ── native wheel command assembly ────────────────────────────────────────────


def test_wheel_command_shape() -> None:
    defines = resolve_defines(InstallConfig(modules={"md": True, "pw": True}))
    cmd = build_wheel_command(
        Path("/repo/python"), Path("/b"), defines, python="/usr/bin/python"
    )
    assert cmd[:4] == ["/usr/bin/python", "-m", "pip", "wheel"]
    assert str(Path("/repo/python")) in cmd
    assert "--no-build-isolation" in cmd and "--no-deps" in cmd
    assert cmd[cmd.index("--wheel-dir") + 1] == str(Path("/b/dist"))
    settings = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-C"]
    assert "wheel.cmake=true" in settings
    assert f"build-dir={Path('/b/cmake')}" in settings
    assert "cmake.define.ATV_BUILD_MD=ON" in settings
    assert "cmake.define.ATV_BUILD_PW=ON" in settings
    assert "cmake.define.ATV_BUILD_KSPACE=OFF" in settings
    assert "cmake.define.ATV_BUILD_PYTHON=ON" in settings


def test_wheel_command_requires_python_define() -> None:
    with pytest.raises(ConfigError, match="ATV_BUILD_PYTHON"):
        build_wheel_command(Path("/p"), Path("/b"), {"ATV_BUILD_MD": "ON"})


def test_molpy_wheel_command_builds_local_checkout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ATV_MOLPY_PATH", raising=False)
    root = _fake_checkout(tmp_path / "atv", "0.1.0")
    cmd = build_molpy_wheel_command(root, Path("/b"), python="/usr/bin/python")
    assert cmd[:4] == ["/usr/bin/python", "-m", "pip", "wheel"]
    assert str(root.parent / "molpy" / "molpy") in cmd
    assert "--no-build-isolation" in cmd and "--no-deps" in cmd
    assert cmd[cmd.index("--wheel-dir") + 1] == str(Path("/b/dist"))


def test_install_command_shape() -> None:
    cmd = build_install_command(
        [Path("/b/dist/x.whl"), Path("/b/dist/m.whl")], python="/usr/bin/python"
    )
    assert cmd == [
        "/usr/bin/python",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--force-reinstall",
        str(Path("/b/dist/x.whl")),
        str(Path("/b/dist/m.whl")),
    ]


def test_build_state_roundtrip(tmp_path: Path) -> None:
    state = BuildState(
        version="0.1.0",
        source_root="/repo",
        python_dir="/repo/python",
        build_dir=str(tmp_path),
        modules=["md"],
        wheel="/w.whl",
        molpy_wheel="/m.whl",
    )
    path = state.save()
    assert BuildState.load(path) == state


def test_build_state_missing_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no build state"):
        BuildState.load(tmp_path / "absent.json")


# ── source resolution ────────────────────────────────────────────────────────


def _fake_checkout(root: Path, version: str, *, molpy: bool = True) -> Path:
    root = root / "checkout"
    pkg = root / "python" / "src" / "atomiverse"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    (root / "python" / "pyproject.toml").write_text("", encoding="utf-8")
    (root / "CMakeLists.txt").write_text("project(x)\n", encoding="utf-8")
    if molpy:
        dependency = root.parent / "molpy"
        for relative in ("molcxx/Cargo.toml", "molpy/pyproject.toml", "crates/molrs-ffi/Cargo.toml"):
            target = dependency / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("", encoding="utf-8")
    return root


def test_resolve_source_ok(tmp_path: Path) -> None:
    root = _fake_checkout(tmp_path, "1.2.3")
    spec = resolve_source(str(root), version="1.2.3")
    assert spec.root == root.resolve()
    assert spec.python_dir == root.resolve() / "python"
    assert spec.version == "1.2.3"
    assert resolve_source(str(root / "python"), version="1.2.3") == spec


def test_resolve_source_version_mismatch(tmp_path: Path) -> None:
    root = _fake_checkout(tmp_path, "9.9.9")
    with pytest.raises(ConfigError, match="version mismatch"):
        resolve_source(str(root), version="0.1.0")


def test_resolve_source_needs_merged_molpy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ATV_MOLPY_PATH", raising=False)
    root = _fake_checkout(tmp_path, "0.1.0", molpy=False)
    with pytest.raises(ConfigError, match="molpy checkout missing"):
        resolve_source(str(root), version="0.1.0")


def test_resolve_source_not_a_checkout(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="checkout"):
        resolve_source(str(tmp_path), version="0.1.0")


def test_package_version_is_str() -> None:
    assert isinstance(package_version(), str)


def test_config_dry_run_prints_wheel_command(capsys, tmp_path: Path) -> None:
    root = _fake_checkout(tmp_path, package_version())
    rc = main(["config", "md", "--mpi", "--src", str(root), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "pip wheel" in out
    assert "cmake.define.ATV_BUILD_MD=ON" in out
    assert "cmake.define.ATV_USE_MPI=ON" in out
    assert "dry-run" in out.lower()
    assert not (root / "build-python").exists()


# ── status ───────────────────────────────────────────────────────────────────


def test_status_nothing() -> None:
    text = format_status(None, None, extension=False)
    assert "not installed" in text
    assert "config" in text.lower()
    assert run_status() == 0


def test_status_installed_and_active() -> None:
    installed = {"defines": {"ATV_BUILD_MD": "ON", "ATV_BUILD_PW": "OFF",
                             "ATV_USE_MPI": "ON"}}
    active = {"build_dir": "/b", "source_root": "/repo", "modules": ["md"],
              "wheel": "/b/dist/x.whl", "installed": True}
    text = format_status(installed, active, extension=True)
    assert "modules:   md" in text
    assert "ATV_USE_MPI=ON" in text
    assert "/b/dist/x.whl" in text
    assert "installed: True" in text


def test_status_broken_extension_names_loader_error() -> None:
    err = ImportError("undefined symbol: _ZTVN3ATV3cpu9minimizer2CGE")
    text = format_status(None, None, extension=False, load_error=err)
    assert "BROKEN" in text and "undefined symbol" in text
    assert "not installed" not in text


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


def test_run_script_implicit(tmp_path: Path, capsys) -> None:
    script = tmp_path / "hello.py"
    script.write_text("print('hello-atv')\n", encoding="utf-8")
    rc = main([str(script)])
    assert rc == 0
    assert "hello-atv" in capsys.readouterr().out


def test_run_script_missing() -> None:
    rc = main(["/no/such/script.py"])
    assert rc == 2


def test_run_subcommand(tmp_path: Path, capsys) -> None:
    script = tmp_path / "hello.py"
    script.write_text("print('hello-run')\n", encoding="utf-8")
    rc = main(["run", str(script)])
    assert rc == 0
    assert "hello-run" in capsys.readouterr().out


def test_local_molpy_override_is_shared_by_wheel_build(tmp_path: Path, monkeypatch) -> None:
    root = _fake_checkout(tmp_path / "default", "0.1.0")
    other = _fake_checkout(tmp_path / "override", "0.1.0").parent / "molpy"
    monkeypatch.setenv("ATV_MOLPY_PATH", str(other))
    command = build_molpy_wheel_command(root, tmp_path / "build")
    assert str(other / "molpy") in command


@pytest.mark.parametrize('flags,mpi,cuda', [
    (['--hf'], 'OFF', 'OFF'),
    (['--hf', '--mpi'], 'ON', 'OFF'),
    (['--hf', '--cuda'], 'OFF', 'ON'),
    (['--hf', '--cuda', '--mpi'], 'ON', 'ON'),
    (['hf', 'cuda', '--mpi'], 'ON', 'ON'),
])
def test_hf_accelerators_compose(flags, mpi, cuda, capsys, tmp_path):
    root = _fake_checkout(tmp_path, package_version())
    assert main(['config', *flags, '--src', str(root), '--dry-run']) == 0
    output = capsys.readouterr().out
    for key, value in [('ATV_BUILD_HF', 'ON'), ('ATV_BUILD_CUDA', cuda), ('ATV_USE_MPI', mpi)]:
        assert f'cmake.define.{key}={value}' in output
    assert 'cmake.define.ATV_BUILD_MD=OFF' in output
