"""Pure-Python / soft-extension import contract (spec 02-pure-wheel).

Without the ``_atomiverse`` C++ extension the package must still import,
expose ``__version__``, and fail compute paths with an install-oriented
``ImportError``. With the extension present, the existing surface remains
reachable.

Pure-path cases run in an isolated subprocess that puts the pure sources
first on ``sys.path`` and blocks the extension via ``sys.modules`` ``None``
sentinels, so real dependencies (typer, numpy) stay importable.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
_INSTALL_SNIPPET = "python -m atomiverse install"

_ISOLATE_PREAMBLE = f"""\
import sys
src = {str(_SRC_ROOT)!r}
sys.path.insert(0, src)
for key in list(sys.modules):
    if key == "atomiverse" or key.startswith("atomiverse.") or key == "_atomiverse":
        del sys.modules[key]
# A None entry makes `import _atomiverse` raise ImportError, so the pure
# path is exercised even when the built extension is installed or on
# PYTHONPATH. Real dependencies (typer, numpy) stay importable.
sys.modules["_atomiverse"] = None
sys.modules["atomiverse._atomiverse"] = None
"""


def _run_isolated(body: str) -> subprocess.CompletedProcess[str]:
    """Execute ``body`` with pure sources first and site-packages stripped."""
    script = _ISOLATE_PREAMBLE + "\n" + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def test_soft_import_without_extension() -> None:
    proc = _run_isolated(
        """
        import atomiverse as av
        from atomiverse._binding import extension_available
        assert av.__version__ == "0.1.0", av.__version__
        assert extension_available() is False
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_binding_module_imports_without_extension() -> None:
    proc = _run_isolated(
        f"""
        import atomiverse._binding as binding
        assert binding.extension_available() is False
        try:
            binding.require_cpp()
        except ImportError as e:
            assert {_INSTALL_SNIPPET!r} in str(e), e
        else:
            raise SystemExit("require_cpp did not raise")
        try:
            _ = binding._cpp.MD
        except ImportError as e:
            assert {_INSTALL_SNIPPET!r} in str(e), e
        else:
            raise SystemExit("_cpp.MD did not raise")
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_compute_getattr_raises_install_message() -> None:
    proc = _run_isolated(
        f"""
        import atomiverse as av
        try:
            _ = av.MD
        except ImportError as e:
            assert {_INSTALL_SNIPPET!r} in str(e), e
        else:
            raise SystemExit("av.MD did not raise")
        try:
            _ = av.single_point
        except ImportError as e:
            assert {_INSTALL_SNIPPET!r} in str(e), e
        else:
            raise SystemExit("av.single_point did not raise")
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_main_stub_runs_without_extension() -> None:
    """``python -m atomiverse`` must not traceback when the .so is absent."""
    proc = _run_isolated(
        """
        import runpy, sys
        sys.argv = ["atomiverse"]
        runpy.run_module("atomiverse", run_name="__main__")
        """
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout + proc.stderr
    assert "atomiverse" in out.lower()
    assert "install" in out.lower()
    assert "Traceback" not in out


def _is_package_module(key: str) -> bool:
    return key == "atomiverse" or (
        key.startswith("atomiverse.") and key != "atomiverse._atomiverse"
    )


@pytest.fixture
def fresh_package(monkeypatch: pytest.MonkeyPatch):
    """Re-import the package from the source tree, then put the original back.

    Without the restore, later tests would hold the installed package while
    its submodules resolve to the source copy: two ``_binding`` modules, two
    distinct exception classes.
    """
    saved = {k: v for k, v in sys.modules.items() if _is_package_module(k)}
    for key in saved:
        del sys.modules[key]
    monkeypatch.syspath_prepend(str(_SRC_ROOT))
    yield
    for key in [k for k in sys.modules if _is_package_module(k)]:
        del sys.modules[key]
    sys.modules.update(saved)


def test_extension_present_path_still_loads(fresh_package: None) -> None:
    """When the native extension is available, soft-load must not hide it.

    ``_atomiverse`` itself is deliberately kept in ``sys.modules``:
    re-initializing a nanobind extension in the same process aborts on
    duplicate type registrations, and the soft-import probe finds the
    cached module either way.
    """

    av = importlib.import_module("atomiverse")
    binding = importlib.import_module("atomiverse._binding")

    if not binding.extension_available():
        pytest.skip("C++ extension not built in this environment")
    mod = binding.require_cpp()
    assert mod is not None
    # Any build carries some public name; MD itself is optional.
    assert av.__all__
    assert getattr(av, av.__all__[0]) is not None
    if hasattr(binding._cpp, "PW"):
        assert hasattr(av, "PW")
    else:
        assert "PW" not in getattr(av, "__all__", [])


def test_single_point_uses_public_method_entry(monkeypatch) -> None:
    from unittest.mock import Mock
    import atomiverse.driver as driver

    require = Mock()
    monkeypatch.setattr(driver, "require_cpp", require)
    method = Mock()
    frame = object()
    assert driver.single_point(frame, method) is None
    require.assert_called_once_with()
    method.run.assert_called_once_with(frame)


def test_single_point_rejects_non_evaluator(monkeypatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock
    import atomiverse.driver as driver

    monkeypatch.setattr(driver, "require_cpp", Mock())
    monkeypatch.setattr(driver, "has_domain", lambda _: False)
    monkeypatch.setattr(driver, "_cpp", SimpleNamespace(_Component=SimpleNamespace))
    with pytest.raises(TypeError, match="method with run"):
        driver.single_point(object(), object())
