"""Pure-Python / soft-extension import contract (spec 02-pure-wheel).

Without the ``_atomiverse`` C++ extension the package must still import,
expose ``__version__``, and fail compute paths with an install-oriented
``ImportError``. With the extension present, the existing surface remains
reachable.

Pure-path cases run in an isolated subprocess whose ``sys.path`` puts pure
sources first and drops site-packages (stdlib kept).
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
sys.path = [src] + [
    p for p in sys.path
    if p and "site-packages" not in p and "dist-packages" not in p
]
for key in list(sys.modules):
    if key == "atomiverse" or key.startswith("atomiverse.") or key == "_atomiverse":
        del sys.modules[key]
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


def test_extension_present_path_still_loads() -> None:
    """When the native extension is available, soft-load must not hide it."""
    for key in list(sys.modules):
        if key == "atomiverse" or key.startswith("atomiverse.") or key == "_atomiverse":
            del sys.modules[key]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

    av = importlib.import_module("atomiverse")
    binding = importlib.import_module("atomiverse._binding")

    if not binding.extension_available():
        pytest.skip("C++ extension not built in this environment")
    mod = binding.require_cpp()
    assert mod is not None
    assert hasattr(av, "MD")
    assert av.MD is not None
    if hasattr(binding._cpp, "PWSolver"):
        assert hasattr(av, "PWSolver")
    else:
        assert "PWSolver" not in getattr(av, "__all__", [])
