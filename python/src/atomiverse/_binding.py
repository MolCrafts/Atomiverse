"""Internal shim: soft-imports the C++ extension module as ``_cpp``.

Two valid import paths when the extension is present:
  - ``atomiverse._atomiverse`` — installed wheel (relative import)
  - ``_atomiverse``           — development (bare .so on PYTHONPATH)

When BOTH fail the package still imports: ``extension_available()`` is
False and attribute access on the ``_cpp`` sentinel (or an explicit
``require_cpp()`` call) raises an install-oriented ``ImportError``.

Important: never use ``from atomiverse import _atomiverse`` here — that
re-enters package ``__init__`` while ``_binding`` is still loading and
creates a circular import.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Optional

__all__ = [
    "_cpp",
    "extension_available",
    "require_cpp",
    "INSTALL_HINT",
]

INSTALL_HINT = (
    "The Atomiverse C++ extension (_atomiverse) is not installed.\n"
    "Build it with the staged cmake CLI:\n"
    "  python -m atomiverse config md --src /path/to/Atomiverse\n"
    "  python -m atomiverse compile\n"
    "  python -m atomiverse install\n"
    "Or configure/build manually:\n"
    "  cmake -B build -DATV_BUILD_PYTHON=ON && cmake --build build\n"
    "Then install into the env or add the build directory to PYTHONPATH."
)


class _MissingExtension:
    """Sentinel standing in for ``_atomiverse`` when the extension is absent.

    Any attribute access raises ``ImportError`` with the install hint so
    compute paths fail loud instead of with an opaque ``AttributeError``.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        raise ImportError(INSTALL_HINT)

    def __repr__(self) -> str:
        return "<atomiverse._binding: C++ extension not installed>"


def _try_import_extension() -> Optional[ModuleType]:
    # Relative import only — does not re-enter package __init__.
    try:
        from . import _atomiverse as mod  # type: ignore[attr-defined]

        return mod
    except ImportError:
        pass
    try:
        import _atomiverse as mod  # type: ignore[import-untyped]

        return mod
    except ImportError:
        return None


_ext: Optional[ModuleType] = _try_import_extension()
_cpp: Any = _ext if _ext is not None else _MissingExtension()


def extension_available() -> bool:
    """Return True when the ``_atomiverse`` C++ extension is importable."""
    return _ext is not None


def require_cpp() -> ModuleType:
    """Return the C++ extension module, or raise an install-oriented ImportError.

    @return The loaded ``_atomiverse`` module.
    @raises ImportError: when the extension has not been built/installed.
    """
    if _ext is None:
        raise ImportError(INSTALL_HINT)
    return _ext
