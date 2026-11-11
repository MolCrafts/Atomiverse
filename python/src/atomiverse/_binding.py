"""Internal shim: soft-imports the C++ extension module as ``_cpp``.

Two valid import paths when the extension is present:
  - ``atomiverse._atomiverse`` — installed wheel (relative import)
  - ``_atomiverse``           — development (bare .so on PYTHONPATH)

When the extension is absent the package still imports:
``extension_available()`` is False and attribute access on the ``_cpp``
sentinel (or an explicit ``require_cpp()`` call) raises an install-oriented
``ImportError``. An extension that is present but fails to load (undefined
symbol, missing shared library) is a different failure: the package still
imports — so the CLI can rebuild it — but every compute access re-raises the
loader's own error rather than claiming nothing is installed.

Important: never use ``from atomiverse import _atomiverse`` here — that
re-enters package ``__init__`` while ``_binding`` is still loading and
creates a circular import.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Optional

__all__ = [
    "_cpp",
    "extension_available",
    "extension_load_error",
    "require_cpp",
    "INSTALL_HINT",
    "DOMAINS",
    "MissingDomainError",
    "has_domain",
    "missing_domain_message",
    "require_domain",
]

INSTALL_HINT = (
    "The Atomiverse C++ extension (_atomiverse) is not installed.\n"
    "Build it with an Atomiverse checkout and the local molpy checkout:\n"
    "  export ATV_MOLPY_PATH=/path/to/merged/molpy\n"
    "  pip install 'atomiverse[native]'\n"
    "  python -m atomiverse config md --src /path/to/Atomiverse\n"
    "  python -m atomiverse compile\n"
    "  python -m atomiverse install"
)


class _MissingExtension:
    """Sentinel standing in for ``_atomiverse`` when the extension is absent.

    Any attribute access raises ``ImportError`` with the install hint so
    compute paths fail loud instead of with an opaque ``AttributeError``.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        require_cpp()

    def __repr__(self) -> str:
        return "<atomiverse._binding: C++ extension not installed>"


_EXTENSION_NAMES = ("atomiverse._atomiverse", "_atomiverse")


def _try_import_extension() -> tuple[Optional[ModuleType], Optional[ImportError]]:
    """Return ``(module, None)``, ``(None, None)`` if absent, or
    ``(None, error)`` if an extension was found but failed to load."""
    # import_module, not `from . import`: inside the half-initialized package
    # the latter reports an absent submodule as a plain ImportError ("cannot
    # import name"), which would read as a broken extension.
    try:
        mod = importlib.import_module("atomiverse._atomiverse")

        return mod, None
    except ModuleNotFoundError as exc:
        if exc.name not in _EXTENSION_NAMES:
            return None, exc
    except ImportError as exc:
        return None, exc
    try:
        import _atomiverse as mod  # type: ignore[import-untyped]

        return mod, None
    except ModuleNotFoundError as exc:
        if exc.name not in _EXTENSION_NAMES:
            return None, exc
        return None, None
    except ImportError as exc:
        return None, exc


_ext, _load_error = _try_import_extension()
_cpp: Any = _ext if _ext is not None else _MissingExtension()


def extension_available() -> bool:
    """Return True when the ``_atomiverse`` C++ extension is importable."""
    return _ext is not None


def extension_load_error() -> Optional[ImportError]:
    """Return the loader error of an extension that is present but broken.

    @return The ``ImportError`` raised while loading ``_atomiverse``, or None
        when it loaded or is simply not installed.
    """
    return _load_error


def require_cpp() -> ModuleType:
    """Return the C++ extension module, or raise an install-oriented ImportError.

    @return The loaded ``_atomiverse`` module.
    @raises ImportError: when the extension has not been built/installed, or
        (chained from the loader's error) when it is present but fails to load.
    """
    if _ext is not None:
        return _ext
    if _load_error is not None:
        raise ImportError(
            f"The Atomiverse C++ extension failed to load: {_load_error}"
        ) from _load_error
    raise ImportError(INSTALL_HINT)


#: Product module (the name ``python -m atomiverse config`` takes) -> its
#: CMake flag and a symbol its binding file always registers. An extension
#: holds only the modules it was configured with; the probe is how the
#: package tells which.
DOMAINS: dict[str, tuple[str, str]] = {
    "potentials": ("ATV_BUILD_POTENTIALS", "PairLJCut"),
    "md": ("ATV_BUILD_MD", "MD"),
    "minimize": ("ATV_BUILD_MINIMIZER", "Minimize"),
    "mc": ("ATV_BUILD_MC", "MC"),
    "vib": ("ATV_BUILD_VIB", "Hessian"),
    "am1": ("ATV_BUILD_SEMIEMPIRICAL", "AM1"),
    "hf": ("ATV_BUILD_HF", "RHF"),
    "pw": ("ATV_BUILD_PW", "PW"),
    "kspace": ("ATV_BUILD_KSPACE", "KSpacePME"),
}


class MissingDomainError(ImportError):
    """A deep module whose product module this extension was built without.

    @param what: The module that was imported.
    @param module: The ``DOMAINS`` key it needs, kept as ``self.module``.
    """

    def __init__(self, what: str, module: str) -> None:
        super().__init__(missing_domain_message(what, module), name=what)
        self.module = module


def has_domain(module: str) -> bool:
    """Return True when the installed extension carries product *module*.

    @param module: A ``DOMAINS`` key, e.g. ``"md"``.
    @return False on a pure install, a broken extension, or a build that
        deselected the module.
    """
    return _ext is not None and hasattr(_ext, DOMAINS[module][1])


def missing_domain_message(what: str, module: str) -> str:
    """Explain that *what* needs product *module* and how to build it in.

    @param what: The module or name that was asked for.
    @param module: The ``DOMAINS`` key it needs.
    @return One message naming the module, its flag and the rebuild command.
    """
    flag = DOMAINS[module][0]
    return (
        f"{what} needs the `{module}` module ({flag}=ON), which this "
        "extension was built without. Rebuild with it selected:\n"
        f"  python -m atomiverse config {module} ... --src /path/to/Atomiverse\n"
        "  python -m atomiverse compile\n"
        "  python -m atomiverse install"
    )


def require_domain(name: str, module: str) -> None:
    """Fail the import of deep module *name* unless *module* is built.

    Call it at the top of a deep module, before it reads any binding.

    @param name: The importing module's ``__name__``.
    @param module: The ``DOMAINS`` key its bindings come from.
    @raises ImportError: with the install hint when no extension is present.
    @raises MissingDomainError: when the extension lacks *module*.
    """
    require_cpp()
    if not has_domain(module):
        raise MissingDomainError(name, module)


def backend_type(device: str, cpu_name: str, cuda_name: str) -> type:
    """Select a bound native type without translating method parameters."""
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unknown device {device!r}; expected 'cpu' or 'cuda'")
    extension = require_cpp()
    name = cpu_name if device == "cpu" else cuda_name
    native = getattr(extension, name, None)
    if native is None:
        raise ValueError(f"{device.upper()} binding {name} is unavailable in this build")
    return native
