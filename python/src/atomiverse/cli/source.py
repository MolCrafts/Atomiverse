"""Source resolution for the native build.

The native extension is built from a local Atomiverse **checkout** (git clone
with a local merged molpy checkout); there is no remote-source path. The checkout's
version (``__version__`` in ``python/src/atomiverse/__init__.py``) must equal
the installed package version, so a native build never silently swaps the
release the user installed for a different tree.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from atomiverse.cli.config import ConfigError

_VERSION_RE = re.compile(
    r'(?m)^__version__\s*=\s*["\']([^"\']+)["\']\s*$'
)


@dataclass(frozen=True)
class SourceSpec:
    """A checkout the native wheel is built from.

    @param root: Repo root (holds ``CMakeLists.txt``).
    @param python_dir: The ``python/`` project pip builds.
    @param version: Package version this checkout carries.
    """

    root: Path
    python_dir: Path
    version: str


def package_version() -> str:
    """Return the installed ``atomiverse`` distribution version.

    Prefers ``importlib.metadata.version("atomiverse")``. When the
    distribution is not installed (pure src on ``PYTHONPATH``), falls back
    to ``atomiverse.__version__``.

    @return Version string.
    @raises ConfigError: when neither metadata nor ``__version__`` is available.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("atomiverse")
    except PackageNotFoundError:
        pass
    try:
        from atomiverse import __version__

        return __version__
    except Exception as exc:  # pragma: no cover
        raise ConfigError(
            "cannot determine atomiverse version "
            "(package not installed and __version__ unavailable)"
        ) from exc


def read_source_version(python_dir: Path) -> str:
    """Read ``__version__`` from a checkout's ``python/`` project.

    @param python_dir: The checkout's ``python/`` directory.
    @return Version string.
    @raises ConfigError: if the file is missing or has no ``__version__``.
    """
    init = python_dir / "src" / "atomiverse" / "__init__.py"
    if not init.is_file():
        raise ConfigError(f"not an atomiverse python project: {init} missing")
    match = _VERSION_RE.search(init.read_text(encoding="utf-8"))
    if not match:
        raise ConfigError(f"no __version__ in {init}")
    return match.group(1)


def resolve_source(src: str, *, version: Optional[str] = None) -> SourceSpec:
    """Resolve and version-check a local checkout.

    @param src: Checkout root (or its ``python/`` directory).
    @param version: Override the expected version (tests only; production
        uses :func:`package_version`).
    @return :class:`SourceSpec` for the checkout.
    @raises ConfigError: if *src* is not a checkout, lacks the merged molpy
        checkout, or its version differs from the installed one.
    """
    path = Path(src).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"--src path does not exist: {path}")
    root = path.parent if path.name == "python" else path
    python_dir = root / "python"
    if not (root / "CMakeLists.txt").is_file() or not (
        python_dir / "pyproject.toml"
    ).is_file():
        raise ConfigError(
            f"--src does not look like an Atomiverse checkout: {path} "
            "(expected CMakeLists.txt and python/pyproject.toml)"
        )
    molpy_source_root(root)

    expected = version if version is not None else package_version()
    found = read_source_version(python_dir)
    if found != expected:
        raise ConfigError(
            f"source version mismatch: checkout has {found!r}, installed "
            f"atomiverse is {expected!r} — check out tag v{expected} or "
            f"`pip install atomiverse=={found}` first"
        )
    return SourceSpec(root=root, python_dir=python_dir, version=found)


def molpy_source_root(source_root: Path) -> Path:
    """Locate the merged local molpy checkout shared by CXX and Python builds."""
    root = Path(os.environ.get("ATV_MOLPY_PATH", str(source_root.parent / "molrs"))).expanduser().resolve()
    for relative in ("molcxx/Cargo.toml", "molpy/pyproject.toml", "crates/molrs-ffi/Cargo.toml"):
        if not (root / relative).is_file():
            raise ConfigError(
                f"merged molpy checkout missing {relative} under {root}; "
                "set ATV_MOLPY_PATH to the local MolCrafts/molpy repository"
            )
    return root
