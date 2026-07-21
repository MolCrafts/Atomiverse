"""Source resolution for on-demand native installs.

Version is always taken from the already-installed pure-Python package
(``importlib.metadata.version("atomiverse")``). There is **no**
``install --version`` flag. Default tarball URL:

    https://github.com/Roy-Kid/Atomiverse/archive/refs/tags/v{version}.tar.gz
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from atomiverse.cli.config import ConfigError

_DEFAULT_TARBALL = (
    "https://github.com/Roy-Kid/Atomiverse/archive/refs/tags/v{version}.tar.gz"
)

# Minimal version field reader — avoids requiring tomllib just to gate source.
_VERSION_RE = re.compile(
    r'(?m)^\s*version\s*=\s*["\']([^"\']+)["\']\s*$'
)


@dataclass(frozen=True)
class SourceSpec:
    """Where to build the native extension from.

    @param location: Local path or URL pip will install from.
    @param version: Package version this source must match.
    @param is_url: True when *location* is a remote URL.
    """

    location: str
    version: str
    is_url: bool = False


def package_version() -> str:
    """Return the installed ``atomiverse`` distribution version.

    Prefers ``importlib.metadata.version("atomiverse")``. When the
    distribution is not installed (pure src on ``PYTHONPATH``), falls back
    to ``atomiverse.__version__``.

    @return Version string.
    @raises ConfigError: when neither metadata nor ``__version__`` is available.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - py3.9 has importlib.metadata
        from importlib_metadata import PackageNotFoundError, version  # type: ignore

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


def default_tarball_url(version: str) -> str:
    """Build the default GitHub source tarball URL for *version*."""
    return _DEFAULT_TARBALL.format(version=version)


def read_pyproject_version(pyproject: Path) -> str:
    """Read ``version = "…"`` from a ``pyproject.toml``.

    @param pyproject: Path to ``python/pyproject.toml``.
    @return Version string.
    @raises ConfigError: if the file is missing or has no version field.
    """
    if not pyproject.is_file():
        raise ConfigError(f"pyproject.toml not found: {pyproject}")
    text = pyproject.read_text(encoding="utf-8")
    match = _VERSION_RE.search(text)
    if not match:
        raise ConfigError(f"no version field in {pyproject}")
    return match.group(1)


def _find_python_pyproject(root: Path) -> Path:
    """Locate ``python/pyproject.toml`` under a source tree or package dir."""
    candidates = [
        root / "python" / "pyproject.toml",
        root / "pyproject.toml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise ConfigError(
        f"cannot find python/pyproject.toml under source tree {root}"
    )


def _pip_location_for_src(src: Path) -> str:
    """Return the path pip should install (the ``python/`` package dir)."""
    src = src.resolve()
    if (src / "pyproject.toml").is_file() and (src / "src" / "atomiverse").is_dir():
        return str(src)
    if (src / "python" / "pyproject.toml").is_file():
        return str(src / "python")
    if src.name == "python" and (src / "pyproject.toml").is_file():
        return str(src)
    raise ConfigError(
        f"--src path does not look like an Atomiverse checkout or python/: {src}"
    )


def verify_source_version(src_root: Path, expected: str) -> None:
    """Fail if source ``python/pyproject.toml`` version ≠ *expected*.

    @param src_root: Checkout root or ``python/`` directory.
    @param expected: Installed package version.
    @raises ConfigError: on mismatch.
    """
    pyproject = _find_python_pyproject(src_root)
    found = read_pyproject_version(pyproject)
    if found != expected:
        raise ConfigError(
            f"source version mismatch: pyproject has {found!r}, "
            f"installed atomiverse is {expected!r} "
            f"(refusing install before pip)"
        )


def resolve_source(
    *,
    src: Optional[str] = None,
    url: Optional[str] = None,
    version: Optional[str] = None,
) -> SourceSpec:
    """Resolve install source from ``--src`` / ``--url`` / default tarball.

    @param src: Local checkout path (optional).
    @param url: Source tarball/URL override (optional).
    @param version: Override package version (tests only; production uses
        :func:`package_version`).
    @return :class:`SourceSpec` ready for pip.
    @raises ConfigError: if both src and url are set, or version mismatches.
    """
    if src and url:
        raise ConfigError("use either --src or --url, not both")

    ver = version if version is not None else package_version()

    if src:
        root = Path(src).expanduser().resolve()
        if not root.exists():
            raise ConfigError(f"--src path does not exist: {root}")
        verify_source_version(root, ver)
        location = _pip_location_for_src(root)
        return SourceSpec(location=location, version=ver, is_url=False)

    if url:
        # Remote URL: version gate runs only when the tree is local.
        # (Tarball contents are checked by the build backend / user.)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https", "file"):
            raise ConfigError(f"unsupported --url scheme: {url!r}")
        # scikit-build lives under python/; hint subdirectory when pointing
        # at a full-repo GitHub archive.
        location = url
        if "github.com" in url and "subdirectory=" not in url:
            sep = "&" if "#" in url else "#"
            location = f"{url}{sep}subdirectory=python"
        return SourceSpec(location=location, version=ver, is_url=True)

    tarball = default_tarball_url(ver)
    sep = "&" if "#" in tarball else "#"
    location = f"{tarball}{sep}subdirectory=python"
    return SourceSpec(location=location, version=ver, is_url=True)
