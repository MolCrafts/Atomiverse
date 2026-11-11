"""Which product modules the installed extension carries.

A native wheel holds only the modules it was configured with, so a test that
needs a domain says so and is skipped, naming the flag, on an extension built
without it (or on a pure-Python install).
"""

from __future__ import annotations

import pytest

from atomiverse._binding import DOMAINS, extension_available, has_domain

# The package's own probe table, keyed by flag for the skip reasons.
_BY_FLAG: dict[str, str] = {flag: module for module, (flag, _) in DOMAINS.items()}


def built(*flags: str) -> bool:
    """True when the extension is installed and carries every ``flags`` domain."""
    return extension_available() and all(has_domain(_BY_FLAG[f]) for f in flags)


def needs(*flags: str) -> pytest.MarkDecorator:
    """Skip mark for a test, class or parameter that needs ``flags``."""
    return pytest.mark.skipif(not built(*flags), reason="needs " + " and ".join(flags))


def skip_module_unless(*flags: str) -> None:
    """Skip the calling test file unless the extension carries ``flags``.

    Call it before the file's compute imports, which fail without the domain.
    """
    if not built(*flags):
        pytest.skip("needs " + " and ".join(flags), allow_module_level=True)
