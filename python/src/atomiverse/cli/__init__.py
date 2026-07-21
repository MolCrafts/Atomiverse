"""On-demand native-extension install CLI (``python -m atomiverse``).

This package must import without the C++ extension (``_atomiverse`` /
``atomiverse._binding``). Keep imports here pure-Python only.
"""

from __future__ import annotations

__all__ = ["main"]


def __getattr__(name: str):
    if name == "main":
        from atomiverse.cli.main import main as _main

        return _main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
