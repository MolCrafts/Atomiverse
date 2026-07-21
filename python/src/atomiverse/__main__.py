"""``python -m atomiverse`` entry point.

Dispatches to :func:`atomiverse.cli.main.main`. The CLI package must load
without importing the C++ extension (``_binding`` / ``_atomiverse``).
"""

from __future__ import annotations

import sys

from atomiverse.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
