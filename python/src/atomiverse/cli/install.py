"""Compatibility re-exports — native pipeline lives in :mod:`atomiverse.cli.build`.

Prefer importing from ``atomiverse.cli.build`` for new code.
"""

from __future__ import annotations

from atomiverse.cli.build import (  # noqa: F401
    build_cmake_compile_command,
    build_cmake_configure_command,
    build_cmake_install_command,
    dry_run_defines,
    run_compile,
    run_config,
    run_install_stage,
    site_packages_prefix,
    state_path,
    write_install_state,
)

# Historical name used by older tests / status.
from atomiverse.cli.build import run_install_stage as run_install  # noqa: F401
