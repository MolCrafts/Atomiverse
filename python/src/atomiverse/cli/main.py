"""``python -m atomiverse`` Typer entry — CMake-aligned stages.

Pipeline (matches cmake)::

    python -m atomiverse config md pw --mpi --src /path/to/Atomiverse
    python -m atomiverse compile -j 8
    python -m atomiverse install

``--mpi``, ``--real-storage``, … are **global** config options (not per-module).
Modules are positional: ``config md pw``.

Raw cmake stdout/stderr is streamed to the terminal.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence

import typer

from atomiverse.cli.build import run_compile, run_config, run_install_stage
from atomiverse.cli.config import (
    MODULE_TO_DEFINE,
    ConfigError,
    InstallConfig,
    apply_cli_modules,
    load_toml_config,
    merge_unbound_cli,
    validate_mode,
)
from atomiverse.cli.source import resolve_source
from atomiverse.cli.status import run_status

app = typer.Typer(
    name="atomiverse",
    help=(
        "Atomiverse native build CLI (cmake config → compile → install). "
        "Pure-Python wheels ship without the C++ extension."
    ),
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
)


class RealPrecision(str, Enum):
    float = "float"
    double = "double"


class IdSize(str, Enum):
    bigbig = "bigbig"
    smallbig = "smallbig"


def _print_banner() -> None:
    try:
        from atomiverse import __version__
    except Exception:  # pragma: no cover
        __version__ = "unknown"
    print(f"atomiverse {__version__}")
    print()
    print("CMake-aligned native build (raw cmake output is streamed):")
    print("  python -m atomiverse config md pw --mpi --src /path/to/Atomiverse")
    print("  python -m atomiverse compile -j 8")
    print("  python -m atomiverse install")
    print("  python -m atomiverse status")
    print()
    print("Or from a TOML recipe:")
    print("  python -m atomiverse config --config build.toml --src /path/to/Atomiverse")
    print()
    print("Run `python -m atomiverse --help` for full options.")


def _global_options(
    mpi: Optional[bool],
    real_storage: Optional[RealPrecision],
    real_work: Optional[RealPrecision],
    id_size: Optional[IdSize],
) -> dict:
    return {
        "mpi": mpi,
        "real_storage": real_storage.value if real_storage is not None else None,
        "real_work": real_work.value if real_work is not None else None,
        "id_size": id_size.value if id_size is not None else None,
    }


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="print the installed atomiverse package version and exit",
    ),
) -> None:
    if version:
        from atomiverse.cli.source import package_version

        try:
            typer.echo(package_version())
        except ConfigError:
            from atomiverse import __version__

            typer.echo(__version__)
        raise typer.Exit(0)
    if ctx.invoked_subcommand is None:
        _print_banner()
        raise typer.Exit(0)


@app.command("config")
def config_cmd(
    modules: Optional[List[str]] = typer.Argument(
        None,
        help=(
            "product modules to enable: md, minimize, mc, evaluate, vib, path, "
            "pw, gpw, am1, cuda, kspace, potentials, neighbor"
        ),
        metavar="MODULE",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="unified TOML recipe (XOR with positional modules)",
        exists=False,
        dir_okay=False,
    ),
    mpi: Optional[bool] = typer.Option(
        None,
        "--mpi/--no-mpi",
        help="global: enable MPI (ATV_USE_MPI) — not a per-module option",
    ),
    real_storage: Optional[RealPrecision] = typer.Option(
        None,
        "--real-storage",
        help="global: ATV_REAL_STORAGE",
    ),
    real_work: Optional[RealPrecision] = typer.Option(
        None,
        "--real-work",
        help="global: ATV_REAL_WORK",
    ),
    id_size: Optional[IdSize] = typer.Option(
        None,
        "--id-size",
        help="global: ATV_ID_SIZE",
    ),
    hf: bool = typer.Option(
        False,
        "--hf",
        help="(rejected) HF/SCF bindings are not available yet",
    ),
    src: Optional[Path] = typer.Option(
        None,
        "--src",
        help="Atomiverse checkout root (required for local cmake config)",
        exists=False,
        file_okay=False,
        dir_okay=True,
    ),
    url: Optional[str] = typer.Option(
        None,
        "--url",
        help="(reserved) remote source URL — use --src for staged cmake builds",
    ),
    define: Optional[List[str]] = typer.Option(
        None,
        "-D",
        "--define",
        help="unbound CMake -D KEY=VALUE (bound keys rejected)",
    ),
    build_dir: Optional[Path] = typer.Option(
        None,
        "--build-dir",
        help="cmake -B directory (default: <src>/build)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="print cmake configure command without running it",
    ),
) -> None:
    """Configure the native build (``cmake -S -B -D…``). Streams raw cmake output."""
    mod_list = list(modules or [])
    for m in mod_list:
        if m not in MODULE_TO_DEFINE and m != "hf":
            typer.echo(
                f"error: unknown module {m!r}; known: {sorted(MODULE_TO_DEFINE)}",
                err=True,
            )
            raise typer.Exit(2)

    cli_options = _global_options(mpi, real_storage, real_work, id_size)
    has_cli_options = any(v is not None for v in cli_options.values())

    try:
        validate_mode(
            config_path=str(config) if config is not None else None,
            modules=mod_list,
            hf=hf,
            has_cli_options=has_cli_options,
        )
        if config is not None:
            cfg = load_toml_config(config)
            # Allow global CLI overrides on top of TOML only when no XOR conflict
            # — validate_mode already forbids mixing modules; for pure --config
            # with global flags, fold globals in.
            if has_cli_options:
                apply_cli_modules(cfg, [], options=cli_options)
        else:
            cfg = InstallConfig()
            apply_cli_modules(cfg, mod_list, options=cli_options)
        if define:
            merge_unbound_cli(cfg, define)
        if src is None and url is None:
            raise ConfigError(
                "config requires --src /path/to/Atomiverse "
                "(staged cmake pipeline)"
            )
        source = resolve_source(
            src=str(src) if src is not None else None,
            url=url,
        )
        code = run_config(
            cfg,
            source,
            build_dir=build_dir,
            dry_run=dry_run,
        )
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    if code != 0:
        raise typer.Exit(code)


@app.command("compile")
def compile_cmd(
    jobs: Optional[int] = typer.Option(
        None,
        "-j",
        "--jobs",
        help="parallel build jobs (cmake --build -j)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="print cmake --build command without running it",
    ),
) -> None:
    """Compile the native extension (``cmake --build``). Streams raw cmake output."""
    try:
        code = run_compile(jobs=jobs, dry_run=dry_run)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    if code != 0:
        raise typer.Exit(code)


@app.command("install")
def install_cmd(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="print cmake --install command without running it",
    ),
    prefix: Optional[Path] = typer.Option(
        None,
        "--prefix",
        help="cmake install prefix (default: this env's site-packages)",
    ),
) -> None:
    """Install built artifacts into the environment (``cmake --install``).

    Streams raw cmake output. Requires a prior successful ``config`` + ``compile``.
    """
    try:
        code = run_install_stage(dry_run=dry_run, prefix=prefix)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    if code != 0:
        raise typer.Exit(code)


@app.command("status")
def status_cmd() -> None:
    """Show modules / fingerprint from the last config or install."""
    code = run_status()
    if code != 0:
        raise typer.Exit(code)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for tests and ``python -m atomiverse``."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        result = app(args=args, standalone_mode=False)
        if isinstance(result, int):
            return result
        return 0
    except typer.Exit as exc:
        code = getattr(exc, "exit_code", None)
        if code is None:
            code = getattr(exc, "code", 0)
        return int(code) if code is not None else 0
    except SystemExit as exc:  # pragma: no cover
        code = exc.code
        if code is None:
            return 0
        return int(code) if isinstance(code, int) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
