"""Install configuration: binding table, TOML/inline parse, auto-enable DAG.

Maps user-facing module flags and friendly option keys onto the CMake
``ATV_*`` defines declared in ``cmake/AtvProductModules.cmake`` (phase 01).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Set

# ── Module flags (CLI / [modules] section) → product ATV_BUILD_* ──────────
# Names match the user-facing CLI surface. ``am1`` maps to SEMIEMPIRICAL;
# ``minimize`` maps to MINIMIZER. ``neighbor`` is optional-explicit (also
# pulled in by the auto-enable closure).

MODULE_TO_DEFINE: Dict[str, str] = {
    "md": "ATV_BUILD_MD",
    "minimize": "ATV_BUILD_MINIMIZER",
    "mc": "ATV_BUILD_MC",
    "evaluate": "ATV_BUILD_EVALUATE",
    "vib": "ATV_BUILD_VIB",
    "path": "ATV_BUILD_PATH",
    "pw": "ATV_BUILD_PW",
    "gpw": "ATV_BUILD_GPW",
    "am1": "ATV_BUILD_SEMIEMPIRICAL",
    "cuda": "ATV_BUILD_CUDA",
    "kspace": "ATV_BUILD_KSPACE",
    "potentials": "ATV_BUILD_POTENTIALS",
    "neighbor": "ATV_BUILD_NEIGHBOR",
}

# Product modules that are set ON/OFF explicitly on every resolve so monorepo
# CMake defaults (mostly ON) do not leak into an on-demand install.
PRODUCT_MODULES: tuple[str, ...] = tuple(MODULE_TO_DEFINE.keys())

# Auto-enable closure — same one-way DAG as cmake/AtvProductModules.cmake:
#   MD, MC, MINIMIZER → POTENTIALS + NEIGHBOR
#   PATH → MINIMIZER  (then inherits minimizer → potentials+neighbor)
#   KSPACE → POTENTIALS
AUTO_ENABLE: Dict[str, tuple[str, ...]] = {
    "md": ("potentials", "neighbor"),
    "mc": ("potentials", "neighbor"),
    "minimize": ("potentials", "neighbor"),
    "path": ("minimize",),
    "kspace": ("potentials",),
}

# Global-only bound options (CLI: top-level ``--mpi``, ``--real-storage``, …;
# TOML: only under ``[global]`` — never under ``[md]`` / ``[pw]``).
GLOBAL_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "mpi",
        "real_storage",
        "real_work",
        "id_size",
    }
)

# Friendly option keys → ATV_* CMake cache names (01 names).
# Module flags themselves are also bound (cannot pass as -D / unbound).
BINDING_TABLE: Dict[str, str] = {
    # module enable flags
    **MODULE_TO_DEFINE,
    # global build options
    "mpi": "ATV_USE_MPI",
    "real_storage": "ATV_REAL_STORAGE",
    "real_work": "ATV_REAL_WORK",
    "id_size": "ATV_ID_SIZE",
}

# Reverse: any ATV_* name that is bound (cannot appear under cmake.unbound / -D).
BOUND_DEFINES: Set[str] = set(BINDING_TABLE.values()) | {"ATV_BUILD_PYTHON"}

_BOOL_TRUE = frozenset({"yes", "true", "on", "1"})
_BOOL_FALSE = frozenset({"no", "false", "off", "0"})

HF_REJECT_MSG = (
    "HF/SCF Python bindings are not available yet. "
    "Remove --hf / modules.hf until SCF bindings ship."
)


class ConfigError(ValueError):
    """User-facing configuration error (fail-loud)."""


@dataclass
class InstallConfig:
    """Resolved install request before define mapping.

    @param modules: Module name → enabled bool (only explicitly requested /
        auto-enabled modules need be present; resolve fills the rest).
    @param options: Bound friendly-key → value (bool or str).
    @param unbound: Unbound CMake cache key → value (stringified).
    """

    modules: Dict[str, bool] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    unbound: Dict[str, str] = field(default_factory=dict)


def parse_bool(raw: str) -> Optional[bool]:
    """Parse a bool synonym; return None if not a recognised bool token."""
    key = raw.strip().lower()
    if key in _BOOL_TRUE:
        return True
    if key in _BOOL_FALSE:
        return False
    return None


def _load_toml(path: Path) -> Mapping[str, Any]:
    try:
        import tomllib  # py>=3.11
    except ModuleNotFoundError:  # pragma: no cover - exercised on 3.9/3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise ConfigError(
                "TOML config requires Python 3.11+ (tomllib) or the 'tomli' package"
            ) from exc
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_toml_config(path: Path | str) -> InstallConfig:
    """Load unified TOML config (``[modules]``, per-module, ``[global]``, unbound).

    @param path: Path to a ``.toml`` file.
    @return Parsed :class:`InstallConfig`.
    @raises ConfigError: on unknown sections/keys, hf enable, bound-as-unbound.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    data = dict(_load_toml(p))
    cfg = InstallConfig()

    modules_tbl = data.pop("modules", None) or {}
    if not isinstance(modules_tbl, Mapping):
        raise ConfigError("[modules] must be a table")
    for name, enabled in modules_tbl.items():
        name = str(name)
        if name == "hf":
            if enabled:
                raise ConfigError(HF_REJECT_MSG)
            continue
        if name == "global":
            raise ConfigError("'global' is not a product module; use [global]")
        if name not in MODULE_TO_DEFINE:
            raise ConfigError(f"unknown module in [modules]: {name!r}")
        cfg.modules[name] = bool(enabled)

    global_tbl = data.pop("global", None) or {}
    if not isinstance(global_tbl, Mapping):
        raise ConfigError("[global] must be a table")
    for key, value in global_tbl.items():
        _set_bound_option(cfg, str(key), value, where="[global]")

    cmake_tbl = data.pop("cmake", None) or {}
    if not isinstance(cmake_tbl, Mapping):
        raise ConfigError("[cmake] must be a table")
    unbound_tbl = cmake_tbl.get("unbound", {}) if cmake_tbl else {}
    # allow only the unbound sub-table
    extra_cmake = set(cmake_tbl.keys()) - {"unbound"}
    if extra_cmake:
        raise ConfigError(
            f"[cmake] only supports [cmake.unbound], not {sorted(extra_cmake)}"
        )
    if not isinstance(unbound_tbl, Mapping):
        raise ConfigError("[cmake.unbound] must be a table")
    for key, value in unbound_tbl.items():
        _set_unbound(cfg, str(key), value)

    # remaining top-level tables are per-module option sections
    for section, body in data.items():
        if section in ("modules", "global", "cmake"):
            continue
        if section == "hf":
            raise ConfigError(HF_REJECT_MSG)
        if section not in MODULE_TO_DEFINE:
            raise ConfigError(f"unknown config section [{section}]")
        if not isinstance(body, Mapping):
            raise ConfigError(f"[{section}] must be a table")
        # presence of a per-module table does not alone enable the module;
        # [modules] does. But if the module is listed with options, require
        # it be enabled (or enable it implicitly when the table is non-empty
        # only if modules already said so). Spec: modules table enables.
        for key, value in body.items():
            key_s = str(key)
            if key_s in GLOBAL_OPTION_KEYS:
                raise ConfigError(
                    f"{key_s!r} is a global option — put it under [global], "
                    f"not [{section}]"
                )
            _set_bound_option(cfg, key_s, value, where=f"[{section}]")

    return cfg


def _stringify_cmake_value(value: Any) -> str:
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    return str(value)


def _set_bound_option(
    cfg: InstallConfig, key: str, value: Any, *, where: str
) -> None:
    if key not in BINDING_TABLE:
        raise ConfigError(
            f"unknown bound key {key!r} in {where}; "
            f"known keys: {sorted(BINDING_TABLE)}"
        )
    # module names used as options are not valid under [global]/module]
    if key in MODULE_TO_DEFINE:
        raise ConfigError(
            f"{key!r} is a module flag, not an option key (set it under [modules])"
        )
    if isinstance(value, str):
        b = parse_bool(value)
        cfg.options[key] = b if b is not None else value
    else:
        cfg.options[key] = value


def _set_unbound(cfg: InstallConfig, key: str, value: Any) -> None:
    # Reject friendly bound keys and their ATV_* targets.
    if key in BINDING_TABLE or key in BOUND_DEFINES:
        raise ConfigError(
            f"bound key {key!r} cannot be passed as cmake.unbound / -D; "
            "use the friendly module flag or [global] option instead"
        )
    # Also reject if someone passes the friendly name that maps to a define
    for friendly, define in BINDING_TABLE.items():
        if key == define or key == friendly:
            raise ConfigError(
                f"bound key {key!r} cannot be passed as cmake.unbound / -D"
            )
    cfg.unbound[key] = _stringify_cmake_value(value)


def apply_cli_modules(
    cfg: InstallConfig,
    modules: Iterable[str],
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> InstallConfig:
    """Enable product modules from CLI positionals and apply bound options.

    Scheme B (Typer): modules are bare names (``md``, ``pw``, …), not
    ``--md k:v`` strings. Options are native flags (``--mpi``,
    ``--real-storage``, …) collected into *options*.

    @param cfg: Config to mutate (returned for chaining).
    @param modules: Product module names to enable.
    @param options: Bound friendly-key → value (bool or str); ``None`` values skipped.
    """
    for name in modules:
        name = str(name).strip()
        if not name:
            continue
        if name == "hf":
            raise ConfigError(HF_REJECT_MSG)
        if name not in MODULE_TO_DEFINE:
            raise ConfigError(
                f"unknown module {name!r}; "
                f"known: {sorted(MODULE_TO_DEFINE)}"
            )
        cfg.modules[name] = True
    if options:
        for key, value in options.items():
            if value is None:
                continue
            _set_bound_option(cfg, str(key), value, where="CLI option")
    return cfg


def apply_auto_enable(modules: MutableMapping[str, bool]) -> Set[str]:
    """Close *modules* under the product-module DAG (01 semantics).

    @param modules: Module enable map (mutated in place).
    @return Set of module names that are enabled after closure.
    """
    # iterate to fixed point (path → minimize → potentials+neighbor)
    changed = True
    while changed:
        changed = False
        enabled = [m for m, on in modules.items() if on]
        for m in enabled:
            for dep in AUTO_ENABLE.get(m, ()):
                if not modules.get(dep, False):
                    modules[dep] = True
                    changed = True
    return {m for m, on in modules.items() if on}


def resolve_defines(cfg: InstallConfig) -> Dict[str, str]:
    """Map an :class:`InstallConfig` to CMake ``ATV_*`` define strings.

    Always forces ``ATV_BUILD_PYTHON=ON``. Every product module is set
    explicitly ON or OFF so monorepo defaults do not leak.

    @param cfg: Parsed install configuration.
    @return Ordered-stable dict of define name → ``ON``/``OFF``/value.
    @raises ConfigError: empty install (no modules enabled).
    """
    modules: Dict[str, bool] = {m: False for m in PRODUCT_MODULES}
    for name, on in cfg.modules.items():
        if name == "hf" and on:
            raise ConfigError(HF_REJECT_MSG)
        if name not in MODULE_TO_DEFINE:
            raise ConfigError(f"unknown module {name!r}")
        modules[name] = bool(on)

    enabled = apply_auto_enable(modules)
    if not enabled:
        raise ConfigError(
            "empty install: enable at least one module "
            "(e.g. `config md`, or [modules] in --config)"
        )

    defines: Dict[str, str] = {}
    for name in PRODUCT_MODULES:
        defines[MODULE_TO_DEFINE[name]] = "ON" if modules.get(name) else "OFF"

    defines["ATV_BUILD_PYTHON"] = "ON"

    for key, value in cfg.options.items():
        if key not in BINDING_TABLE:
            raise ConfigError(f"unknown bound option {key!r}")
        define = BINDING_TABLE[key]
        defines[define] = _stringify_cmake_value(value)

    for key, value in cfg.unbound.items():
        if key in BOUND_DEFINES or key in BINDING_TABLE:
            raise ConfigError(
                f"bound key {key!r} cannot be passed as cmake.unbound / -D"
            )
        defines[key] = value

    return defines


def enabled_modules_from_defines(defines: Mapping[str, str]) -> List[str]:
    """List product modules whose define is ON (stable sorted)."""
    on: List[str] = []
    for name, define in MODULE_TO_DEFINE.items():
        if defines.get(define, "OFF").upper() in ("ON", "TRUE", "1"):
            on.append(name)
    return on


def define_fingerprint(defines: Mapping[str, str]) -> str:
    """Stable fingerprint of a define map (sorted ``k=v`` joined)."""
    parts = [f"{k}={defines[k]}" for k in sorted(defines)]
    return "|".join(parts)


def validate_mode(
    *,
    config_path: Optional[str],
    modules: Iterable[str],
    hf: bool,
    has_cli_options: bool = False,
) -> None:
    """Enforce XOR of ``--config`` vs positional modules; reject ``--hf`` / empty.

    Global CLI options (``--mpi``, ``--real-storage``, …) may accompany either
    mode. They are **not** per-module.

    @param config_path: Path from ``--config``, or None.
    @param modules: Positional module names (scheme B).
    @param hf: Whether ``--hf`` was passed.
    @param has_cli_options: True if any global CLI option was set. Alone
        (without modules or --config) still counts as empty.

    @raises ConfigError: on mutual exclusion, empty config, or --hf.
    """
    if hf:
        raise ConfigError(HF_REJECT_MSG)

    mod_list = [m for m in modules if m]
    if config_path and mod_list:
        raise ConfigError(
            "mutual exclusion: use either --config PATH.toml or CLI modules "
            "(e.g. `config md pw`), not both"
        )
    if not config_path and not mod_list:
        raise ConfigError(
            "empty config: pass --config PATH.toml or at least one module "
            "(e.g. `config md` or `config md pw --mpi`)"
        )
    # has_cli_options alone is insufficient — already covered when no modules
    # and no config_path above. Keep the parameter for call-site clarity.
    _ = has_cli_options


def merge_unbound_cli(cfg: InstallConfig, pairs: Iterable[str]) -> None:
    """Apply ``-D KEY=VALUE`` / ``--define KEY=VALUE`` CLI unbound pairs."""
    for raw in pairs:
        if "=" not in raw:
            raise ConfigError(f"-D expects KEY=VALUE, got {raw!r}")
        key, _, value = raw.partition("=")
        key = key.strip()
        # strip optional leading -D
        if key.startswith("-D"):
            key = key[2:]
        _set_unbound(cfg, key, value.strip())
