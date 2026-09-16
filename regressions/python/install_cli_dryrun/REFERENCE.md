# install_cli_dryrun — public CLI dry-run contract

## What this pins

`python -m atomiverse install` / `status` on the pure-Python surface:
configuration parsing, mutual exclusion, empty reject, the `hf` module
resolving `ATV_BUILD_HF=ON`, and the define set for `--md` (auto-enable
closure + `ATV_BUILD_PYTHON=ON`). **No C++ compile and no real
`pip install`.**

Physical literature N/A — packaging / CLI contract for chain
`python-on-demand-install` phase 03.

## Reference (contract)

| Check | Expected |
|---|---|
| `python -m atomiverse --help` / bare invocation | exit 0; mentions `install` |
| `install --config X.toml --md` | non-zero; mutual exclusion |
| `install` (no modules) | non-zero; empty install |
| `config md --hf --dry-run --src <repo>` | exit 0; defines include `ATV_BUILD_HF=ON` (normal module) |
| `install --md --dry-run --src <repo>` | exit 0; defines include `ATV_BUILD_MD=ON`, `ATV_BUILD_POTENTIALS=ON`, `ATV_BUILD_NEIGHBOR=ON`, `ATV_BUILD_PYTHON=ON` |
| `status` | exit 0; prints modules + fingerprint lines |

## Tolerance

Exact substring match for define keys and error keywords; no numeric
tolerance.
