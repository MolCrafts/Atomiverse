# install_cli_dryrun — public CLI dry-run contract

## What this pins

`python -m atomiverse install` / `status` on the pure-Python surface:
configuration parsing, mutual exclusion, HF/empty reject, and the define
set for `--md` (auto-enable closure + `ATV_BUILD_PYTHON=ON`). **No C++
compile and no real `pip install`.**

Physical literature N/A — packaging / CLI contract for chain
`python-on-demand-install` phase 03.

## Reference (contract)

| Check | Expected |
|---|---|
| `python -m atomiverse --help` / bare invocation | exit 0; mentions `install` |
| `install --config X.toml --md` | non-zero; mutual exclusion |
| `install` (no modules) | non-zero; empty install |
| `install --hf` / `install --md --hf` | non-zero; HF/SCF unavailable |
| `install --md --dry-run --src <repo>` | exit 0; defines include `ATV_BUILD_MD=ON`, `ATV_BUILD_POTENTIALS=ON`, `ATV_BUILD_NEIGHBOR=ON`, `ATV_BUILD_PYTHON=ON` |
| `status` | exit 0; prints modules + fingerprint lines |

## Tolerance

Exact substring match for define keys and error keywords; no numeric
tolerance.
