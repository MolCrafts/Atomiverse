# install_cli (docs / primary CLI contract)

Literature: N/A.

Expected exit semantics (no full native compile required):

| Case | Exit | Notes |
|------|------|--------|
| `install --help` | 0 | Mentions `--config`, MODULE, `--src`, `--url` |
| `--config` with positional `md` | ≠0 | Mutual exclusion (scheme B) |
| `--hf` | ≠0 | SCF/HF bindings not available |
| Source pyproject version ≠ installed | ≠0 | Before pip/compile |

CLI shape (scheme B): `install md pw --mpi --real-storage double` — no k:v strings.

Canonical product docs: `python/docs/getting_started.md`.
