# pure_import — pure-Python package import contract

## What this pins

A default `pip install atomiverse` delivers a **pure-Python** package: no
`_atomiverse` shared library is required for the package to import. Compute
entry points fail loud with an install-oriented `ImportError`;
`python -m atomiverse` prints install guidance without traceback.

## Reference (contract, not literature)

| Check | Expected |
|---|---|
| `import atomiverse` without extension | succeeds |
| `atomiverse.__version__` | non-empty string (matches packaging version) |
| `atomiverse.MD` / `from atomiverse import MD` | `ImportError` containing `python -m atomiverse install` |
| `python -m atomiverse` | exit 0; stdout/stderr mention install guidance; no uncaught traceback |

Physical literature N/A — this is a packaging / import contract regression
for chain `python-on-demand-install` phase 02.

## Tolerance

Exact substring match for the install hint; no numeric tolerance.
