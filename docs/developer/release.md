# Releasing the Python package

What ships, how a release is cut, and why the native extension is not on
PyPI.

## What ships

PyPI carries one distribution, `atomiverse`, as an **sdist plus a pure
`py3-none-any` wheel** built from `python/`. It holds the Python API, the
`python -m atomiverse` CLI and no compiled code (`wheel.cmake = false` in
`python/pyproject.toml`).

The C++ extension `_atomiverse` is built on the user's machine from the
**git tag matching the installed version**:

| Stage | Runs | Result |
|---|---|---|
| `config` | resolves modules → `ATV_*` defines, records `<build>/atomiverse_cli_state.json` | nothing compiled |
| `compile` | `pip wheel <src>/python --no-build-isolation -C wheel.cmake=true -C build-dir=<build>/cmake -C cmake.define.…`, then `pip wheel <ATV_MOLPY_PATH>/molpy --no-build-isolation` with `CARGO_TARGET_DIR=<build>/molpy-target` | the native wheel and a molpy wheel in `<build>/dist/` |
| `install` | `pip install --force-reinstall --no-deps <wheel> <molpy-wheel>` | the native wheel replaces the pure one; the local molpy wheel replaces the installed one |

Build the molpy wheel and the `molcxx` bridge from the same local molpy
checkout selected by `ATV_MOLPY_PATH` (default `../molpy`). Capsule
validation remains required. No molpy version is pinned.

Because pip installs the native wheel, `_atomiverse*.so` and the
`_build_info.json` CMake writes beside it are RECORD-tracked files of the
distribution: `pip uninstall` removes them and a `pip install -U` replaces
them. Nothing is copied into site-packages behind pip's back. The only
file the CLI writes outside the build tree is the per-environment pointer
`<sys.prefix>/share/atomiverse/active_build.json`.

Every product module is passed explicitly ON or OFF, so a native wheel
contains exactly the modules the user listed plus their DAG closure;
`python/pyproject.toml` fixes only the non-product switches
(`ATV_BUILD_PYTHON=ON`, tests/benchmarks/examples OFF).

### Why not prebuilt native wheels

The extension links BLAS/LAPACK, FFTW, optionally MPI and CUDA, the Rust
molpy static library, and is compiled with `-march=native` for the SCF ERI
kernels. A single manylinux binary would have to drop `-march=native` and
freeze one module set; building from the tag keeps the user's module choice
and full CPU speed. Revisit if a CPU-only, fixed-module, portable-ISA wheel
is wanted for quick starts.

## Version

The single source is `__version__` in `python/src/atomiverse/__init__.py`;
`pyproject.toml` reads it through scikit-build-core's regex provider. The
CLI's version lock (`config` refuses a checkout whose `__version__` differs
from the installed package) depends on this being the only copy.

## Cutting a release

1. On `dev`, bump `__version__` and land it.
2. Merge `dev` into `master` (never commit to `master` directly).
3. Tag `master` and push the tag:
   ```bash
   git tag -a vX.Y.Z -m "atomiverse X.Y.Z" master
   git push upstream vX.Y.Z
   ```
4. `.github/workflows/release.yml` then:
   - fails unless the tag equals `__version__` and the tagged commit is on
     `master`;
   - builds the sdist + wheel, runs `twine check --strict`, installs the
     wheel into a clean venv and runs the pure-Python tests against it;
   - publishes to PyPI by **trusted publishing** (OIDC, no stored token) from
     the `pypi` environment;
   - creates the GitHub release for the tag with the same files.

Native builds require a local molpy checkout.

### One-time setup

- PyPI: register project `atomiverse` and add a trusted publisher —
  repository `MolCrafts/Atomiverse`, workflow `release.yml`, environment
  `pypi`.
- GitHub: create the `pypi` environment (add required reviewers to gate
  publishing on a human).

## Checking a release locally

```bash
python -m build python --outdir /tmp/dist
python -m twine check --strict /tmp/dist/*
python -m venv /tmp/v && /tmp/v/bin/pip install /tmp/dist/*.whl
/tmp/v/bin/python -m atomiverse --version
```
