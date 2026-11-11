# Validation and CI

Atomiverse uses the immutable MolCrafts CI revision and `cpp-python` profile
declared in `.github/partners.env`. The profile owns gate commands, dependencies,
tool probes and platform lanes. `.pre-commit-config.yaml` installs both hooks.

```bash
CI_REF=$(sed -n 's/^CI_REF=//p' .github/partners.env)
uv tool install --force "molcrafts-ci @ git+https://github.com/MolCrafts/molcrafts-ci@$CI_REF"
molci run partners
molci doctor --install-hooks
molci check
molci verify
molci verify --full
molci run native
molci explain native-build
```

Commit checks preserve the existing clang-format and hygiene rules and validate
workflow policy, dependency locks and release versions. Push checks additionally
build the complete CPU product set, all registered serial CPU units, examples
and performance harnesses, the native Python wheel, Python tests, benchmark
framework units, strict documentation and pure-Python release packages. Examples
and performance harnesses are compiled; scientific campaigns run separately.

The native gate calls the existing `atomiverse.cli` configuration and wheel
builders. It builds one MolPy wheel and one Atomiverse native wheel and reuses
those exact artifacts for subsequent checks. Their hashes are checked before a
cached success is accepted. Unknown source paths invalidate native results;
there is no unmapped-path fallback to a small core test.

Verification resolves one immutable MolPy source under `build/ci-partners/molpy`.
The Python runtime and CXX bridge use that same tree and Rust toolchain. Normal
interactive CMake builds can still use `ATV_MOLPY_PATH` and the sibling checkout.
The validation source is separate so hooks do not reset a developer's MolPy
working tree. Modified validation dependencies fail with their path.

Python tools and scientific dependencies share `python/uv.lock`. Dependencies
needed only by a scientific campaign are installed for that campaign. Repeat
hooks reuse only successful results with matching source, dependency, toolchain,
environment, platform and artifact hashes. `--full` forces every applicable
native gate to execute again. Push hooks reject a dirty source tree or any
pushed ref that differs from the verified commit.

## GitHub lifecycle

| Event | Verification |
| --- | --- |
| Development-mirror push | No compilation |
| Upstream PR, including external and Dependabot | Independent full native validation plus platform checks |
| Upstream dev merge | Integration was checked by its required PR; no second matrix |
| Upstream master push | Independent native validation and platform checks |
| Manual test dispatch | Independent native validation and platform checks |
| Weekly or manual nightly | Independent HF scientific campaign; artifacts only |
| Release tag | Exact master-commit verification, pure package build, trusted publication |
| Manual release dispatch | Package dry run; publication disabled |

Linux validates the full CPU product surface; macOS checks native C++ and Python
integration; Windows checks the supported pure-Python delivery. CUDA is not
scheduled on untrusted or privileged GitHub runners. Required jobs must succeed:
skipped, missing, cancelled or failed required checks cannot satisfy `test / result`.
Dev changes must use PRs; the master-push run independently verifies the exact
commit that can become a release, even when its PR was already green.
Hooks are a development gate, not proof that an external contributor ran tests.

`Roy-Kid/Atomiverse` is an independent mirror, outside the upstream GitHub fork
network. Push a feature branch to `MolCrafts/Atomiverse` and open its PR against
`dev`; GitHub cannot open a direct cross-repository PR from that mirror. This
profile grants no trusted-fork exception to independent native validation.

Compiler caches include CPU identity and instruction flags because the native
build uses `-march=native`. They also account for the compiler, Rust, CMake and
BLAS/FFTW versions. GitHub restores build artifacts, never local success receipts.
Gate failure logs are retained as artifacts.

## CUDA and cluster execution

`molci run cuda` is the dedicated CUDA unit gate. It requires the project's
aarch64 GH200 environment and fails on missing GPU/toolchain prerequisites.
Submit cluster work through the public `molq` interface and the authorized
resource budget described in the project contract. The hook's
`MOLCRAFTS_HOOK_RUNNER` transport must wait for validation and preserve its exit
status; submission alone is not a pass. Do not expose a privileged runner to
untrusted PRs. MPI and other optional backends retain their explicit build and
test procedures; the hosted CPU gate does not attest them.

## Scientific campaigns and releases

`molci run benchmark-hf` runs the existing independent HF comparison inventory
and build-attestation protocol. Set `MOLCI_BENCHMARK_TIER=acceptance` for the
acceptance tier. `molci run benchmark-report` preserves a numerical failure in
the report rather than discarding its evidence. Nightly checks out one resolved
dev commit for the whole observation and does not publish development packages.

`molci run package --full` builds and checks the pure wheel and sdist and exports
them to `dist/`. Release guards accept only a successful upstream master-push
verification of the exact release commit. A local hook, an unrelated PR run or
a skipped job is insufficient. Publication still uses the `pypi` environment and
OIDC, and native wheels remain machine-local.

After the workflow migration lands, require `test / result` in the upstream dev
and master branch rules. Do not enable that requirement before the workflow is
present. Merge through dev; keep tags and publication separate from CI migration.
