---
name: atomiverse-release
description: Cuts an Atomiverse release of the pure atomiverse wheel and sdist to PyPI by bumping the version on dev, merging into master and tagging. Use when the operator asks to bump the version, rehearse or publish a release.
---

# atomiverse-release

PyPI gets the pure `atomiverse` wheel and sdist; the native extension builds
from the matching tag. AGENTS.md `## Workflow` owns the procedure,
`docs/developer/release.md` the detail.

## Steps

1. **Bump the version on `dev`.** Edit `__version__` in
   `python/src/atomiverse/__init__.py`, the single version source, and keep
   `project(Atomiverse VERSION ...)` in `CMakeLists.txt` consistent.
2. **Verify fully.** Run `molci verify --full` (see `atomiverse-verify`);
   `molci run package --full` builds the wheel and sdist into `dist/`.
3. **Rehearse if wanted.** A manual dispatch of `release.yml` is a dry run.
4. **Merge into `master`.** Merge `dev` into `master`; never commit to
   `master` directly. Wait for the master-push `test / result` to succeed.
5. **Tag `master` and push the tag.** After the operator confirms the
   MolCrafts remote:

   ```bash
   git tag -a vX.Y.Z -m "atomiverse X.Y.Z" master
   git push <molcrafts-remote> vX.Y.Z
   ```

6. **Watch the publish.** `release.yml` checks tag = `__version__`, the
   commit on `master` and a green `test / result`, then publishes by trusted
   publishing from the `pypi` environment.

## Done when

- the tag equals `v` + `__version__` and sits on `master`
- `release.yml` finished green and PyPI shows the version
- `dev` and `master` agree on the released version
