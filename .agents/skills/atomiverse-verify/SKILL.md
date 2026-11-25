---
name: atomiverse-verify
description: Runs the Atomiverse commit and push gates through the pinned MolCrafts CI and fixes every failure without bypassing hooks. Use when a change is ready to commit, push, open a PR against dev, or hand off as finished.
---

# atomiverse-verify

Hooks and CI run the pinned `molci` (profile `cpp-python`); AGENTS.md
`## Build and gates` owns the gate set and `### Iron laws` (No silent debt)
owns how failures are treated.

## Steps

1. **Install the pinned CI once.** From the repository root:

   ```bash
   CI_REF=$(sed -n 's/^CI_REF=//p' .github/partners.env)
   uv tool install --force "molcrafts-ci @ git+https://github.com/MolCrafts/molcrafts-ci@$CI_REF"
   molci run partners
   molci doctor --install-hooks
   ```

2. **Run the commit gate.** `molci check` (clang-format, hygiene, workflow
   policy, locks, release versions).
3. **Run a targeted gate for fast feedback.** `molci run native` for the
   native build and tests; `molci run cuda` only on the GH200 node (see the
   `atomiverse-cluster` skill).
4. **Run the push gate.** `molci verify`; add `--full` to rerun every gate.
5. **Fix each failure at its cause.** Fix it locally or stop and report
   `path:line`; never skip-mark, weaken an assert or special-case a test.
6. **Rerun until green.** Never use `--no-verify` and never kill a
   `git commit` / `git push` mid-hook.
7. **Open the PR as AGENTS.md `## Workflow` says.** Push a feature branch to
   `MolCrafts/Atomiverse` and target `dev`.

## Done when

- `molci check` and `molci verify` pass
- every failure found is named in the summary as found / fixed / blocking
- no hook was bypassed or interrupted
