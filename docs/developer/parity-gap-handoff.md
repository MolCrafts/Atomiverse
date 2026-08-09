# parity-gap-loop handoff

Current understanding for the next wave's agents. Rewrite, never append.

- Wave 3 (2026-08-09) staffed 1, merged 1: `pw-perf-fair-walls-batch1`
  (`2a7bd70b`). Four PW cases (`pw-pbe-gga-si`, `pw-analytic-stress-si`,
  `pw-uks-kmesh-fe`, `bands-dos-si`) now carry measured same-settings QE 7.6
  NC-UPF walls with inputs pinned under `tools/qe_oracle/inputs/`. The QE
  oracle pipeline is proven end-to-end — batch repins are now routine work.
- **Ready P2 item: `pw-perf-fair-walls-batch2`.** Post-batch-1 census counts
  18 PW `REFERENCE.md` files still `perf_exempt`. Next slice: `pw-lda-si`,
  `pw-spin-gga`, `pw-spin-polarized-o2`, `pw-uks-fe-smearing`, same shape as
  batch 1. Skip `pw-variable-cell-si` — it overlaps the in-flight
  variable-cell tree.
- Three GH200 slugs remain deferred `[~]` with live worktrees and branches —
  **resume them, do not restaff from scratch** (none were touched in wave 3;
  tips and job ids unchanged):
  - `reg-cuda-grad-honesty` — worktree `.claude/worktrees/wf_f384f427-0b3-11`
    @ `wf/reg-cuda-grad-honesty` (now `ca4d62c7`). Recover round 1 fixed the
    nvcc syntax errors from job 1035228 and resubmitted as job **1039997**
    (success line: `ALL CUDA LEGS PASS` in its log). Harvest that log, close
    the gate, merge.
  - `cuda-pw-large-cell-measure` — worktree
    `.claude/worktrees/wf_f384f427-0b3-13` @ `wf/cuda-pw-large-cell-measure`
    (`d574d65d`). Job 1035278 was **still RUNNING at wave-3 close** (~48 min
    into its 1 h limit — it has finished or hit walltime by now; attempts
    1035136 / 1035238 failed earlier). Build dir `build-cuda-pw-cell` was
    deliberately kept. Harvest `.claude/cuda-pw-large-cell-1035278.{out,err}`
    and write the measurement note in `.claude/notes/` — no perf claim from
    small cells; if it hit walltime, shrink the case or split the run.
  - `cuda-pw-pbe-device` — worktree `.claude/worktrees/wf_f384f427-0b3-14`
    @ `wf/cuda-pw-pbe-device` (`e1147831`). Job 1035270 failed at link (the
    aarch64 link pulled x86_64 `libxtb.so` into `test_cuda_pw_xc_pbe`);
    recover round 1 scrubbed the lib path and resubmitted as job **1039999**.
    Harvest its log, close the gate (LDA-parity-pattern tolerances), merge.
- `cuda-md-first-regression` stays **pending** (not deferred): its wave-2
  attempt is parked unmerged at branch `wf/cuda-md-first-regression`
  (`ea17e11a`, LJ argon NVE multi-device case; GH200 jobs 1035232 / 1035271
  failed). Next staffing should branch from / cherry-pick that tip.
- `test-gpw-timeout-triage` is **done** (verdict in the queue's ## done):
  `mbd` 34 s / `hirshfeld` 98 s @ 4.8 GB / `stress` 13.6 s genuinely heavy vs
  the 10 s per-test registration; `forces` + `test_mpi_pw_kpoint_n4` are
  load-flakes. Follow-up shrink work is the new P0 pending
  `test-gpw-unit-shrink`.
- Found, unresolved tension (surface, do not "fix" without a decided rule):
  `tests/cuda/scf/test_rhf.cu:649-800` runs full SCF solves inside `tests/`,
  in tension with iron law 1, yet is today's only CUDA force-parity coverage.
  Needs an owner decision: move to `regressions/` vs sanction as CUDA
  correctness twins (surfaced by the wave-2 grad-honesty worker).
- GH200 batch discipline unchanged: job scripts start `#!/bin/bash -l`, load
  `GPU/buildenv-gcccuda/2026.03-cu13.0`, never pipe `module load`, keep the
  default aarch64 Rust toolchain, and a job not started within ~20 min means
  defer with the job id recorded.
- In-flight tree law still holds: the parent working tree carries the
  ~105 uncommitted variable-cell/stress files; never stage or touch them.
