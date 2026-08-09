# parity-gap-loop handoff

Current understanding for the next wave's agents. Rewrite, never append.

- Wave 2 (2026-08-09) staffed 5, merged 1: `pw-perf-qe-install` (`6d13a436`).
  `tools/qe_oracle/` is live (README + provenance rules + smoke-si input +
  `pin_notes.md` with QE version/build line), so **`pw-perf-fair-walls-batch1`
  is unblocked** and is the ready P2 item.
- Three GH200 slugs are deferred `[~]` with live worktrees and branches —
  **resume them, do not restaff from scratch**:
  - `reg-cuda-grad-honesty` — worktree `.claude/worktrees/wf_f384f427-0b3-11`
    @ `wf/reg-cuda-grad-honesty` (`9fa88e9e`). Job 1035228 failed on nvcc
    syntax errors in the new
    `regressions/scf/h2o_rhf_sto3g_grad/reg_h2o_rhf_sto3g_grad.cu`; fix the
    compile, resubmit one batch run.
  - `cuda-pw-large-cell-measure` — worktree
    `.claude/worktrees/wf_f384f427-0b3-13` @ `wf/cuda-pw-large-cell-measure`
    (`d574d65d`). Job 1035278 was RUNNING at wave close (attempts 1035136 /
    1035238 failed); its build dir `build-cuda-pw-cell` was deliberately kept.
    Harvest `.claude/cuda-pw-large-cell-1035278.{out,err}` and write the
    measurement note in `.claude/notes/` — no perf claim from small cells.
  - `cuda-pw-pbe-device` — worktree `.claude/worktrees/wf_f384f427-0b3-14`
    @ `wf/cuda-pw-pbe-device` (`e1147831`). Job 1035270 failed at link:
    the aarch64 link pulled x86_64 `libxtb.so` ("file in wrong format") into
    `test_cuda_pw_xc_pbe` — scrub the xtb lib path from the GH200 build env,
    resubmit.
- `cuda-md-first-regression` stayed **pending** (not deferred): its wave-2
  attempt is parked unmerged at branch `wf/cuda-md-first-regression`
  (`ea17e11a`, LJ argon NVE multi-device case; GH200 jobs 1035232 / 1035271
  failed). Next staffing should branch from / cherry-pick that tip.
- Remaining P0 is still `test-gpw-timeout-triage` (ctest slug, not docs-only).
  The 5-timeout evidence (`test_cpu_gpw_{mbd,hirshfeld,forces,stress}`,
  `test_mpi_pw_kpoint_n4`) came from a loaded login node; load-flake stays
  the default hypothesis until the serial re-run with raised `--timeout` says
  otherwise.
- GH200 batch discipline unchanged: job scripts start `#!/bin/bash -l`, load
  `GPU/buildenv-gcccuda/2026.03-cu13.0`, never pipe `module load`, keep the
  default aarch64 Rust toolchain, and a job not started within ~20 min means
  defer with the job id recorded.
- In-flight tree law still holds: the parent working tree carries the
  ~105 uncommitted variable-cell/stress files; never stage or touch them.
