# Arrhenius and Mac workbench — what an agent needs to resume

Machine facts for the GAFF / regression work. Open work list:
[`../specs/gaff-m1-open-work.md`](../specs/gaff-m1-open-work.md).

## Arrhenius (NSC/NAISS)

- SSH host alias `Arrhenius`, user `jicli594`. Login is keyboard-interactive:
  an agent cannot connect until the user opens the ControlMaster socket
  (`! ssh Arrhenius` in the session). The login node is sometimes taken down
  ("This login node is currently not available") — wait, do not retry in a
  loop.
- Slurm accounts: `naiss2026-1-53-cpu` (x86 EPYC-9005, 256 cores/node,
  `-p cpu`) and `naiss2026-1-53-gpu` (aarch64 GH200).
- Lmod needs explicit versions (`GPU/Miniforge/26.3.2-2-eb`,
  `Python/3.13.5-bare-gcc-2025b-eb`, `buildenv-gcc/2026.03-mpich`).
  `bash script.sh` is not a login shell: scripts that call `module` must
  `source /etc/profile` first (the `benchmarks/gaff/*.sh` scripts do).
- Working checkout for this work: `~/work/molcrafts/Atomiverse-gaff`
  (rsynced from the Mac including `.git`; now that everything is pushed,
  `git pull` there instead). Do not touch `~/work/molcrafts/Atomiverse` on
  Arrhenius — it is someone else's tree.
  Full product build (HF/PW/GPW/KSPACE/MPI on): `build-fullreg/` in it.
- LAMMPS reference: `~/.local/x86_64/bin/lmp` ("30 Mar 2026 - Development",
  has MOLECULE, EXTRA-MOLECULE, KSPACE, RIGID, OPENMP).
- Data root `/nobackup/proj/disk/teoroo/personal/jicli594/work/gaff-bench/`:
  - `base rep2 rep3 rep4` — system B at 24,840 / 198,720 / 670,680 /
    1,589,760 atoms (LAMMPS data + ff + prmtop);
  - `omm-env` — OpenMM 8.6.1 CUDA conda env built on the GH200 node;
  - `atv-pyenv` — Python 3.13 with numpy, typer, molcrafts-molpy 0.18.x;
    pass it as `-DPython3_EXECUTABLE=.../atv-pyenv/bin/python` or the
    `regressions/python/*` cases fail on missing deps (they pass with it);
  - `runs/` — job outputs; `runs/regdiff/{current,head}-<case>.log` are the
    last current-vs-HEAD regression logs.
- Jobs at hand-off: GAFF CPU matrix `4125867` (submitted with pre-fix code;
  cancel and resubmit). Regdiff `4131285` finished.

### Current-vs-HEAD regression diff

To tell a new red from a pre-existing one, build the same targets on the
working tree and on a detached `HEAD` worktree and run them serially in one
`-n 64` CPU job (`OMP_NUM_THREADS=8`, `timeout 1800` each). Running the full
regression set at `ctest -j16` × 8 threads oversubscribes the node and
produced false reds (`gpw-dftu-kpoint-forces`). Flags:
`-DATV_BUILD_CUDA=OFF -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON
-DATV_BUILD_GPW=ON -DATV_BUILD_KSPACE=ON -DATV_BUILD_REGRESSIONS=ON
-DATV_USE_MPI=ON -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=$OPENBLAS_ROOT`;
copy `molrs/` into the worktree (submodule not initialised there).

## Mac (M4) workbench — machine-local, may be gone after a reboot

- ATV build `/tmp/atv-gaff-build` (MD, KSPACE, MOLRS, SEMIEMPIRICAL …;
  HF/PW/GPW off; `Python3_EXECUTABLE=/tmp/atv-gaff-build/pyenv/bin/python`),
  configured by `/tmp/atv-configure.sh <dir>`. GPW-on trees:
  `/tmp/atv-architecture-pw`, `/tmp/atv-agent-gpw`, `/tmp/atv-agent-vib`.
- LAMMPS a0e2bd6e with the GAFF packages: `/tmp/atv-lammps-gaff-build/lmp`
  (source `/tmp/atv-lammps-neighbor-reference`).
- Conda env `molcrafts-tools`: AmberTools + OpenMM 8.6.1 (CPU platform);
  envs `xtb-oracle`, `tblite-oracle` for GFN pins.
- molpy venv `~/work/molcrafts/molpy/.venv` carries molrs built from the
  standalone `~/work/molcrafts/molrs` checkout (prmtop fix included).
- Mac timings vary ±30 % run to run; direction only.

## Git hooks

`pre-commit` stashes unstaged changes into `~/.cache/pre-commit/patch*` for
both the commit and the pre-push hook and restores them only on a normal
exit. Killing a `git commit` / `git push` mid-hook leaves the working tree
stripped; recover with `git apply` of the newest patch file. The pre-push
hook builds and runs the path-scoped unit gate and takes many minutes.
