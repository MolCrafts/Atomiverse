# Arrhenius and Mac workbench

Machine facts for GAFF / regression work. Open work:
[`../specs/gaff-m1-open-work.md`](../specs/gaff-m1-open-work.md).

## Arrhenius (NSC/NAISS)

- SSH alias `Arrhenius`, user `jicli594`. Login is keyboard-interactive: an
  agent cannot connect until the user opens the ControlMaster socket
  (`! ssh Arrhenius`). If the login node reports "not available", wait — do not
  retry in a loop.
- Slurm accounts: `naiss2026-1-53-cpu` (x86 EPYC-9005, 256 cores/node,
  `-p cpu`) and `naiss2026-1-53-gpu` (aarch64 GH200).
- Lmod needs explicit versions (`GPU/Miniforge/26.3.2-2-eb`,
  `Python/3.13.5-bare-gcc-2025b-eb`, `buildenv-gcc/2026.03-mpich`). `bash
  script.sh` is not a login shell: scripts calling `module` must
  `source /etc/profile` first (the `benchmarks/gaff/*.sh` scripts do).
- Login-node CPU affinity changes during a session; print
  `len(os.sched_getaffinity(0))` next to any login-node timing and prefer batch.
- Working checkout: `~/work/molcrafts/Atomiverse-gaff` (update with
  `git pull`); full product build (HF/PW/GPW/KSPACE/MPI) in `build-fullreg/`.
  Never touch `~/work/molcrafts/Atomiverse` there — someone else's tree.
- LAMMPS reference: `~/.local/x86_64/bin/lmp` (MOLECULE, EXTRA-MOLECULE,
  KSPACE, RIGID, OPENMP).
- Data root `/nobackup/proj/disk/teoroo/personal/jicli594/work/gaff-bench/`:
  - `base rep2 rep3 rep4` — system B at 24,840 / 198,720 / 670,680 / 1,589,760
    atoms (LAMMPS data + ff + prmtop);
  - `omm-env` — OpenMM 8.6.1 CUDA conda env (built on the GH200 node);
  - `atv-pyenv` — Python 3.13 + numpy, typer, molcrafts-molpy 0.18.x; pass
    `-DPython3_EXECUTABLE=.../atv-pyenv/bin/python` or `regressions/python/*`
    fail on missing deps;
  - `runs/` — job outputs; `runs/regdiff/{current,head}-<case>.log`.
- CP2K: see [gpw-perf-baseline.md](gpw-perf-baseline.md).

### Current-vs-HEAD regression diff

To tell a new red from a pre-existing one, build the same targets on the
working tree and on a detached `HEAD` worktree and run them serially in one
`-n 64` CPU job (`OMP_NUM_THREADS=8`, `timeout 1800` each). `ctest -j16` × 8
threads oversubscribes the node and produces false reds. Flags:
`-DATV_BUILD_CUDA=OFF -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON
-DATV_BUILD_GPW=ON -DATV_BUILD_KSPACE=ON -DATV_BUILD_REGRESSIONS=ON
-DATV_USE_MPI=ON -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=$OPENBLAS_ROOT`;
set `ATV_MOLPY_PATH` to the merged molpy checkout for both trees.

## Mac (M4) workbench — machine-local, may vanish on reboot

- ATV build `/tmp/atv-gaff-build` (MD, KSPACE, MOLRS, SEMIEMPIRICAL; HF/PW/GPW
  off; `Python3_EXECUTABLE=/tmp/atv-gaff-build/pyenv/bin/python`), configured by
  `/tmp/atv-configure.sh <dir>`. GPW-on trees: `/tmp/atv-architecture-pw`,
  `/tmp/atv-agent-gpw`, `/tmp/atv-agent-vib`.
- LAMMPS with the GAFF packages: `/tmp/atv-lammps-gaff-build/lmp` (source
  `/tmp/atv-lammps-neighbor-reference`).
- Conda env `molcrafts-tools`: AmberTools + OpenMM 8.6.1 (CPU); envs
  `xtb-oracle`, `tblite-oracle` for GFN pins.
- molpy venv `~/work/molcrafts/molpy/.venv` carries molrs built from
  `~/work/molcrafts/molrs`.
- OpenMPI on macOS: `OMPI_MCA_mpi_yield_when_idle=1`. One heavy job at a time.
- Mac timings vary ±30 % run to run; direction only.

## Git hooks

`pre-commit` stashes unstaged changes into `~/.cache/pre-commit/patch*` for the
commit and pre-push hooks and restores them only on a normal exit. A killed
`git commit` / `git push` leaves the tree stripped; recover with `git apply` of
the newest patch. The pre-push hook builds and runs the path-scoped unit gate
(many minutes).
