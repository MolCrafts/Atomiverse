---
name: atomiverse-cluster
description: Sets up the cluster build environment for Atomiverse and runs CPU or CUDA builds and tests as batch jobs on the cluster and the aarch64 GH200 node. Use when building on the cluster, hitting libopenblas.so.0 or GLIBCXX errors, or running anything CUDA.
---

# atomiverse-cluster

Interactive cluster builds need Lmod and the right runtime paths; CUDA runs
only on the GH200 node. AGENTS.md `## Build and gates` (Cluster) owns the
rules, `docs/developer/project-contract.md` the full recipe.

## Steps

1. **Use a login shell.** Run interactive builds with `bash -lc '...'` so
   `module` is defined; never pipe `module load`.
2. **Load the CPU environment.** Inside that shell:

   ```bash
   module load buildenv-gcc/2026.03-mpich
   export OPENBLAS_ROOT="$OPENBLASDIR"
   export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
   export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
   ```

3. **Point at molpy.** Keep `ATV_MOLPY_PATH` (default `../molpy`) on one
   checkout for both `molcxx/` and the `molpy/` wheel.
4. **Send full builds to a batch job.** Use `tools/cpu_job.sh`, configured
   by `ATV_CMAKE_FLAGS`, `ATV_TARGETS` and `ATV_CTEST_RE` (knobs in its
   header); never write a new script per task.
5. **Run CUDA work on GH200 only.** Use `tools/gpu_job.sh` with the same
   knobs; it loads `GPU/buildenv-gcccuda/2026.03-cu13.0`. `molci run cuda`
   is the CUDA unit gate there.
6. **Submit and inspect through `molq`.** Use the public `molq` CLI (see
   `molq --help`) within the authorized budget; reuse a running job and
   never resubmit when unsure whether a submission landed.
7. **Time BLAS paths unbound.** Never set `OMP_PROC_BIND` / `OMP_PLACES`
   for SCF, GPW, plane-wave or DFTB SCC timing.

## Done when

- the build links and runs without `libopenblas.so.0` / `GLIBCXX` errors
- the job's ctest selection passed and its job reference is reported
- no duplicate job was submitted
