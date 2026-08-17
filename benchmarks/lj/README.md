# LJ 12-6 standing benchmark

Not registered in ctest. Build with `-DATV_BUILD_BENCHMARKS=ON`
(or the CUDA claim job, which builds `cuda_nonbonded_style_bench`).

## CUDA tile engine (GH200)

```
# on GH200 after a CUDA build
./cuda_nonbonded_style_bench 10000 80 15
```

Recorded GH200 job **1332540** (n68, `GPU/buildenv-gcccuda/2026.03-cu13.0`):

```
path=openmm_cluster n_side=22 N=10648 steps=80  ms/step=0.1098  Matom*step/s=96.997  energy=-39062.893592
```

Job **1331961** (same node class): `96.740 Matom*step/s`.

## LAMMPS twin

Deck: `tools/bench_lj_vs_lammps/in.lj_nve.kk` (same density / cut / dt).
A perf pin is the number `lmp` prints — never a reimplementation.

```
lmp -in tools/bench_lj_vs_lammps/in.lj_nve.kk -var N 10648 -var WARM 15 -var STEPS 80
```

Record binary version + the `Loop time` / `Performance` line next to the
ATV `Matom*step/s` line. Keep the roadmap ⚡ only when that same-node ratio
supports it.

## CPU xsimd

The login-node CPU build of `PairLJCut` is the xsimd path (`-march=native`).
A same-size CPU wall is optional supporting evidence; the ⚡ claim is the
CUDA tile engine vs LAMMPS Kokkos on GH200.
