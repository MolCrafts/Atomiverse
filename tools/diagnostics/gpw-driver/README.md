# GPW Driver and public-method checks

This opt-in suite executes complete methods outside CTest. The working queue
and completion rules live in [the DFT execution checklist](../../../docs/roadmap/dft.md#execution-checklist).

Configure the existing CPU or CUDA build with
`-DCMAKE_PROJECT_Atomiverse_INCLUDE=$repo/tools/diagnostics/gpw-driver/register.cmake`.
The hook also retains the preceding PW/PBE-family diagnostic targets.

| Target | Required physical envelope |
| --- | --- |
| `gpw-driver-cpu` | Six original OT outer paths (Γ RKS/UKS, multi-k RKS/UKS with/without +U), OT/DIIS energy agreement, geometry reuse, actual stage/criterion measurements, total standalone/nested budgets, warm zero-SCF mode, exceptional teardown, double-child energy/force/virial accumulation and parent-loop restoration. Default: 11 named cases. |
| `gpw-driver-cpu --kpoints-only` | Four multi-k paths. Run with one and two MPI ranks and compare the original and shifted-geometry energies. Multi-rank nested full-Frame execution remains outside the supported envelope. |
| `gpw-driver-preflight` | Same four fixtures and two cutoffs as the CUDA diagnostic. CPU convergence, nonzero full forces, ionic energy differences, independently reconverged xx strain differences, Frame units, exhaustion and failed-derivative publication. Eight named cases. |
| `gpw-driver-cuda` | Γ RKS, Γ UKS, RKS multigrid and multi-k RKS. Original CPU/device gates, solve-then-move reuse, actual newly attached stages/criteria, three parent request modes, two different-cutoff children, two geometries, live-bound move rejection, parent services and exception/retry checks. Four named cases. |
| `gpw-driver-uks-pbe` | Gamma UKS single-grid PBE/PBEsol/revPBE/RPBE, each H3 alpha2/beta1 and H2 alpha2/beta0. Eight cases preserve full E/F/diagonal-stress CPU gates, standalone run/forces/stress lifecycle, three nested request modes, two cutoffs, two geometries, borrowed stream, second-child failure cleanup and retry. `--case PBE-h3` selects one diagnostic case. |
| `gpw-driver-oracles` | Existing CP2K 2026.1 water OT energy pin at 250 Ry, plus OT/DIIS comparison, through public Kind/Frame calls. Two named cases. |

`fixtures.h` supplies the same physical input to CUDA and CPU preflight.
H₂ uses DZV-GTH-PADE/GTH-PADE-q1 in a `(12,11,10)`-Bohr cell. The two
cutoffs are 80 and 100 Ry. The multigrid case uses two levels and 20 Ry
relative cutoff; the UKS case has two alpha electrons and no beta electrons;
the k-point case uses an unfolded `2×1×1` mesh. RKS/MG use PBE; UKS/k use
Padé. CPU/device gates retain `5e-6` Ha, `5e-5` Ha/Bohr and
`5e-5` Ha/Bohr³. Independent CPU ionic FD uses `1e-4` Å and `1e-3` eV/Å;
the reconverged xx strain uses `.002` and `2e-4` eV/Å³.

The original water oracle settings, energy and tolerances are copied from
`regressions/gpw/orbital-transformation/ot-minimizer-water.cpp` and its
reference: `-17.212848291913666` Ha, `1.5e-3` Ha versus CP2K and `1e-4` Ha
OT/DIIS. The prior frozen CPU archive is run independently before the new
library. Diagnostic wall times are recorded without a matched-engine speedup
claim. No new CP2K run is supplied by this suite.

`collect.py` requires exact complete case sets, matched native XML/inventories,
unfiltered native unit scopes, stable source fingerprints, binary provenance
and the correct hardware. CPU includes explicit MPI comparisons; missing,
skipped, failed, duplicated or incomplete evidence fails the envelope.
Results and failed attempts are kept under `results/2026-10-27`.

Current [CPU](results/2026-10-27/cpu/validation.json),
[x86 CUDA build](results/2026-10-27/cuda-build/validation.json) and
[native GH200](results/2026-10-27/gh200/validation.json) envelopes pass.
The first CPU nested-UKS virial failures and first CUDA lifecycle-diagnostic
failure remain as separate attempts. Γ diagnostic SCF precision was tightened
to `tol=1e-11`, `residual_tol=1e-7`, retaining the original physical inputs,
production defaults and `1e-4` eV virial comparison. The CUDA correction changes
only the expected build/destroy count of the second, unexecuted child after
an injected parent failure; the production archive hash is unchanged.

The UKS PBE-family extension uses the same physical H3 fixture as the
[actual CP2K UKS oracle](../pbe-periodic-oracles/cp2k-uks-2026-10-28/reference.json),
and a separate empty-beta compatibility fixture. It sets both backends to
`tol=1e-10`, `residual_tol=1e-9`; H3 retains its original 100-step budget
and H2 its original 200-step budget. Public CUDA `residual_tol` defaults to
`1e-5`, matching its preceding hard-coded gate, and rejects nonpositive or
nonfinite settings before output publication. Gamma UKS now shares one
Pulay coefficient vector across both spin channels, matching CPU DIIS;
Gamma RKS's extrapolation algorithm is preserved. The retained independent-
spin DIIS failures, temporary 200-step H3 investigation and coupled-history
primitive checks are separate evidence, not changed physical gates.

`collect.py --profile gh200 --uks-pbe` additionally requires the exact eight
new complete-method cases. External engine comparisons are collected by
`pbe-periodic-oracles`, independently of the nested Driver receipt. Empty-beta
CPU/device compatibility does not qualify the point law's negative/floor
variational extension; see the [spin-domain contract](../../../docs/developer/xc-spin-domain.md).

Reusable CPU/GPU allocations are recorded in
`.molq/gpw-driver/allocations.json`. While they are running, commands attach
directly with no new batch submission:

```sh
tools/diagnostics/gpw-driver/run-allocation.sh cpu -- cmake --build build -j16
tools/diagnostics/gpw-driver/run-allocation.sh cpu --tasks 2 --cpus 1 -- \
  build/bin/gpw-driver-cpu --kpoints-only
tools/diagnostics/gpw-driver/run-allocation.sh cuda -- \
  build-gpu-job/bin/gpw-driver-cuda
```

The runner selects the matching toolchain and rejects an expired/pending
allocation. Current pools reserve 16 CPU cores and one GH200 with eight host
cores, respectively, for two hours. Each run uses its own native-output
directory; the allocation ID alone is not an attempt identifier.

CPU commands use Slurm's native nonoverlapping step allocation: `--exact`
with `--cpu-bind=threads`, without `--overlap`. `--cpus` requests cores per
MPI rank; the default remains one rank with 16 cores. Slurm allocates available
cores and queues requests until resources are free. The CUDA branch retains
its existing `--overlap` behavior. See the
[srun step-allocation contract](https://slurm.schedmd.com/srun.html#OPT_exact).
There is no extra stdin-consuming discovery process or application scheduler.
Module selection, one-thread OpenMP/BLAS defaults, input/output and child exit
status retain their original behavior. Native tool probes cover concurrent
single-rank, MPI2 and two-core steps; scientific MPI qualification remains
specific to each diagnostic's evidence.

CUDA GPW still has host setup and fixed-P diagonal stress work. Gamma RKS
single-grid AO Pulay and grid/core force assembly consume live device state;
UKS/multigrid/k force branches still use existing host assembly. Nested RKS
single-grid forces accumulate directly on the parent device frame; existing
host force branches upload at their publication boundary. These checks establish
their stated consistency envelope, without qualifying complete device
residency, general-cell shear or closed-form stress. Formal source/build
qualification remains false.
