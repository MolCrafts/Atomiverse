# ES CPU + CUDA optimization queue

Worklist for workflow `es-cpu-cuda-opt`. **Two tracks** on one loop:

| Track | Goal | Gate |
| --- | --- | --- |
| **cpu** | Cut wall on landed GPW/PW CPU (fair twin vs real CP2K) | pins green; `T_atv/T_cp2k` not worse; FD-force if energy path |
| **cuda** | Implement / polish **corresponding** CUDA for landed ES | CPU twin green first; CUDA↔CPU energy/forces tol stated; device-only kernels |

**Not this queue:** new electronic methods (use `cp2k-method-loop` / `pyscf-method-loop`); classical MD CUDA (OpenMM track).

**Disk law:** wipe `build` / `build-*` after every cmake/ctest on every tree. No multi‑GB leftovers on nobackup.

**CUDA runtime:** login nodes have **no** CUDA toolkit. CUDA build/test only via batch on **GH200** (`sbatch -p gpu`, module `GPU/buildenv-gcccuda/2026.03-cu13.0`). CPU track stays on login with `buildenv-gcc/2026.03-mpich`.

Architecture / order: `CLAUDE.md` → Electronic structure optimization order;
`.claude/notes/es-optimization-order.md`; GPW walls `.claude/notes/gpw-perf-baseline.md`;
CUDA RHF / PW envelope `.claude/notes/cuda-rhf-device-only.md`.

## Fair twins (CPU)

| Twin | Role | Baseline ratio (authority) |
| --- | --- | ---: |
| `gpw-multigrid-kpoints` | multi-k MG products | **2.33×** (worst) |
| `gpw-kpoints-si` | multi-k dense χ^k | **1.18×** |
| `gpw-uks-o2-gamma` | Γ UKS DIIS | **1.077×** (≤1.2× closed) |
| `ot-minimizer-water` | OT iter + per-KS residual | OT ~**1.20×**; SCF body **≤1.0×** |

Wall numbers: [gpw-perf-baseline.md](../../.claude/notes/gpw-perf-baseline.md)
(arrhenius1, 1 MPI × OMP=4, CP2K 2026.1 quiet TIMING MAXIMUM). **Do not**
quote superseded ~1.25× MAIN figures from wave-7 claims.

## CUDA correctness twins

| Twin | Role |
| --- | --- |
| `tests/cuda/scf/test_rhf.cu` | device energy + s/p forces + level_shift invariant (H₂ pure-s + H₂O s/p) |
| H₂O STO-3G RHF | CPU oracle for CUDA energy/forces (same settings) |
| `regressions/pw/pw-cuda-lda-si` | CUDA↔CPU PW LSDA energy parity (Si; small cell, correctness only) |
| `regressions/pw/cuda-pw-forces` | CUDA↔CPU PW analytic ionic forces (Si dimer stretch; LSDA dual-gate) |
| `regressions/pw/cuda-pw-forces-pbe` | CUDA↔CPU PW analytic ionic forces under **PBE** (Si dimer stretch) |
| `regressions/pw/cuda-pw-stress` | CUDA↔CPU PW Nielsen–Martin cell stress (Si; LSDA dual-gate) |
| `regressions/pw/cuda-pw-stress-pbe` | CUDA↔CPU PW Nielsen–Martin cell stress under **PBE** (Si FCC) |
| `tests/cuda/planewave/test_pw_force_local.cu` | device local HF force kernel vs CPU fragment |
| `tests/cuda/planewave/test_pw_cell_stress.cu` | empty/open-shell reject + H₂ CUDA↔CPU stress parity |
| `tests/cuda/gpw/test_device_poisson.cu` | Γ Poisson/FFT foothold vs CPU oracle |
| `tests/cuda/gpw/test_xc_lda.cu` | Padé LDA XC grid foothold vs CPU oracle |
| `tests/cuda/gpw/test_product_collocate.cu` | device product ρ collocate foothold vs CPU collint |
| `tests/cuda/gpw/test_product_integrate.cu` | device product V·shape integrate foothold vs CPU collint |
| `tests/cuda/gpw/test_ks_chain.cu` + `reg_cuda-gpw-ks-chain` | composed ρ→V_eff stage (LDA) vs CPU `ks_grid_build_rks` |
| `tests/cuda/gpw/test_ks_gga.cu` + `reg_cuda-gpw-ks-gga` | composed ρ→V_eff **PBE-GGA** stage vs CPU `ks_grid_build_rks(PBE)` |
| `tests/cuda/gpw/test_ks_multigrid.cu` + `reg_cuda-gpw-ks-multigrid` | multi-grid product collocate/integrate KS stage vs CPU Γ MG fragment |
| `reg_cuda-pw-collinear-spin` | CUDA↔CPU collinear LSDA O₂ dual-gate (fixed occupation; no smear) |
| `reg_cuda-pw-spin-pbe` | CUDA↔CPU collinear **spin-PBE** O₂ dual-gate (fixed occupation; energy only) |

## pending

**GapCensus wave 7 on `dev` @ `d83de986` (post Assign wave 6 staff):**
Re-diff tree + baseline + notes + roadmap vs base (not worker tips). Fair
walls **held** (authority [gpw-perf-baseline.md](../../.claude/notes/gpw-perf-baseline.md)):
MG MAIN **2.33×** worst; OT water SCF body **≤1.0×** / OT-only ~**1.20×**;
UKS DIIS **1.077×** closed; k-si MAIN **1.18×** hold. **cpu_pending=0** —
pin-safe collint/structure/sphere **exhausted**; no promote from
[gpw-perf-queue.md](gpw-perf-queue.md). Public `cuda::gpw::GpwDft` first slice
**held** (feat `31d24207`; GH200 **1120817**). Device mid-SCF collocate/
integrate **on base** (merge `69296751` / feat `430a2268`; GH200 **1125709**).
Tree scan `src/cuda/gpw/gpw_solver.cu` **on base**: device product
collocate/KS/integrate via `KsChainDevice::run` (stream still null); host
`build_STVpp` / P→weight pack H2D / integrate-acc D2H + `scatter_product_acc`
/ `GpwDiis` / `diagonalize_fock_shifted`; free-fn forces **CPU-backed**
(`cpu::gpw::gpw_forces_from_scf`); **no** `cuda::gpw::gpw_stress`. Dual-gate
unit/reg still **PADE H₂ only** on base (PBE accepted, ungated). Fail-closed
**on base**: UKS / OT / multi-k / multigrid / MT / non-PADE-PBE /
`local_pp_on_grid=false`. PW reject matrix re-confirmed: spin∧PBE energy
green; forces/stress reject spin/smear; spin∧smear rejects; spin dual-gates
**Γ-only**; closed-shell force dual-gate **`k_mesh=1` only**
(`reg_cuda-pw-forces`) while `run()` accepts multi-k. **Three `[~]` P1 tips
exist** (not merged): `device-forces` tip GH200 **1125926**; `multigrid-run`
tip **1125894**; `uks-gamma` tip **1125905** — keep `## in_progress` until
merge; GapCensus does **not** claim base closed.

**Autonomous discovery residual (wave 7 — foothold ≠ closed):** Wave-6 joint
rows **held**. Tree×CPU-reg re-diff opens **deeper** residual not already
pending: (1) **Γ MG RKS stress** after `multigrid-run` + `stress` — forces
row alone leaves σ open (CPU `reg_gpw-analytic-stress` / stress-water class);
(2) **PBE stress dual-gate** after `pbe-product-gate` + `stress` (mirror
`pbe-forces-gate`); (3) **UKS×MG stress** after `uks-multigrid`; (4) **triple
UKS×MG×k F/σ** after `uks-multigrid-kpoints` energy (forces/stress rows were
pairwise-only); (5) **UKS-PBE F/σ** after `uks-pbe-gate` + UKS F/σ parents;
(6) **H₂O×PBE product dual-gate** after `product-h2o-gate` +
`pbe-product-gate`; (7) **MG×PBE** and **multi-k×PBE** product energy gates
after path parents + `pbe-product-gate` (PADE-only path gates leave GGA
ungated); (8) PW **closed-shell multi-k F/σ dual-gates** — force/stress APIs
take `k_mesh` but production dual-gates are Γ (`k_mesh=1`); (9) after
`spin-kmesh` → **spin-kmesh-PBE energy** + **spin-kmesh F/σ**. Host residual
post-collocate still: weight pack + D2H acc + DIIS/eigen →
`device-diis-eigen`; setup STVnl → `device-stvnl`. Do **not** invent
hybrid/GAPW/PAW/meta-GGA; OT/MT/vdW/metallic-GPW stay same-capability
follow-on after DIIS spine. **No gpw-perf-queue promote**. Staff `wf/<slug>`
off current `dev`.

| Fact | Evidence |
| --- | --- |
| Public-shape freeze | **green** merge `0147fe02` |
| First product | **landed** feat `31d24207` — GH200 **1120817** |
| Device mid-SCF collocate/integrate | **landed** merge `69296751` / feat `430a2268` — GH200 **1125709** |
| Product hybrid (base) | host STVnl/DIIS/eigen + weight pack; device product collocate/KS/integrate (stream null); forces free fn **CPU-backed** |
| Fail-closed (base) | UKS / OT / multi-k / multigrid / MT / non-PADE-PBE / `local_pp_on_grid=false` |
| Claimed-but-ungated | product `XcKind::PBE` accepted; only PADE H₂ dual-gate reg on base |
| Closed-shell PW stress | **LSDA + PBE** dual-gate **landed** (`cuda-pw-stress` + `cuda-pw-pbe-stress`) |
| Collinear spin-PBE energy | **landed** merge `118d3003` / GH200 **1124645** (Γ dual-gate only) |
| Staffed tips (not base) | wave 6 `[~]`: `device-forces` (**1125926**), `multigrid-run` (**1125894**), `uks-gamma` (**1125905**) |

**Counts:** **pending_count=51**; **in_progress=3**; **cpu_pending=0**;
**cuda_pending=51**; **gaps_remaining≈55** (36 GPW depth + 18 PW depth + MG
residual; staffed tips still open). Queue **not** exhausted.

### CUDA — GPW product depth (product on base — staff depth)

method=cuda throughout. Product is **on base** — staff depth into public
`GpwDft`. Do **not** open lone new kernel footholds — compose landed stage units.

**Compose / path lifts (held + re-ranked + wave-5 joint paths):**

`cuda-gpw-device-collocate` **landed** (merge `69296751` / GH200 **1125709**)
— mid-SCF device product collocate/integrate on base; residual host weight
pack + D2H acc + DIIS/eigen → `device-diis-eigen`.

- [ ] `cuda-gpw-pbe-product-gate` | **P1** Public product accepts `XcKind::PBE`
  but e2e dual-gate is **PADE-only** H₂ (unit + `reg_cuda-gpw-full-product`).
  Track: **cuda**. Twin: same-settings CPU Γ RKS PBE (H₂/water class, GTH-PBE).
  Gate: GH200 dual-gate energy (+ forces if free-fn path used); stated
  CUDA↔CPU tols; **perf_exempt** small cell. Lever: extend unit/reg under PBE;
  reuses landed `KsChainDevice` PBE stage (`cuda-gpw-ks-gga` green). Staff with
  or after collocate — not a new kernel foothold.
  *(wave 6 staff: held pending — parallel OK next wave after P1 trio)*
- [ ] `cuda-gpw-nondefault-stream` | **P2** Product constructs
  `KsChainDevice(..., /*stream=*/nullptr)`; mid-SCF H2D/D2H use default stream
  (collocate tip still stream-null). Track: **cuda**. Twin: same dual-gate
  energy after stream bind (PADE class). Gate: GH200 + tols unchanged;
  stream-scoped fences. Lever: own `cudaStream_t` on product / pass into
  `KsChainDevice` + device collocate when composed (mirror
  `cuda-pw-nondefault-stream`). Staff after or with `device-collocate`.
- [ ] `cuda-gpw-device-diis-eigen` | **P2** Mid-SCF host residual after device
  KS / collocate tip: every iter host packs `DeviceProduct` density weights
  (P→w H2D), D2H integrate-acc, host scatter → V_grid, then host
  `GpwDiis::step` + `diagonalize_fock_shifted` on dense AO (`gpw_solver.cu`).
  Track: **cuda**. Twin: same-settings CPU Γ RKS DIIS energy dual-gate. Gate:
  GH200 + stated energy tol; fail-closed paths unchanged. Lever: **device
  P→product-weight pack** + device-resident integrate acc → H (or device
  scatter) + device Pulay (RHF CDIIS pattern) + cuSOLVER dense eigen on
  orthonormal AO; staff after `device-collocate` merges.
- [ ] `cuda-gpw-product-h2o-gate` | **P2** Product dual-gate is **H₂ only**
  (unit + `reg_cuda-gpw-full-product`; collocate tip same H₂ class);
  production-class AO/grid depth missing (foothold ≠ closed). Track: **cuda**.
  Twin: CPU `GpwDft` water Γ RKS PADE (then PBE after `pbe-product-gate`).
  Gate: GH200 dual-gate E (+ F if free-fn); **perf_exempt** small molecule;
  stated tols. Lever: new/extended reg under water geometry + GTH-PADE; no
  new kernel.
- [ ] `cuda-gpw-kpoints` | **P2** Multi-k RKS DIIS product path (lift
  `k_mesh=={1,1,1}` reject). Track: **cuda**. Twin: CPU `reg_gpw-kpoints-si`
  class. Gate: GH200 dual-gate energy. Lever: phased STVnl/products host or
  device compose; not OT / not UKS multi-k in this slug. Forces are
  `kpoint-forces` after energy; joint MG multi-k is `multigrid-kpoints`.
- [ ] `cuda-gpw-stress` | **P2** Public cell-stress twin missing entirely on
  CUDA GPW (CPU `gpw_stress` / Frame evaluator product). Track: **cuda**.
  Twin: CPU Γ RKS fixed-\(P\) stress + dual-gate. Gate: GH200 + stated
  max‖Δσ‖; fail-closed multi-k/UKS/OT until owned. Lever: compose SCF
  `last_result` + device/host stress assembly under `cuda::gpw::gpw_stress`
  matching freeze optional depth.
- [ ] `cuda-gpw-uks-forces` | **P3** Analytic free-fn forces under Γ UKS
  product after `uks-gamma` energy (energy slug does not own forces; base
  free-fn is RKS CPU-backed only). Track: **cuda**. Twin: CPU
  `reg_gpw-uks-forces` / O₂ UKS forces + dual-gate. Gate: GH200 + stated
  max\|ΔF\| after UKS energy green. Lever: dual-spin density + force assembly
  on product force free-fn (compose with `device-forces` patterns).
- [ ] `cuda-gpw-multigrid-forces` | **P3** Analytic forces under product
  multigrid energy after `multigrid-run` (stage MG KS green; product forces
  still single-grid / CPU-backed). Track: **cuda**. Twin: CPU MG force
  fragment / water-class MG forces + dual-gate. Gate: GH200 after
  `multigrid-run`. Lever: multi-level veff/cbg force path on free-fn after MG
  `run` lands.
- [ ] `cuda-gpw-uks-pbe-gate` | **P3** Γ UKS product dual-gate under **PBE**
  after `uks-gamma` energy (PADE O₂ twin alone leaves spin-GGA product
  un-gated; CPU `reg_gpw-uks-pbe` is true UKS-PBE). Track: **cuda**. Twin:
  CPU Γ UKS PBE energy dual-gate. Gate: GH200 + stated CUDA↔CPU energy tol;
  **perf_exempt** small cell; fail-closed non-PBE UKS path if split. Lever:
  product UKS density + device spin/GGA KS path already used by PW spin-PBE
  + GPW `KsChainDevice` PBE; extend unit/reg only — no new foothold.
- [ ] `cuda-gpw-kpoint-forces` | **P3** Analytic free-fn forces under multi-k
  RKS product after `kpoints` energy (energy slug does not own forces; base
  free-fn is Γ RKS CPU-backed). Track: **cuda**. Twin: CPU
  `reg_gpw-kpoint-forces` class + dual-gate. Gate: GH200 + stated max\|ΔF\|
  after multi-k energy green. Lever: phased cbg/kbg force assembly on free-fn
  (compose `device-forces` patterns with multi-k SCF density).
- [ ] `cuda-gpw-uks-stress` | **P3** **new (wave 5)** — Cell stress under Γ
  UKS product after `uks-gamma` energy + `stress` RKS pattern (energy/force
  alone leave UKS σ open; CPU `reg_gpw-uks-stress`). Track: **cuda**. Twin:
  CPU Γ UKS fixed-\(P\) stress dual-gate. Gate: GH200 + stated max‖Δσ‖ after
  UKS energy. Lever: `cuda::gpw::gpw_stress` dual-spin payload after RKS
  stress lands.
- [ ] `cuda-gpw-kpoint-stress` | **P3** **new (wave 5)** — Cell stress under
  multi-k RKS product after `kpoints` + `stress` (CPU
  `reg_gpw-kpoint-stress`). Track: **cuda**. Twin: CPU multi-k RKS stress
  dual-gate. Gate: GH200 after multi-k energy. Lever: k-summed fixed-\(P_k\)
  stress assembly on free-fn / Frame path.
- [ ] `cuda-gpw-uks-kpoints` | **P3** **new (wave 5)** — Multi-k collinear UKS
  DIIS **energy** after `uks-gamma` + `kpoints` (each axis alone leaves joint
  open; CPU `reg_gpw-kpoints-uks`). Track: **cuda**. Twin: CPU multi-k UKS
  energy dual-gate. Gate: GH200 + stated energy tol; **perf_exempt** small
  cell. Lever: dual-spin phased STVnl/products compose on product `run`; not
  forces/stress in this slug.
- [ ] `cuda-gpw-uks-multigrid` | **P3** **new (wave 5)** — Γ UKS **multigrid**
  product energy after `uks-gamma` + `multigrid-run` (CPU
  `reg_gpw-uks-multigrid`). Track: **cuda**. Twin: CPU Γ UKS MG fragment +
  dual-gate. Gate: GH200 after both parents. Lever: dual-spin +
  `KsMultiGridChainDevice` compose on product.
- [ ] `cuda-gpw-multigrid-kpoints` | **P3** **new (wave 5)** — Multi-k **MG**
  RKS DIIS product energy after `multigrid-run` + `kpoints` (CPU worst-twin
  class `reg_gpw-multigrid-kpoints`; stage MG + multi-k energy alone leave
  joint open). Track: **cuda**. Twin: CPU multi-k MG energy dual-gate. Gate:
  GH200; **perf_exempt** small cell (not fair-wall claim). Lever: product
  compose multi-level products × k-mesh; not forces in this slug.
- [ ] `cuda-gpw-device-stvnl` | **P3** **new (wave 5)** — Host residual after
  collocate + diis-eigen spine: product still runs host `build_STVpp` +
  Löwdin orth once per `run` (`gpw_solver.cu`). Track: **cuda**. Twin:
  same-settings CPU Γ RKS energy dual-gate. Gate: GH200 after
  `device-diis-eigen`; stated energy tol. Lever: device Hermite / contracted
  S/T/Vnl + orth upload once (or device orth); compose not a lone foothold
  kernel campaign. Does **not** own mid-SCF collocate/DIIS.
- [ ] `cuda-gpw-pbe-forces-gate` | **P3** **new (wave 6)** — Free-fn analytic
  forces dual-gate under product **PBE** after `pbe-product-gate` energy +
  `device-forces` (PADE force dual-gate alone leaves GGA F product ungated;
  energy PBE gate does not own forces). Track: **cuda**. Twin: same-settings
  CPU Γ RKS PBE forces + dual-gate. Gate: GH200 + stated max\|ΔF\| after both
  parents; **perf_exempt** small cell. Lever: free-fn force path + PBE
  density/veff already on product; extend unit/reg under PBE — no new kernel
  foothold.
- [ ] `cuda-gpw-uks-multigrid-forces` | **P3** **new (wave 6)** — Analytic
  forces under Γ UKS **multigrid** product after `uks-multigrid` energy +
  `multigrid-forces` / `uks-forces` patterns (pairwise force rows leave the
  joint open). Track: **cuda**. Twin: CPU Γ UKS MG forces class + dual-gate.
  Gate: GH200 after UKS-MG energy. Lever: dual-spin multi-level force
  assembly on free-fn.
- [ ] `cuda-gpw-multigrid-kpoint-forces` | **P3** **new (wave 6)** — Analytic
  forces under multi-k **MG** RKS product after `multigrid-kpoints` energy +
  `kpoint-forces` (energy joint alone leaves F open). Track: **cuda**. Twin:
  CPU multi-k MG forces class + dual-gate. Gate: GH200 after multi-k MG
  energy. Lever: multi-level × k-summed force assembly on free-fn.
- [ ] `cuda-gpw-multigrid-kpoint-stress` | **P3** **new (wave 6)** — Cell
  stress under multi-k **MG** RKS product after `multigrid-kpoints` +
  `kpoint-stress` (or `stress` + multi-k MG energy). Track: **cuda**. Twin:
  CPU multi-k MG stress class + dual-gate. Gate: GH200 after multi-k MG
  energy. Lever: k-summed multi-level fixed-\(P_k\) stress on free-fn /
  Frame path.
- [ ] `cuda-gpw-uks-multigrid-kpoints` | **P3** **new (wave 6)** — **Triple**
  multi-k UKS **multigrid** DIIS **energy** after pairwise
  `uks-multigrid` / `multigrid-kpoints` / `uks-kpoints` (each pair alone
  leaves the three-way open; CPU `reg_gpw-uks-multigrid-kpoints`). Track:
  **cuda**. Twin: CPU multi-k UKS MG energy dual-gate. Gate: GH200 + stated
  energy tol; **perf_exempt** small cell. Lever: dual-spin × multi-level ×
  k-mesh product compose on `run`; not forces/stress in this slug.
- [ ] `cuda-gpw-kpoint-uks-forces` | **P3** **new (wave 6)** — Analytic free-fn
  forces under multi-k collinear UKS product after `uks-kpoints` energy +
  `kpoint-forces` / `uks-forces` patterns (CPU `reg_gpw-kpoint-uks-forces`).
  Track: **cuda**. Twin: CPU multi-k UKS forces dual-gate. Gate: GH200 after
  multi-k UKS energy. Lever: dual-spin phased force assembly on free-fn.
- [ ] `cuda-gpw-kpoint-uks-stress` | **P3** **new (wave 6)** — Cell stress
  under multi-k collinear UKS product after `uks-kpoints` + `kpoint-stress`
  / `uks-stress` (CPU `reg_gpw-kpoint-uks-stress`). Track: **cuda**. Twin:
  CPU multi-k UKS fixed-\(P_k\) stress dual-gate. Gate: GH200 after multi-k
  UKS energy. Lever: dual-spin k-summed stress assembly on free-fn / Frame.
- [ ] `cuda-gpw-multigrid-stress` | **P3** **new (wave 7)** — Cell stress
  under Γ **multigrid** RKS product after `multigrid-run` + `stress`
  (`multigrid-forces` alone leaves σ open; CPU stress-water / analytic-
  stress MG class). Track: **cuda**. Twin: CPU Γ MG RKS fixed-\(P\) stress
  dual-gate. Gate: GH200 after MG energy. Lever: multi-level fixed-\(P\)
  stress on free-fn / Frame after RKS stress lands.
- [ ] `cuda-gpw-pbe-stress-gate` | **P3** **new (wave 7)** — Free-fn cell
  stress dual-gate under product **PBE** after `pbe-product-gate` + `stress`
  (PADE stress gate alone leaves GGA σ ungated; mirror `pbe-forces-gate`).
  Track: **cuda**. Twin: same-settings CPU Γ RKS PBE stress + dual-gate.
  Gate: GH200 + stated max‖Δσ‖ after both parents; **perf_exempt** small
  cell. Lever: free-fn stress path + PBE density already on product;
  extend unit/reg under PBE — no new kernel.
- [ ] `cuda-gpw-uks-multigrid-stress` | **P3** **new (wave 7)** — Cell stress
  under Γ UKS **multigrid** after `uks-multigrid` energy +
  `uks-stress` / `multigrid-stress` patterns (UKS-MG forces alone leave σ
  open). Track: **cuda**. Twin: CPU Γ UKS MG stress class + dual-gate.
  Gate: GH200 after UKS-MG energy. Lever: dual-spin multi-level stress
  assembly on free-fn / Frame.
- [ ] `cuda-gpw-uks-multigrid-kpoint-forces` | **P3** **new (wave 7)** —
  Analytic forces under **triple** multi-k UKS MG product after
  `uks-multigrid-kpoints` energy (pairwise F rows leave three-way F open;
  CPU UKS-MG-k force class if present, else compose UKS-k + MG-k F). Track:
  **cuda**. Twin: CPU multi-k UKS MG forces dual-gate. Gate: GH200 after
  triple energy. Lever: dual-spin × multi-level × k-summed force assembly.
- [ ] `cuda-gpw-uks-multigrid-kpoint-stress` | **P3** **new (wave 7)** —
  Cell stress under **triple** multi-k UKS MG after `uks-multigrid-kpoints`
  + stress parents (same three-way gap as forces). Track: **cuda**. Twin:
  CPU multi-k UKS MG stress dual-gate. Gate: GH200 after triple energy.
  Lever: dual-spin × multi-level × k-summed fixed-\(P_k\) stress.
- [ ] `cuda-gpw-uks-pbe-forces` | **P3** **new (wave 7)** — Free-fn forces
  under Γ UKS **PBE** after `uks-pbe-gate` energy + `uks-forces` (PADE UKS
  F alone leaves spin-GGA F open). Track: **cuda**. Twin: CPU
  `reg_gpw-uks-pbe` / UKS-PBE forces class + dual-gate. Gate: GH200 after
  UKS-PBE energy. Lever: dual-spin PBE veff + force assembly on free-fn.
- [ ] `cuda-gpw-uks-pbe-stress` | **P3** **new (wave 7)** — Cell stress under
  Γ UKS **PBE** after `uks-pbe-gate` + `uks-stress` (same GGA joint gap).
  Track: **cuda**. Twin: CPU Γ UKS PBE stress dual-gate. Gate: GH200 after
  UKS-PBE energy. Lever: dual-spin PBE stress assembly on free-fn / Frame.
- [ ] `cuda-gpw-product-h2o-pbe-gate` | **P3** **new (wave 7)** — Water-class
  product dual-gate under **PBE** after `product-h2o-gate` +
  `pbe-product-gate` (H₂ PADE + H₂ PBE + water PADE alone leave production
  AO/grid × GGA ungated). Track: **cuda**. Twin: CPU water Γ RKS PBE
  dual-gate. Gate: GH200 E (+ F if free-fn); **perf_exempt**; stated tols.
  Lever: extend unit/reg only — no new kernel.
- [ ] `cuda-gpw-multigrid-pbe-gate` | **P3** **new (wave 7)** — Product MG
  energy dual-gate under **PBE** after `multigrid-run` + `pbe-product-gate`
  (MG stage/product gates are PADE-class today). Track: **cuda**. Twin: CPU
  Γ MG RKS PBE fragment + dual-gate. Gate: GH200 after both parents;
  **perf_exempt**. Lever: `KsMultiGridChainDevice` PBE compose + reg —
  no new foothold.
- [ ] `cuda-gpw-kpoint-pbe-gate` | **P3** **new (wave 7)** — Multi-k RKS
  product energy dual-gate under **PBE** after `kpoints` +
  `pbe-product-gate` (multi-k energy gate alone may stay PADE). Track:
  **cuda**. Twin: CPU multi-k RKS PBE class + dual-gate. Gate: GH200 after
  both parents; **perf_exempt**. Lever: product multi-k path + device PBE
  KS; extend unit/reg.

### CUDA — PW depth (landed surface incomplete) — parallel staff OK

Reject matrix re-confirmed on `dev` @ `d83de986` (method=cuda throughout):

**Closed-shell axis** (energy/forces/PBE-stress landings green; smear depth open):

- [ ] `cuda-pw-smear-forces` | **P2** Lift restricted `smearing != none` reject on
  `pw_forces_analytic` (energy smear landed; forces still throw). Track:
  **cuda**. Twin: CPU forces under metallic occ + dual-gate. Gate: GH200.
- [ ] `cuda-pw-smear-stress` | **P2** Lift restricted smear reject on
  `pw_stress`. Track: **cuda**. Twin: CPU `pw_stress` + dual-gate. Gate: GH200.

**Collinear spin axis** (LSDA + **spin-PBE energy** landed; derivatives open):

- [ ] `cuda-pw-spin-forces` | **P2** Analytic ionic forces under collinear **LSDA**
  spin (currently forces reject spin). Track: **cuda**. Twin: CPU
  `pw_forces_analytic` + dual-gate. Gate: GH200; fail-closed non-spin path
  unchanged. Spin-PBE forces are a separate deeper row.
- [ ] `cuda-pw-spin-pbe-forces` | **P3** Analytic ionic forces under collinear
  **spin-PBE** (energy parent **landed** `cuda-pw-spin-pbe`; LSDA spin-forces
  alone leaves GGA force depth open). Track: **cuda**. Twin: CPU spin-PBE
  forces + dual-gate. Gate: GH200. Lever: `pw_forces_analytic` spin path +
  device PBE spin XC reuse.
- [ ] `cuda-pw-spin-stress` | **P2** Cell stress under collinear LSDA spin
  (currently stress reject spin). Track: **cuda**. Twin: CPU `pw_stress`.
  Gate: GH200.
- [ ] `cuda-pw-spin-pbe-stress` | **P3** Cell stress under collinear
  **spin-PBE** (energy parent **landed**; base stress still rejects all spin;
  LSDA spin-stress alone leaves GGA stress depth open). Track: **cuda**. Twin:
  CPU spin-PBE `pw_stress` + dual-gate. Gate: GH200 (prefer after
  `spin-stress` LSDA pattern). Lever: `pw_stress` spin path + device PBE spin
  XC on NSCF strain-FD density remap.
- [ ] `cuda-pw-spin-smear` | **P2** Collinear spin + metallic smear free energy
  (currently spin∧smear throws). Track: **cuda**. Twin: CPU
  `reg_pw-uks-fe-smearing` class. Gate: GH200; optional after spin-PBE /
  energy spin path stable. Forces/stress are separate deeper rows; spin∧smear∧PBE
  is `spin-smear-pbe`.
- [ ] `cuda-pw-spin-smear-forces` | **P3** Analytic ionic forces under collinear
  spin **+** metallic smear after `spin-smear` energy (restricted smear forces
  and LSDA spin forces alone leave the joint path open). Track: **cuda**. Twin:
  CPU UKS+smear forces class + dual-gate. Gate: GH200 after `spin-smear`
  energy. Lever: `pw_forces_analytic` joint gate lift + fractional occ force
  assembly.
- [ ] `cuda-pw-spin-smear-stress` | **P3** Cell stress under collinear spin +
  metallic smear after `spin-smear` energy (same joint-path gap as forces).
  Track: **cuda**. Twin: CPU UKS+smear `pw_stress` + dual-gate. Gate: GH200
  after `spin-smear` energy (prefer after LSDA spin-stress / restricted
  smear-stress patterns). Lever: `pw_stress` joint gate + Mermin free-energy
  stress terms.
- [ ] `cuda-pw-spin-smear-pbe` | **P3** **new (wave 5)** — Collinear spin +
  metallic smear free energy under **PBE** after `spin-smear` (first joint
  energy may be LSDA-only; CPU `reg_pw-uks-fe-smearing` / spin-GGA class leaves
  spin∧smear∧PBE open on device). Track: **cuda**. Twin: CPU UKS+smear PBE
  free energy dual-gate. Gate: GH200 after `spin-smear` energy; stated tol;
  **perf_exempt** small cell. Lever: `pw_solver` joint gate +
  `DeviceDensity` spin-PBE XC already landed for fixed-occ.
- [ ] `cuda-pw-spin-kmesh` | **P2** **new (wave 6)** — Multi-k collinear
  fixed-occ UKS **energy dual-gate** (device `run()` already accepts
  `k_mesh`×`spin_polarized` for LSDA/PBE; production dual-gates are **Γ-only**
  — foothold path ≠ gated capability; CPU `reg_pw-uks-kmesh-fe`). Track:
  **cuda**. Twin: CPU multi-k collinear LSDA (then optional spin-PBE) energy
  dual-gate. Gate: GH200 + stated energy tol; **perf_exempt** small cell;
  not forces/stress/smear. Lever: unit/reg under `k_mesh>1` + spin; no new
  XC kernel.
- [ ] `cuda-pw-spin-smear-pbe-forces` | **P3** **new (wave 6)** — Analytic
  ionic forces under collinear spin + metallic smear + **PBE** after
  `spin-smear-pbe` energy (LSDA spin-smear forces alone leave GGA joint F
  open). Track: **cuda**. Twin: CPU UKS+smear PBE forces + dual-gate. Gate:
  GH200 after `spin-smear-pbe` energy. Lever: `pw_forces_analytic` triple
  gate lift + device spin-PBE XC reuse.
- [ ] `cuda-pw-spin-smear-pbe-stress` | **P3** **new (wave 6)** — Cell stress
  under collinear spin + metallic smear + **PBE** after `spin-smear-pbe`
  energy (same triple joint gap as forces). Track: **cuda**. Twin: CPU
  UKS+smear PBE `pw_stress` + dual-gate. Gate: GH200 after `spin-smear-pbe`
  energy. Lever: `pw_stress` triple gate + Mermin free-energy stress terms +
  spin-PBE XC on NSCF strain-FD.
- [ ] `cuda-pw-kmesh-forces` | **P2** **new (wave 7)** — Closed-shell
  multi-k analytic ionic **forces dual-gate** (`reg_cuda-pw-forces` is
  **`k_mesh=1` only** while API accepts `k_mesh>1` — foothold path ≠ gated
  multi-k F). Track: **cuda**. Twin: CPU multi-k closed-shell forces +
  dual-gate (Si class). Gate: GH200 + stated max\|ΔF\|; not spin/smear.
  Lever: unit/reg under `k_mesh>1` + existing `pw_forces_analytic` closed-
  shell path; no new kernel.
- [ ] `cuda-pw-kmesh-stress` | **P2** **new (wave 7)** — Closed-shell multi-k
  cell **stress dual-gate** (production stress regs are small-cell /
  effectively Γ-class coverage vs multi-k energy `pw-cuda-lda-si` k=2).
  Track: **cuda**. Twin: CPU multi-k closed-shell `pw_stress` + dual-gate.
  Gate: GH200 + stated max‖Δσ‖; not spin/smear. Lever: unit/reg under
  `k_mesh>1` + existing stress path.
- [ ] `cuda-pw-spin-kmesh-pbe` | **P3** **new (wave 7)** — Multi-k collinear
  **spin-PBE** energy dual-gate after `spin-kmesh` (LSDA multi-k spin) +
  `spin-pbe` (Γ spin-PBE) — joint open (CPU `reg_pw-uks-kmesh-fe` / spin-GGA
  class). Track: **cuda**. Twin: CPU multi-k spin-PBE energy dual-gate.
  Gate: GH200 after `spin-kmesh`; **perf_exempt**; not F/σ. Lever:
  `run()` joint path + device spin-PBE XC already landed; unit/reg only.
- [ ] `cuda-pw-spin-kmesh-forces` | **P3** **new (wave 7)** — Analytic forces
  under multi-k collinear spin after `spin-kmesh` + `spin-forces` (Γ spin F
  + multi-k energy alone leave joint F open). Track: **cuda**. Twin: CPU
  multi-k UKS forces class + dual-gate. Gate: GH200 after both parents.
  Lever: `pw_forces_analytic` joint gate lift + multi-k spin density.
- [ ] `cuda-pw-spin-kmesh-stress` | **P3** **new (wave 7)** — Cell stress
  under multi-k collinear spin after `spin-kmesh` + `spin-stress` (same
  joint gap as forces). Track: **cuda**. Twin: CPU multi-k UKS stress +
  dual-gate. Gate: GH200 after both parents. Lever: `pw_stress` joint gate
  + multi-k spin density / NSCF strain-FD.

### Fair-wall rank (1 MPI × OMP=4; worst residual first)

| Twin | ATV | CP2K twin | ratio | lever class |
| --- | ---: | ---: | ---: | --- |
| `gpw-multigrid-kpoints` MAIN | **1.6 s** | **0.688 s** quiet | **2.33×** | **worst_ratio** — coll/int ~2.0×/2.8× per-call; pin-safe kernel + structure + sphere **exhausted** |
| `ot-uks-o2` OT | **~11.8 s** / 27 it (post OUTER polish) | **8.68 s** / 30 SCF | **~1.4×** | ≤3× aim **met**; not primary vs MG |
| `ot-minimizer-water` OT-only | **~2.99 s** (10 it) | **2.50 s** | **~1.20×** | control + per-KS residual closed (SCF body **≤1.0×**) |
| `gpw-uks-o2-gamma` | **14.75 s** | **13.691 s** | **1.077×** | ≤1.2× **closed** — do not re-open dual-spin fuse |

Secondary (not in primary four; still baseline): `gpw-kpoints-si` MAIN **1.18×**
— hold products-vs-dense A/B until MG nearer parity.

MG budget (ngrid=37800, ~13 SCF): setup STVnl+products ~0.42 s once +
~13×(coll 30 + int 48 + XC/Poisson ~5) ≈ **1.5 s** → matches 1.6 s pin. CP2K
coll/int alone ~13×(14.5+17.1) ≈ **0.41 s**. Closing MAIN is **blocked**
`mg-main-structural` (non-pin-safe windows or task-list rewrite) — not more
collint micro-opts (`cpu-collint-int-residual-2` wave-8 **no-land**). Arch
campaign landed single-owner multi-level path; **fair-wall ~1× still residual**.

### CPU — wall vs CP2K (contribution to worst ratio)

**cpu_pending = 0.** Honest staffable set empty: worst twin has no pin-safe
lever left. Authority walls unchanged vs baseline
([gpw-perf-baseline.md](../../.claude/notes/gpw-perf-baseline.md)). Pure-s +
mixed-angular 2-center SIMD **landed** (slab-class; invisible on four fair
twins). No further CPU polish staffed here without a pin-safe wall lever.
**No promote** from [gpw-perf-queue.md](gpw-perf-queue.md) P1/P2 holds.

*Not pending (residual mass still counted in `gaps_remaining`, not staffable):*

| Row | Why not pending |
| --- | --- |
| `cpu-collint-int-residual-2` | wave-8 **no-land** — fused segment accumulate noise-flat; CP2K-style per-chunk hsum **regressed** |
| MG MAIN → ~1× (`mg-main-structural`) | **blocked** — pin-safe collint/structure/sphere campaigns closed; arch single-owner path landed; needs non-pin-safe windows or task-list rewrite beyond pin law |
| `p1-products-default-k` | stay on gpw-perf-queue; A/B only after MG nearer parity + products ≤ dense |
| `p2-mpi-product-grid` | multi-rank; not 1-MPI fair-wall lever |
| OT / UKS further polish | water OT ~1.20× / SCF body ≤1.0×; UKS DIIS 1.077× closed; OT UKS ≤3× met |

*Already landed this loop (CPU):* `cpu-xc-spectral-fft`, `cpu-ot-diis-outer`,
`cpu-ot-residual`, `cpu-stvnl-residual`, `cpu-collint-residual`,
`cpu-products-mg-setup`, `cpu-collint-shell-tasks`, `cpu-collint-integrate-residual`,
`cpu-simd-2center-setup`, `cpu-simd-2center-angular`
(+ gpw-perf OT match / sphere / UKS fuse / STVnl setup — walls in baseline).

### CUDA — corresponding ES (open-gate readiness × user value)

**Tree fact on `dev` @ `d83de986`:** stage units + PW landings + first public
`cuda::gpw::GpwDft` + **device mid-SCF product collocate/integrate** +
closed-shell **PBE stress** + **collinear spin-PBE energy** dual-gates
confirmed **on base**. `gpw_solver.cu` (base): device `KsChainDevice::run`
collocate/KS/integrate (stream-null); host weight pack + integrate-acc D2H +
DIIS/eigen + `build_STVpp`; free-fn forces **CPU-backed**. `pw_solver` accepts
spin∧PBE energy (+ `k_mesh`); forces/stress still reject spin/smear;
spin∧smear still rejects; spin dual-gates Γ-only; closed-shell F dual-gate
**Γ (`k_mesh=1`)**. **Three open worker tips** (not base): `device-forces` /
`multigrid-run` / `uks-gamma` with GH200 tip evidence (see `## in_progress`).

| Slice on `dev` | Status | Depth gap |
| --- | --- | --- |
| RHF s/p energy + forces + level_shift + path-select + device CDIIS + **non-default stream D2H** | **landed** | higher-L host-fill remains (out of product scope) |
| Γ GPW stage units (Poisson/FFT + Padé LDA + product collocate/integrate + `KsChainDevice` LDA/PBE + `KsMultiGridChainDevice`) | **landed** | stage ≠ full product depth |
| Public `cuda::gpw::GpwDft` first slice + **device mid-SCF collocate/integrate** (Γ RKS DIIS PADE/PBE; host STVnl/DIIS/eigen; device product KS chain; CPU force free fn) | **landed** (`31d24207` / **1120817** + collocate `69296751` / **1125709**) | **pending:** **PBE + H₂O (+ H₂O×PBE) product dual-gates**, device forces (+ **PBE F/σ gates**), multigrid run + **MG F/σ** + **MG×PBE**, UKS Γ + **UKS F/σ** + **UKS-PBE (+ F/σ)**, multi-k + **kpoint F/σ** + **k×PBE**, **UKS multi-k** + **kpoint-UKS F/σ**, **UKS MG** + **UKS-MG F/σ**, **multi-k MG** + **MG-k F/σ** + **UKS-MG-k triple (+ F/σ)**, stress, **product stream**, **device DIIS/eigen** (+ P→weight), **device STVnl** residual |
| `cuda::pw::PW` LSDA + device PBE-GGA + residency + stream + large-cell + **forces (LSDA+PBE Γ)** + **cell stress (LSDA + PBE dual-gate)** + **collinear LSDA + spin-PBE energy** + **restricted metallic smear energy** | **landed** | **pending:** **closed-shell multi-k F/σ dual-gates**; smear F/σ; spin F/σ / **spin-PBE F/σ** / **spin multi-k dual-gate** (+ **spin-kmesh-PBE** + **spin-kmesh F/σ**) / spin+smear energy + **spin-smear F/σ** + **spin-smear-pbe** + **spin-smear-pbe F/σ** |

**Staff order:** (1) finish/merge wave-6 `[~]` P1 tips (`device-forces` /
`multigrid-run` / `uks-gamma`) then residual P1 `pbe-product-gate`; (2)
`nondefault-stream` + `device-diis-eigen` + `product-h2o-gate` + `kpoints` +
`stress` + PW **P2** (`smear-forces` / `smear-stress` / `spin-forces` /
`spin-stress` / `spin-smear` / **`spin-kmesh`** / **`kmesh-forces`** /
**`kmesh-stress`**); (3) wave-4/5/6/7 joint paths (UKS/MG/k F+σ + multi-k
MG/UKS + triple UKS-MG-k + F/σ + PBE path gates + UKS-PBE F/σ +
product-h2o-pbe + device-stvnl + spin-pbe/smear F/σ + spin-smear-pbe F/σ +
spin-kmesh-PBE/F/σ) **P3** after parent energy rows. Do **not** staff lone
new GPW kernel footholds. Higher-L RHF stays product-out-of-scope. OT /
vdW / MT / metallic-GPW stay later same-capability follow-on.

Open gate: CPU twin green + public `run`/`forces`/`stress` stable where owned
+ stated CUDA↔CPU tols + device-only kernels + **GH200 log evidence**.

**Next Assign (post-wave-7):** after Merge lands the three `[~]` P1 tips →
residual P1 `pbe-product-gate` + P2 stack (`nondefault-stream` /
`device-diis-eigen` / `product-h2o-gate` / `kpoints` / `stress` + PW
`smear-forces` / `smear-stress` / `spin-forces` / `spin-stress` /
`spin-smear` / `spin-kmesh` / `kmesh-forces` / `kmesh-stress`); wave-7 joint
rows stay pending until parents land. Do **not** staff new while capacity is
full with unmerged green tips.

*Not pending this loop:*

| Row | Why |
| --- | --- |
| CUDA d-shell / higher-L RHF product | out of loop product scope (host-fill remains for non-s/p STO-3G) |
| Further lone GPW kernel footholds | collocate↔integrate + LDA + GGA + MG chain complete; depth is product compose — not another lone foothold |
| `cuda-pw-pbe-large-cell` | large-cell LSDA ladder **closed**; optional PBE remeasure not a correctness gap |
| Classical MD CUDA | OpenMM track — not this loop |
| hybrid/GAPW/PAW/meta-GGA | method loops |
| CUDA GPW OT / MT / vdW first cut | same-capability follow-on after DIIS product spine depth; not open-gate P0/P1 this wave |

*Already landed this loop (CUDA):* `cuda-rhf-pshell-forces`,
`cuda-rhf-device-only-polish`, `cuda-rhf-level-shift`, `cuda-rhf-path-select-host`,
`cuda-rhf-cdiis-device`, `cuda-rhf-nondefault-stream`, `cuda-gpw-foothold`,
`cuda-gpw-xc-lda-foothold`, `cuda-gpw-rho-collocate-foothold`,
`cuda-gpw-product-integrate-foothold`, `cuda-gpw-ks-chain`, `cuda-gpw-ks-gga`,
`cuda-gpw-ks-multigrid`, `cuda-pw-method-slice` (+ `cuda-pw-kpoint-hoist`,
`cuda-pw-zheevd`, `cuda-pw-devptr-nrm2` residency polish on `dev`),
`cuda-pw-pbe-device`, `cuda-pw-large-cell-measure`, `cuda-pw-nondefault-stream`,
`cuda-pw-forces`, `cuda-pw-forces-pbe`, `cuda-pw-stress`,
`cuda-pw-collinear-spin`, `cuda-pw-metallic-smear`, **`cuda-gpw-full-product`**,
**`cuda-pw-pbe-stress`**, **`cuda-pw-spin-pbe`**,
**`cuda-gpw-device-collocate`**.

## in_progress

**Assign wave 7** (parent `dev` @ `9f5a76ed`; GapCensus wave 7 already
refilled pending): priority **[~] first** — capacity **3/3** filled by wave-6
P1 tips still **not** on base. **No new staff** this wave (held residual P1
`cuda-gpw-pbe-product-gate` + P2 stack + wave-7 joint rows stay pending until
parents merge). **in_progress=3**; **new_assigned=0**;
**incomplete_remaining=0** (tips claim GH200 green — **Merge owns land**).
Queue **not** exhausted.

- [~] `cuda-gpw-device-forces` | **P1** Free-function forces call **CPU**
  `cpu::gpw::gpw_forces_from_scf` on **base** (no device force product).
  Track: **cuda**. Twin: same-settings CPU analytic forces + dual-gate vs
  CUDA SCF density. Gate: GH200 + stated max\|ΔF\| ≤ **5e-5** Ha/Bohr,
  \|ΔE\| ≤ **5e-6** Ha; keep free-function public shape. Lever: `gpw_forces_*`
  in `gpw_solver.cu` + device grid HF / Pulay after landed `device-collocate`.
  Tip: `wf/cuda-gpw-device-forces` (`5fa33be9` + `0f70ad52`); GH200
  **1125926** acceptance_met tip claim (max\|ΔF\|=**2.360e-6** Ha/Bohr) —
  not on base. Spec: `.claude/specs/cuda-gpw-device-forces/` (tip only).
  Resume: `is_resume=true` from tip `0f70ad52`.
- [~] `cuda-gpw-multigrid-run` | **P1** Lift `ngrids!=1` reject on public
  `GpwDft::run` — wire landed `KsMultiGridChainDevice` into product energy
  (stage unit green; **base** still single-grid). Track: **cuda**. Twin: CPU
  Γ MG fragment / water-class MG. Gate: GH200 dual-gate energy (tol class
  **5e-6** Ha). Lever: `validate_slice_` + product `run` compose
  `KsMultiGridChainDevice`. Tip: `wf/cuda-gpw-multigrid-run` (`b9236f17`);
  GH200 **1125894** \|dE\|=**1.587e-7** Ha tip claim — not on base. Spec:
  `.claude/specs/cuda-gpw-multigrid-run/` (tip only). Resume:
  `is_resume=true` from tip `b9236f17`.
- [~] `cuda-gpw-uks-gamma` | **P1** Lift collinear UKS reject
  (`n_alpha`/`n_beta`) on public product — Γ UKS DIIS **energy** (**base**
  still RKS-only). Track: **cuda**. Twin: CPU `reg_gpw-uks-o2-gamma` / O₂ /
  high-spin H₂ UKS PADE class. Gate: GH200 + \|ΔE\| ≤ **5e-6** Ha. Lever:
  dual-spin density + `KsChainDevice::run_uks`. Tip: `wf/cuda-gpw-uks-gamma`
  (`6deea27e`); GH200 **1125905** \|ΔE\|=**4.760e-6** Ha tip claim — not on
  base. Spec: `.claude/specs/cuda-gpw-uks-gamma/` (tip only). Resume:
  `is_resume=true` from tip `6deea27e`.

## blocked

- [!] `global-xtensor-xsimd` | never enable global `XTENSOR_USE_XSIMD`
- [!] `mg-main-structural` | MG MAIN **2.33×** residual after pin-safe collint + structure + sphere campaigns; further cut needs non-pin-safe windows or task-list rewrite beyond this loop’s pin law — do not staff as soft-tol or self-golden
  (arch-campaign landed single-owner multi-level path `arch-mg-main-structural`;
  fair-wall ~1× is still a **perf** residual, not a soft-tol staff)

## done

Harness wave 5 land 2026-08-10 (`dev` @ `69296751`):

- [x] `cuda-gpw-device-collocate` | **P1** Compose landed `product_collocate` /
  `product_integrate` into public `cuda::gpw::GpwDft` mid-SCF
  (`build_H_and_energy` via `KsChainDevice::run`); drop host cbg
  collocate/integrate + mid-SCF H2D ρ / D2H V_eff for the H path | track
  cuda | twin: same-settings CPU `GpwDft` H₂ Γ RKS PADE dual-gate +
  `reg_cuda-gpw-device-collocate` | files: `gpw_solver.{cuh,cu}`, reg, job,
  REFERENCE, public-shape note | merge: `69296751` / feat `430a2268` |
  evidence: GH200 **1125709** \|ΔE\|=**1.617e-7** Ha max\|ΔF\|=**2.137e-6**
  Ha/Bohr (tol 5e-6 / 5e-5); units 3/3 PASS; prior **1124643** / **1123960**
  / **1122743**; **perf_exempt** small cell; residual host weight pack + D2H
  acc + host DIIS/eigen → `device-diis-eigen` |

Harness wave 4 land 2026-08-10 (`dev` @ `118d3003`):

- [x] `cuda-pw-spin-pbe` | **P1** Lift collinear `spin_polarized` ∧ `XCType::PBE`
  reject on `cuda::pw::PW::run` — device true spin-GGA energy
  (`xc_pbe_spin_grid` + spectral ∇ρ_σ in `DeviceDensity`) | track cuda | twin:
  unit `test_cuda_pw_xc_pbe_spin` + `reg_cuda-pw-spin-pbe` O₂ triplet
  ONCV-PBE | files: `xc_pbe.{cu,cuh}`, `device_density.{cu,cuh}`,
  `pw_solver.{cu,cuh}`, unit, reg, job | merge: `118d3003` / feat `7949e29d` |
  evidence: GH200 **1124645** \|ΔE\|=**1.14e-13** Ha (tol 2e-3); E both
  **−28.7585210635** Ha mag=2; \|E_PBE−E_LSDA\|_cpu=**0.129** Ha; prior
  **1123961** / **1122874**; **perf_exempt** small cell; not forces/stress/
  spin-smear |

Harness wave 3 land 2026-08-10 (`dev` @ `18663d13`):

- [x] `cuda-pw-pbe-stress` | **P1** Dual-gate closed-shell **PBE** Nielsen–Martin
  cell stress — `pw_stress` path already took `XCType`; product dual-gate pin
  under PBE (LSDA stress already landed) | track cuda | twin: unit
  `test_pw_cell_stress` H₂ PBE + `reg_cuda-pw-stress-pbe` Si FCC GTH-PBE-q4 |
  files: `pw_stress.cuh`, unit, reg, job, REFERENCE | merge: `18663d13` / feat
  `b0523de0` | evidence: GH200 re-verify **1123959** max‖Δσ‖=**5.585e-12**
  eV/Å³ (tol 5e-2); E both **−13.90403280** Ry; prior **1122705** / **1122143**;
  **perf_exempt** small cell; not spin/smear stress |

Post-loop land 2026-08-10 (`dev` @ `fb2c2951`):

- [x] `cuda-gpw-full-product` | **P0** First public `cuda::gpw::GpwDft` — Γ RKS
  DIIS PADE/PBE, single-grid periodic; host STVnl/cbg/DIIS + device
  `KsChainDevice::run_from_rho`; free-function forces (CPU-backed first slice);
  fail-closed UKS/OT/multi-k/MG/MT | track cuda | twin: H₂ CPU dual-gate |
  files: `gpw_solver.{cuh,cu}`, unit, reg, job | merge: **FF** tip
  `wf/cuda-gpw-full-product` → `fb2c2951` / feat `31d24207` | evidence: GH200
  **1120817** \|ΔE\|=**4.796e-14** Ha max\|ΔF\|=**2.643e-09** Ha/Bohr (prior
  **1120312**); tols 5e-6 / 5e-5; **perf_exempt** small cell; depth residual
  (device collocate/forces, MG run, UKS, k-points, stress) stays pending |

Harness wave 7 land 2026-08-10 (`dev` @ `018dca0d`):

- [x] `cuda-pw-metallic-smear` | **P2** Lift restricted `smearing != none` — Fermi–Dirac / Gaussian / Methfessel–Paxton occupations + `extra_bands` on device band path; multi-k device Ritz cache; Mermin free energy; spin+smear / forces / stress still reject | track cuda | twin: unit guards + `reg_cuda-pw-metallic-smear` Al FD dual-gate | files: `device_solver.{cu,cuh}`, `pw_solver.{cu,cuh}`, reg, guards | merge: `d2ccc627` / feat `2891e824` | evidence: GH200 **1051119** ctest 9/9; dual-gate E both **−7.5496489573 Ry** n_occ=3 \|ΔE\|=**1.24e-14** Ry (tol 5e-3); **perf_exempt** small cell; not spin+smear / forces / stress |
- [x] `cuda-rhf-nondefault-stream` | **P3** Residual/V_nuc D2H on Evaluate non-default stream (not default stream); host force H2D fallback stream-bound | track cuda | twin: `tests/cuda/scf/test_rhf.cu` + device integrals | files: `scf_solver.cu`, `rhf.cu`, test | merge: `018dca0d` / feat `a8e23747` | evidence: GH200 **1051227** `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS** (2.98 + 0.86 s); tols unchanged |

Harness wave 6 land 2026-08-10 (`dev` @ `61f873c9`):

- [x] `cuda-pw-collinear-spin` | **P1** Lift `spin_polarized` throw on `cuda::pw::PW` — device nspin=2 density + collinear LSDA (VWN5 spin); fixed-occupation UKS; forces/stress still reject spin; no smear | track cuda | twin: unit XC spin parity + `reg_cuda-pw-collinear-spin` O₂ LSDA dual-gate | files: `device_density.{cu,cuh}`, `xc_lda.{cu,cuh}`, `pw_solver.{cu,cuh}`, `device_solver.{cu,cuh}` | merge: `0482e668` / feat `9c78896b` | evidence: GH200 **1051041** ctest 9/9; E both **−28.6294634700 Ha** mag=2 \|ΔE\|=**1.07e-13** Ha (tol 2e-3); post-merge combined **1051075** same parity; prior COMPLETED **1050941** exit 0; **perf_exempt** small cell; not spin-PBE / spin forces / smear |
- [x] `cuda-gpw-ks-multigrid` | **P2** Extend composed GPW KS stage with multi-grid product collocate/integrate (`MultigridTransferDevice` spectral prolong/restrict + `KsMultiGridChainDevice`) — **stage unit only**, not `GpwDft::run` | track cuda | twin: `test_cuda_gpw_ks_multigrid` + `reg_cuda-gpw-ks-multigrid` vs CPU Γ MG fragment | files: `ks_chain.{cu,cuh}`, `multigrid_transfer.{cu,cuh}` | merge: `61f873c9` / feat `ffd67884` | evidence: GH200 **1051042** unit 3/3 PASS; dual-gate max\|ΔV_eff\|=**8.095e-10**; post-merge combined **1051075** same; prior COMPLETED **1051020** exit 0; not full product |

Harness wave 5 land 2026-08-10 (`dev` @ `721ac286`):

- [x] `cuda-pw-forces-pbe` | **P1** Dual-gate analytic ionic forces under **device PBE-GGA** SCF — stream share `den.stream()` on force path; reg `cuda-pw-forces-pbe` | track cuda | twin: unit `test_pw_force_local` + `reg_cuda-pw-forces-pbe` Si dimer GTH-PBE-q4 | files: `pw_forces.{cu,cuh}`, `pw_solver.cu`, reg | merge: `a92d1af0` / feat `c5ae8fa2` | evidence: GH200 rebased **1050675** max\|ΔF\|=2.634e-13 + post-merge **1050801** max\|ΔF\|=**2.628e-13** eV/Å (tol 5e-3); E both −13.89672028 Ry; **perf_exempt** small cell; not stress/spin/smear |
- [x] `cuda-gpw-ks-gga` | **P1** Extend composed GPW KS stage from Padé LDA to **closed-shell PBE-GGA** (`SpectralFftDevice` + `KsXcKind::PBE`) — **stage unit only**, not `GpwDft::run` | track cuda | twin: `test_cuda_gpw_ks_gga` + `reg_cuda-gpw-ks-gga` vs CPU `ks_grid_build_rks(PBE)` | files: `ks_chain.{cu,cuh}`, `spectral_fft.{cu,cuh}` | merge: `721ac286` / feat `ac713518` | evidence: GH200 rebased **1050680** dual-gate e_H/e_xc ~1e-15 max\|ΔV_eff\|=8.0e-16 max\|Δacc\|=3.9e-14 + post-merge **1050825**; not full product / not multigrid |

Harness wave 4 land 2026-08-10 (`dev` @ `2987d20f`):

- [x] `cuda-pw-stress` | **P0** Device Nielsen–Martin analytic cell stress for `cuda::pw::PW` — public `cuda::pw::pw_stress`; density remap ρ·Ω₀/Ω + NSCF strain-FD twin of CPU | track cuda | twin: unit `test_pw_cell_stress` + `reg_cuda-pw-stress --device=both` | files: `pw_stress.{cu,cuh}`, `device_density.{cu,cuh}`, `pw_solver.{cu,cuh}` | merge: `98bd743a` / feat `2c662eab` | evidence: GH200 worker **1050263** + post-merge unit **1050357** + reg **1050604** — unit PASS; max\|Δσ\| **1.36e-12** eV/Å³ (tol 5e-2); E both **−13.86296636** Ry \|ΔE\|=2.132e-14 Ry; **perf_exempt** small cell; not spin/smear/PBE-force |
- [x] `cpu-simd-2center-angular` | **P3** Portable xsimd mixed-angular 2-center S/T Hermite (L≤d image-axis McMurchie–Davidson; pure-s path retained) | track cpu | twin: unit one_electron angular + `reg_cpu-simd-2center-angular` vs prim image sum | files: `one_electron.cpp` | merge: `2987d20f` / feat `1add075a` | evidence: unit+slab reg PASS; fused vs prim rel **1e-11**; FD `gpw-forces-water` max\|dF\|=3.243e-04; **ratio_before=ratio_after=2.33**; not bit-exact (xsimd exp/reduce); not fair-wall lever |

Harness wave 3 land 2026-08-10 (`dev` @ `1ba1678f`):

- [x] `cuda-pw-forces` | **P0** Device analytic ionic forces for `cuda::pw::PW` — local HF + Ewald + KB nonlocal; public `cuda::pw::pw_forces_analytic`; CPU force-path XC fix (`method.xc`) | track cuda | twin: unit `test_pw_force_local` + `reg_cuda-pw-forces` | files: `pw_forces.{cu,cuh}`, `device_solver.cu`, `pw_solver.{cu,cuh}`, CPU `pw_forces.cpp` | merge: `1ba1678f` / feat `a82a0758` | evidence: GH200 **1050105** max\|ΔF\| **2.86e-13** eV/Å (tol 5e-3); E both −13.85556302 Ry; not stress/spin/smear |
- [x] `cuda-gpw-ks-chain` | **P1** Compose landed GPW device footholds into one ρ→V_eff **stage unit** (collocate → Poisson → Padé LDA → integrate) — **not** `GpwDft::run` | track cuda | twin: `test_cuda_gpw_ks_chain` + `reg_cuda-gpw-ks-chain` | files: `ks_chain.{cu,cuh}` | merge: `9b19991b` / feat `47781d68` | evidence: GH200 **1049858** — unit 5/5; dual-gate e_H/e_xc ~1e-15, max\|ΔV_eff\|=4.4e-16, max\|Δacc\|=6.9e-14; not full product / not GGA |
- [x] `cpu-simd-2center-setup` | **P3** Portable xsimd pure-s analytic 2-center S/T (slab-class; invisible on four fair twins) | track cpu | twin: unit one_electron + `reg_cpu-simd-2center-setup` literature closed form | files: `one_electron.cpp` | merge: `e49e0757` / feat `b363e5e6` | evidence: unit 65/65; slab \|dS\|=2.2e-16 \|dT\|=4.4e-16; microbench ~24–26×; FD `gpw-forces-water` PASS max\|dF\|=3.243e-04; **ratio_before=ratio_after=2.33**; not bit-exact (xsimd exp/reduce) |

Harness wave 2 2026-08-10 (`dev` @ `8280ed89`):

- [x] `cuda-pw-nondefault-stream` | **P2** Non-default stream on PW density+band SCF path — `DeviceDensity` owns `cudaStream_t`; band solver borrows; kernels/cuFFT/cuBLAS/cuSOLVER bound; mid-SCF fences stream-scoped | track cuda | twin: unit stream handle + `pw-cuda-lda-si` + `reg_cuda-pw-nondefault-stream` | files: `device_density.{cu,cuh}`, `device_solver.{cu,cuh}`, `pw_solver.cu`, job shield | merge: `8280ed89` / feat `05b52ba2` | evidence: GH200 **1049681** — ctest **9/9 PASS**; MAX_EIG_DIFF **5.274e-15 Ha**; \|E_cuda−E_cpu\| **1.243e-14 Ry** (both regs); tols unchanged; **perf_exempt** small cell; wipe-shield `BUILD=_pw_stream_work` |

GapCensus wave 4 recovery 2026-08-10 (`dev` @ `e22325b7` / large-cell merge
`948d0d86`):

- [x] `cuda-pw-large-cell-measure` | **P1** ES order step 4 — first large-cell CPU↔CUDA wall (Si 8/64/216 ladder; small cells correctness-only) | track cuda | twin: `tools/bench_cuda_pw_cell.cu` + `pw-cuda-lda-si` | files: harness + `tools/gpu_job_cuda_pw_large_cell.sh` + note | merge: `948d0d86` | evidence: GH200 **1035278** (primary) + **1048264** reconfirm; 64-atom CUDA **~9.65 s** vs CPU **>5400 s** (\(T_{\mathrm{cu}}/T_{\mathrm{cp}} < 0.0018\)); slice \|ΔE\|=1.243e-14 Ry; note [cuda-pw-large-cell.md](../../.claude/notes/cuda-pw-large-cell.md); measurement-only (no energy-path change) |

Harness wave 1 2026-08-10 (`dev` @ `948d0d86`, PBE land `3f901034`):

- [x] `cuda-pw-pbe-device` | **P0** Device PBE-GGA for `cuda::pw::PW` — lift LSDA-only gate; closed-shell PBE-X + PW92-based PBE-C at ζ=0; spectral ∇ρ / −∇·flux in `DeviceDensity`; same-settings CPU oracle | track cuda | twin: unit `test_cuda_pw_xc_pbe` + DeviceDensity stage parity + e2e `reg_cuda-pw-pbe-device` | files: `xc_pbe.{cu,cuh}`, `device_density.{cu,cuh}`, `pw_solver.{cu,cuh}` | merge: `3f901034` | evidence: GH200 job **1048320** — units **4/4 PASS**; dual-gate \(E_\mathrm{cpu}=E_\mathrm{cuda}=-14.84955549\) Ry (7 it); \(\lvert\Delta\rvert=1.421\times10^{-14}\) Ry ≪ 1e-3 Ry; tols **1e-12·max(1,\|ref\|)** pointwise; roadmap PBE CUDA ✓ (PW closed-shell only); **perf_exempt** small cell; not forces/stress/spin/PBE-family |

Wave 7 land 2026-08-06 (`dev` @ `8b76cbe7`):

- [x] `cpu-collint-integrate-residual` | **P1** Residual integrate toward CP2K ~20 ms class — width-4 segment accumulate on AVX-512 (CP2K AVX2 idiom; avoid width-8 ZMM spill), short-segment scalar, Ltot triangular skip on rectangular monom walks, zero-pol reverse, cab shared-alpha from n≥4 | track cpu | twin: `gpw-multigrid-kpoints` MAIN | files: `product_collint.cpp` | merge: `8b76cbe7` / feat `c6659278` | evidence: int **~58→~45** ms/iter; coll ~29; MAIN **~1.6 s**; post-baseline CP2K quiet **0.688 s** → ratio **2.33×** (supersedes contemporaneous ~1.25× claim); pin \|ΔE\|=2.029e-05 Ha; unit collocate+multigrid PASS; FD `gpw-forces-water` PASS max\|dF\|=3.243e-04; not bit-exact (SIMD order) |

Wave 6 land 2026-08-06 (`dev` @ `a9f87206`):

- [x] `cpu-collint-shell-tasks` | **P0** Shell-level collint task list (`CollintTask` ranges once in `build_products`; collocate/integrate drive contiguous monom runs; cab-style shared-alpha reverse contract on integrate) | track cpu | twin: `gpw-multigrid-kpoints` MAIN | files: `multigrid.{h,cpp}`, `product_collint.cpp` | merge: `f3571435` / feat `2debae55` | evidence: MAIN **~1.9→~1.6 s**; coll **~63→~30** / int **~76→~58** ms/iter; pin unit multigrid/collocate PASS; FD `gpw-forces-water` PASS max\|dF\|=3.243e-04 (post-merge log) |
- [x] `cuda-gpw-product-integrate-foothold` | **P1** Thin device product V·shape integrate foothold (V·shape → per-product acc) vs same-settings CPU `collint_integrate_product` / continuous host — transpose of landed ρ collocate; reuses `DeviceProduct` / `CollocateGridDesc` | track cuda | twin: `test_cuda_gpw_product_integrate` | files: `product_integrate.{cu,cuh}`, unit test | merge: `a9f87206` / feat `2b317beb` | GH200 post-merge **915825**: `test_cuda_gpw_product_integrate` **Passed 0.95 s** + collocate **2.00 s** + `test_cuda_scf_rhf` **2.25 s** (**3/3 PASS**); tols device≡host continuous **1e-12·peak + 1e-14**, vs collint **1e-10·peak + 1e-14** (p-type **1e-9·peak + 1e-13**); not full GPW product |

Wave 5 land 2026-08-06 (`dev` @ `759ca486`):

- [x] `cpu-products-mg-setup` | **P0** Geometry-batch `build_products` / products_mg — monoms sharing collint geometry compute K/pc once; compact batch sort + flat exclusive-scan contrib packing | track cpu | twin: `gpw-multigrid-kpoints` setup mark `products_mg` | files: `multigrid.cpp` (`build_products`) | merge: `72926013` / feat `bbf1f167` | evidence: products_mg **~285→~18 ms**; MAIN **~2.4→~1.9 s**; pin bit-stable; FD `gpw-forces-water` PASS; no 5th collocation layout |
- [x] `cuda-rhf-cdiis-device` | **P1** Device CDIIS (Pulay) Fock mixer on s/p RHF path — CPU MixStage / DIISMixer twin; e = Xᵀ[F,P]_S X; history on device; tiny Pulay solve host-only | track cuda | twin: `tests/cuda/scf/test_rhf.cu` + H₂O STO-3G CUDA↔CPU | files: `diis_mixer.{cu,cuh}`, `scf_solver.cu`, `rhf_fock.cu` | merge: `d575e50b` / feat `f21a1173` | GH200 worker **915746** + post-merge **915778**: `test_cuda_scf` + `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS** (0.90 + 2.02 + 0.82 s); tols unchanged (ha_to_ev(1e-6) / 1e-8 Ha class) |
- [x] `cuda-gpw-rho-collocate-foothold` | **P2** Thin device product ρ collocate foothold vs same-settings CPU collint oracle (Γ small grid; s/p + multi-center s–s) | track cuda | twin: `test_cuda_gpw_product_collocate` | files: `product_collocate.{cu,cuh}`, unit test; AVX-512 collint residual gated for aarch64 GH200 | merge: `759ca486` / feat `0d2868e7` | GH200 worker **915754** + post-merge **915778**: `test_cuda_gpw_product_collocate` **Passed 0.80 s** (+ poisson/XC/RHF suite **6/6 PASS**); tols device≡host continuous **1e-14**, vs collint **1e-10·peak + 1e-14** (p-type **1e-9·peak**); not full GPW product |

Wave 4 land 2026-08-06 (`dev` @ collint merge `5fa8b0c6` + CUDA `44a5d858`):

- [x] `cpu-collint-residual` | **P0** Residual collint: hybrid xsimd FMA + 4-wide residual, lp_z≤9, cab-style alpha for large groups, TLS multi-imr sphere, gamma group_idx nthread private grids, integrate \|V(pc)\| once per geometry | track cpu | twin: `gpw-multigrid-kpoints` MAIN | files: `product_collint.cpp`, `multigrid.cpp` | merge: `5fa8b0c6` / feat `6da97a85` | evidence: MAIN remeasure **~2.4 s**; coll **~63** / int **~76** ms/iter; pin bit-stable \|ΔE\|=5.7e-6 Ha; FD `gpw-forces-water` PASS max\|dF\|=3.2e-4; OT water 25 it + UKS O₂ 27 it / ~11.8 s PASS; shell-level tasks still open |
- [x] `cuda-gpw-xc-lda-foothold` | **P1** Thin device Padé LDA XC grid vs same-settings CPU oracle (closed-shell + LSD + energy reductions) | track cuda | twin: `test_cuda_gpw_xc_lda` | files: `src/cuda/gpw/internal/xc_lda.{cu,cuh}`, unit test | merge: `cc450b59` / feat `90acb1d5` | GH200 worker **915678** + post-merge **915704**: `test_cuda_gpw_xc_lda` **Passed 0.86 s**; tols ~**1e-12** per-point, energy rel **~1e-10**; not full GPW product |
- [x] `cuda-rhf-path-select-host` | **P2** Host-Z path select without setup type[] D2H (`cache_host_types` before upload) | track cuda | twin: `tests/cuda/scf/test_rhf.cu` | files: `rhf.cu`, `rhf.cuh` | merge: `44a5d858` / feat `6f179574` | GH200 worker **915654** + post-merge **915704**: `test_cuda_scf` + `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS** (0.95 + 2.03 + 0.87 s); tols unchanged |

Wave 3 land 2026-08-06 (`dev` @ `9f2d791c`):

- [x] `cpu-ot-residual` | **P0** OUTER_SCF polish (fixed-H eigen+mix after residual moderate; no DFT+U) | track cpu | twin: `ot-uks-o2` then `ot-minimizer-water` | files: `gpw_solver.cpp`, `ot.h` | merge: `4249fd3f` / feat `867c1add` | evidence: UKS O₂ **87→27** it, OT wall **~52→13 s** / CP2K **8.68 s** → **~1.53×** (was ~4.4×; aim ≤3× **met**); water **32→25** it; pins + FD `gpw-forces-water` PASS; no tol soften |
- [x] `cpu-stvnl-residual` | **P1** STVnl Hermite geometry-key reuse + lattice phase flatten | track cpu | twin: `gpw-multigrid-kpoints` | files: `one_electron.cpp`, `kpoints.cpp` | merge: `dfe68fdd` / feat `76239cfc` | evidence: STVnl_k **~507→435 ms**; MG MAIN quiet best **~2.6→2.3 s**; pin bit-stable; forces-water PASS |
- [x] `cuda-rhf-level-shift` | **P2** Saunders–Hillier λ on device s/p Fock (no host-fill when λ≠0) | track cuda | twin: `tests/cuda/scf/test_rhf.cu` `CUDALevelShiftEnergyInvariant` | files: `rhf.cu`, `rhf_fock.cu`, `scf_solver.cu` | merge: `ba3a8145` / feat `77eb7f79` | GH200 worker job **913709** + post-merge **915584**: `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **PASS**; λ=0.5 energy matches λ=0 and CPU @ 1e-8 Ha |
- [x] `cuda-gpw-foothold` | **P3** Thin Γ-point device Poisson/FFT vs CPU oracle | track cuda | twin: `test_cuda_gpw_device_poisson` | files: `src/cuda/gpw/internal/poisson_fft.{cu,cuh}`, unit test | merge: `9f2d791c` / feat `f604577c` | GH200 worker **915518** + post-merge **915584**: `test_cuda_gpw_device_poisson` **Passed 0.97 s** + `test_cuda_pw_poisson` **Passed 1.78 s** (**100%** of 4); tols max\|ΔV\| **1e-9…1e-10**, Hartree **1e-10…1e-12** rel; not full GPW product |

Wave 2 land 2026-08-06 (`dev` @ `fb9e7ed8`):

- [x] `cpu-ot-diis-outer` | **P0** Tighten Γ OT MINIMIZER DIIS outer count (g·e B-matrix + SAFE_DIIS/Armijo + CG fallback) | track cpu | twin: `ot-uks-o2` then `ot-minimizer-water` | files: `src/cpu/gpw/internal/ot.cpp`, `gpw_solver.cpp` | merge: `25c2268c` / feat `660a50d1` | evidence: water **93→32** OT it pin PASS; UKS O₂ **180→87–94** it, OT wall **~38.5 s** / CP2K **8.68 s** → **~4.4×** (was ~7.1×; land claim 3.74×); FD `gpw-forces-water` PASS max\|dF\|=3.2e-4; \|E_OT−E_DIIS\|~1e-10; no tol soften |
- [x] `cuda-rhf-device-only-polish` | **P1** Wave-7: drop mid-build V_nuc D2H/H2D (energy ← `d_V_nuc`); ERI `__launch_bounds__(256,2)` + occupancy-capped grid-stride | track cuda | twin: `tests/cuda/scf/test_rhf.cu` | files: `device_integrals.cu`, `rhf_fock.cu`, `rhf_density.cu` | merge: `fb9e7ed8` / feat `1ae27395` | GH200 job log `.claude/merge-cuda-polish-913612.out`: `test_cuda_scf_rhf` **Passed 2.29 s** + `test_cuda_scf_device_integrals` **Passed 0.82 s** (**100%**); need `-DCMAKE_CUDA_ARCHITECTURES=90` on GH200; tols unchanged from wave 6 |

Wave 1 land 2026-08-06 (`dev` @ `8337251b`):

- [x] `cpu-xc-spectral-fft` | **P1** GGA KS spectral gradient/div: fused multi-out FFT + FFTW_MEASURE plans | track cpu | twin: `ot-minimizer-water` | files: `grid_fft.cpp`, `xc_grid_gga.cpp`, `fft_cache.{h,cpp}` | merge: `977710fb` / feat `db132bb8` | evidence: XC ~108→**95** ms/iter; quiet wall ~16.5→**16.2 s**; energy pin stable; FD `gpw-forces-water` PASS; non-bit-exact MEASURE FP with stated tol |
- [x] `cuda-rhf-pshell-forces` | **P0** Device analytic forces for l≤1 s/p | track cuda | twin: H₂O STO-3G CUDA↔CPU + `tests/cuda/scf/test_rhf.cu` | files: `device_integrals.cu`, `rhf.cu`, force kernels | merge: `8337251b` / feat `e6f984d8` | GH200 job log `.claude/merge-cuda-rhf-913053.out`: `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS**; tols: H₂ |ΔF| **1e-5** eV/Å, H₂O |ΔF| **1e-4** eV/Å, energy **ha_to_ev(1e-6)**; `last_force_on_device()` fail-closed; no host RHF for covered shells |

Post-loop CUDA PW residency (on `dev`, evidence in `.claude/notes/cuda-rhf-device-only.md`):

- [x] `cuda-pw-method-slice` | `cuda::pw::PW` + device density side; GH200 **948152** parity |
- [x] `cuda-pw-kpoint-hoist` | per-k tables once per run; bit-identical correctness |
- [x] `cuda-pw-zheevd` | cuSOLVER subspace eigen; CUDA↔CPU E at roundoff |
- [x] `cuda-pw-devptr-nrm2` | device-pointer residual norms; bit-identical to prior |
