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

**GapCensus wave 8 on `dev` @ `31405991` (re-diff post wave-7 queue; no
feature land on base since opt-3):** Fair walls **held** (authority
[gpw-perf-baseline.md](../../.claude/notes/gpw-perf-baseline.md)): MG MAIN
**2.33×** worst; OT water SCF body **≤1.0×** / OT-only ~**1.20×**; UKS DIIS
**1.077×** closed; k-si MAIN **1.18×** hold. **cpu_pending=0** — pin-safe
collint/structure/sphere **exhausted**; no promote from
[gpw-perf-queue.md](gpw-perf-queue.md). Tree re-diff vs base (foothold ≠
closed; `src/cuda/gpw/gpw_solver.cu` + `src/cuda/planewave/pw_solver.cu`):

(1) Multi-k **on base** still **host** χ^k collocate + H2D ρ +
`run_from_rho` + **host** `integrate_potential_k` + host `KDiis` + phased
STVnl → `kpoint-device-*` held. (2) Multi-k dual-gate is **H₂ k=2×1×1
only** → `kpoint-si-gate` + `kpoint-h2o-gate` + production multi-k F/σ held;
**wave 8 opens** production multi-k **×PBE** energy gates
(`kpoint-si-pbe-gate` / `kpoint-h2o-pbe-gate`) — PADE production energy +
H₂ `kpoint-pbe-gate` alone leave production AO/grid × k × GGA ungated.
(3) MG e2e still **H₂ ngrids=4** → `multigrid-si/h2o-gate` + water MG F/σ
held; **wave 8 opens** **Si** MG F/σ (`multigrid-si-forces` /
`multigrid-si-stress`) + Si MG×PBE energy (`multigrid-si-pbe-gate`) —
water MG F/σ alone leaves MP-class Si multi-level F/σ ungated; MG PADE
production ≠ MG×PBE production. (4) Mid-SCF re-H2Ds full `DeviceProduct`
packs → `device-product-residency` after staffed `device-weight-pack`.
(5) UKS free-fn F **reject**; stress free-fn host fixed-\(P\); O₂ PADE
energy green → `uks-forces` / `uks-o2-forces` / `uks-stress` /
`uks-o2-stress` held. (6) **UKS-PBE** product still Padé-only (`run_uks`
reject) → `uks-pbe-gate` + `uks-o2-pbe-gate` held; **wave 8 opens**
production O₂ UKS-PBE **F/σ** (`uks-o2-pbe-forces` / `uks-o2-pbe-stress`)
— H₂/O₂ energy gates alone leave production spin-GGA F/σ ungated.
(7) Water Γ RKS **E+F** dual-gate green (`product-h2o-gate`); residual
water **σ** + water×PBE energy held; **wave 8 opens** water×PBE **F/σ**
(`product-h2o-pbe-forces` / `product-h2o-pbe-stress`). (8) Staffed `[~]`
tips now **have feat commits** (not on base): `uks-device-diis`
`549d2abe` GH200 **1130279**; `device-weight-pack` `94256cf0` GH200
**1130344**; `smear-forces` `bafa9c3f` GH200 **1130359** — keep `[~]`;
Merge owns land; do **not** re-assign. Residual after tips: host STVnl +
`device-product-residency` + force hybrid depth. (9) PW: spin LSDA forces
**on base**; residual spin-stress / spin-PBE F/σ / smear F/σ (staffed tip)
/ spin∧smear / multi-k F/σ + multi-k PBE / **spin×kmesh×smear** held.
**No new method domains.** Queue **not** exhausted.

**GapCensus wave 7 on `dev` @ `c149a80a` / `31405991`:** production multi-k/MG
F/σ residual + close `product-h2o-forces` — **held**; detail in git
(`31405991`).

**Autonomous discovery residual (wave 8 — foothold ≠ closed):**
(1) multi-k host residual spine → `kpoint-device-*`; (2) multi-k H₂ ≠ Si /
water energy → `kpoint-si/h2o-gate`; (3) production multi-k F/σ →
`kpoint-h2o/si-forces/stress`; (4) multi-k PADE production + H₂ PBE alone
leave **production multi-k ×PBE** open → **`kpoint-si-pbe-gate` /
`kpoint-h2o-pbe-gate`** (new); (5) MG H₂ ≠ Si/water energy →
`multigrid-si/h2o-gate`; (6) water MG F/σ held; **Si** MG F/σ →
**`multigrid-si-forces` / `multigrid-si-stress`** (new); (7) Si MG×PBE
energy → **`multigrid-si-pbe-gate`** (new); (8) UKS-PBE O₂ energy alone
leaves production spin-GGA F/σ open → **`uks-o2-pbe-forces` /
`uks-o2-pbe-stress`** (new); (9) water×PBE energy alone leaves F/σ open →
**`product-h2o-pbe-forces` / `product-h2o-pbe-stress`** (new); (10) staffed
spine `[~]` feat tips (GH200 evidence) still unmerged; (11) hybrid free-fn
+ path×PBE + 4-way + PW spin×kmesh×smear. Do **not** invent
hybrid/GAPW/PAW/meta-GGA; OT/MT/vdW/metallic-GPW stay same-capability
follow-on. **No gpw-perf-queue promote**.

| Fact | Evidence |
| --- | --- |
| Public-shape freeze | **green** merge `0147fe02` |
| First product | **landed** feat `31d24207` — GH200 **1120817** |
| Device mid-SCF collocate/integrate | **landed** merge `69296751` / feat `430a2268` — GH200 **1125709** |
| Product multigrid energy | **landed** feat `2d780cff` / `e201a6cb` — GH200 **1126433** |
| Γ UKS DIIS PADE energy | **landed** feat `f679807c` / `254ff67c` — GH200 **1126430** (H₂ foothold); O₂ gate **1127173** |
| Device grid forces free-fn | **landed** GH200 **1127167** (host AO Pulay residual; H₂ dual-gate; default-stream packs) |
| Product stress free-fn | **landed** GH200 **1128195** (CUDA density + **full host** fixed-\(P\); H₂ only) |
| Product non-default stream | **landed** GH200 **1128196** (SCF mid-path; force free-fn still default) |
| Device mid-SCF DIIS + eigen (RKS) | **landed** merge `e164b767` / harness `c2fd4ae8` — GH200 **1129719** \|dE\|=**1.616e-07** Ha |
| Water product dual-gate | **landed** FF `30bb49b4` / feat `9124615d` — GH200 **1130195** \|dE\|=**7.5e-7** max\|dF\|=**3.2e-6** (**E+F**; σ separate) |
| Multi-k RKS DIIS energy | **landed** merge `adc7c7d7` / feat `b2778a83` — GH200 **1130141** \|dE\|=**3.375e-14** (H₂ k=2×1×1; host residual spine) |
| Collinear LSDA spin forces | **landed** merge `ab7a95e1` / feat `44995ff6` — GH200 **1130196** max\|ΔF\|=**9.25e-13** eV/Å |
| Product hybrid (base) | host STVnl + weight pack + scatter; **device** RKS DIIS/eigen + product KS (RKS+UKS+MG; non-default stream); multi-k **host** χ^k + device ρ→V_eff; UKS host dual GpwDiis; forces hybrid + H2D handoff; stress host assembly (multi-k stress reject only) |
| Fail-closed (base) | OT / MT / non-PADE-PBE / `local_pp_on_grid=false`; multi-k UKS / multi-k MG; UKS free-fn F; multi-k free-fn F; MG free-fn F; multi-k stress; UKS still host dual GpwDiis; `run_uks` Padé only |
| Claimed-but-ungated | multi-k H₂ → Si/water energy + F/σ + **Si/water multi-k×PBE**; MG H₂ → Si/water energy + **water/Si MG F/σ** + **Si MG×PBE**; **uks-device-diis** / **device-weight-pack** / **device-product-residency** / **kpoint-device-*** / **uks-o2-forces/stress** / **uks-o2-pbe-gate** + **uks-o2-pbe F/σ** / device STVnl/AO-Pulay/**stress**/handoffs/force-stream; water **σ** + water×PBE **E+F/σ**; path×PBE + 4-way; PW multi-k PBE + spin-stress + smear F/σ + **spin×kmesh×smear** |
| Closed-shell PW stress | **LSDA + PBE** dual-gate **landed** (Γ-class coverage; multi-k F/σ + multi-k PBE energy open) |
| Collinear spin-PBE energy | **landed** merge `118d3003` / GH200 **1124645** (Γ dual-gate only) |
| Staffed tips (not base) | Assign wave 6 `[~]` **feat tips**: `uks-device-diis` `549d2abe` GH200 **1130279**; `device-weight-pack` `94256cf0` GH200 **1130344**; `smear-forces` `bafa9c3f` GH200 **1130359** — unmerged |

**Counts (post GapCensus wave 8):** **pending_count=105**; **in_progress=3**
(`uks-device-diis` + `device-weight-pack` + `smear-forces`);
**cpu_pending=0**; **cuda_pending=108** (pending + in_progress);
**gaps_remaining=109** (pending + in_progress + blocked MG fair-wall
`mg-main-structural`). Queue **not** exhausted.
**new_slugs_added=9** this GapCensus wave 8 (`kpoint-si-pbe-gate` +
`kpoint-h2o-pbe-gate` + `multigrid-si-forces` + `multigrid-si-stress` +
`multigrid-si-pbe-gate` + `uks-o2-pbe-forces` + `uks-o2-pbe-stress` +
`product-h2o-pbe-forces` + `product-h2o-pbe-stress`).
**Assign wave 6 staff still live** (feat tips green — Merge owns land) —
do not re-assign until Worktrees returns.

### CUDA — GPW product depth (product on base — staff depth)

method=cuda throughout. Product is **on base** — staff depth into public
`GpwDft`. Do **not** open lone new kernel footholds — compose landed stage units.

**Compose / path lifts (held + re-ranked + wave-5 joint paths):**

`cuda-gpw-device-collocate` **landed** (merge `69296751` / GH200 **1125709**)
— mid-SCF device product collocate/integrate on base; residual host weight
pack + D2H acc → deeper pack/`device-stvnl`; DIIS/eigen → **landed**.

`cuda-gpw-device-diis-eigen` **landed** (merge `e164b767` / GH200 **1129719**)
— RKS mid-SCF Pulay DIIS + cuSOLVER eigen via `AoDiisEigenDevice`; residual
host weight pack + scatter + STVnl; UKS host dual GpwDiis →
**`uks-device-diis`** + **`device-weight-pack`** + `device-stvnl` child rows.

`cuda-gpw-multigrid-run` **landed** (feat `2d780cff` / GH200 **1126433**) —
product `ngrids>1` energy via `KsMultiGridChainDevice`; residual MG F/σ +
MG×PBE / multi-k MG → child rows.

`cuda-gpw-uks-gamma` **landed** (feat `f679807c` / GH200 **1126430**) — Γ
UKS DIIS PADE energy via `KsChainDevice::run_uks`; residual UKS F/σ +
UKS-PBE + joint paths → child rows; UKS **host** DIIS residual →
`uks-device-diis` (wave 3).



- [ ] `cuda-gpw-uks-forces` | **P3** Analytic free-fn forces under Γ UKS
  product after `uks-gamma` energy (energy slug does not own forces; base
  free-fn is RKS CPU-backed only). Track: **cuda**. Twin: CPU
  `reg_gpw-uks-forces` / O₂ UKS forces + dual-gate. Gate: GH200 + stated
  max\|ΔF\| after UKS energy green. Lever: dual-spin density + force assembly
  on product force free-fn (compose with `device-forces` patterns).
- [ ] `cuda-gpw-uks-o2-forces` | **P3** **new (wave 5)** — Production **O₂**
  dual-gate for Γ UKS free-fn forces after `uks-forces` (H₂-class force
  foothold alone leaves production AO/grid × spin F ungated; mirror
  `uks-o2-gate` after `uks-gamma` energy). Track: **cuda**. Twin: CPU
  `reg_gpw-uks-forces` O₂ class + dual-gate. Gate: GH200 after `uks-forces`;
  stated max\|ΔF\|; **perf_exempt**. Lever: extend unit/reg under O₂ UKS —
  no new kernel.
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
  Production O₂ spin-GGA is **`uks-o2-pbe-gate`** (child).
- [ ] `cuda-gpw-uks-o2-pbe-gate` | **P3** **new (wave 7)** — Production **O₂**
  dual-gate for Γ UKS **PBE** energy after `uks-pbe-gate` (H₂-class UKS-PBE
  foothold alone leaves production AO/grid × spin-GGA ungated; mirror
  `uks-o2-gate` after `uks-gamma` PADE). Track: **cuda**. Twin: CPU
  `reg_gpw-uks-pbe` O₂ class + dual-gate. Gate: GH200 after `uks-pbe-gate`;
  stated energy tol; **perf_exempt**. Lever: extend unit/reg under O₂ UKS
  PBE — no new kernel. Production O₂ UKS-PBE F/σ → `uks-o2-pbe-forces` /
  `uks-o2-pbe-stress` (wave 8).
- [ ] `cuda-gpw-uks-o2-pbe-forces` | **P3** **new (wave 8)** — Production **O₂**
  dual-gate for Γ UKS **PBE** free-fn forces after `uks-o2-pbe-gate` energy +
  `uks-pbe-forces` (H₂-class UKS-PBE F + O₂ energy alone leave production
  AO/grid × spin-GGA F ungated; mirror `uks-o2-forces`). Track: **cuda**.
  Twin: CPU `reg_gpw-uks-pbe` / UKS-PBE forces O₂ class + dual-gate. Gate:
  GH200 after parents; stated max\|ΔF\|; **perf_exempt**. Lever: extend
  unit/reg only — no new kernel.
- [ ] `cuda-gpw-uks-o2-pbe-stress` | **P3** **new (wave 8)** — Production **O₂**
  dual-gate for Γ UKS **PBE** cell stress after `uks-o2-pbe-gate` +
  `uks-pbe-stress` (same production gap as forces for σ). Track: **cuda**.
  Twin: CPU Γ UKS PBE stress O₂ class + dual-gate. Gate: GH200 after parents;
  stated max‖Δσ‖; **perf_exempt**. Lever: extend unit/reg only.
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
- [ ] `cuda-gpw-uks-o2-stress` | **P3** **new (wave 5)** — Production **O₂**
  dual-gate for Γ UKS cell stress after `uks-stress` (H₂-class stress
  foothold alone leaves production spin σ ungated; mirror `uks-o2-gate` /
  `uks-o2-forces`). Track: **cuda**. Twin: CPU `reg_gpw-uks-stress` O₂
  class + dual-gate. Gate: GH200 after `uks-stress`; stated max‖Δσ‖;
  **perf_exempt**. Lever: extend unit/reg under O₂ UKS — no new kernel.
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
  kernel campaign. Does **not** own mid-SCF collocate/DIIS; mid-SCF pack is
  `device-weight-pack` (wave 3).
- [~] `cuda-gpw-uks-device-diis` | **P2** **new (wave 3)** — Γ UKS product still
  runs **host dual** `cpu::gpw::internal::GpwDiis` + host eigen after RKS
  `AoDiisEigenDevice` landed (`gpw_solver.cu` UKS branch — foothold energy ≠
  device SCF spine). Track: **cuda**. Twin: same-settings CPU Γ UKS PADE
  energy dual-gate (H₂/O₂ class). Gate: GH200 after `device-diis-eigen` +
  `uks-gamma` / `uks-o2-gate`; stated energy tol; **perf_exempt** small cell.
  Lever: dual-spin device Pulay DIIS + cuSOLVER eigen compose
  `AoDiisEigenDevice` pattern on product UKS path; not forces/stress/PBE in
  this slug (those stay child rows).
- [~] `cuda-gpw-device-weight-pack` | **P2** **new (wave 3)** — Mid-SCF every
  iter still host `product_density_weight` + H2D `DeviceProduct` pack + D2H
  integrate acc + host `scatter_product_acc` + `download_P` after device DIIS
  (`build_H_and_energy` in `gpw_solver.cu` — distinct from one-shot STVnl).
  Track: **cuda**. Twin: same-settings CPU Γ RKS energy dual-gate. Gate:
  GH200 after `device-diis-eigen`; stated energy tol; **perf_exempt**. Lever:
  device P→weight (+ dual-spin for UKS path) and/or device acc→V_AO scatter
  keeping P resident; compose not a lone foothold campaign. Does **not** own
  STVnl setup (`device-stvnl`) or DIIS/eigen.
- [ ] `cuda-gpw-device-product-residency` | **P3** **new (wave 5)** — Mid-SCF
  still re-H2Ds full `DeviceProduct` packs every iter while product geometry
  is fixed for the run (only density weights change — residual after
  `device-weight-pack` compute). Track: **cuda**. Twin: same-settings CPU Γ
  RKS energy dual-gate. Gate: GH200 after `device-weight-pack`; stated energy
  tol; **perf_exempt**. Lever: keep geometry-fixed product descriptors
  device-resident; update weight scalars only (or device P→weight without
  full pack rewrite); compose not a lone foothold campaign. Does **not**
  own STVnl (`device-stvnl`) or DIIS/eigen.
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
  Lever: extend unit/reg only — no new kernel. Production water×PBE F/σ →
  `product-h2o-pbe-forces` / `product-h2o-pbe-stress` (wave 8).
- [ ] `cuda-gpw-product-h2o-pbe-forces` | **P3** **new (wave 8)** — Analytic
  free-fn forces under **water-class Γ RKS PBE** after `product-h2o-pbe-gate`
  energy + `pbe-forces-gate` (H₂ PBE F + water PBE energy alone leave
  production AO/grid × GGA F ungated). Track: **cuda**. Twin: CPU water Γ
  RKS PBE forces dual-gate. Gate: GH200 after parents; stated max\|ΔF\|;
  **perf_exempt**. Lever: extend unit/reg only — no new kernel.
- [ ] `cuda-gpw-product-h2o-pbe-stress` | **P3** **new (wave 8)** — Cell stress
  under **water-class Γ RKS PBE** after `product-h2o-pbe-gate` +
  `pbe-stress-gate` / `product-h2o-stress`. Track: **cuda**. Twin: CPU water
  Γ RKS PBE stress dual-gate. Gate: GH200 after parents; stated max‖Δσ‖;
  **perf_exempt**. Lever: extend unit/reg only.
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
  KS; extend unit/reg. Production Si/water multi-k×PBE →
  `kpoint-si-pbe-gate` / `kpoint-h2o-pbe-gate` (wave 8).
- [ ] `cuda-gpw-kpoint-si-pbe-gate` | **P3** **new (wave 8)** — Production
  **Si-class multi-k RKS × PBE** energy dual-gate after `kpoint-si-gate` +
  `kpoint-pbe-gate` (Si multi-k PADE + H₂ multi-k PBE alone leave MP-mesh
  production × GGA ungated). Track: **cuda**. Twin: CPU `reg_gpw-kpoints-si`
  PBE-class + dual-gate. Gate: GH200 after parents; stated energy tol;
  **perf_exempt** (not fair-wall claim). Lever: extend unit/reg under Si
  GTH + k_mesh>1 + PBE — no new kernel.
- [ ] `cuda-gpw-kpoint-h2o-pbe-gate` | **P3** **new (wave 8)** — Production
  **water-class multi-k RKS × PBE** energy dual-gate after `kpoint-h2o-gate`
  + `kpoint-pbe-gate` (water multi-k PADE + H₂ multi-k PBE alone leave
  molecular production × k × GGA ungated). Track: **cuda**. Twin: CPU water
  multi-k RKS PBE fragment + dual-gate. Gate: GH200 after parents; stated
  energy tol; **perf_exempt**. Lever: extend unit/reg only.

- [ ] `cuda-gpw-multigrid-h2o-gate` | **P3** **new (wave 8)** — Water-class
  product **multigrid** dual-gate after `multigrid-run` + `product-h2o-gate`
  (MG e2e is H₂ ngrids=4 only). Track: **cuda**. Twin: CPU water Γ RKS
  PADE MG fragment + dual-gate. Gate: GH200 E (+ F if free-fn);
  **perf_exempt**. Lever: extend unit/reg only — no new kernel. Production
  water MG F/σ → `multigrid-h2o-forces` / `multigrid-h2o-stress` (wave 7).
- [ ] `cuda-gpw-multigrid-si-gate` | **P3** **new (wave 5)** — Production
  **Si-class** product multigrid dual-gate after `multigrid-run` (e2e is
  H₂ ngrids=4 only; water-class is `multigrid-h2o-gate` — Si/MP production
  AO/grid × MG ungated; mirror `kpoint-si-gate` after multi-k energy
  foothold). Track: **cuda**. Twin: CPU `reg_gpw-multigrid-kpoints` /
  Γ MG Si fragment class + dual-gate. Gate: GH200 after `multigrid-run`;
  stated energy tol; **perf_exempt** small cell (not fair-wall claim).
  Lever: extend unit/reg under Si GTH + ngrids>1 — no new kernel. Production
  Si MG F/σ → `multigrid-si-forces` / `multigrid-si-stress`; Si MG×PBE →
  `multigrid-si-pbe-gate` (wave 8).
- [ ] `cuda-gpw-multigrid-si-forces` | **P3** **new (wave 8)** — Analytic forces
  under **Si-class multigrid** RKS after `multigrid-si-gate` energy +
  `multigrid-forces` (H₂ MG F foothold + Si MG energy + water MG F alone
  leave MP-class multi-level F ungated). Track: **cuda**. Twin: CPU Si Γ MG
  forces class + dual-gate. Gate: GH200 after both parents; stated max\|ΔF\|;
  **perf_exempt**. Lever: extend unit/reg only — no new kernel.
- [ ] `cuda-gpw-multigrid-si-stress` | **P3** **new (wave 8)** — Cell stress
  under **Si-class multigrid** RKS after `multigrid-si-gate` +
  `multigrid-stress`. Track: **cuda**. Twin: CPU Si Γ MG stress dual-gate.
  Gate: GH200 after both parents; stated max‖Δσ‖; **perf_exempt**. Lever:
  extend unit/reg only.
- [ ] `cuda-gpw-multigrid-si-pbe-gate` | **P3** **new (wave 8)** — Production
  **Si-class multigrid × PBE** energy dual-gate after `multigrid-si-gate` +
  `multigrid-pbe-gate` (Si MG PADE + H₂ MG×PBE alone leave production
  AO/grid × multi-level GGA ungated). Track: **cuda**. Twin: CPU Si Γ MG
  RKS PBE fragment + dual-gate. Gate: GH200 after parents; **perf_exempt**.
  Lever: extend unit/reg only — no new kernel.
- [ ] `cuda-gpw-multigrid-pbe-forces` | **P3** **new (wave 8)** — Analytic
  forces under product **MG×PBE** after `multigrid-pbe-gate` energy +
  `multigrid-forces` / `pbe-forces-gate` (path×PBE energy alone leaves F
  open). Track: **cuda**. Twin: CPU Γ MG RKS PBE forces + dual-gate. Gate:
  GH200 after parents. Lever: free-fn multi-level PBE force assembly.
- [ ] `cuda-gpw-multigrid-pbe-stress` | **P3** **new (wave 8)** — Cell stress
  under product **MG×PBE** after `multigrid-pbe-gate` + `multigrid-stress`
  / `pbe-stress-gate`. Track: **cuda**. Twin: CPU Γ MG RKS PBE stress +
  dual-gate. Gate: GH200. Lever: multi-level PBE fixed-\(P\) stress.
- [ ] `cuda-gpw-kpoint-pbe-forces` | **P3** **new (wave 8)** — Analytic
  forces under multi-k RKS **PBE** after `kpoint-pbe-gate` +
  `kpoint-forces` / `pbe-forces-gate`. Track: **cuda**. Twin: CPU multi-k
  RKS PBE forces dual-gate. Gate: GH200. Lever: k-summed PBE force assembly
  on free-fn.
- [ ] `cuda-gpw-kpoint-pbe-stress` | **P3** **new (wave 8)** — Cell stress
  under multi-k RKS **PBE** after `kpoint-pbe-gate` + `kpoint-stress` /
  `pbe-stress-gate`. Track: **cuda**. Twin: CPU multi-k RKS PBE stress
  dual-gate. Gate: GH200. Lever: k-summed PBE stress assembly.
- [ ] `cuda-gpw-uks-pbe-multigrid` | **P3** **new (wave 8)** — Γ UKS
  **multigrid PBE** energy after `uks-multigrid` + `uks-pbe-gate` +
  `multigrid-pbe-gate` (pairwise leave three-way open; device `run_uks` is
  Padé-only today — needs spin-GGA on KS chain + MG compose). Track:
  **cuda**. Twin: CPU Γ UKS MG PBE fragment + dual-gate. Gate: GH200 +
  stated energy tol; **perf_exempt**. Lever: product dual-spin MG +
  spin-PBE XC on `KsMultiGridChainDevice` / `run_uks`.
- [ ] `cuda-gpw-uks-kpoints-pbe` | **P3** **new (wave 8)** — Multi-k
  collinear UKS **PBE** energy after `uks-kpoints` + `uks-pbe-gate` +
  `kpoint-pbe-gate`. Track: **cuda**. Twin: CPU multi-k UKS PBE class +
  dual-gate. Gate: GH200; **perf_exempt**. Lever: dual-spin phased path +
  spin-PBE on product `run`.
- [ ] `cuda-gpw-multigrid-kpoints-pbe` | **P3** **new (wave 8)** — Multi-k
  **MG RKS PBE** energy after `multigrid-kpoints` + `multigrid-pbe-gate` +
  `kpoint-pbe-gate` (worst-twin class under GGA). Track: **cuda**. Twin:
  CPU multi-k MG PBE energy dual-gate. Gate: GH200; **perf_exempt** (not
  fair-wall claim). Lever: multi-level × k-mesh × PBE product compose.
- [ ] `cuda-gpw-uks-pbe-multigrid-forces` | **P3** **new (wave 8)** —
  Analytic forces under Γ UKS **MG×PBE** after `uks-pbe-multigrid` energy.
  Track: **cuda**. Twin: CPU Γ UKS MG PBE forces dual-gate. Gate: GH200.
  Lever: dual-spin multi-level PBE force assembly.
- [ ] `cuda-gpw-uks-pbe-multigrid-stress` | **P3** **new (wave 8)** — Cell
  stress under Γ UKS **MG×PBE** after `uks-pbe-multigrid` + stress parents.
  Track: **cuda**. Twin: CPU Γ UKS MG PBE stress dual-gate. Gate: GH200.
  Lever: dual-spin multi-level PBE stress.
- [ ] `cuda-gpw-uks-kpoint-pbe-forces` | **P3** **new (wave 8)** — Analytic
  forces under multi-k UKS **PBE** after `uks-kpoints-pbe` energy. Track:
  **cuda**. Twin: CPU multi-k UKS PBE forces dual-gate. Gate: GH200.
  Lever: dual-spin k-summed PBE force assembly.
- [ ] `cuda-gpw-uks-kpoint-pbe-stress` | **P3** **new (wave 8)** — Cell
  stress under multi-k UKS **PBE** after `uks-kpoints-pbe` + stress parents.
  Track: **cuda**. Twin: CPU multi-k UKS PBE stress dual-gate. Gate: GH200.
  Lever: dual-spin k-summed PBE stress.

- [ ] `cuda-gpw-force-stream` | **P3** **new (wave 10)** — Free-fn force path
  still uses **default stream** + blocking `cudaMemcpy` while product SCF owns
  a non-default stream (`cuda-gpw-nondefault-stream` landed; force free-fn not
  wired). Track: **cuda**. Twin: same-settings CPU Γ RKS force dual-gate (H₂
  class). Gate: GH200 after `device-forces` + `nondefault-stream`; stated
  max\|ΔF\|; **perf_exempt**. Lever: bind `gpw_forces_from_scf` grid HF kernels
  + packs to an owned / borrowed non-default stream; fence that stream only —
  no new kernel foothold.
- [ ] `cuda-gpw-device-ao-pulay` | **P3** **new (wave 10)** — Free-fn forces
  still run **host AO Pulay** (`W·dS`, `P·dT`, `P·dVnl` + op-center) while
  device owns only grid V_eff / ionic / local-PP HF (`gpw_forces.cu` —
  foothold ≠ full force product). Track: **cuda**. Twin: CPU
  `gpw_forces_from_scf` dual-gate. Gate: GH200 after `device-forces` (+ prefer
  after `device-stvnl` Hermite reuse patterns); stated max\|ΔF\|. Lever: device
  Hermite/derivative STVnl force assembly on free-fn; compose not a lone
  foothold campaign.
- [ ] `cuda-gpw-product-h2o-stress` | **P3** **new (wave 10)** — Product stress
  dual-gate is **H₂ only** (`reg_cuda-gpw-stress`); production water σ ungated
  (water **F** already dual-gated on `product-h2o-gate` — σ is the residual).
  Track: **cuda**. Twin: CPU water Γ RKS PADE stress + dual-gate. Gate: GH200
  after `product-h2o-gate` + landed `stress`; **perf_exempt**; stated max‖Δσ‖.
  Lever: extend unit/reg only — host fixed-\(P\) free-fn already on base.
- [ ] `cuda-gpw-device-stress` | **P3** **new (wave 11)** — Free-fn stress is
  still **100% host** `cpu::gpw::gpw_stress` after CUDA SCF density payload
  (`gpw_solver.cu` `gpw_stress` — foothold hybrid ≠ device stress product).
  Track: **cuda**. Twin: same-settings CPU Γ RKS fixed-\(P\) stress dual-gate.
  Gate: GH200 after landed `stress` (+ prefer after `device-ao-pulay` /
  `device-stvnl` patterns where reusable); stated max‖Δσ‖. Lever: device
  fixed-\(P\) analytic assembly on free-fn (or device density-resident path);
  compose not a lone foothold campaign.
- [ ] `cuda-gpw-stress-stream` | **P3** **new (wave 11)** — Stress free-fn has
  no owned non-default stream binding for any device work (SCF stream ends at
  `run`; stress assembly host today). Track: **cuda**. Twin: CPU Γ RKS stress
  dual-gate. Gate: GH200 after `device-stress` (+ `nondefault-stream` parent);
  stated max‖Δσ‖; **perf_exempt**. Lever: bind device stress kernels/packs to
  owned or borrowed non-default stream; fence that stream only.
- [ ] `cuda-gpw-force-device-handoff` | **P3** **new (wave 11)** — Free-fn forces
  **re-H2D** V_eff / V_es / ρ / P / cbg after SCF already D2H'd store_grids
  (`gpw_forces_from_scf` blocking packs — residency round-trip residual after
  `device-forces` + `nondefault-stream`). Track: **cuda**. Twin: CPU Γ RKS
  force dual-gate. Gate: GH200 after `device-forces` (+ prefer with
  `force-stream`); stated max\|ΔF\|; **perf_exempt**. Lever: keep final KS
  grids/density device-resident for free-fn (or zero-copy handoff API) —
  drop force-path H2D re-upload; no new physics kernel.
- [ ] `cuda-gpw-uks-multigrid-kpoints-pbe` | **P3** **new (wave 11)** —
  **Four-way** multi-k UKS **multigrid PBE** DIIS **energy** after pairwise /
  triple parents (`uks-multigrid-kpoints` + `uks-pbe-multigrid` +
  `uks-kpoints-pbe` + `multigrid-kpoints-pbe` — each lower joint alone leaves
  four-way open). Track: **cuda**. Twin: CPU multi-k UKS MG PBE energy
  dual-gate. Gate: GH200 + stated energy tol; **perf_exempt** small cell.
  Lever: dual-spin × multi-level × k-mesh × spin-PBE product compose on
  `run`; not forces/stress in this slug.
- [ ] `cuda-gpw-uks-multigrid-kpoint-pbe-forces` | **P3** **new (wave 11)** —
  Analytic forces under **four-way** multi-k UKS MG **PBE** after
  `uks-multigrid-kpoints-pbe` energy (triple F + path×PBE F rows leave
  four-way F open). Track: **cuda**. Twin: CPU multi-k UKS MG PBE forces
  dual-gate. Gate: GH200 after four-way energy. Lever: dual-spin ×
  multi-level × k-summed PBE force assembly on free-fn.
- [ ] `cuda-gpw-uks-multigrid-kpoint-pbe-stress` | **P3** **new (wave 11)** —
  Cell stress under **four-way** multi-k UKS MG **PBE** after
  `uks-multigrid-kpoints-pbe` + stress parents (same four-way gap as forces).
  Track: **cuda**. Twin: CPU multi-k UKS MG PBE stress dual-gate. Gate:
  GH200 after four-way energy. Lever: dual-spin × multi-level × k-summed
  fixed-\(P_k\) PBE stress on free-fn / Frame.
- [ ] `cuda-gpw-stress-device-handoff` | **P3** **new (wave 3)** — Stress free-fn
  re-enters density / fixed-\(P\) assembly from host after SCF already
  materialised grids on device (mirror of `force-device-handoff`; hybrid
  stress foothold ≠ residency product). Track: **cuda**. Twin: CPU Γ RKS
  fixed-\(P\) stress dual-gate. Gate: GH200 after `device-stress` (+ prefer
  with `stress-stream`); stated max‖Δσ‖; **perf_exempt**. Lever: keep
  density/grids device-resident into stress free-fn (or zero-copy handoff) —
  drop stress-path re-upload; no new physics kernel.
- [ ] `cuda-gpw-kpoint-device-collocate` | **P3** **new (wave 4)** — Multi-k
  energy tip still **host** χ^k collocate/integrate + H2D ρ +
  `KsChainDevice::run_from_rho` while Γ product owns device product
  collocate mid-SCF (`wf/cuda-gpw-kpoints` residual — foothold energy ≠
  device multi-k KS spine). Track: **cuda**. Twin: same-settings CPU multi-k
  RKS energy dual-gate (H₂ k-class then Si after `kpoint-si-gate`). Gate:
  GH200 after `kpoints` energy; stated energy tol; **perf_exempt**. Lever:
  device multi-k density / product-collocate compose into multi-k `run`
  (not forces/stress; not host KDiis — that is `kpoint-device-diis`).
- [ ] `cuda-gpw-kpoint-device-integrate` | **P3** **new (wave 5)** — Multi-k
  tip still **D2H V_eff** + host `integrate_potential_k` (V→H_k) every SCF
  iter after device `run_from_rho` (density collocate is
  `kpoint-device-collocate`; integrate half is a distinct residual —
  foothold energy ≠ device multi-k H assembly). Track: **cuda**. Twin:
  same-settings CPU multi-k RKS energy dual-gate. Gate: GH200 after
  `kpoints` + prefer after `kpoint-device-collocate`; stated energy tol;
  **perf_exempt**. Lever: device χ^k / product integrate of V_eff into H_k
  keeping veff resident; compose not a lone foothold campaign. Does **not**
  own KDiis (`kpoint-device-diis`) or phased STVnl (`kpoint-device-stvnl`).
- [ ] `cuda-gpw-kpoint-device-diis` | **P3** **new (wave 4)** — Multi-k tip
  still runs **host** `KDiis` after Γ RKS `AoDiisEigenDevice` landed
  (`gpw_solver.cu` multi-k branch on tip — foothold ≠ device SCF spine).
  Track: **cuda**. Twin: same-settings CPU multi-k RKS energy dual-gate.
  Gate: GH200 after `kpoints` + `device-diis-eigen`; stated energy tol;
  **perf_exempt**. Lever: multi-k / per-k device Pulay DIIS + cuSOLVER eigen
  compose `AoDiisEigenDevice` patterns; not forces/stress/UKS-k in this slug.
- [ ] `cuda-gpw-kpoint-device-stvnl` | **P3** **new (wave 4)** — Multi-k tip
  still builds **host** phased \(S_k/T_k/V_{\mathrm{nl}}^k\) once per `run`
  (Γ `device-stvnl` does not own multi-k lattice sums). Track: **cuda**.
  Twin: same-settings CPU multi-k RKS energy dual-gate. Gate: GH200 after
  `kpoints` + prefer after Γ `device-stvnl`; stated energy tol;
  **perf_exempt**. Lever: device phased Hermite / contracted STVnl^k upload
  once; compose not a lone foothold campaign. Does **not** own mid-SCF
  collocate (`kpoint-device-collocate`) or KDiis (`kpoint-device-diis`).
- [ ] `cuda-gpw-kpoint-si-gate` | **P3** **new (wave 4)** — Multi-k product
  dual-gate on tip is **H₂ k=2×1×1** only; production Si MP mesh class
  ungated after energy foothold (mirror `product-h2o-gate` after H₂
  product). Track: **cuda**. Twin: CPU `reg_gpw-kpoints-si` class + dual-gate.
  Gate: GH200 after `kpoints` energy; stated energy tol; **perf_exempt**
  small cell (not fair-wall claim). Lever: extend unit/reg under Si GTH +
  multi-k; no new kernel.

### CUDA — PW depth (landed surface incomplete) — parallel staff OK

Reject matrix re-confirmed on `dev` @ `e13cdb91` (method=cuda throughout):

**Closed-shell axis** (energy/forces/PBE-stress landings green; smear depth open):

- [ ] `cuda-gpw-kpoint-h2o-gate` | **P3** **new (wave 6)** — Water-class
  multi-k RKS product dual-gate after `kpoints` (e2e multi-k is **H₂
  k=2×1×1** only on base; Si-class is `kpoint-si-gate` — molecular
  production AO/grid × multi-k ungated; mirror `product-h2o-gate` after
  first product energy). Track: **cuda**. Twin: CPU water multi-k RKS
  PADE fragment + dual-gate. Gate: GH200 after `kpoints`; stated energy
  tol; **perf_exempt** small cell (not fair-wall claim). Lever: extend
  unit/reg under water GTH + k_mesh>1 — no new kernel. Production multi-k
  F/σ under water → `kpoint-h2o-forces` / `kpoint-h2o-stress` (wave 7).
- [ ] `cuda-gpw-kpoint-h2o-forces` | **P3** **new (wave 7)** — Analytic free-fn
  forces under **water-class multi-k** RKS after `kpoint-h2o-gate` energy +
  `kpoint-forces` (H₂ multi-k F foothold + water multi-k energy alone leave
  production AO/grid × k F ungated). Track: **cuda**. Twin: CPU water multi-k
  RKS forces dual-gate. Gate: GH200 after both parents; stated max\|ΔF\|;
  **perf_exempt**. Lever: extend unit/reg under water + k_mesh>1 — no new
  kernel.
- [ ] `cuda-gpw-kpoint-h2o-stress` | **P3** **new (wave 7)** — Cell stress under
  **water-class multi-k** RKS after `kpoint-h2o-gate` + `kpoint-stress`
  (same production gap as forces for σ). Track: **cuda**. Twin: CPU water
  multi-k RKS stress dual-gate. Gate: GH200 after both parents; stated
  max‖Δσ‖; **perf_exempt**. Lever: extend unit/reg only.
- [ ] `cuda-gpw-kpoint-si-forces` | **P3** **new (wave 7)** — Analytic free-fn
  forces under **Si-class multi-k** RKS after `kpoint-si-gate` energy +
  `kpoint-forces` (H₂ multi-k F + Si multi-k energy alone leave MP-mesh
  production F ungated). Track: **cuda**. Twin: CPU `reg_gpw-kpoints-si` /
  multi-k Si forces class + dual-gate. Gate: GH200 after both parents;
  stated max\|ΔF\|; **perf_exempt**. Lever: extend unit/reg under Si GTH +
  multi-k — no new kernel.
- [ ] `cuda-gpw-kpoint-si-stress` | **P3** **new (wave 7)** — Cell stress under
  **Si-class multi-k** RKS after `kpoint-si-gate` + `kpoint-stress`. Track:
  **cuda**. Twin: CPU multi-k Si stress dual-gate. Gate: GH200 after both
  parents; stated max‖Δσ‖; **perf_exempt**. Lever: extend unit/reg only.
- [ ] `cuda-gpw-multigrid-h2o-forces` | **P3** **new (wave 7)** — Analytic forces
  under **water-class multigrid** RKS after `multigrid-h2o-gate` energy +
  `multigrid-forces` (H₂ MG F foothold + water MG energy alone leave
  production multi-level F ungated). Track: **cuda**. Twin: CPU water Γ MG
  forces dual-gate. Gate: GH200 after both parents; stated max\|ΔF\|;
  **perf_exempt**. Lever: extend unit/reg only — no new kernel.
- [ ] `cuda-gpw-multigrid-h2o-stress` | **P3** **new (wave 7)** — Cell stress
  under **water-class multigrid** RKS after `multigrid-h2o-gate` +
  `multigrid-stress`. Track: **cuda**. Twin: CPU water Γ MG stress dual-gate.
  Gate: GH200 after both parents; stated max‖Δσ‖; **perf_exempt**. Lever:
  extend unit/reg only.

- [~] `cuda-pw-smear-forces` | **P2** Lift restricted `smearing != none` reject on
  `pw_forces_analytic` (energy smear landed; forces still throw). Track:
  **cuda**. Twin: CPU forces under metallic occ + dual-gate. Gate: GH200.
- [ ] `cuda-pw-smear-stress` | **P2** Lift restricted smear reject on
  `pw_stress`. Track: **cuda**. Twin: CPU `pw_stress` + dual-gate. Gate: GH200.

**Collinear spin axis** (LSDA + **spin-PBE energy** landed; derivatives open):


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
- [ ] `cuda-pw-kmesh-pbe` | **P2** **new (wave 11)** — Closed-shell multi-k
  **PBE energy dual-gate** (`pw-cuda-lda-si` multi-k is **LSDA-class**;
  `cuda-pw-pbe-device` dual-gate is Γ/small-cell — foothold path accepts
  `k_mesh>1`∧PBE but production multi-k PBE energy ungated; parent for
  `kmesh-pbe-forces` / `kmesh-pbe-stress`). Track: **cuda**. Twin: CPU
  multi-k closed-shell PBE energy dual-gate (Si class). Gate: GH200 +
  stated energy tol; **perf_exempt** small cell; not F/σ/spin/smear.
  Lever: unit/reg under `k_mesh>1` + `XCType::PBE` + existing `run()` path —
  no new kernel.
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
- [ ] `cuda-pw-smear-pbe` | **P2** **new (wave 8)** — Restricted metallic
  smear free energy under **PBE** dual-gate (landed `metallic-smear` reg is
  **LSDA-class Al FD** — foothold path accepts PBE but production dual-gate
  missing). Track: **cuda**. Twin: CPU restricted smear PBE free energy +
  dual-gate. Gate: GH200 + stated tol; **perf_exempt**; not spin/smear-F.
  Lever: unit/reg under `XCType::PBE` + smear; reuses device PBE XC + smear
  band path — no new kernel.
- [ ] `cuda-pw-smear-pbe-forces` | **P3** **new (wave 8)** — Analytic ionic
  forces under restricted smear **+ PBE** after `smear-pbe` energy +
  `smear-forces` (LSDA smear F alone leaves GGA joint open). Track:
  **cuda**. Twin: CPU restricted smear PBE forces dual-gate. Gate: GH200.
  Lever: `pw_forces_analytic` joint gate + fractional occ + PBE XC.
- [ ] `cuda-pw-smear-pbe-stress` | **P3** **new (wave 8)** — Cell stress
  under restricted smear **+ PBE** after `smear-pbe` + `smear-stress`.
  Track: **cuda**. Twin: CPU restricted smear PBE `pw_stress` dual-gate.
  Gate: GH200. Lever: `pw_stress` joint gate + Mermin terms + PBE XC on
  NSCF strain-FD.
- [ ] `cuda-pw-kmesh-pbe-forces` | **P3** **new (wave 8)** — Closed-shell
  multi-k analytic **PBE** forces after `kmesh-forces` + landed PBE force
  path (LSDA multi-k F alone leaves GGA multi-k F open). Track: **cuda**.
  Twin: CPU multi-k closed-shell PBE forces dual-gate. Gate: GH200. Lever:
  unit/reg under `k_mesh>1` + `XCType::PBE` + existing force path.
- [ ] `cuda-pw-kmesh-pbe-stress` | **P3** **new (wave 8)** — Closed-shell
  multi-k **PBE** cell stress after `kmesh-stress` + PBE stress path.
  Track: **cuda**. Twin: CPU multi-k closed-shell PBE stress dual-gate.
  Gate: GH200. Lever: unit/reg under `k_mesh>1` + PBE stress path.
- [ ] `cuda-pw-spin-kmesh-pbe-forces` | **P3** **new (wave 8)** — Analytic
  forces under multi-k collinear **spin-PBE** after `spin-kmesh-pbe` +
  `spin-kmesh-forces` / `spin-pbe-forces`. Track: **cuda**. Twin: CPU
  multi-k spin-PBE forces dual-gate. Gate: GH200. Lever: `pw_forces_analytic`
  triple joint + device spin-PBE XC.
- [ ] `cuda-pw-spin-kmesh-pbe-stress` | **P3** **new (wave 8)** — Cell
  stress under multi-k collinear **spin-PBE** after `spin-kmesh-pbe` +
  `spin-kmesh-stress` / `spin-pbe-stress`. Track: **cuda**. Twin: CPU
  multi-k spin-PBE stress dual-gate. Gate: GH200. Lever: `pw_stress` triple
  joint + spin-PBE XC on NSCF strain-FD.
- [ ] `cuda-pw-spin-kmesh-smear` | **P3** **new (wave 3)** — Multi-k collinear
  spin **+** metallic smear free energy (pairwise `spin-kmesh` + `spin-smear`
  leave the triple open; base `run()` still throws spin∧smear). Track:
  **cuda**. Twin: CPU multi-k UKS+smear free energy class
  (`reg_pw-uks-fe-smearing` / k-mesh class). Gate: GH200 after both parents;
  stated tol; **perf_exempt**; not F/σ/PBE. Lever: lift joint gate on
  `pw_solver` + fractional occ on multi-k spin density path.
- [ ] `cuda-pw-spin-kmesh-smear-forces` | **P3** **new (wave 3)** — Analytic
  ionic forces under multi-k collinear spin + metallic smear after
  `spin-kmesh-smear` energy (spin-smear F + spin-kmesh F alone leave triple F
  open). Track: **cuda**. Twin: CPU multi-k UKS+smear forces dual-gate. Gate:
  GH200 after triple energy. Lever: `pw_forces_analytic` triple gate +
  fractional occ force assembly.
- [ ] `cuda-pw-spin-kmesh-smear-stress` | **P3** **new (wave 3)** — Cell stress
  under multi-k collinear spin + metallic smear after `spin-kmesh-smear`
  energy (same triple joint gap as forces). Track: **cuda**. Twin: CPU
  multi-k UKS+smear `pw_stress` dual-gate. Gate: GH200 after triple energy.
  Lever: `pw_stress` triple gate + Mermin free-energy stress terms.
- [ ] `cuda-pw-spin-kmesh-smear-pbe` | **P3** **new (wave 3)** — Multi-k
  collinear spin + metallic smear free energy under **PBE** after
  `spin-kmesh-smear` + `spin-smear-pbe` (LSDA triple alone leaves GGA joint
  open). Track: **cuda**. Twin: CPU multi-k UKS+smear PBE free energy
  dual-gate. Gate: GH200 after parents; **perf_exempt**; not F/σ. Lever:
  `pw_solver` joint gate + device spin-PBE XC already landed for fixed-occ.
- [ ] `cuda-pw-spin-kmesh-smear-pbe-forces` | **P3** **new (wave 3)** —
  Analytic forces under multi-k collinear spin + smear + **PBE** after
  `spin-kmesh-smear-pbe` energy. Track: **cuda**. Twin: CPU multi-k UKS+smear
  PBE forces dual-gate. Gate: GH200. Lever: `pw_forces_analytic` four-way
  joint + device spin-PBE XC.
- [ ] `cuda-pw-spin-kmesh-smear-pbe-stress` | **P3** **new (wave 3)** — Cell
  stress under multi-k collinear spin + smear + **PBE** after
  `spin-kmesh-smear-pbe` energy. Track: **cuda**. Twin: CPU multi-k
  UKS+smear PBE `pw_stress` dual-gate. Gate: GH200. Lever: `pw_stress`
  four-way joint + Mermin terms + spin-PBE XC on NSCF strain-FD.

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

**Tree fact on `dev` @ `31405991`:** stage units + PW landings + public
`cuda::gpw::GpwDft` + device mid-SCF collocate/integrate + multigrid energy
+ Γ UKS PADE energy (+ O₂ gate) + device grid forces + **product stress** +
**product non-default stream** + **device mid-SCF RKS DIIS/eigen** +
**water Γ RKS E+F dual-gate** + **multi-k RKS DIIS energy (H₂ foothold)** +
closed-shell PW PBE stress + collinear spin-PBE energy + **collinear LSDA
spin forces** **on base**. No feat land on base after opt-3. `gpw_solver.cu`:
device KS chain + MG on owned stream + **`AoDiisEigenDevice`** (RKS only);
host weight pack + D2H acc scatter + `build_STVpp`; UKS host dual `GpwDiis`;
multi-k branch host χ^k collocate/integrate + H2D ρ + `run_from_rho` + host
`KDiis` + phased STVnl; free-fn forces hybrid (host AO Pulay + device grid
HF; **default-stream blocking packs** + **H2D re-upload** after SCF
store_grids); stress free-fn = **full host** fixed-\(P\) (reject multi-k
only); `run_uks` Padé LSD only; multi-k UKS / multi-k MG / UKS+MG / free-fn
UKS F / multi-k F / MG F reject. PW: spin∧PBE energy OK; **spin LSDA forces
OK**; stress reject spin/smear; smear forces reject on base; spin dual-gates
Γ-only; closed-shell F dual-gate `k_mesh=1`; multi-k PBE energy dual-gate
missing; spin×kmesh×smear joint open. **Staff `[~]` Assign wave 6 (feat
tips, unmerged):** `uks-device-diis` **1130279** + `device-weight-pack`
**1130344** + `smear-forces` **1130359**.

| Slice on `dev` | Status | Depth gap |
| --- | --- | --- |
| RHF s/p energy + forces + level_shift + path-select + device CDIIS + **non-default stream D2H** | **landed** | higher-L host-fill remains (out of product scope) |
| Γ GPW stage units (Poisson/FFT + Padé LDA + product collocate/integrate + `KsChainDevice` LDA/PBE + `KsMultiGridChainDevice`) | **landed** | stage ≠ full product depth |
| Public `cuda::gpw::GpwDft` + device collocate + MG energy + Γ UKS + device forces + stress + non-default stream + **device RKS DIIS/eigen** + **water E+F dual-gate** + **multi-k energy (H₂)** (Γ RKS DIIS PADE/PBE; host STVnl + weight pack; hybrid F/σ free-fns; multi-k host residual spine) | **landed** (through opt-3: h2o **1130195**, kpoints **1130141**, spin-forces **1130196**; diis-eigen **1129719**) | **pending:** **`kpoint-device-*` + `kpoint-si/h2o-gate` + water/Si multi-k F/σ + multi-k×PBE production**, **`multigrid-si/h2o-gate` + water/Si MG F/σ + Si MG×PBE**, **`uks-device-diis` + `device-weight-pack` + `device-product-residency`**, **`uks-o2-forces/stress` + `uks-o2-pbe-gate` + O₂ UKS-PBE F/σ**, **force-stream + handoffs + AO-Pulay + device-stress**, **water σ + water×PBE E+F/σ**, **PBE F/σ + MG/UKS F/σ + joints + 4-way**, **device STVnl** |
| `cuda::pw::PW` LSDA + PBE + residency + stream + large-cell + forces (LSDA+PBE Γ) + stress (LSDA+PBE) + collinear LSDA + spin-PBE energy + **spin LSDA forces** + restricted metallic smear energy | **landed** | **pending:** multi-k F/σ + **multi-k PBE energy** (+ multi-k PBE F/σ); smear F/σ (tip **1130359**) + smear-PBE; spin-stress + spin-PBE F/σ / spin multi-k (+ spin-kmesh-PBE F/σ) / spin+smear (+ spin-smear-pbe F/σ) / **spin×kmesh×smear** (+ PBE F/σ) |

**Staff order:** (1) Merge Assign wave 6 P2 product lifts — `uks-device-diis`
+ `device-weight-pack` + PW `smear-forces` (staffed `[~]` **feat tips**);
(2) remaining **P2** PW (`smear-stress` / `smear-pbe` / `spin-stress` /
`spin-smear` / `spin-kmesh` / `kmesh-forces` / `kmesh-stress` /
**`kmesh-pbe`**); (3) multi-k device spine + production energy gates
(`kpoint-si/h2o-gate`, `multigrid-si/h2o-gate`) then **production multi-k/MG
F/σ** + **production multi-k/MG×PBE** + **water×PBE F/σ** +
**O₂ UKS-PBE F/σ** + hybrid free-fn depth + joints + **4-way** + PW
spin/smear/multi-k + **spin×kmesh×smear** **P3** after parent energy rows.
Do **not** staff lone new GPW kernel footholds. Higher-L RHF stays
product-out-of-scope. OT / vdW / MT / metallic-GPW stay later same-capability
follow-on.

Open gate: CPU twin green + public `run`/`forces`/`stress` stable where owned
+ stated CUDA↔CPU tols + device-only kernels + **GH200 log evidence**.

**Assign wave 6 (parent `dev` @ `d03327a8`):** capacity **3/3** free after
opt-3 land → **NEW×3** (theme: real product spine lifts, not dual-gate
matrix). Staff: `cuda-gpw-uks-device-diis`, `cuda-gpw-device-weight-pack`,
`cuda-pw-smear-forces`. Held next free-slot: multi-k device collocate /
integrate, spin-stress, product-h2o-forces. Do **not** re-staff landed
collocate/MG/UKS/PBE/forces/stress/stream/DIIS-eigen/h2o/kpoints/spin-forces.

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
**`cuda-gpw-device-collocate`**, **`cuda-gpw-multigrid-run`**,
**`cuda-gpw-uks-gamma`**, **`cuda-gpw-device-forces`**,
**`cuda-gpw-pbe-product-gate`**, **`cuda-gpw-uks-o2-gate`**,
**`cuda-gpw-stress`**, **`cuda-gpw-nondefault-stream`**,
**`cuda-gpw-device-diis-eigen`**, **`cuda-gpw-product-h2o-gate`**, **`cuda-gpw-kpoints`**, **`cuda-pw-spin-forces`**.

## in_progress

Assign wave 6 (parent `dev` @ `d03327a8`; still live after GapCensus wave 8)
— capacity 3/3; **feat tips green, unmerged** (Merge owns land):

| slug | track | tip branch | tip SHA / GH200 |
| --- | --- | --- | --- |
| `cuda-gpw-uks-device-diis` | cuda | `wf/cuda-gpw-uks-device-diis` | `549d2abe` / **1130279** \|dE\|=**4.760e-6** Ha |
| `cuda-gpw-device-weight-pack` | cuda | `wf/cuda-gpw-device-weight-pack` | `94256cf0` / **1130344** \|dE\|=**1.616e-7** Ha |
| `cuda-pw-smear-forces` | cuda | `wf/cuda-pw-smear-forces` | `bafa9c3f` / **1130359** max\|ΔF\|=**1.36e-11** eV/Å |

Rows marked `[~]` in `## pending` above; do **not** re-assign.



## blocked

- [!] `global-xtensor-xsimd` | never enable global `XTENSOR_USE_XSIMD`
- [!] `mg-main-structural` | MG MAIN **2.33×** residual after pin-safe collint + structure + sphere campaigns; further cut needs non-pin-safe windows or task-list rewrite beyond this loop’s pin law — do not staff as soft-tol or self-golden
  (arch-campaign landed single-owner multi-level path `arch-mg-main-structural`;
  fair-wall ~1× is still a **perf** residual, not a soft-tol staff)

## done

GapCensus wave 7 close (no feat land — dual-gate already met):

- [x] `cuda-gpw-product-h2o-forces` | **P3** Water Γ RKS PADE free-fn force dual-gate — **met by** `product-h2o-gate` reg (GH200 **1130195** max\|dF\|=**3.2e-6** Ha/Bohr; E+F in one gate). Closed wave 7; residual water **σ** stays `product-h2o-stress`; hybrid force depth (stream / AO-Pulay / handoff) stays separate rows

Post-opt-3 land 2026-08-11 (`dev` after tip merges):

- [x] `cuda-gpw-product-h2o-gate` | **P1** Water Γ RKS PADE production dual-gate (**E+F**) | GH200 **1130195** / **1130140** \|dE\|=**7.5e-7** max\|dF\|=**3.2e-6** | FF `30bb49b4` / feat `9124615d`
- [x] `cuda-gpw-kpoints` | **P2** Multi-k RKS DIIS energy (H₂ k=2×1×1) | GH200 **1130141** \|dE\|=**3.375e-14** | merge `adc7c7d7` / feat `b2778a83` | residual host integrate/DIIS/STVnl
- [x] `cuda-pw-spin-forces` | **P2** Collinear LSDA spin analytic ionic forces | GH200 **1130196** / **1130145** max\|ΔF\|=**9.25e-13** eV/Å | merge spin-forces / feat `44995ff6`


Harness wave 2 land 2026-08-11 (`dev` @ `e164b767`):

- [x] `cuda-gpw-device-diis-eigen` | **P2** Device mid-SCF Pulay DIIS + cuSOLVER dense eigen on RKS AO (`AoDiisEigenDevice`) | track cuda | twin: H₂ Γ RKS PADE energy dual-gate + `reg_cuda-gpw-device-diis-eigen` + unit `test_cuda_gpw_diis_eigen` | files: `ao_diis_eigen.{cu,cuh}`, `gpw_solver.{cu,cuh}`, unit, reg, REFERENCE | merge: `e164b767` / feat `e8170b10` + re-verify `5ed40df9` | evidence: GH200 **1129719** \|dE\|=**1.616e-07** Ha (tol 5e-6); prior tip **1129285**; **perf_exempt** small cell; residual host weight pack + scatter + STVnl; UKS host dual GpwDiis |

Post-wave-9 land 2026-08-11 (`dev` @ wave-9 merges):

- [x] `cuda-gpw-stress` | **P2** Public `cuda::gpw::gpw_stress` free-fn (CUDA Γ density + host fixed-P analytic) | GH200 **1128195** \|dE\|=**1.617e-07** Ha max\|dσ\|=**2.242e-10** Ha/Bohr³ | merge stress + feat `d58e9815`
- [x] `cuda-gpw-nondefault-stream` | **P2** Product SCF owns non-default stream for mid-SCF + KS | GH200 **1128196** \|dE\|=**1.617e-07** max\|dF\|=**2.329e-06** Ha/Bohr | merge stream + feat `5018ed7c`

*no-land wave 9:* `cuda-gpw-product-h2o-gate` — CUDA water RKS SCF non-convergence on device path (GH200 **1128194**/**1128217**/**1128269**/**1128295**); tip dropped then recovered on Assign wave 2 (tip green **1129720**; still unmerged).

Post-wave-8 land 2026-08-10/11 (`dev` @ `4f78d716`):

- [x] `cuda-gpw-device-forces` | **P1** Device grid force product on free-fn forces (host AO Pulay + device HF grid) | GH200 **1127167** max\|ΔF\|=**2.056e-6** Ha/Bohr \|ΔE\|=**1.617e-7** Ha | FF `81b74dec` / feat `d6eee397`
- [x] `cuda-gpw-pbe-product-gate` | **P1** H₂ Γ RKS **PBE** e2e dual-gate on public GpwDft | GH200 **1127166** \|dE\|=**1.658e-7** max\|dF\|=**5.232e-7** | merge `d38eb0e5` / feat `37c663c2`
- [x] `cuda-gpw-uks-o2-gate` | **P2** O₂ Γ UKS PADE production dual-gate after H₂ foothold | GH200 **1127173** \|dE\|=**3.52e-6** Ha | merge `4f78d716` / feat `dbb07ad3`


Harness wave 7 land 2026-08-10 (`dev` @ `f679807c`):

- [x] `cuda-gpw-multigrid-run` | **P1** Lift `ngrids!=1` reject on public
  `GpwDft::run` — wire landed `KsMultiGridChainDevice` into product energy
  | track cuda | twin: H₂ Γ RKS PADE ngrids=4 rel_cutoff=40 dual-gate +
  `reg_cuda-gpw-multigrid-run` | files: `gpw_solver.{cuh,cu}`, unit, reg,
  REFERENCE | feat: `2d780cff` / `e201a6cb` | evidence: GH200 **1126433**
  \|dE\|=**1.587e-7** Ha (tol 5e-6); prior **1125894**; units full_product +
  ks_multigrid PASS; **perf_exempt** small cell; residual MG F/σ + MG×PBE /
  multi-k MG → child rows |
- [x] `cuda-gpw-uks-gamma` | **P1** Lift collinear UKS reject — Γ UKS DIIS
  **energy** (Padé LSD; `KsChainDevice::run_uks`) | track cuda | twin:
  high-spin H₂ Γ UKS PADE dual-gate + `reg_cuda-gpw-uks-gamma` | files:
  `gpw_solver.{cuh,cu}`, `ks_chain.{cu,cuh}`, unit, reg, REFERENCE | feat:
  `f679807c` / `254ff67c` | evidence: GH200 **1126430** \|ΔE\|=**4.760e-6**
  Ha (tol 5e-6) ⟨S²⟩=2.0 both; prior **1125905**; unit full_product PASS;
  **perf_exempt** small cell; residual UKS F/σ + UKS-PBE + joints → child
  rows |

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
