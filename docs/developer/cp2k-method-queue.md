# CP2K capability queue

Durable worklist for workflow `cp2k-method-loop`. Do not delete done rows.

**Recover first:** every shift starts by finishing `## in_progress` `[~]`, open
`.claude/specs/` (GPW/PW/periodic), and unmerged `wf/<slug>` branches. Census
only rewrites **pending**. Assign priority: **incomplete ≫ pending**. Spec is
source of truth for workers.

Census rewrites **pending** from `docs/roadmap/dft.md` §10–12 (periodic / GPW /
PW / response / infra) and related MD ○ footholds that **CP2K can oracle
offline** (Quickstep GPW, plane-wave DFT, forces/stress, OT, D3, BOMD, phonons).
**done** / **blocked** / **in_progress** history is preserved (Census must not
erase incomplete).

Out of scope for this loop (use `pyscf-method-queue.md`): molecular Gaussian
post-HF, pure PySCF continuum/charge methods without CP2K twin, classical MD/FF,
CUDA-Kokkos product tracks.

Census: 2026-07-24 (parent `dev`). Source: `docs/roadmap/dft.md` §10–12 +
§3/§8 GGA/dispersion ○, `docs/roadmap/index.md` GPW/PW caveats,
`docs/roadmap/molecular-dynamics.md` §AIMD ○, `docs/roadmap/optimization.md`
variable-cell productization.
**Harness wave 1 2026-07-23:** 10 staff-batch slugs merged → `## done`.
**Assign wave 2 2026-07-23:** 6 incomplete resumes + 4 dep-safe pending →
`## in_progress` (staff 10; includes `gpw-kpoints-si`).
**Harness wave 2 2026-07-24:** 9 merged slugs → `## done` (stress, BOMD, PW
analytic stress, FD phonons, DFT+U, bands/DOS, k-UKS, PBEsol, D4). Incomplete
remaining: **`gpw-kpoints-si`** only (not in harness wave-2 staff list; no
merge/reg).
**Assign wave 3 2026-07-24:** 1 incomplete resume (`gpw-kpoints-si`) + 9
dep-safe pending → `## in_progress` (staff 10).
**Recover 2026-07-24:** incomplete_count=**10** (all wave-3 `[~]`); open
CP2K/GPW/PW specs=**0**; unmerged CP2K `wf/*` ahead of `dev`=**0**.
**Census 2026-07-24:** rewrote **pending only** from remaining roadmap ○ with
offline CP2K twins; **excluded** every slug already in `## in_progress` /
`## done` / `## blocked` (no demote of incomplete; no promote into staff).
Excluded also molecular post-HF, pure PySCF, classical MD/FF, CUDA-Kokkos,
GAPW/PAW/periodic hybrid/meta-GGA. Assign must finish incomplete first.
**Census 2026-07-25:** no CP2K/GPW/PW merges into `dev` since wave-2 harness
(only metalwalls waves + molecular scf-review chain — out of scope); all 11
prior pending re-verified as true gaps on `dev` (GPW `XcKind` still
PADE|PBE|PBEsol, GPW stress still strain-FD, no TS/MBD/vdW-DF, no
DFPT/dispersions, no periodic MP2/RPA/GW); wave-3 `[~]` ×10 preserved (no
regressions/specs/`wf/*` for any — still incomplete). Added 2 slugs from
remaining §3/§8 ○ with offline CP2K twins: `gpw-rpbe-water`, `gpw-vdw-df-ice`.
Open specs `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp` are **PySCF-queue**
deferred items (per `.claude/specs/INDEX.md`) — not this loop's. Roadmap
already matches `dev` — no ○/✓ flips.
**Assign wave (claude) 2026-07-25:** staffed 6 of the 10 wave-3 `[~]` resumes
(no pending promoted; all picks already in_progress): `gpw-kpoints-si`
(oldest resume, wave-2 leftover), `gpw-variable-cell-ice`, `dfpt-phonon-si`,
`xl-bomd-water`, `berry-polarization`, `periodic-mp2-si`. Held `[~]` for next
wave: `wannier-centers` / `cpmd-water` (surface overlap with
`berry-polarization` / `xl-bomd-water` picks — k-path and AIMD propagator
files), `mpi-gpw-scaling-4rank` / `mpi-pw-scaling-4rank` (infra; contend with
`gpw-kpoints-si` / `periodic-mp2-si` solver trees). No pick depends on
another pick. All picks are write-first: no open spec / no `wf/*` / no
regression exists for any.
**Harness wave 1 (claude) 2026-07-25:** 5 merged → done (k-points GPW,
variable-cell, DFPT Γ, Berry polarization, periodic MP2 foothold);
xl-bomd-water WIP-checkpointed; 71 merged wf/* pruned.
**Census 2026-07-25 (post harness wave 1):** no CP2K/GPW/PW merges since
`859d6e28` (HEAD); all 13 prior pending re-verified as true gaps on `dev`
(GPW `XcKind` still PADE|PBE|PBEsol; `gpw_stress` analytic fixed-P; no
TS/MBD/vdW-DF; DFPT q = 0 only; no Wannier/CPMD/RPA/GW; no PW variable-cell
or Fe-smearing regressions). Wave-3 `[~]` ×5 preserved (xl-bomd-water WIP
`61d235f4` untested; others no spec/`wf/*`/regression). Added 2 slugs for
declared-but-unqueued follow-ups: `periodic-ri-mp2-si` (foothold ≠ capability
closed — §10 Periodic MP2 stays ○ until RI/k-point lands) and
`gpw-variable-cell-molopt` (the "MOLOPT re-twin queued" claim in the
`gpw-variable-cell-ice` done row now has a real queue row). §3 BP86/B97/OLYP/
HCTH/SOGGA remain molecular-path (PySCF-loop oracle) — not added. Roadmap
already matches `dev` (harness `859d6e28` synced Berry/DFPT/k-points/
variable-cell; MP2 ○ kept) — no ○/✓ flips.
**Assign wave (claude) 2026-07-25 (wave 2):** staffed 6 dep-safe picks — all
5 wave-3 `[~]` resumes (`xl-bomd-water` = WIP `61d235f4` on
`wf/xl-bomd-water`, resume build+test first; `wannier-centers` / `cpmd-water`
now free of the prior surface holds — `berry-polarization` merged, xl-bomd
impl checkpointed; `mpi-gpw-scaling-4rank` / `mpi-pw-scaling-4rank` — the
contending `gpw-kpoints-si` / `periodic-mp2-si` solver trees merged) + 1
pending promoted: `gpw-analytic-stress` (dep `gpw-stress-water` ✓).
Skipped `second-gen-cpmd-water` (dep `cpmd-water` is itself a pick this
wave). Open specs `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp` stay
PySCF-loop deferred — not staffed. Note: `cpmd-water` and `xl-bomd-water`
may both touch AIMD propagator files — harness resolves at merge.

**Bootstrap + GapCensus 2026-07-25 (parent `dev` @ `5bdae27d`):** recover
clean — incomplete=`0`, open CP2K/GPW/PW specs=`0` (only PySCF deferred
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`), unmerged `wf/*`=`0`, isolation
worktrees=`0`, leftover `build*`=`none`. All 14 prior pending re-verified as
true gaps on tree: GPW `XcKind` still `{PADE,PBE,PBEsol}` (no BLYP/revPBE/RPBE);
no TS/MBD/vdW-DF symbols; DFPT remains Γ-only (`phonon-dispersion-si` open);
no bulk-Fe + spin+smearing reg; no PW CELL_OPT product reg; Periodic MP2
capability still ○ (Γ foothold `periodic-mp2-si` ✓ — RI/k-point + RPA/GW
pending); second-gen CP after CPMD ✓. All 30 done-slug regression dirs present
under `regressions/{gpw,pw,response,dynamics,scf}/`. Roadmap matches `dev`
(no ○/✓ flips). **Pending unchanged (14).** Ready for Assign wave — prefer
surface-disjoint staff (one XcKind GGA slug/wave; TS before MBD/vdW-DF;
`second-gen-cpmd-water` free now that `cpmd-water` ✓).
**Recover 2026-07-25 (parent `dev` @ `f6838676`):** reconfirmed clean —
incomplete_count=**0**; `## in_progress` empty (no `[~]`); open CP2K/GPW/PW
specs=**0** (PySCF deferred only); unmerged `wf/*`=**0**; isolation
worktrees=**0**; leftover builds=**none**; all 30 done-slug regression dirs
present; no half-regs for any of the 14 pending. Do **not** invent methods —
next is GapCensus/Assign from **pending**.

**GapCensus wave 1 2026-07-25 (parent `dev` @ `f6838676`):** recover clean
(incomplete=`0`, CP2K specs=`0`, `wf/*`=`0`, builds=`none`). Tree re-diff vs
`docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: all **14** prior pending still true
gaps (no half-regs; `XcKind` still `{PADE,PBE,PBEsol}`; no TS/MBD/vdW-DF;
DFPT Γ-only; Periodic MP2 capability still ○; no Fe-smearing / PW CELL_OPT
regs). **Added 4** tree-declared follow-ons with offline CP2K twins:
`gpw-uks-pbe`, `gpw-kpoints-uks`, `gpw-kpoint-forces` (from `gpw_solver.h`
scope: "UKS PBE-GGA, k-point UKS, and analytic k-point forces are follow-on"),
`periodic-wannier-si` (roadmap §11 MLWF residual after molecular Boys
foothold). **Not added:** BP86/B97/OLYP/HCTH (molecular PySCF-oracle path);
XDM/VV10 (hold behind TS/MBD/vdW-DF order); GAPW/PAW/periodic hybrid/meta-GGA
(blocked); molecular post-HF / classical MW / CUDA ports (other loops).
Roadmap: **no ○/✓ table flips** (rows already match tree); fixed stale
`docs/roadmap/index.md` GPW blurb (still said "no Gaussian k-points /
strain-FD stress" after k-points + analytic stress landed). **Pending = 18.**

**Harness wave 1 2026-07-25 (post GapCensus staff; parent `dev` @ `83a100df`):**
9 staffed slugs merged → `## done` (`gpw-blyp-water`, `gpw-uks-pbe`,
`pw-uks-fe-smearing`, `pw-variable-cell-si`, `gpw-ts-ice`,
`phonon-dispersion-si`, `periodic-wannier-si`, `second-gen-cpmd-water`,
`periodic-ri-mp2-si`). Specs deleted on close. **Pending residual = 9**
(revPBE/RPBE, k-UKS/k-forces, MBD/vdW-DF, MOLOPT re-twin, RPA/GW).
`## in_progress` empty. Roadmap flips: TS ✓, Periodic MP2 ✓ (Γ RI product;
multi-k residual), MLWF ✓ (molecular Boys + periodic PW MV), 2nd-gen CP ✓,
phonon dispersion productized (IFC Fourier; finite-q DFPT still ○), GPW BLYP
+ UKS-PBE + bulk Fe smear + PW CELL_OPT prose. **Queue not exhausted** —
GapCensus owns next discovery.

**GapCensus wave 2 2026-07-25 (parent `dev` @ `507bb5f8`):** recover —
incomplete_count was **0** in queue, but unmerged `wf/gpw-kpoint-forces`
(ahead 1: `d15a475d` feat analytic k-point forces + reg + oracle input;
merge-base `9ff2f9a8` pre-harness) → promote to **`## in_progress`**
(incomplete ≫ pending; do not leave half-landed `wf/*` as pending). Open
CP2K/GPW/PW specs on `dev` = **0** (PySCF deferred only); isolation
worktrees = **0**; leftover builds = **none**. Tree re-diff vs
`docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending (minus
k-forces) still true gaps — `XcKind` still `{PADE,PBE,PBEsol,BLYP}` (no
revPBE/RPBE); no MBD/vdW-DF symbols; DFPT remains **q = 0** only; multi-k
MP2 still throws; `gpw_stress` throws multi-k + UKS; force assembly is
closed-shell Γ RKS (UKS forces unpinned). **Added 4** tree-declared
follow-ons with offline CP2K twins: `dfpt-finite-q-si`,
`periodic-mp2-kpoints`, `gpw-kpoint-stress`, `gpw-uks-forces`. **Not
added:** BP86/B97/OLYP/HCTH (molecular PySCF path); XDM/VV10 (hold behind
MBD/vdW-DF order); GAPW/PAW/periodic hybrid/meta-GGA (blocked); molecular
post-HF / classical MW / CUDA ports (other loops). Roadmap: **no ○/✓ table
flips** (nothing new merged since harness); stale §10 intro + DFPT prose
refreshed. **Pending = 12**; **in_progress = 1**.

**Harness wave 2 2026-07-25 (parent `dev` @ `f5739506`):** 10 staffed
slugs merged → `## done` (`gpw-kpoint-forces`, `gpw-revpbe-si`,
`gpw-kpoints-uks`, `gpw-uks-forces`, `gpw-kpoint-stress`, `gpw-mbd-ice`,
`periodic-mp2-kpoints`, `periodic-rpa-si`, `dfpt-finite-q-si`,
`gpw-variable-cell-molopt`). Specs deleted on close. **Pending residual =
3** (`gpw-rpbe-water`, `gpw-vdw-df-ice`, `periodic-gw-si`).
`## in_progress` empty. Roadmap flips: revPBE ✓, free-atom MBD ✓, Periodic
RPA ✓ (Γ dRPA@KS product), finite-q DFPT productized; multi-k MP2 residual
closed; GPW k-UKS / k-forces / k-stress / Γ-UKS forces / MOLOPT CELL_OPT
prose. **Queue not exhausted** — GapCensus owns next discovery.

**GapCensus wave 3 2026-07-25 (parent `dev` @ `def040bd`):** recover
clean — incomplete_count=**0**; `## in_progress` empty; open CP2K/GPW/PW
specs=**0** (PySCF deferred only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`);
unmerged `wf/*`=**0**; isolation worktrees=**0**; leftover builds=**none**.
Tree re-diff vs `docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD +
`tools/cp2k_oracle` + `include/atomiverse/cpu/{gpw,planewave}`: prior
pending **3** still true gaps — `XcKind` still
`{PADE,PBE,PBEsol,RevPBE,BLYP}` (no RPBE); no vdW-DF / NON_LOCAL symbols;
no periodic GW. **Added 5** tree-declared / roadmap residuals with offline
CP2K twins: `gpw-kpoint-uks-forces` (forces throw “k-point UKS not
implemented”; header follow-on after Γ-UKS + k-RKS forces ✓),
`gpw-uks-stress` (analytic stress throws UKS at Γ and multi-k),
`periodic-rpa-kpoints` (multi-k RPA residual — `direct_rpa` throws nk>1;
Γ dRPA foothold ≠ multi-k capability), `gpw-mbd-hirshfeld` (roadmap §8
Hirshfeld volumes / density-scaled MBD after free-atom MBD ✓),
`dfpt-incommensurate-si` (finite-q product is commensurate supercell;
incommensurate Sternheimer still header follow-on). **Not added:**
BP86/B97/OLYP/HCTH (molecular PySCF path); XDM/VV10/vdW-DF2 (hold behind
vdW-DF order); GAPW/PAW/periodic hybrid/meta-GGA (blocked); molecular
post-HF / classical MW / CUDA ports (other loops). Roadmap: **no ○/✓
table flips** (nothing new merged); open-follow-on + forces/stress/RPA
prose refreshed; stale index Cell “GPW strain-FD” → analytic. **Pending
= 8**; **in_progress = 0**.

**Harness wave 3 2026-07-25 (parent `dev` @ `0a642122`):** 8 staffed
slugs merged → `## done` (`gpw-rpbe-water`, `gpw-kpoint-uks-forces`,
`gpw-uks-stress`, `gpw-vdw-df-ice`, `gpw-mbd-hirshfeld`,
`periodic-rpa-kpoints`, `periodic-gw-si`, `dfpt-incommensurate-si`).
Specs deleted on close. **Pending residual = 0** (staff batch emptied
the census pending list — **not** queue exhausted; GapCensus owns next
discovery). `## in_progress` empty. Roadmap flips: RPBE ✓, vdW-DF ✓
(DRSLL), Periodic GW ✓ (Γ spectral G0W0@PBE; multi-k residual),
Hirshfeld-scaled MBD/TS prose; multi-k dRPA residual closed; Γ-UKS
stress productized (multi-k UKS stress residual); multi-k UKS forces
productized; incommensurate DFPT productized. **Blocked = 5** unchanged.

**GapCensus wave 4 2026-07-26 (parent `dev` @ `decf917f`):** recover
clean — incomplete_count=**0**; `## in_progress` empty; open CP2K/GPW/PW
specs=**0** (PySCF deferred only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`);
unmerged `wf/*`=**0**; isolation worktrees=**0**; leftover builds=**none**.
Tree re-diff vs `docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD +
`tools/cp2k_oracle` + `include/atomiverse/cpu/{gpw,planewave}`: prior
pending residual **0** (harness emptied list — not capability-closed).
**True residuals re-opened (9)** with offline CP2K / literature twins:
`gpw-kpoint-uks-stress` (throws multi-k UKS; after Γ-UKS + k-RKS stress ✓),
`periodic-gw-kpoints` (Γ G0W0 foothold ≠ multi-k; `g0w0` throws nk>1),
`gpw-vdw-df2-ice` / `gpw-rvv10-ice` (§8 vdW-DF2 / rVV10 ○ after DRSLL ✓),
`gpw-mbd-periodic` (free-atom MBD ✓; periodic Ewald throw),
`pw-spin-gga` (PW `step_spin` σ=0 LSDA-limit; true collinear spin-GGA
deferred), `gpw-uks-blyp` (UKS spin-GGA wrappers only PBE/PBEsol; RKS
BLYP ✓), `periodic-rpa-finite-q` (k-diagonal multi-k dRPA ✓; full
q-resolved dielectric follow-on), `gpw-dftu-stress` (PLUS_U energy ✓;
stress throws DFT+U). **Not added:** BP86/B97/OLYP/HCTH (molecular
PySCF path); XDM (hold behind rVV10/vdW-DF2 product order — weaker stock
CP2K twin); shear/off-diagonal stress; OT-UKS; CPMD RATTLE; GPW metallic
smearing; UMP2/UGW/URPA; GAPW/PAW/periodic hybrid/meta-GGA (blocked);
molecular post-HF / classical MW / CUDA ports (other loops). Roadmap:
**no ○/✓ table flips** (nothing new merged since harness wave 3); §10/§8
open-residual prose + index Cell/GPW residual line refreshed to match
queue. **Pending = 9**; **in_progress = 0**; **blocked = 5**.

**Harness wave 4 2026-07-26 (parent `dev` @ `85eaa2d9`):** 9 staffed
slugs merged → `## done` (`gpw-kpoint-uks-stress`, `periodic-gw-kpoints`,
`gpw-vdw-df2-ice`, `gpw-rvv10-ice`, `gpw-mbd-periodic`, `pw-spin-gga`,
`gpw-uks-blyp`, `periodic-rpa-finite-q`, `gpw-dftu-stress`). Specs
deleted on close. **Pending residual = 0** (staff batch emptied the
census pending list — **not** queue exhausted; GapCensus owns next
discovery). `## in_progress` empty. Roadmap flips: vdW-DF2 ✓ (LMKLL),
rVV10 ✓; multi-k G0W0 residual closed; periodic Ewald MBD productized;
PW true collinear spin-GGA productized; Γ-UKS BLYP; multi-k UKS stress;
finite-q inter-k dRPA; DFT+U analytic stress. **Blocked = 5** unchanged.

**GapCensus wave 5 2026-07-26 (parent `dev` @ `4437d80f`):** recover
clean — incomplete_count=**0**; `## in_progress` empty; open CP2K/GPW/PW
specs=**0** (PySCF deferred only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`);
unmerged `wf/*`=**0**; isolation worktrees=**0**; leftover builds=**none**.
Tree re-diff vs `docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD +
`tools/cp2k_oracle` + `include/atomiverse/cpu/{gpw,planewave}`: prior
pending residual **0** (harness emptied list — not capability-closed).
**True residuals re-opened (9)** with offline CP2K / literature twins:
`gpw-dftu-kpoints` (`build` rejects k-mesh + `plus_u`; Γ DFT+U energy +
stress ✓ ≠ multi-k PLUS_U), `periodic-gw-finite-q` (multi-k G0W0 is
**k-diagonal** only; full inter-\(k\) / finite-\(q\) \(W\) header
follow-on after `periodic-gw-kpoints` ✓ / `periodic-rpa-finite-q` ✓),
`gpw-uks-revpbe` (UKS XcKind still PADE|PBE|PBEsol|BLYP; RKS RevPBE ✓),
`gpw-xdm-ice` (§8 XDM ○ — order hold behind rVV10/vdW-DF2 now cleared),
`gpw-vv10-ice` (stock VV10 ○, distinct from product rVV10),
`cpmd-rattle-water` (CPMD SHAKE-free foothold; RATTLE follow-on),
`gpw-metallic-smearing` (GPW fixed occupations only; PW metallic smear ✓),
`periodic-ump2-si` (`mp2` throws closed-shell; UMP2 follow-on),
`ot-uks-o2` (OT + UKS throws). **Not added:** BP86/B97/OLYP/HCTH
(molecular PySCF path); `gpw-uks-rpbe` (thin after revPBE UKS); shear /
off-diagonal stress; optB88-vdW (needs exchange + nonlocal combo);
UGW/URPA (hold behind UMP2); Hirshfeld×k-points; DFPT-GGA (DFPT LSDA
only); XL-BOMD UKS/k; nonlocal vdW SCF potential / forces; GAPW/PAW/
periodic hybrid/meta-GGA (blocked); molecular post-HF / classical MW /
CUDA ports (other loops). Roadmap: **no ○/✓ table flips** (nothing new
merged since harness wave 4); §10 open-residual prose + index Cell
stale “multi-k UKS residual” refreshed. **Pending = 9**; **in_progress
= 0**; **blocked = 5**.

**Harness wave 5 2026-07-26 (parent `dev` @ `2d43d1c2`):** 6 staffed
slugs merged → `## done` (`gpw-dftu-kpoints`, `periodic-gw-finite-q`,
`gpw-uks-revpbe`, `cpmd-rattle-water`, `periodic-ump2-si`, `ot-uks-o2`).
Specs deleted on close. **Pending residual = 1** (`gpw-xdm-ice`);
**in_progress = 2** unmerged staff `[~]` (`gpw-metallic-smearing`
ahead on `wf/gpw-metallic-smearing`; `gpw-vv10-ice` docs/block note on
`wf/gpw-vv10-ice` — CP2K stock VV10 oracle gap). `## incomplete` kept
as `[~]` (do not demote). Roadmap residual prose: multi-k DFT+U,
finite-q G0W0 \(W\), Γ-UKS revPBE, CPMD electronic RATTLE, UMP2,
OT-UKS productized; **no new ○→✓ table flips** (rows already ✓ or
still-open XDM/VV10). **Blocked = 5** unchanged. **Queue not
exhausted** — GapCensus owns next discovery.

**GapCensus wave 6 2026-07-26 (parent `dev` @ `5847a35a`):** recover —
incomplete_count=**2** (`[~] gpw-metallic-smearing` unmerged
`wf/gpw-metallic-smearing` @ `31b00478` feat FD/MP multi-k RKS + reg;
`[~] gpw-vv10-ice` unmerged `wf/gpw-vv10-ice` @ `0197163f` docs/block —
CP2K 2026.1 `TYPE VV10` enum reject, only DRSLL|LMKLL|RVV10; do not
demote). Open CP2K/GPW/PW specs on `dev`=**0** (PySCF deferred only:
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation worktrees=**0**;
leftover builds=**none**. Tree re-diff vs `docs/roadmap/dft.md` §3/§8/
§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending **1**
(`gpw-xdm-ice`) still true (no XDM symbols; §8 ○). **Added 8**
tree-declared follow-ons with offline CP2K / literature twins:
`gpw-uks-rpbe` (Γ/k UKS XcKind still PADE|PBE|PBEsol|RevPBE|BLYP — no
RPBE spin path / no `xc_grid_*_rpbe_spin`; RKS RPBE ✓),
`gpw-dftu-kpoint-stress` (multi-k DFT+U SCF ✓; `gpw_stress` k-path
throws `plus_u`), `periodic-ump2-kpoints` (Γ UMP2 ✓; multi-k UMP2
throws), `periodic-ugw-si` / `periodic-urpa-si` (closed-shell G0W0/dRPA
✓; UKS throws “UGW/URPA follow-on” after UMP2 foothold),
`gpw-hirshfeld-kpoints` (Hirshfeld TS/MBD Γ RKS ✓; k-points + UKS
throw), `dfpt-gga-si` (PW DFPT LSDA-only gate; GGA kernel residual),
`xl-bomd-uks` (XL-BOMD closed-shell RKS Γ only; UKS/`use_ot` throw).
**Not added:** BP86/B97/OLYP/HCTH (molecular PySCF path); shear /
off-diagonal stress (orthorhombic design); optB88-vdW (exchange +
nonlocal combo after XDM/rVV10 product depth); OT multi-k / OT+DFT+U
(build rejects); second-gen CP UKS; nonlocal vdW SCF potential /
forces (post-SCF E_disp product); multi-k UKS stress GGA expand
(RevPBE/BLYP — thin after PADE/PBE/PBEsol k-UKS stress ✓);
GAPW/PAW/periodic hybrid/meta-GGA (blocked); molecular post-HF /
classical MW / CUDA ports (other loops). Roadmap: **no ○/✓ table
flips** (nothing new merged since harness wave 5); §3/§8/§10–12 open-
residual prose + index GPW residual line refreshed for wave-6 queue.
**Pending = 9**; **in_progress = 2**; **blocked = 5**.

**Harness wave 6 2026-07-26 (parent `dev` @ `0869833a`):** 9 staffed
slugs merged → `## done` (`gpw-metallic-smearing`, `gpw-xdm-ice`,
`gpw-uks-rpbe`, `gpw-dftu-kpoint-stress`, `periodic-ump2-kpoints`,
`periodic-ugw-si`, `gpw-hirshfeld-kpoints`, `dfpt-gga-si`, `xl-bomd-uks`).
Specs deleted on close. **Pending residual = 1** (`periodic-urpa-si`);
**in_progress = 1** unmerged staff `[~]` (`gpw-vv10-ice` on
`wf/gpw-vv10-ice` — CP2K stock VV10 oracle gap; do not demote).
Roadmap: **XDM ✓** table flip; residual prose closed GPW metallic smear,
Γ-UKS RPBE, multi-k DFT+U stress, multi-k UMP2, UGW, multi-k Hirshfeld,
DFPT-GGA, XL-BOMD UKS; stock VV10 / URPA stay open. **Blocked = 5**
unchanged. **Queue not exhausted** — GapCensus owns next discovery.

**GapCensus wave 7 2026-07-26 (parent `dev` @ `9bdeddf1`):** recover —
incomplete_count=**1** (`[~] gpw-vv10-ice` unmerged `wf/gpw-vv10-ice` @
`9be3808a` docs/block — CP2K 2026.1 has no `TYPE VV10`; do not demote).
Open CP2K/GPW/PW specs on `dev`=**0** (PySCF deferred only:
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation worktrees=**0**;
leftover builds=**none**. Tree re-diff vs `docs/roadmap/dft.md` §3/§8/
§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending **1**
(`periodic-urpa-si`) still true (`direct_rpa` throws UKS — “URPA is a
follow-on”; closed-shell multi-k dRPA + Γ UGW ✓). **Added 8**
tree-declared follow-ons with offline CP2K / literature twins:
`periodic-ugw-kpoints` (Γ UGW ✓; `g0w0` throws multi-k UKS — “multi-k
UGW is a follow-on”), `gpw-uks-smearing` (multi-k RKS FD/MP ✓; build
rejects UKS+smear), `gpw-hirshfeld-uks` (Hirshfeld TS/MBD Γ+multi-k RKS
✓; UKS throws), `cpmd-uks` (CPMD + RATTLE RKS Γ ✓; evaluator rejects
UKS), `second-gen-cpmd-uks` (SGCP ASPC+partial SCF RKS ✓; UKS throw),
`ot-kpoints` (OT Γ RKS+UKS ✓; k-mesh rejects OT), `ot-dftu` (OT and
DFT+U each product; OT+PLUS_U throws on UKS OT path), `dfpt-spin-si`
(DFPT Γ/finite-q/incommensurate/GGA ✓; collinear spin unsupported).
**Not added:** BP86/B97/OLYP/HCTH (molecular PySCF path); stock VV10
(kept `[~]` incomplete — no demote; no CP2K twin); optB88-vdW (exchange
+ nonlocal combo); shear/off-diagonal stress (orthorhombic design);
XL-BOMD multi-k / OT-XL (hold behind OT-k + XL depth); nonlocal vdW SCF
potential / forces (post-SCF E_disp product depth); DFPT metallic smear
(hold behind spin DFPT); GAPW/PAW/periodic hybrid/meta-GGA (blocked);
molecular post-HF / classical MW / CUDA ports (other loops). Roadmap:
**no ○/✓ table flips** (nothing new merged since harness wave 6); §8/§10
open-residual prose + index GPW residual line refreshed for wave-7
queue. **Pending = 9**; **in_progress = 1**; **blocked = 5**.

**Harness wave 7 2026-07-26 (parent `dev` @ `8cb47664`):** 9 staffed
slugs merged → `## done` (`periodic-urpa-si`, `periodic-ugw-kpoints`,
`gpw-uks-smearing`, `gpw-hirshfeld-uks`, `cpmd-uks`, `second-gen-cpmd-uks`,
`ot-kpoints`, `ot-dftu`, `dfpt-spin-si`). Specs deleted on close.
**Pending residual = 0** (staff batch emptied the census pending list —
**not** queue exhausted; GapCensus owns next discovery).
**in_progress = 1** unmerged staff `[~]` (`gpw-vv10-ice` on
`wf/gpw-vv10-ice` — CP2K stock VV10 oracle gap; do not demote).
Roadmap: **no new ○→✓ table flips** (Periodic RPA/GW/DFPT/OT/smearing/
CPMD/SGCP rows already ✓); residual prose closed URPA@UKS, multi-k UGW,
GPW UKS smear, UKS Hirshfeld, CPMD/SGCP UKS, OT+k, OT+DFT+U, collinear-
spin DFPT; stock VV10 stays open. **Blocked = 5** unchanged. **Queue not
exhausted** — GapCensus owns next discovery.

**GapCensus wave 8 2026-07-26 (parent `dev` @ `9c57a290`):** recover —
incomplete_count=**1** (`[~] gpw-vv10-ice` unmerged `wf/gpw-vv10-ice` @
`d15cd804` docs/block — CP2K 2026.1 has no `TYPE VV10`; do not demote).
Open CP2K/GPW/PW specs on `dev`=**0** (PySCF deferred only:
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation worktrees=**0**;
leftover builds=**none**. Tree re-diff vs `docs/roadmap/dft.md` §3/§8/
§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending residual **0**
(harness emptied list — not capability-closed). **True residuals
re-opened (9)** with offline CP2K / literature twins:
`periodic-urpa-kpoints` (`direct_rpa` throws multi-k UKS — “multi-k URPA
is a follow-on”; Γ URPA + multi-k closed-shell dRPA ✓),
`ot-smearing` (OT and metallic smear each product; `build` throws
OT+smear — fixed integer OT occupations),
`ot-uks-kpoints` (Γ-UKS OT ✓ + multi-k RKS OT ✓; multi-k UKS OT reject),
`ot-dftu-kpoints` (Γ OT+DFT+U ✓ + multi-k OT / multi-k DFT+U DIIS ✓;
multi-k OT+PLUS_U not wired),
`gpw-d4-stress` (D4 energy ✓ `reg_gpw-d4-water`; analytic
`gpw_stress` throws D4 on Γ and multi-k),
`gpw-vdw-scf-potential` (vdW-DF/DF2/rVV10 post-SCF \(E_{\mathrm{disp}}\)
✓; nonlocal potential back-propagation / forces follow-on per
`vdw_df.h`),
`dfpt-metallic-si` (DFPT Γ/finite-q/incommensurate/GGA/spin ✓; metallic
smearing unsupported on Sternheimer path — hold cleared after spin DFPT),
`dfpt-spin-incommensurate` (collinear spin DFPT ✓ at Γ/commensurate;
incommensurate-\(q\) + spin throws),
`gpw-optb88-vdw` (§8 optB88-vdW ○ — exchange + nonlocal combo after
vdW-DF/rVV10 product depth). **Not added:** BP86/B97/OLYP/HCTH
(molecular PySCF path); stock VV10 (kept `[~]` incomplete — no demote;
no CP2K twin); shear/off-diagonal stress (orthorhombic design);
D4/MBD analytic forces (hold behind D4 stress / SCF-vdW order);
XL-BOMD multi-k / OT-XL / SGCP-OT corrector (hold behind OT+smear /
OT-k-UKS depth this wave); UKS multigrid product collocation (infra
multigrid ✓ single-grid product; O(mHa) residual vs CP2K NGRIDS);
spin-polarized / metallic DOS foothold; UGW@UKS-RPBE XC throw (thin
after UKS RPBE SCF); GAPW/PAW/periodic hybrid/meta-GGA (blocked);
molecular post-HF / classical MW / CUDA ports (other loops). Roadmap:
**no ○/✓ table flips** (nothing new merged since harness wave 7); §8/
§10–12 open-residual prose + index GPW residual line refreshed for
wave-8 queue. **Pending = 9**; **in_progress = 1**; **blocked = 5**.

**Harness wave 8 2026-07-26 (parent `dev` @ `a6752916`):** 9 staffed
slugs merged → `## done` (`periodic-urpa-kpoints`, `ot-smearing`,
`ot-uks-kpoints`, `ot-dftu-kpoints`, `gpw-d4-stress`,
`gpw-vdw-scf-potential`, `dfpt-metallic-si`, `dfpt-spin-incommensurate`,
`gpw-optb88-vdw`). Specs deleted on close. **Pending residual = 0**
(staff batch emptied the census pending list — **not** queue exhausted;
GapCensus owns next discovery). **in_progress = 1** unmerged staff
`[~]` (`gpw-vv10-ice` on `wf/gpw-vv10-ice` — CP2K stock VV10 oracle gap;
do not demote). Roadmap: **optB88-vdW ✓**; residual prose closed multi-k
URPA, OT+smear, multi-k UKS OT, multi-k OT+DFT+U, D4 analytic stress,
nonlocal vdW SCF potential/forces, metallic DFPT, collinear-spin
incommensurate DFPT; stock VV10 stays open. **Blocked = 5** unchanged.
**Queue not exhausted** — GapCensus owns next discovery.

**GapCensus wave 9 2026-07-26 (parent `dev` @ `07dd7bf3`):** recover —
incomplete_count=**1** (`[~] gpw-vv10-ice` unmerged `wf/gpw-vv10-ice` @
`b8a2e6ca` docs/block — CP2K 2026.1 has no `TYPE VV10`; do not demote).
Open CP2K/GPW/PW specs on `dev`=**0** (PySCF deferred only:
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation worktrees=**0**;
leftover builds=**none**. Tree re-diff vs `docs/roadmap/dft.md` §3/§8/
§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending residual **0**
(harness emptied list — not capability-closed). **True residuals
re-opened (9)** with offline CP2K / literature twins:
`ot-uks-smearing` (RKS OT+smear ✓ + multi-k UKS OT ✓; `build` /
k-UKS OT path throws UKS OT+smear — integer occupations only),
`ot-uks-dftu` (Γ OT+DFT+U ✓ + multi-k RKS OT+DFT+U ✓ + multi-k UKS OT ✓;
multi-k UKS OT + PLUS_U throws “not wired”),
`ot-dftu-smearing` (OT+smear ✓ + multi-k OT+DFT+U ✓; multi-k OT path
throws smear+PLUS_U together),
`gpw-d4-forces` (D4 energy ✓ + analytic stress ✓; Γ forces skip D4 —
energy-only policy; `dftd4_energy` has no force twin),
`gpw-mbd-forces` (MBD energy ✓ + Γ/periodic stress ✓; forces energy-only
v1 per `mbd_dispersion.h`),
`gpw-uks-multigrid` (RKS multigrid product collocation ✓ when `ngrids>1`;
UKS throws “multigrid product collocation is not wired”),
`dfpt-spin-smearing` (spin DFPT ✓ + metallic DFPT ✓ separately; collinear
spin + smear rejected on Γ Sternheimer path),
`dfpt-finite-q-smearing` (finite-q DFPT ✓ fixed-occ + Γ metallic DFPT ✓;
finite-q + smear throws — “use Γ restricted smear”),
`second-gen-cpmd-ot` (SGCP ASPC+partial SCF RKS/UKS ✓; OT corrector
throws — diagonalization SCF required). **Not added:** BP86/B97/OLYP/
HCTH (molecular PySCF path); stock VV10 (kept `[~]` incomplete — no
demote; no CP2K twin); shear/off-diagonal stress (orthorhombic design);
multi-k UKS stress XcKind expand RevPBE/RPBE/BLYP (thin after
PADE/PBE/PBEsol k-UKS stress ✓); UGW@UKS-RPBE XC throw (thin after UKS
RPBE SCF); spin/metallic bands-DOS footholds (hold behind DFPT
spin+smear depth); XL-BOMD multi-k / OT-XL (hold behind OT combo depth
this wave); multi-k dispersion forces on k-path (hold behind D4/MBD
force product); GAPW/PAW/periodic hybrid/meta-GGA (blocked); molecular
post-HF / classical MW / CUDA ports (other loops). Roadmap: **no ○/✓
table flips** (nothing new merged since harness wave 8); §8/§10–12 +
MD AIMD open-residual prose + index GPW residual line refreshed for
wave-9 queue. **Pending = 9**; **in_progress = 1**; **blocked = 5**.

**Harness wave 9 2026-07-26 (parent `dev` @ `dab2b190`):** 8 staffed
slugs merged → `## done` (`ot-uks-smearing`, `ot-uks-dftu`,
`ot-dftu-smearing`, `gpw-d4-forces`, `gpw-mbd-forces`,
`gpw-uks-multigrid`, `dfpt-finite-q-smearing`, `second-gen-cpmd-ot`).
Specs deleted on close. **Pending residual = 0** (staff batch emptied
the landed list — **not** queue exhausted; GapCensus owns next
discovery). **in_progress = 2** unmerged staff `[~]`: `gpw-vv10-ice` on
`wf/gpw-vv10-ice` (CP2K stock VV10 oracle gap; do not demote) +
`dfpt-spin-smearing` on `wf/dfpt-spin-smearing` @ `d7576dea` (feat + reg
on branch; not merged to `dev` — keep incomplete). Roadmap: **no new
○→✓ table flips** (D4/MBD/OT/multigrid rows already ✓; stock VV10 stays
○); residual prose closed UKS OT+smear, multi-k UKS OT+DFT+U,
OT+DFT+U+smear, D4/MBD analytic forces, UKS multigrid, finite-q DFPT
smear, SGCP OT corrector; **DFPT spin+smear** stays open incomplete;
stock VV10 stays open. **Blocked = 5** unchanged. **Queue not
exhausted** — GapCensus owns next discovery.

**GapCensus wave 10 2026-07-26 (parent `dev` @ `94c2401d`):** recover —
incomplete_count=**2** (`[~] gpw-vv10-ice` unmerged `wf/gpw-vv10-ice` @
`2f1d90e2` docs/block — CP2K 2026.1 has no `TYPE VV10`; do not demote;
`[~] dfpt-spin-smearing` unmerged `wf/dfpt-spin-smearing` @ `d7576dea`
feat+reg — not merged; do not demote). Open CP2K/GPW/PW specs on
`dev`=**0** (PySCF deferred only: `periodic-rhf-gamma` /
`rsh-eri-cam-b3lyp`); isolation worktrees=**0**; leftover builds=**none**.
Tree re-diff vs `docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD +
`tools/cp2k_oracle` + `include/atomiverse/cpu/{gpw,planewave}`: prior
pending residual **0** (harness emptied list — not capability-closed).
**True residuals re-opened (9)** with offline CP2K / literature twins:
`ot-uks-dftu-smearing` (multi-k UKS OT + DFT+U + metallic smear —
`build` / k-UKS OT path reject; RKS OT+U+smear + UKS OT+smear + UKS
OT+U product),
`second-gen-cpmd-uks-ot` (SGCP RKS OT corrector ✓; UKS OT + ASPC
throws “follow-on”),
`xl-bomd-kpoints` (Γ RKS/UKS XL ✓; multi-k XL product depth / pin after
wave-9 OT depth),
`bands-spin-si` (`Bands::compute` throws spin-polarized foothold),
`dos-metallic-si` (`Dos::compute` throws metallic smearing foothold),
`periodic-wannier-uks` (`PeriodicWannier` throws spin-polarized MLWF),
`berry-spin-si` (`BerryPolarization` throws spin-polarized strings),
`gpw-mbd-kpoint-forces` (Γ MBD ionic forces ✓; multi-k force path folds
d3/d4/ts/xdm but **skips MBD**),
`ugw-uks-rpbe` (`g0w0` UKS RPBE spin XC throws — not product; PBE/revPBE
UKS UGW ✓). **Not added:** BP86/B97/OLYP/HCTH (molecular PySCF path);
stock VV10 (kept `[~]` incomplete — no demote; no CP2K twin); shear /
off-diagonal stress (orthorhombic design); electronic CPMD thermostat
(hold behind SGCP UKS OT / XL×k depth); TS/XDM dedicated force regs
(Γ force assembly already folds ts/xdm — thin after MBD×k); spin DOS /
metallic bands thin siblings (hold behind bands-spin + dos-metallic
first); finite-q DFPT spin+smear (hold behind incomplete Γ spin+smear
merge); XL+OT architectural exclusion (density vs orbital parameter);
GAPW/PAW/periodic hybrid/meta-GGA (blocked); molecular post-HF /
classical MW / CUDA ports (other loops). Roadmap: **no ○/✓ table
flips** (nothing new merged since harness wave 9); §8/§10–12 + MD AIMD
open-residual prose + index GPW residual line refreshed for wave-10
queue. **Pending = 9**; **in_progress = 2**; **blocked = 5**.

**Harness wave 10 2026-07-26 (parent `dev` @ `06ecaa66`):** 9 staffed
slugs merged → `## done` (`dfpt-spin-smearing`, `ot-uks-dftu-smearing`,
`second-gen-cpmd-uks-ot`, `xl-bomd-kpoints`, `bands-spin-si`,
`dos-metallic-si`, `periodic-wannier-uks`, `berry-spin-si`,
`gpw-mbd-kpoint-forces`). Specs deleted on close. **Pending residual =
1** (`ugw-uks-rpbe` — not staffed this wave; **not** queue exhausted;
GapCensus owns next discovery). **in_progress = 1** unmerged staff
`[~]`: `gpw-vv10-ice` on `wf/gpw-vv10-ice` (CP2K stock VV10 oracle gap;
do not demote). `dfpt-spin-smearing` incomplete resolved by merge into
`dev`. Roadmap: **no new ○→✓ table flips** (DFPT / Berry / Wannier /
bands-DOS / MBD / OT / SGCP / XL rows already ✓; stock VV10 stays ○);
residual prose closed collinear DFPT spin+smear, multi-k UKS OT+U+smear,
SGCP UKS OT, multi-k XL-BOMD, spin bands, metallic DOS, spin MLWF,
spin Berry, multi-k MBD forces; **UGW@UKS-RPBE** + stock VV10 stay open.
**Blocked = 5** unchanged. **Queue not exhausted** — GapCensus owns next
discovery.

**GapCensus wave 11 2026-07-26 (parent `dev` @ `96a0a7d2`):** recover —
incomplete_count=**1** (`[~] gpw-vv10-ice` unmerged `wf/gpw-vv10-ice` @
`a60c73d5` docs/block — CP2K 2026.1 `TYPE VV10` enum reject, only
DRSLL|LMKLL|RVV10; **resolved → blocked**, no force-merge / no product
`GpwDft::vv10`). Open CP2K/GPW/PW specs on `dev`=**0** (PySCF deferred
only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation worktrees=**0**;
leftover builds=**none**. Tree re-diff vs `docs/roadmap/dft.md` §3/§8/
§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending residual **1**
(`ugw-uks-rpbe`) still true (`g0w0` `eval_xc_grid_spin` throws UKS RPBE
despite SCF `xc_grid_rpbe_spin` product). **Added 8** tree-declared
follow-ons with offline CP2K / literature twins:
`xl-bomd-uks-kpoints` (multi-k RKS XL ✓ + Γ-UKS XL ✓; multi-k UKS XL
throws dual-spin `P_k^σ`),
`bands-metallic-si` (`Bands::compute` throws metallic smearing foothold
after `dos-metallic-si` ✓),
`dos-spin-si` (`Dos::compute` throws spin-polarized foothold after
`bands-spin-si` ✓),
`dfpt-finite-q-spin-smearing` (Γ spin+smear ✓ + finite-q smear RKS ✓ +
finite-q spin fixed-occ ✓; finite-q collinear spin+smear throws),
`gpw-kpoint-uks-stress-gga` (multi-k UKS stress product for PADE/PBE/
PBEsol only; Γ UKS stress already RevPBE/RPBE/BLYP — expand k-path
XcKind),
`cpmd-electronic-thermostat` (CPMD + RATTLE + UKS/SGCP depth ✓;
electronic thermostat residual per MD AIMD roadmap),
`gpw-ts-forces` / `gpw-xdm-forces` (force assembly already folds TS/XDM
on Γ + multi-k; dedicated offline pins after MBD×k forces ✓). **Not
added:** BP86/B97/OLYP/HCTH (molecular PySCF path); stock VV10 (moved
**blocked** — no CP2K twin); shear/off-diagonal stress (orthorhombic
design); multi-k SGCP / multi-k CPMD (no explicit throw; unpinned
depth hold); GAPW/PAW/periodic hybrid/meta-GGA (blocked); molecular
post-HF / classical MW / CUDA ports (other loops). Roadmap: **no ○/✓
table flips** (nothing new merged since harness wave 10); §8/§10–12 +
MD AIMD open-residual prose + index GPW residual line refreshed;
**VV10** remains ○ with blocked note. **Pending = 9**; **in_progress
= 0**; **blocked = 6**.

**Harness wave 11 2026-07-26 (parent `dev` @ `09298298`):** 9 staffed
slugs merged → `## done` (`ugw-uks-rpbe`, `gpw-ts-forces`, `gpw-xdm-forces`,
`gpw-kpoint-uks-stress-gga`, `cpmd-electronic-thermostat`,
`xl-bomd-uks-kpoints`, `bands-metallic-si`, `dos-spin-si`,
`dfpt-finite-q-spin-smearing`). Specs deleted on close. **Pending residual
= 0** (staff batch emptied the census pending list — **not** queue
exhausted; GapCensus owns next discovery). **in_progress = 0** (no `[~]`;
stock VV10 stays **blocked**). Roadmap: **no new ○→✓ table flips**
(capability rows already ✓ or VV10 still ○/blocked); residual prose closed
UGW@UKS-RPBE, dedicated TS/XDM force pins, multi-k UKS stress GGA expand,
CPMD electronic thermostat, multi-k UKS XL-BOMD, metallic bands, spin DOS,
finite-q DFPT spin+smear. **Blocked = 6** unchanged. **Queue not
exhausted** — GapCensus owns next discovery.

**GapCensus wave 12 2026-07-26 (parent `dev` @ `087df536`):** recover
clean — incomplete_count=**0**; `## in_progress` empty (no `[~]`); open
CP2K/GPW/PW specs on `dev`=**0** (PySCF deferred only:
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation worktrees=**0**;
leftover builds=**none**; `wf/gpw-vv10-ice` tip remains for blocked stock
VV10 docs only (not incomplete). Tree re-diff vs `docs/roadmap/dft.md`
§3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending residual **0**
(harness wave 11 emptied list — not capability-closed). **True residuals
re-opened (9)** with offline CP2K / literature twins:
`bands-spin-metallic-si` (`Bands::compute` throws collinear spin +
metallic smear — follow-on after fixed-occ spin bands ✓ + restricted
metallic bands ✓),
`dos-spin-metallic-si` (`Dos::compute` throws spin + metallic smear —
follow-on after spin DOS ✓ + metallic DOS ✓),
`gpw-d3-atm` (D3 zero / BJ two-body product; three-body C9 / ATM off —
CP2K `DISPERSION_FUNCTIONAL` ATM),
`gpw-d4-atm` (D4 two-body BJ product + forces/stress ✓; three-body ATM
off — dftd4 / CP2K s9 path),
`xl-bomd-linearized-forces` (SCF-converged XL product at Γ/multi-k RKS/UKS;
zero-SCF linearized-functional force stays ○ — Niklasson XL residual),
`cpmd-kpoints` (CPMD Γ RKS/UKS product; seed requires real `C` —
multi-k Bloch CPMD unpinned),
`second-gen-cpmd-kpoints` (SGCP ASPC on real Γ \(P\) / dual UKS \(P^\sigma\);
multi-k complex \(P_k\) SGCP unpinned),
`periodic-wannier-gpw` (PW Marzari–Vanderbilt MLWF ✓; GPW-grid MLWF
residual still unqueued per §11),
`gpw-fd-phonon-si` (FD supercell phonons product on SW twin only;
GPW analytic-force supercell vs CP2K `VIBRATIONAL_ANALYSIS` residual).
**Not added:** BP86/B97/OLYP/HCTH (molecular PySCF path); stock VV10
(**blocked** — no CP2K twin); shear/off-diagonal stress (orthorhombic
design); periodic hybrid / meta-GGA / GAPW / PAW (**blocked**); multi-k
UMP2 header “follow-on” is **stale** (product `reg_periodic-ump2-kpoints`);
UKS OT+U+smear header residual is **stale** (`reg_ot-uks-dftu-smearing`);
density-scaled TS already product via `hirshfeld`+`ts`; molecular post-HF /
classical MW / CUDA ports (other loops). Roadmap: **no ○/✓ table flips**
(nothing new merged since harness wave 11); §8/§10–12 + MD AIMD open-
residual prose + index GPW residual line refreshed for wave-12 queue.
**Pending = 9**; **in_progress = 0**; **blocked = 6**.

**Assign wave 12 2026-07-26 (parent `dev` @ `cbe74bdf`):** recover clean —
incomplete_count=**0** (no `[~]`; stock VV10 **blocked**); open CP2K/GPW/PW
specs=**0** (PySCF deferred only); unmerged feature `wf/*`=**0**
(`wf/gpw-vv10-ice` tip is blocked docs only). Staffed **all 9** pending
residuals → `## in_progress` (staff 9 ≤ 10; no open-spec resumes; no
pending held). Surface-disjoint notes for workers: `bands-spin-metallic-si`
∥ `dos-spin-metallic-si` share PW `bands_dos` — serialize edits or
split patches carefully; `gpw-d3-atm` ∥ `gpw-d4-atm` share dispersion
tables / force-assembly fold — careful dual surface; `cpmd-kpoints` ∥
`second-gen-cpmd-kpoints` share AIMD propagator / k-seed paths —
coordinate Bloch \(C_k\) / \(P_k\) design; `xl-bomd-linearized-forces`
free of multi-k CPMD/SGCP surface; `periodic-wannier-gpw` free of
`gpw-fd-phonon-si` (MLWF vs vib FD consumer). All picks write-first: no
open `.claude/specs/<slug>/`, no `wf/<slug>` feature tip, no regression
dir yet. **Pending residual after assign = 0** (staff emptied list —
**not** queue exhausted; GapCensus owns next discovery after harness).

**GapCensus wave 13 2026-07-26 (parent `dev` @ `663f4804`):** recover —
incomplete_count=**9** (all wave-12 `[~]` preserved; do not demote);
open CP2K/GPW/PW specs on `dev`=**0** (PySCF deferred only:
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation worktrees=**9**;
unmerged feature `wf/*`=**9** (+ blocked `wf/gpw-vv10-ice`); leftover
builds=**none** on parent. Tree re-diff vs `docs/roadmap/dft.md` §3/§8/
§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending residual **0**
(assign wave 12 emptied list — not capability-closed). Wave-12 staff still
true gaps on `dev` (no half-regs on parent). **True residuals re-opened
(9)** with offline CP2K / literature twins (exclude all done / blocked /
in_progress):
`gpw-d3-forces` / `gpw-d3bj-forces` (force assembly folds D3 zero/BJ on Γ
+ multi-k; only E_disp pins — same dedicated-force pattern as TS/XDM after
energy footholds),
`gpw-d3bj-stress` (analytic stress folds D3(BJ) energy under strain-FD;
dedicated pin missing — D4 has `reg_gpw-d4-stress`),
`gpw-mbd-stress` (Γ stress folds MBD E_disp; forces product; no stress
reg twin),
`gpw-mbd-kpoint-stress` (multi-k `gpw_stress` path folds d3/d3bj/d4/ts/xdm
but **silently skips MBD** — multi-k MBD forces ✓ / Γ MBD stress path
exists; k-path MBD stress missing),
`gpw-dftu-kpoint-forces` (multi-k DFT+U SCF + analytic stress ✓;
`gpw_forces` only copies `plus_u` into cfg — no dedicated multi-k U force
pin / explicit U force depth),
`dfpt-gga-spin-si` (DFPT GGA kernel closed-shell ✓ + collinear spin LSDA ✓;
joint GGA × spin unpinned),
`dfpt-gga-finite-q-si` (finite-q DFPT pins are LSDA; Γ GGA DFPT ✓ —
finite-q GGA residual),
`gpw-bomd-kpoints` (Γ BOMD ✓ + multi-k XL ✓; plain multi-k BOMD unpinned
vs CP2K Quickstep AIMD k-mesh).
**Not added:** BP86/B97/OLYP/HCTH (molecular PySCF path); stock VV10
(**blocked**); shear/off-diagonal stress (orthorhombic design); periodic
hybrid / meta-GGA / GAPW / PAW (**blocked**); multigrid×k (hold behind
BOMD×k / MBD×k stress depth this wave); D3-zero dedicated stress (thin
after D3BJ stress); TS/XDM dedicated stress (thin after MBD stress /
TS-XDM force pins); SCS-MP2 periodic (members exist; hold behind
dispersion stress depth); Berry/Wannier metallic (insulator requirement
physics); ATM force/stress deeper slugs (hold behind incomplete
`gpw-d3-atm` / `gpw-d4-atm`); molecular post-HF / classical MW / CUDA
(other loops). Roadmap: **no ○/✓ table flips** (nothing new merged since
harness wave 11); §8/§10–12 + MD AIMD residual prose + index GPW residual
line refreshed for wave-13 queue. **Pending = 9**; **in_progress = 9**;
**blocked = 6**.

**Harness wave 12 2026-07-26 (parent `dev` @ `4ab58f97`):** 6 of 9 wave-12
staff merged → `## done` (`xl-bomd-linearized-forces`,
`second-gen-cpmd-kpoints`, `gpw-d3-atm`, `gpw-d4-atm`,
`bands-spin-metallic-si`, `dos-spin-metallic-si`). Specs deleted on close.
**Pending residual = 9** (GapCensus wave 13 list unchanged — **not** queue
exhausted). **in_progress = 3** unmerged staff `[~]` (`cpmd-kpoints` @
`3425c7c6`, `periodic-wannier-gpw` @ `5c59a97f`, `gpw-fd-phonon-si` @
`7c3e1712` — do not demote). Roadmap: **no new ○→✓ table flips** (XL /
SGCP / bands-DOS / dispersion rows already ✓); residual prose closed
zero-SCF linearized XL, multi-k SGCP, D3/D4 ATM, collinear spin+metallic
bands/DOS; multi-k CPMD + GPW MLWF + GPW FD phonon stay open incomplete.
**Blocked = 6** unchanged. Clearup already ran (no isolation worktrees; no
builds). **Queue not exhausted** — next wave Recover incomplete then
continue.

**GapCensus wave 14 2026-07-26 (parent `dev` @ `cee90c43`):** recover —
incomplete_count=**3** (`[~] cpmd-kpoints` @ `3425c7c6` multi-k Bloch
\(C_k\) feat; `[~] periodic-wannier-gpw` @ `5c59a97f` GPW MV MLWF feat;
`[~] gpw-fd-phonon-si` @ `7c3e1712` Phonon+GPW wiring — Si force gate
blocks acceptance; **do not demote**). Open CP2K/GPW/PW specs on
`dev`=**0** (PySCF deferred only: `periodic-rhf-gamma` /
`rsh-eri-cam-b3lyp`); isolation worktrees=**0**; leftover builds=**none**;
unmerged feature `wf/*`=**3** (+ blocked `wf/gpw-vv10-ice`). Tree re-diff
vs `docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending **9** still true
gaps (no half-regs on `dev`; multi-k `gpw_stress` still omits MBD in
dispersion fold; D3/D3BJ force assembly product but no dedicated offline
pins; DFT+U multi-k forces unpinned; DFPT GGA×spin / GGA×finite-q unpinned
though kernels claim support; multi-k BOMD ○). **Added 6** tree-declared
follow-ons with offline CP2K / literature twins (exclude done / blocked /
in_progress):
`gpw-multigrid-kpoints` (`build` rejects `ngrids>1` under k-mesh — multi-k
× REL_CUTOFF multigrid residual after Γ/UKS multigrid ✓),
`xl-bomd-linearized-uks` / `xl-bomd-linearized-kpoints` (zero-SCF
linearized XL is Γ RKS only — evaluator throws UKS and multi-k follow-ons
after `reg_xl-bomd-linearized-forces` ✓),
`gpw-d3-stress` (D3-zero energy folds into fixed-\(P\) stress; dedicated
stress pin missing — same pattern as queued D3(BJ) stress / shipped D4
stress),
`gpw-d3-atm-forces` / `gpw-d4-atm-forces` (ATM energy product; force
assembly passes `d3_atm` / `d4_atm` into D3/D4 grads — dedicated offline
force pins after energy footholds, same dedicated-pin pattern as
TS/XDM/D4 two-body).
**Not added:** BP86/B97/OLYP/HCTH (molecular PySCF path); stock VV10
(**blocked**); shear/off-diagonal stress (orthorhombic design); periodic
hybrid / meta-GGA / GAPW / PAW (**blocked**); TS/XDM dedicated stress
(thin after MBD stress queue + force pins); ATM stress deeper (hold
behind ATM force pins); metallic Berry/Wannier (insulator requirement);
SCS-MP2 periodic (hold); molecular post-HF / classical MW / CUDA ports
(other loops). Roadmap: **no ○/✓ table flips** (nothing new merged since
harness wave 12); §8/§10–12 + MD AIMD residual prose + index GPW residual
line refreshed for wave-14 queue. **Pending = 15**; **in_progress = 3**;
**blocked = 6**.

**Assign wave 14 2026-07-26 (parent `dev` @ `b06568f2`):** recover —
incomplete_count=**3** (wave-12 residual `[~]` preserved; unmerged
`wf/cpmd-kpoints` @ `3425c7c6`, `wf/periodic-wannier-gpw` @ `5c59a97f`,
`wf/gpw-fd-phonon-si` @ `7c3e1712` — each behind=17 / ahead=1 vs `dev`;
do not demote); open CP2K/GPW/PW specs=**0** (PySCF deferred only);
isolation worktrees=**0**. Priority incomplete ≫ pending. Staffed
**staff 10** = 3 incomplete resumes + 7 dep-safe pending →
`## in_progress`. New promotions (write-first; no open
`.claude/specs/<slug>/`, no `wf/<slug>` tip yet): `gpw-d3-forces`,
`gpw-d3bj-forces`, `gpw-mbd-stress`, `gpw-dftu-kpoint-forces`,
`dfpt-gga-spin-si`, `gpw-bomd-kpoints`, `gpw-multigrid-kpoints`.
Surface notes: `gpw-d3-forces` ∥ `gpw-d3bj-forces` share dispersion /
force-assembly fold — serialize or split patches carefully; `gpw-mbd-stress`
free of multi-k MBD stress (held); `gpw-dftu-kpoint-forces` free of DFPT
tree; `dfpt-gga-spin-si` free of finite-q GGA (held); `gpw-bomd-kpoints`
∥ `gpw-multigrid-kpoints` free of each other per census (BOMD propagator
vs multigrid build gate); incomplete CPMD×k free of multi-k BOMD if
workers coordinate Bloch / MD evaluator paths. **Held pending (8):**
`gpw-d3bj-stress`, `gpw-mbd-kpoint-stress`, `dfpt-gga-finite-q-si`,
`xl-bomd-linearized-uks`, `xl-bomd-linearized-kpoints`, `gpw-d3-stress`,
`gpw-d3-atm-forces`, `gpw-d4-atm-forces` (stress / ATM / linearized XL /
finite-q GGA after force dual depth or next wave). Resume workers:
rebase `wf/*` onto `dev` (behind 17) then dual-gate / finish acceptance.
**Pending residual after assign = 8** (not queue exhausted).

**Recover 2026-07-31 (parent `dev` @ `c0a12f0b`):** clean —
incomplete_count=**0**; `## in_progress` empty (no `[~]`; last close
`gpw-fd-phonon-si` 2026-07-26); open CP2K/GPW/PW specs on `dev`=**0**
(PySCF deferred only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`);
unmerged `wf/*`=**0** local + remote (even the blocked `wf/gpw-vv10-ice`
docs tip is pruned — its row stays `## blocked`); isolation
worktrees=**0**; half regressions=**0** (no regression dir exists for any
of the 12 pending slugs; the only queue-unmatched dirs
`regressions/pw/h_*` are pre-loop PW cases committed on `dev`). Leftover
`build*` dirs + untracked `tools/bench_cuda_rhf_*` / nsys files belong to
the concurrent CUDA RHF campaign (out of loop scope — not wiped
mid-flight). Since wave-14 close, `dev` gained the esmm campaign + science
fixes (`3ee28243` D4 analytic forces / D3 stress on the **molecular scf
path**, OT gtol retune + thread-deterministic GPW grid kernels
`0f78ff6d`, PW spin V_eff fix `e7b7b534`) — GapCensus must re-verify all
12 pending rows against the tree (especially `gpw-d3-stress` /
`gpw-d3-atm-forces` / `gpw-d4-atm-forces` overlap with `3ee28243`) before
Assign. **Pending = 12**; **in_progress = 0**; **blocked = 6**. Do **not**
invent methods — next is GapCensus/Assign from **pending**.

**GapCensus wave 15 2026-07-31 (shift wave 1; parent `dev` @ `e5f55189`):**
recover already clean per Recover 2026-07-31 (incomplete=**0**;
`## in_progress` empty; open CP2K/GPW/PW specs=**0** — PySCF deferred only:
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`; unmerged `wf/*`=**0**; isolation
worktrees=**0**; no regression dir for any pending slug). Tree re-diff vs
`docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}` on post-esmm `dev`: **all 12 prior
pending re-verified as true gaps** — `gpw_stress.cpp:351` still refuses MBD
on the multi-k path; `gpw_evaluator.cpp:181/185` still throw UKS / multi-k
linearized XL; `gpw_solver.cpp:5077` k-path still `reject("multigrid
(ngrids > 1)")`; `gpw_forces.cpp` still only copies `plus_u` into cfg (no U
force fold, Γ or k); no dedicated D3/D3BJ/TS/XDM stress or D3/D4 ATM force
pins; DFPT GGA×spin / GGA×finite-q joint pins absent; no `gpw-bomd-kpoints`
reg. Overlap check vs the science-fix commits: `3ee28243` landed **molecular**
scf/dftd4 analytic two-body D4 (EEQ adjoint) + ATM F2+FD and **DFTB2** D3
stress — the GPW dedicated-pin rows stay open (ATM-force workers should
reuse that machinery); `0f78ff6d` (thread-deterministic grid kernels) and
`e7b7b534` (PW spin V_eff fix) change no capability row. `e7b7b534` also
**voided two fossil self-golden pins** (`pw-uks-kmesh-fe` /
`pw-uks-fe-smearing`, ATV self-freezes from the pre-spin-GGA solver) and
replaced them with exact invariants — correct per pin provenance; no re-pin
row queued (no citable NC-UPF PW twin; both regs now gate identities, not
frozen ATV output). **Added 5** tree-declared residuals with offline CP2K /
literature twins: `gpw-vdw-multigrid` (vdW-DF/DF2/rVV10 × `ngrids>1` throws
"multigrid not supported" at `gpw_solver.cpp:4942`; CP2K default NGRIDS 4
twin), `gpw-ts-stress` / `gpw-xdm-stress` (Γ stress folds ts/xdm energy;
dedicated pins missing — the wave-13/14 hold "thin after MBD stress + TS/XDM
force pins" cleared when both shipped), `cpmd-uks-kpoints` (evaluator throws
"multi-k UKS + cpmd is not productized"; MD-roadmap ○ residual),
`second-gen-cpmd-ot-kpoints` (multi-k SGCP throws "use_ot + P_k density
guess is not productized" after Γ RKS/UKS OT correctors ✓). **Not added:**
CPMD stress / variable-cell CP (deliberate refusal, esmm-cpmd-02 gate — no
classic-CP CP2K twin); PW Mermin fractional spin GS loop (prose depth, no
throw); ATM stress (hold behind pending ATM force pins); SCS-MP2 periodic
(members product-shaped; hold behind dispersion stress depth); multi-k UMP2
`mp2.h` "follow-on" comment is **stale doc**, capability is product
(`reg_periodic-ump2-kpoints`); BP86/B97/OLYP/HCTH (molecular PySCF path);
stock VV10 (**blocked** — no CP2K twin); shear/off-diagonal stress
(orthorhombic design); GAPW/PAW/periodic hybrid/meta-GGA (blocked);
molecular post-HF / classical MW / CUDA ports (other loops). Roadmap: **no
○/✓ table flips** (nothing capability-shaped merged since harness wave 14 —
esmm campaign is molecular/multiscale); fixed stale §8 prose (D3/D3BJ
forces + Γ MBD stress still listed "queued" though shipped), §10 open-
residual line (wave-12 incompletes closed), §11 "GPW-grid MLWF queued" →
product, §11 GPW FD phonon mention; index GPW residual line refreshed to
wave 15; MD roadmap CPMD/SGCP residual prose annotated with the two new
queue slugs. **Pending = 17**; **in_progress = 0**; **blocked = 6**.

**Assign wave 1 (shift) 2026-07-31 (parent `dev` @ `7982f0a5`):** recover
clean — incomplete_count=**0** (no `[~]`); open CP2K/GPW/PW specs=**0**
(PySCF deferred only); unmerged `wf/*`=**0**; isolation worktrees=**0**.
Staffed **10** dep-safe pending → `## in_progress` (no resumes exist — all
picks write-first: no open `.claude/specs/<slug>/`, no `wf/<slug>` tip, no
regression dir). Picks prioritize the four wave-14 staffed-but-unlanded rows
(`gpw-dftu-kpoint-forces`, `dfpt-gga-spin-si`, `gpw-bomd-kpoints`,
`gpw-multigrid-kpoints` — closest to incomplete; branches did not survive,
restart write-first) + oldest census age: `gpw-d3bj-stress`, `gpw-d3-stress`,
`gpw-mbd-kpoint-stress`, `gpw-d3-atm-forces`, `gpw-d4-atm-forces`,
`xl-bomd-linearized-uks`. Surface notes for workers: `gpw-d3bj-stress` ∥
`gpw-d3-stress` share the D3 dispersion stress fold — serialize or split
patches carefully (same pattern as the shipped force pair);
`gpw-mbd-kpoint-stress` edits the **multi-k** `gpw_stress` path (Γ pins
untouched) — coordinate with the two Γ D3 stress workers on `gpw_stress.cpp`;
`gpw-d3-atm-forces` ∥ `gpw-d4-atm-forces` share force-assembly ATM folds —
reuse `3ee28243` molecular ATM F2+FD / EEQ-adjoint machinery, serialize
carefully; `gpw-dftu-kpoint-forces` (multi-k U fold in `gpw_forces`) is
disjoint from the dispersion-grad region but same file — coordinate;
`xl-bomd-linearized-uks` ∥ `gpw-bomd-kpoints` may both touch
`gpw_evaluator.cpp` (linearized-force path vs multi-k MD path) — coordinate;
`gpw-multigrid-kpoints` free of all (solver build gate). **Held pending (7):**
`gpw-ts-stress` / `gpw-xdm-stress` (thin pin-only — after this wave's stress
workers drain `gpw_stress`), `dfpt-gga-finite-q-si` (shares PW DFPT GGA
kernel with `dfpt-gga-spin-si` pick), `xl-bomd-linearized-kpoints` (shares
linearized-XL evaluator throw sites with the UKS pick), `gpw-vdw-multigrid`
(shares `gpw_solver` multigrid gates with `gpw-multigrid-kpoints` pick),
`cpmd-uks-kpoints` / `second-gen-cpmd-ot-kpoints` (wave-15 newest; AIMD
propagator surface contention with XL/BOMD picks). **Pending residual after
assign = 7** (not queue exhausted).

**Harness wave 1 (shift) 2026-07-31 (parent `dev` @ `8a9584f4`):** 9 of 10
staffed slugs merged → `## done` (`gpw-d3bj-stress`, `gpw-mbd-kpoint-stress`,
`gpw-d3-atm-forces`, `gpw-d4-atm-forces`, `gpw-dftu-kpoint-forces`,
`dfpt-gga-spin-si`, `gpw-bomd-kpoints`, `gpw-multigrid-kpoints`,
`xl-bomd-linearized-uks`) — nine `--no-ff` merges `44749b1a`..`268aaf66`;
merge gate green at close (scoped rebuild +
`ctest -R '^test_cpu_(gpw|pw|scf|xc)_|^test_xc_'` 57/57). Specs deleted on
close (`8a9584f4`; the spec-close commit / INDEX cross-name the batch
"harness wave 13"). **Pending residual = 7** (held wave-1 list unchanged —
**not** queue exhausted; GapCensus owns next discovery). **in_progress = 1**
unmerged staff `[~]` (`gpw-d3-stress` on `wf/gpw-d3-stress` @ `3b3872eb` —
wip spec + CP2K NUMERICAL-stress pins + D3 stress gates committed, acceptance
unfinished; do not demote). Roadmap: **no ○/✓ table flips** (capability rows
already ✓); residual prose closed D3(BJ) stress, multi-k MBD stress, D3/D4
ATM forces, multi-k DFT+U forces, joint DFPT GGA×spin, multi-k BOMD, multi-k
multigrid, Γ-UKS linearized XL (dft.md §8/§10/§12 + index GPW line + MD AIMD
BOMD/XL lines); `gpw-kpoints-si` ATV-frozen A7 self-golden pin replaced by
the genuine CP2K 2026.1 twin number (in `966d9b8a`). **Blocked = 6**
unchanged. Clearup already ran (merged `wf/*` pruned; no isolation worktrees;
no loop builds — concurrent CUDA-RHF campaign artifacts left untouched). No
push.

**GapCensus wave 2 (shift) 2026-07-31 (parent `dev` @ `b06944ca`):** recover —
incomplete_count=**1** (`[~] gpw-d3-stress` unmerged `wf/gpw-d3-stress` @
`3b3872eb`, 1 ahead / 20 behind `dev`; wip spec + CP2K NUMERICAL-stress pins +
D3 stress gates; do not demote). Open CP2K/GPW/PW specs on `dev`=**0** (PySCF
deferred only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation
worktrees=**0**; leftover loop build `build/` (CPU merge-gate leftover from
harness wave 1, CMakeCache 03:29) wiped this wave; concurrent CUDA-RHF
artifacts (`tools/bench_*`, nsys) untouched. No merges into `dev` since
harness wave 1 (`b06944ca` = HEAD). Tree re-diff vs `docs/roadmap/dft.md`
§3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: **all 7 prior pending re-verified as
true gaps** — `gpw_evaluator.cpp:182` still throws multi-k linearized XL;
`gpw_solver.cpp:5096` still throws nonlocal vdW × `ngrids>1`;
`gpw_evaluator.cpp:242` multi-k UKS CPMD / `:319` multi-k SGCP OT still throw;
Γ+k `gpw_stress` folds ts/xdm energy but no dedicated pin/reg exists; PW
finite-q DFPT accepts GGA (kernel claims joint support) with no offline pin.
**Added 5** tree-declared residuals with offline CP2K / literature twins:
`gpw-uks-multigrid-kpoints` (`gpw_solver.cpp:5246` k-path rejects
"multigrid (ngrids > 1) with UKS — closed-shell RKS only"; declared by the
`gpw-multigrid-kpoints` merge), `gpw-vdw-kpoints` (`gpw_solver.cpp:5262`
nonlocal vdW not wired on the k path at all — fail-fast added while opening
the k multigrid gate; SCF potential + E_c^nl live only in the Γ runner),
`gpw-d3-atm-stress` / `gpw-d4-atm-stress` (ATM energy folds into the
strained-cell stress rebuild; dedicated pins missing — the wave-15 hold
"ATM stress behind ATM force pins" cleared when `gpw-d3-atm-forces` /
`gpw-d4-atm-forces` shipped), `cpmd-thermostat-kpoints`
(`gpw_evaluator.cpp:251` "NVE electrons only on the k path" after Γ
electronic thermostat ✓ + multi-k RKS CPMD ✓). **Not added:** CPMD stress /
variable-cell CP (deliberate refusal, esmm-cpmd-02 gate); PW Mermin
fractional spin GS loop (prose depth, no throw); OT+XL (architectural
mutual exclusion — OT parameterizes orbitals, not density; SGCP is the
OT-family AIMD); DFPT GGA×spin×finite-q (hold behind pending
`dfpt-gga-finite-q-si`); BP86/B97/OLYP/HCTH (molecular PySCF path); stock
VV10 (blocked — no CP2K twin); shear/off-diagonal stress (orthorhombic
design); GAPW/PAW/periodic hybrid/meta-GGA (blocked); molecular post-HF /
classical MW / CUDA ports (other loops). Roadmap: **no ○/✓ table flips**
(nothing merged since harness wave 1); §8/§10/§12 + index GPW residual line
+ MD AIMD CPMD prose refreshed for the wave-2 queue. **Pending = 12**;
**in_progress = 1**; **blocked = 6**.

**GapCensus wave 2 completion (resumed session) 2026-07-31 (parent `dev` @
`9983b89c`):** the session-limit orphan commit `9983b89c` landed the wave-2
queue rewrite + `dft.md` §8/§10 prose but dropped the claimed
`docs/roadmap/index.md` GPW residual-line and MD-roadmap CPMD refreshes —
both landed now (index residual line lists all 12 pending families incl.
ATM stress / vdW×k / k-UKS multigrid / k CPMD thermostat; MD CPMD prose
names the Γ-only electronic thermostat + queued `cpmd-thermostat-kpoints`).
Source tree unchanged since the census parent (`b06944ca`; only docs on
top), and all 12 pending rows re-verified against it this session: the
seven cited throw/reject sites hold (`gpw_evaluator.cpp` multi-k linearized
XL / multi-k UKS CPMD / k CPMD thermostat / multi-k SGCP OT;
`gpw_solver.cpp` vdW×multigrid, k-UKS multigrid, k-path vdW fail-fast) and
no regression dir exists for any pending slug. `[~] gpw-d3-stress` stays
in_progress on `wf/gpw-d3-stress` (do not demote). **No ○/✓ table flips.**
**Pending = 12**; **in_progress = 1**; **blocked = 6**.

**Assign wave 2 (shift) 2026-07-31 (parent `dev` @ `197468ac`):** recover —
incomplete_count=**1** (`[~] gpw-d3-stress` unmerged `wf/gpw-d3-stress` @
`3b3872eb`, merge-base `03c16eef` — resume staffed first, incomplete ≫
pending). Open CP2K/GPW/PW specs=**0** (PySCF deferred only:
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation worktrees=**0**; all
12 pending re-verified write-first (no spec / no `wf/*` / no regression
dir). Staffed **10**: 1 resume (`gpw-d3-stress` — rebase/merge the
20-behind tip onto `dev`, then finish acceptance; wip spec + CP2K
NUMERICAL-stress pins + D3 stress gates already on branch) + 9 dep-safe
pending by census age: `dfpt-gga-finite-q-si` (w13),
`xl-bomd-linearized-kpoints` (w14), `gpw-vdw-multigrid`, `gpw-ts-stress`,
`gpw-xdm-stress`, `cpmd-uks-kpoints`, `second-gen-cpmd-ot-kpoints` (w15),
`gpw-d3-atm-stress`, `gpw-d4-atm-stress` (w2-shift; ATM-force holds
cleared). Surface notes for workers: five stress slugs share the
`gpw_stress.cpp` dispersion fold (`gpw-d3-stress` resume +
ts/xdm/d3-atm/d4-atm) — serialize or split patches carefully (wave-1
precedent: shipped D3(BJ)/MBD stress pair); `xl-bomd-linearized-kpoints` ∥
`cpmd-uks-kpoints` ∥ `second-gen-cpmd-ot-kpoints` touch distinct
`gpw_evaluator.cpp` throw sites (182/242/319) — coordinate, same file;
`gpw-vdw-multigrid` edits the `gpw_solver.cpp` vdW×multigrid gate alone
this wave; `dfpt-gga-finite-q-si` is PW response — free of all. No pick
depends on another pick. **Held pending (3):** `gpw-uks-multigrid-kpoints`
/ `gpw-vdw-kpoints` (share `gpw_solver.cpp` multigrid/k gates + the
nonlocal-vdW runner with the `gpw-vdw-multigrid` pick; newest census),
`cpmd-thermostat-kpoints` (extends the multi-k CPMD path staffed
`cpmd-uks-kpoints` builds — adjacent throw sites
`gpw_evaluator.cpp:242/251`; newest census). **Pending residual after
assign = 3** (not queue exhausted).

**Harness wave 2 (shift) 2026-07-31 (parent `dev` @ `8b2fda00`):** 8 of 10
staffed slugs merged → `## done` (`gpw-d3-stress` resume finished,
`dfpt-gga-finite-q-si`, `xl-bomd-linearized-kpoints`, `gpw-vdw-multigrid`,
`cpmd-uks-kpoints`, `second-gen-cpmd-ot-kpoints`, `gpw-d3-atm-stress`,
`gpw-d4-atm-stress`) — eight `--no-ff` merges `aa897bb2`..`8b2fda00`; merge
gate green at close (scoped rebuild +
`ctest -R '^test_cpu_(gpw|pw|scf|xc)_|^test_xc_'` 57/57 on merged `dev`). Specs
deleted on close (seven dirs here; the `gpw-d4-atm-stress` spec was already
deleted on its branch, `a2c85a6b`). **Pending residual = 3** (held wave-2
list unchanged — **not** queue exhausted; GapCensus owns next discovery).
**in_progress = 2** unlanded staff `[~]` (`gpw-ts-stress` / `gpw-xdm-stress`
— branch tips still at the assign parent `197468ac`, all WIP uncommitted in
their isolation worktrees: spec + regression + oracle-input scaffolds; do
not demote, do not remove the two worktrees). Roadmap: **no ○/✓ table
flips** (capability rows already ✓); the §8/§10/§12 + MD AIMD residual
prose for all eight ships landed **with the worker merges**; harness fixed
the stale `index.md` GPW ships/residual line (vdW multigrid + multi-k SGCP
OT + D3/D4 ATM stress missing from ships, still listed pending). **Blocked
= 6** unchanged. Clearup already ran (merged `wf/*` pruned; the two dirty
WIP worktrees kept deliberately). No push.

**GapCensus wave 3 (shift) 2026-07-31 (parent `dev` @ `7befaead`):** recover —
incomplete_count=**2** (`[~] gpw-ts-stress` / `[~] gpw-xdm-stress`: branch
tips still at the assign parent `197468ac`, all WIP uncommitted in their two
kept isolation worktrees — spec + regression + oracle-input scaffolds; do not
demote, do not remove the worktrees). Open CP2K/GPW/PW specs on `dev`=**0**
(PySCF deferred only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); unmerged
`wf/*` commits=**0**; leftover loop builds=**none**. Parent tree carries
uncommitted tutorial-doc enrichments for 4 wave-2 slugs (concurrent docs
session) — left untouched, not staged by this census. No merges into `dev`
since harness wave 2 (`7befaead` = docs-index commit on top of `8b2fda00`).
Tree re-diff vs `docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD +
`tools/cp2k_oracle` + `include/atomiverse/cpu/{gpw,planewave}`: **all 3
prior pending re-verified as true gaps** (throw sites moved after the wave-2
merges — citations refreshed): k-UKS multigrid reject now
`gpw_solver.cpp:5309`; k-path nonlocal-vdW reject now `gpw_solver.cpp:5326`;
k-path CPMD thermostat throw now `gpw_evaluator.cpp:291`. **Added 4**
tree-declared residuals with offline CP2K / literature twins:
`gpw-vdw-uks` (Γ UKS + nonlocal vdW-DF/DF2/rVV10 fail-fast at
`gpw_solver.cpp:5188` — "E_c^nl would be silently dropped"; the found-rot
fix the `gpw-vdw-multigrid` merge declared), `second-gen-cpmd-ot-uks-kpoints`
(multi-k UKS SGCP + OT corrector throws at `gpw_evaluator.cpp:361` —
"dual-spin k-OT warm start from P_k^sigma is not productized"; declared by
the `second-gen-cpmd-ot-kpoints` merge), `xl-bomd-linearized-stress`
(linearized XL + `compute_stress` refused at `gpw_evaluator.cpp:223` —
"variable-cell linearized XL is deferred"; refusal added by the
`xl-bomd-linearized-kpoints` merge, was a silent warm-step drop),
`dfpt-gga-spin-finite-q-si` (joint collinear spin × pure GGA × finite-q:
the q-path Sternheimer already applies the spin-GGA FD kernel at
`dfpt.cpp:3303` with **no offline pin** — the wave-2 hold "behind
`dfpt-gga-finite-q-si`" cleared when both GGA×spin and GGA×finite-q pins
shipped). **Not added:** multi-k D3/D3BJ/D4/TS/XDM stress pins (the
geometry-only fold identical to Γ is already in the k Lagrangian
`gpw_stress.cpp:341-544`; hold behind the in-flight Γ TS/XDM stress pins);
multi-k **UKS** CPMD electronic thermostat (thin behind queued
`cpmd-thermostat-kpoints`); vdW × UKS × multigrid / vdW × k higher-order
combos (hold behind `gpw-vdw-uks` + `gpw-vdw-kpoints` +
`gpw-uks-multigrid-kpoints`); CPMD stress / variable-cell CP (deliberate
refusal, esmm-cpmd-02 gate); OT+XL (architectural mutual exclusion); PW
Mermin fractional spin GS loop (prose depth, no throw); BP86/B97/OLYP/HCTH
(molecular PySCF path); stock VV10 (blocked — no CP2K twin);
shear/off-diagonal stress (orthorhombic design); GAPW/PAW/periodic
hybrid/meta-GGA (blocked); molecular post-HF / classical MW / CUDA ports
(other loops). Roadmap: **no ○/✓ table flips** (nothing capability-shaped
merged since harness wave 2); §8 Γ-UKS vdW reject + §12 open-residual list
+ §12 DFPT "census-held joint row" → queued slug + index GPW residual line
+ MD roadmap SGCP-UKS-OT / linearized-XL-stress prose annotated with the
four new queue slugs. **Pending = 7**; **in_progress = 2**; **blocked = 6**.

**Assign wave 3 (shift) 2026-07-31 (parent `dev` @ `4a2f966b`):** recover —
incomplete_count=**2** (`[~] gpw-ts-stress` / `[~] gpw-xdm-stress`: branch
tips still at the wave-2 assign parent `197468ac`, WIP uncommitted in their
two kept isolation worktrees — spec + regression + oracle-input scaffolds;
resumes staffed first, incomplete ≫ pending). Open CP2K/GPW/PW specs=**0**
(PySCF deferred only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); all 7
pending re-verified write-first (no spec on `dev` / no `wf/*` tip beyond
parent / no regression dir). Staffed **9**: 2 resumes (`gpw-ts-stress` /
`gpw-xdm-stress` — merge the `197468ac`-based tips onto `dev`, commit the
scaffolds, finish pins + acceptance; both fold into `gpw_stress.cpp` +
`regressions/CMakeLists.txt` — serialize or split patches carefully, wave-2
precedent: shipped D3/D3-ATM/D4-ATM stress trio) + all **7** pending by
census age: `gpw-uks-multigrid-kpoints`, `gpw-vdw-kpoints`,
`cpmd-thermostat-kpoints` (w2-shift), `gpw-vdw-uks`,
`second-gen-cpmd-ot-uks-kpoints`, `xl-bomd-linearized-stress`,
`dfpt-gga-spin-finite-q-si` (w3-shift). Surface notes for workers:
`gpw-uks-multigrid-kpoints` ∥ `gpw-vdw-kpoints` ∥ `gpw-vdw-uks` touch
distinct `gpw_solver.cpp` gates (5309 / 5326 / 5188) — same file,
coordinate at merge; `cpmd-thermostat-kpoints` ∥
`second-gen-cpmd-ot-uks-kpoints` ∥ `xl-bomd-linearized-stress` touch
distinct `gpw_evaluator.cpp` throw sites (291 / 361 / 223) — same file,
coordinate; `xl-bomd-linearized-stress` also reads `gpw_stress` — the two
stress resumes own the `gpw_stress.cpp` fold region, XL only calls it;
`dfpt-gga-spin-finite-q-si` is PW response (`dfpt.cpp`) — free of all. No
pick depends on another pick. **Pending residual after assign = 0** (staff
emptied the census list — **not** queue exhausted; GapCensus owns next
discovery).


**Harness wave 3 (shift) 2026-07-31 (parent `dev` @ `419c1dc3`):** **9/9**
staffed slugs merged → `## done` (`gpw-ts-stress`, `gpw-xdm-stress`,
`gpw-vdw-kpoints`, `gpw-vdw-uks`, `gpw-uks-multigrid-kpoints`,
`cpmd-thermostat-kpoints`, `second-gen-cpmd-ot-uks-kpoints`,
`xl-bomd-linearized-stress`, `dfpt-gga-spin-finite-q-si`) — nine `--no-ff`
merges + post-merge k-UKS+vdW fail-fast fix. Merge gate: all 9 `run_reg_*`
green on merged `dev`; unit/reg product gates re-verified. Specs closed
(deleted). **Pending residual = 0** (not queue exhausted; GapCensus owns next
discovery). **in_progress = 0**. **Blocked = 6** unchanged. Clearup runs with
harness (worktrees removed, merged `wf/*` pruned, builds wiped). No push.


**Bootstrap + GapCensus wave 4 (shift) 2026-08-01 (parent `dev` @ `87ae7924`):**
recover clean — incomplete_count=**0**; `## in_progress` empty; open
CP2K/GPW/PW specs=**0** (PySCF deferred only: `periodic-rhf-gamma` /
`rsh-eri-cam-b3lyp`); unmerged `wf/*`=**0**; isolation worktrees=**0**;
leftover `build/` (**~10 G**) wiped this census; wave-3 done-slug regression
dirs all present (9/9). CP2K source build OK (`cp2k.psmp` 2026.1
`git:5e54ba2` via `…/cp2k/install/env.sh`). Tree re-diff vs
`docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending residual **0**
(harness wave 3 emptied list — **not** capability-closed). **True residuals
re-opened (11)** with offline CP2K / literature twins:
`gpw-vdw-uks-kpoints` (multi-k UKS + nonlocal vdW-DF/DF2/rVV10 still
fail-fast at `gpw_solver.cpp` build reject — "no dual-spin nonlocal fold";
Γ UKS + multi-k RKS product after wave 3), `gpw-d3-kpoint-stress` /
`gpw-d3bj-kpoint-stress` / `gpw-d3-atm-kpoint-stress` (geometry-only
D3/D3(BJ)/C9 fold already in k Lagrangian `gpw_stress.cpp`; dedicated
multi-k stress pins held behind Γ D3/TS/XDM pins — now free; CP2K
NUMERICAL stress-difference twins), `gpw-d4-kpoint-stress` (k-path D4
fold product; dftd4 strained-cell pin — same no-DFTD4 CP2K policy as Γ
`gpw-d4-stress`), `gpw-ts-kpoint-stress` / `gpw-xdm-kpoint-stress`
(k-path TS/XDM fold product; libMBD / pure-Python strained-cell pins as
Γ), `gpw-bomd-uks-kpoints` (multi-k UKS plain BOMD unpinned — RKS only
`reg_gpw-bomd-kpoints`; CP2K Quickstep UKS AIMD twin),
`gpw-vdw-uks-multigrid` (Γ UKS + nonlocal vdW + REL_CUTOFF multigrid
combo unpinned — each axis product alone; CP2K NGRIDS×DRSLL×UKS twin),
`gpw-hirshfeld-uks-kpoints` (multi-k UKS Hirshfeld TS/MBD fold exists in
`fold_dispersion_k_uks`; no dedicated pin — Γ UKS + multi-k RKS pins
only), `cpmd-uks-thermostat-kpoints` (multi-k UKS electronic Nosé–Hoover
wired on per-(k,σ) `CpmdCplx` after `cpmd-thermostat-kpoints` /
`cpmd-uks-kpoints`; no combo pin; CP2K has no fictitious-mass twin —
literature envelope). **Not added:** variable-cell CPMD stress (deliberate
NVE-family refusal; no fair CP2K CP twin); XL-BOMD + metallic smear /
k-UKS fixed-density+smear (hold — fractional fixed-density path);
GPW NPT AIMD composition (MD barostat + product stress — not a method
domain gap); density-dependent XDM Hirshfeld moments (header out-of-scope);
Γ RKS DFT+U (design reject — UKS or k-path); SCS-MP2 thin scale pins;
stock VV10 (blocked); BP86/B97/OLYP/HCTH (molecular PySCF path);
shear/off-diagonal stress (orthorhombic design); OT+XL (architectural
exclusion); GAPW/PAW/periodic hybrid/meta-GGA (blocked); molecular
post-HF / classical MW / CUDA ports (other loops). **Found rot fixed:**
stale `gpw_solver.h` comment claimed multi-k UKS OT+DFT+U+smear residual
after `reg_ot-uks-dftu-smearing` landed. Roadmap: **no ○/✓ table flips**
(capability rows already ✓); residual prose in `dft.md` §8/§10/§12 +
`index.md` GPW + MD AIMD refreshed for wave-3 ships + wave-4 queue.
**Pending = 11**; **in_progress = 0**; **blocked = 6**.

**Recover 2026-08-01 (parent `dev` @ `446a0a3d`):** reconfirmed clean —
incomplete_count=**0**; `## in_progress` empty (no `[~]`); open CP2K/GPW/PW
specs=**0** (PySCF deferred only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`);
unmerged `wf/*`=**0** local+remote; isolation worktrees=**0**; half-regs=**0**
(no regression/spec dir for any of the 11 pending; all 9 harness wave-3 done
slug dirs present); leftover loop `build*`=**none** on parent. Stock VV10 stays
**blocked**. Do **not** invent methods / do not promote pending into staff here —
next is Assign (or GapCensus if tree drifted) from **pending**.

**GapCensus wave 5 (shift) 2026-08-01 (parent `dev` @ `446a0a3d`):** recover
clean — incomplete_count=**0**; `## in_progress` empty; open CP2K/GPW/PW
specs=**0** (PySCF deferred only); unmerged `wf/*`=**0**; isolation
worktrees=**0**; leftover loop builds=**none**; wave-3 done-slug regs 9/9
present; CP2K source build OK (`cp2k.psmp` 2026.1 `git:5e54ba2`). No capability
merges into `dev` since GapCensus wave 4 (`446a0a3d` = HEAD). Tree re-diff vs
`docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: **all 11 prior pending re-verified as
true gaps** — `GpwDft::build` still rejects multi-k UKS + nonlocal vdW
(`gpw_solver.cpp` reject “with multi-k UKS”); k-path `gpw_stress` still folds
d3/d3bj/d3_atm/d4/ts/xdm with no dedicated multi-k pin/reg; multi-k UKS BOMD
warm-start product (`guess_P_k`+`P_k_beta`) unpinned; Γ-UKS + vdW + multigrid
code path unpinned; `fold_dispersion_k_uks` Hirshfeld unpinned; multi-k UKS
CPMD eNH combo unpinned. **Added 3** tree-declared residuals with offline
CP2K / literature twins (exclude done / blocked / in_progress):
`periodic-wannier-gpw-uks` (`PeriodicWannier::compute(GpwDft&)` throws
“UKS / spin MLWF not in this foothold” after RKS GPW MLWF ✓ and PW spin MLWF
✓ — CP2K Quickstep UKS MLWF twin), `periodic-scs-mp2-si` (`mp2_os_scale` /
`mp2_ss_scale` product-shaped on Γ/multi-k RI-GPW MP2; no dedicated pin —
wave-15 hold “behind dispersion stress depth” cleared when D3/ATM/TS/XDM
stress pins shipped; CP2K SCS-MP2 twin), `gpw-optb88-uks` (Γ-UKS
`xc_grid_optb88_vdw_spin` + OptB88VdW in UKS XcKind allow-list product;
only RKS `reg_gpw-optb88-vdw` pin — same UKS-GGA expand pattern as
revPBE/RPBE/BLYP; CP2K UKS optB88-vdW twin). **Not added:** k-UKS
fixed-density + metallic smear (throw remains; hold — fractional fixed-density
path for XL/CPMD metals); multi-k UKS + vdW multigrid triple combo (hold
behind `gpw-vdw-uks-kpoints`); multi-k ATM force dedicated pins (geometry-only
ATM fold identical to Γ — thin after Γ ATM force pins); CPMD stress /
variable-cell CP (deliberate refusal); OT+XL (architectural exclusion); GPW
Berry (hold — larger surface than pin-only, after GPW MLWF UKS); BP86/B97/
OLYP/HCTH (molecular PySCF path); stock VV10 (**blocked**); shear/off-diagonal
stress (orthorhombic design); GAPW/PAW/periodic hybrid/meta-GGA (blocked);
molecular post-HF / classical MW / CUDA ports (other loops). Roadmap: **no
○/✓ table flips** (nothing capability-shaped merged); residual prose in
`dft.md` §8/§10/§11 + `index.md` GPW refreshed for wave-5 queue (SCS-MP2 +
GPW spin MLWF + OptB88 UKS). **Pending = 14**; **in_progress = 0**;
**blocked = 6**.

**Harness wave 1 (shift) 2026-08-01 (parent `dev` @ `424d6989`):** **6/10**
staffed slugs merged → `## done` (`gpw-vdw-uks-kpoints`,
`gpw-d3-kpoint-stress`, `gpw-d3bj-kpoint-stress`,
`gpw-d3-atm-kpoint-stress`, `gpw-bomd-uks-kpoints`,
`gpw-vdw-uks-multigrid`) — six `--no-ff` merges + post-merge roadmap
residual refresh (`424d6989`). Specs closed (deleted). **Incomplete kept
`[~]` ×4** (unmerged staff tips; do not demote): `gpw-d4-kpoint-stress`
(`wf/gpw-d4-kpoint-stress` @ `95a51e49`), `gpw-ts-kpoint-stress`
(`wf/gpw-ts-kpoint-stress` @ `1a4a5311`), `gpw-xdm-kpoint-stress`
(`wf/gpw-xdm-kpoint-stress` @ `e73e18d7`), `gpw-hirshfeld-uks-kpoints`
(`wf/gpw-hirshfeld-uks-kpoints` @ `bc981d93`) — feat + reg + REFERENCE on
tips; not merged this harness. **Pending residual = 4**
(`cpmd-uks-thermostat-kpoints`, `periodic-wannier-gpw-uks`,
`periodic-scs-mp2-si`, `gpw-optb88-uks` — not queue exhausted; GapCensus
owns next discovery). **in_progress = 4**; **blocked = 6** unchanged.
Clearup already ran (no isolation worktrees; incomplete `wf/*` kept). No
push.

**GapCensus wave 2 (shift) 2026-08-01 (parent `dev` @ `d32a7e40`):** recover —
incomplete_count=**4** (`[~]` multi-k D4/TS/XDM stress + multi-k UKS
Hirshfeld on unmerged `wf/*` tips; do not demote; no isolation worktrees —
tips only). Open CP2K/GPW/PW specs on `dev`=**0** (PySCF deferred only:
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); unmerged feature `wf/*`=**4**
(incomplete only); half-regs on `dev`=**0** (no reg dir for any pending or
incomplete slug); leftover loop builds=**none**; concurrent uncommitted
tutorial docs left untouched. CP2K source build OK (`cp2k.psmp` 2026.1
`git:5e54ba2`). No capability merges into `dev` since harness close
(`d32a7e40` = HEAD staff-close on top of `424d6989`). Tree re-diff vs
`docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: **all 4 prior pending re-verified
as true gaps** — multi-k UKS CPMD eNH combo unpinned; GPW
`PeriodicWannier::compute` still throws UKS
(`periodic_wannier.cpp:818`); SCS scales product-shaped but unpinned
(`mp2_os_scale`/`mp2_ss_scale`); Γ-UKS `xc_grid_optb88_vdw_spin` +
`XcKind::OptB88VdW` allow-list product, only RKS `reg_gpw-optb88-vdw` pin.
**Added 4** tree-declared residuals with offline CP2K / literature twins
(exclude done / blocked / in_progress):
`gpw-vdw-uks-kpoint-multigrid` (multi-k UKS × nonlocal vdW × REL_CUTOFF
multigrid triple combo unpinned — each axis pair product after wave-1
`gpw-vdw-uks-kpoints` + `gpw-vdw-uks-multigrid` + multi-k UKS multigrid;
wave-5 hold “behind multi-k UKS vdW” cleared; CP2K NGRIDS×DRSLL×UKS×MP
twin), `gpw-optb88-kpoints` (multi-k RKS optB88-vdW product-shaped on the
k allow-list / `xc_grid_optb88_vdw`; only Γ RKS pin — same multi-k expand
pattern as vdW-DF k; CP2K KPOINTS + optB88-vdW twin),
`gpw-d3-atm-kpoint-forces` (multi-k force assembly already folds
`d3_atm` identically to Γ; dedicated offline multi-k ATM force pin after
Γ ATM forces ✓ + multi-k ATM stress ✓ — CP2K ENERGY_FORCE + k-mesh +
ATM C9), `gpw-d4-atm-kpoint-forces` (same multi-k ATM fold for
`d4_atm`; dftd4 3.7.0 `--grad` multi-k geometry twin — same no-DFTD4
CP2K policy as Γ `gpw-d4-atm-forces`). **Not added:** multi-k UKS
fixed-density + metallic smear (throw at `gpw_solver.cpp` k-UKS
`max_iter==0` path — real residual, but no fair CP2K twin for ATV’s
fixed-density electronic step; hold / literature-only later);
multi-k UKS optB88 (hold behind pending `gpw-optb88-uks`); multi-k
SCS-MP2 (hold behind Γ `periodic-scs-mp2-si`); GPW Berry (hold behind
`periodic-wannier-gpw-uks`); CPMD stress / variable-cell CP (deliberate
refusal); OT+XL (architectural exclusion); stock VV10 (**blocked**);
BP86/B97/OLYP/HCTH (molecular PySCF path); shear/off-diagonal stress
(orthorhombic design); GAPW/PAW/periodic hybrid/meta-GGA (blocked);
GTH↔NC-UPF cross-representation windows; molecular post-HF / classical
MW / CUDA ports (other loops). **Found rot fixed:** stale
`include/atomiverse/cpu/gpw/mp2.h` claimed multi-k UMP2 “follow-on”
after `reg_periodic-ump2-kpoints` product. Roadmap: **no ○/✓ table
flips** (nothing capability-shaped merged to `dev`); residual prose in
`dft.md` §8/§10 + `index.md` GPW refreshed for wave-2 queue (vdW×k×UKS×mg
+ multi-k optB88 + multi-k ATM force pins). **Pending = 8**;
**in_progress = 4**; **blocked = 6**.

**Harness wave 2 (shift) 2026-08-01 (parent `dev` @ `79bcfa14`):** **8/9**
staffed slugs on `dev` → `## done` (`gpw-d4-kpoint-stress`,
`gpw-xdm-kpoint-stress`, `gpw-hirshfeld-uks-kpoints`,
`cpmd-uks-thermostat-kpoints`, `periodic-wannier-gpw-uks`,
`periodic-scs-mp2-si`, `gpw-optb88-uks`, `gpw-vdw-uks-kpoint-multigrid`,
`gpw-optb88-kpoints` — the latter two already recorded under worker wave 2 /
this harness; D4/XDM/Hirshfeld/optB88/SCS/MLWF landed as feat commits on
`dev`). Specs closed (deleted). **Incomplete kept `[~]` ×1** (do not demote):
`gpw-ts-kpoint-stress` (`wf/gpw-ts-kpoint-stress` @ `65aa0e6d` — feat tip
ahead of `dev`, no reg dir on parent). **Pending residual = 2**
(`gpw-d3-atm-kpoint-forces`, `gpw-d4-atm-kpoint-forces` — not queue exhausted;
GapCensus owns next discovery). **in_progress = 1**; **blocked = 6**
unchanged. Clearup already ran (no isolation worktrees; incomplete `wf/*`
kept). No push.

**GapCensus wave 3 (shift) 2026-08-01 (parent `dev` @ `e0123118`):** recover —
incomplete_count=**1** (`[~] gpw-ts-kpoint-stress` on
`wf/gpw-ts-kpoint-stress` @ `65aa0e6d`, 1 ahead / 22 behind `dev`; feat tip +
reg on branch, no reg dir on parent; do not demote). Open CP2K/GPW/PW specs
on `dev`=**0** (PySCF deferred only: `periodic-rhf-gamma` /
`rsh-eri-cam-b3lyp`); isolation worktrees=**0**; leftover loop builds=**none**;
unmerged feature `wf/*`=**1** (incomplete only); half-regs on `dev`=**0** for
any pending slug; concurrent uncommitted tutorial docs left untouched. CP2K
source build available (`cp2k.psmp` 2026.1 `git:5e54ba2`). No capability
merges into `dev` since harness wave 2 (`e0123118` = HEAD
`chore(harness)` on top of feat commits). Tree re-diff vs
`docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: **both prior pending re-verified as
true gaps** — multi-k force assembly already folds `d3_atm` / `d4_atm`
identically to Γ (`gpw_forces.cpp` k path) but no dedicated multi-k ATM force
reg/pin exists; CP2K ENERGY_FORCE + k + ATM C9 twin (D3) and dftd4 3.7.0
`--grad` (D4; no CP2K DFTD4 — same policy as Γ). **Added 4** tree-declared
residuals with offline CP2K / literature twins (exclude done / blocked /
in_progress):
`gpw-optb88-uks-kpoints` (multi-k UKS allow-list +
`xc_grid_optb88_vdw_spin` + nonlocal `vdw_df` product on the k-UKS path;
only Γ UKS `reg_gpw-optb88-uks` + multi-k RKS `reg_gpw-optb88-kpoints` —
wave-2 hold “behind gpw-optb88-uks” cleared; CP2K KPOINTS + UKS +
OptB88+DRSLL twin),
`periodic-scs-mp2-kpoints` (`mp2_os_scale`/`mp2_ss_scale` product-shaped on
multi-k RI-GPW RKS/UMP2 paths; only Γ `reg_periodic-scs-mp2-si` — wave-2
hold “behind Γ SCS” cleared; CP2K multi-k SCALE_S/SCALE_T twin after
`periodic-mp2-kpoints` ✓),
`periodic-berry-gpw` (Berry-phase polarization is PW-only
`cpu::pw::BerryPolarization`; GPW MLWF RKS+UKS product — wave-2 hold
“behind periodic-wannier-gpw-uks” cleared; CP2K Quickstep `&MOMENTS
PERIODIC` twin class already offline as `si_berry_polarization_note.inp`),
`gpw-hirshfeld-forces` (**found rot / residual:** energy path folds
Hirshfeld volume ratios into TS/MBD (`fold_dispersion*`) but
`gpw_forces` / `gpw_stress` call free-atom `ts_pbe` / `mbd_rsscs_pbe` with
empty volumes — force/stress ignore `hirshfeld`; fix + dedicated pin;
libMBD volume-scaled gradient twin / CP2K TS ENERGY_FORCE when available).
**Not added:** multi-k UKS fixed-density + metallic smear (throw remains;
no fair CP2K twin for ATV fixed-density electronic step); multi-k SCS
UMP2 (hold behind RKS `periodic-scs-mp2-kpoints`); Hirshfeld-scaled
**stress** dedicated pin (same fold as forces — land with
`gpw-hirshfeld-forces` or thin follow-on); CPMD stress / variable-cell CP
(deliberate refusal, esmm-cpmd-02); OT+XL (architectural exclusion); stock
VV10 (**blocked**); BP86/B97/OLYP/HCTH (molecular PySCF path);
shear/off-diagonal stress (orthorhombic design); GAPW/PAW/periodic
hybrid/meta-GGA (blocked); GTH↔NC-UPF cross-representation windows;
molecular post-HF / classical MW / CUDA ports (other loops). Roadmap: **no
○/✓ table flips** (nothing capability-shaped merged; capability rows already
✓ or blocked ○); residual prose in `dft.md` §8/§10 + `index.md` GPW
refreshed for wave-3 queue. **Pending = 6**; **in_progress = 1**;
**blocked = 6**.

**Assign wave 3 (shift) 2026-08-01 (parent `dev` @ `98ab480d`):** recover —
incomplete_count=**1** (`[~] gpw-ts-kpoint-stress` unmerged
`wf/gpw-ts-kpoint-stress` @ `65aa0e6d`, 1 ahead / 23 behind `dev` — resume
staffed first, incomplete ≫ pending). Open CP2K/GPW/PW specs=**0** (PySCF
deferred only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation
worktrees=**0**; all 6 pending re-verified write-first (no
`.claude/specs/<slug>/`, no `wf/<slug>` tip, no regression dir). Staffed
**7** (≤10): 1 resume + all **6** pending → `## in_progress`. New
promotions: `gpw-d3-atm-kpoint-forces`, `gpw-d4-atm-kpoint-forces`,
`gpw-optb88-uks-kpoints`, `periodic-scs-mp2-kpoints`, `periodic-berry-gpw`,
`gpw-hirshfeld-forces`. Surface notes for workers: `gpw-d3-atm-kpoint-forces`
∥ `gpw-d4-atm-kpoint-forces` share multi-k force-assembly ATM folds —
reuse Γ ATM F2+FD / EEQ-adjoint machinery, serialize carefully;
`gpw-hirshfeld-forces` also edits `gpw_forces` / `gpw_stress` dispersion
grad (volume-scaled TS/MBD vs free-atom) — coordinate with ATM force
workers on those files; resume `gpw-ts-kpoint-stress` owns multi-k
`gpw_stress` TS pin (rebase tip onto `dev` then dual-gate) — free of
Hirshfeld force volume fix if stress only asserts free-atom TS fold;
`gpw-optb88-uks-kpoints` is pin-only on product k-UKS optB88 path
(allow-list + spin XC + DRSLL) — free of force/stress; `periodic-scs-mp2-kpoints`
is pin-only multi-k SCS scales on RI-GPW MP2 — free of all;
`periodic-berry-gpw` is new GPW Berry surface (PW-only today) — free of
force/stress/ATM/optB88. No pick depends on another pick. **Held
pending (0):** staff emptied the census list. **Pending residual after
assign = 0** (not queue exhausted; GapCensus owns next discovery after
harness).

**Harness wave 3 (shift) 2026-08-01 (parent `dev` @ `9bd134e0`):** **4/7**
staffed slugs on `dev` → `## done` (`gpw-d4-atm-kpoint-forces`,
`gpw-optb88-uks-kpoints`, `periodic-scs-mp2-kpoints`, `periodic-berry-gpw`
— feat commits on `dev`; specs closed). **Incomplete kept `[~]` ×3** (do
not demote): `gpw-ts-kpoint-stress` (`wf/gpw-ts-kpoint-stress` @
`4afc4185`), `gpw-d3-atm-kpoint-forces` (`wf/gpw-d3-atm-kpoint-forces` @
`a4168475`), `gpw-hirshfeld-forces` (`wf/gpw-hirshfeld-forces` @
`4f3a62c6`) — feat tips ahead of `dev`, no reg dir on parent. **Pending
residual = 0** (staff emptied census list — **not** queue exhausted;
GapCensus owns next discovery). **in_progress = 3**; **blocked = 6**
unchanged. Clearup already ran (no isolation worktrees; incomplete `wf/*`
kept). No push.

**GapCensus wave 4 (shift) 2026-08-01 (parent `dev` @ `b6b2e927`):** recover —
incomplete_count=**3** (`[~] gpw-ts-kpoint-stress` @ `4afc4185`, 1 ahead /
~91 behind `dev`; `[~] gpw-d3-atm-kpoint-forces` @ `a4168475`;
`[~] gpw-hirshfeld-forces` @ `4f3a62c6` — feat tips + reg on branch only;
do **not** demote). Open CP2K/GPW/PW specs on `dev`=**0** (PySCF deferred
only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation worktrees=**0**;
leftover loop builds=**none**; unmerged feature `wf/*`=**3** (incomplete
only); half-regs on `dev`=**0** for any pending slug; concurrent uncommitted
tutorial docs left untouched. CP2K source build available (`cp2k.psmp`
2026.1 `git:5e54ba2`). No capability merges into `dev` since harness wave 3
(`b6b2e927` = HEAD `chore(harness)` on top of feat commits). Tree re-diff
vs `docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending residual **0**
(harness emptied list — **not** capability-closed). **True residuals
re-opened (9)** with offline CP2K / literature twins (exclude done /
blocked / in_progress):
`gpw-d3-kpoint-forces` / `gpw-d3bj-kpoint-forces` / `gpw-d4-kpoint-forces`
(multi-k force assembly already folds d3 / d3bj / d4 two-body identically
to Γ — `gpw_forces.cpp` k-path; dedicated multi-k pins missing after Γ
force pins + multi-k stress pins + multi-k D4 ATM forces ✓; CP2K
ENERGY_FORCE + k-mesh for D3/D3(BJ), dftd4/`libdftd4` for D4),
`gpw-d4-atm-kpoint-stress` (k-path fixed-\(P_k\) stress rebuilds
`d4_atm` via \(s_9\); Γ ATM stress + multi-k D4 two-body stress ✓ —
dedicated multi-k ATM pin missing; dftd4 strained-cell twin),
`gpw-ts-kpoint-forces` / `gpw-xdm-kpoint-forces` (multi-k force assembly
folds free-atom TS/XDM; Γ force + multi-k stress pins product — dedicated
multi-k force pins missing; libMBD / pure-Python twins, same family policy
as Γ),
`periodic-berry-gpw-uks` (`BerryPolarization::compute(GpwDft&)` has a
product-shaped UKS dual-channel string path; only RKS
`reg_periodic-berry-gpw` — CP2K Quickstep `&MOMENTS PERIODIC` UKS twin
class),
`periodic-scs-ump2-si` (`mp2_os_scale`/`mp2_ss_scale` already applied on
Γ UMP2; only closed-shell SCS pins — CP2K UKS SCALE_S/SCALE_T twin after
`periodic-scs-mp2-si` ✓ / `periodic-ump2-si` ✓),
`dfpt-gga-spin-smearing` (PW DFPT product paths cover GGA×spin, spin×smear,
GGA×smear, and finite-q joints separately; joint **Γ collinear spin × pure
GGA × metallic smear** unpinned — CP2K UKS-PBE + smear
`VIBRATIONAL_ANALYSIS` twin).
**Not added:** multi-k UKS fixed-density + metallic smear (throw at
`gpw_solver.cpp` k-UKS `max_iter==0` — real residual, no fair CP2K twin for
ATV fixed-density electronic step); multi-k SCS UMP2 (hold behind Γ
`periodic-scs-ump2-si`); CPMD stress / variable-cell CP (deliberate
refusal, esmm-cpmd-02); OT+XL (architectural exclusion); stock VV10
(**blocked**); BP86/B97/OLYP/HCTH (molecular PySCF path); shear/off-diagonal
stress (orthorhombic design); GAPW/PAW/periodic hybrid/meta-GGA (blocked);
GTH↔NC-UPF cross-representation windows; molecular post-HF / classical MW
/ CUDA ports (other loops); items already `[~]` (`gpw-ts-kpoint-stress`,
`gpw-d3-atm-kpoint-forces`, `gpw-hirshfeld-forces`). Roadmap: **no ○/✓
table flips** (nothing capability-shaped merged; incompletes unmerged);
residual prose in `dft.md` §8/§10 + `index.md` GPW refreshed for wave-4
queue. **Pending = 9**; **in_progress = 3**; **blocked = 6**.

**Harness wave 4 (shift) 2026-08-01 (parent `dev` @ `6f1218b3`):** **9/9**
staffed slugs on `dev` → `## done` (`gpw-d3-atm-kpoint-forces`,
`gpw-hirshfeld-forces`, `gpw-d3-kpoint-forces`, `gpw-d3bj-kpoint-forces`,
`gpw-d4-kpoint-forces`, `gpw-d4-atm-kpoint-stress`, `periodic-berry-gpw-uks`,
`periodic-scs-ump2-si`, `dfpt-gga-spin-smearing` — feat merges
`e52e4396`..`6f1218b3`; specs closed). **Incomplete kept `[~]` ×1** (do
not demote): `gpw-ts-kpoint-stress` (`wf/gpw-ts-kpoint-stress` @
`2d0977c8` — feat tip; no reg dir on parent). **Pending residual = 2**
(`gpw-ts-kpoint-forces`, `gpw-xdm-kpoint-forces` — not staffed this wave;
**not** queue exhausted; GapCensus owns next discovery). **in_progress =
1**; **blocked = 6** unchanged. Clearup already ran (no isolation
worktrees; merged `wf/*` pruned; incomplete `wf/*` kept). No push.

**GapCensus wave 5 (shift) 2026-08-01 (parent `dev` @ `d6d4d02a`):** recover —
incomplete_count=**1** (`[~] gpw-ts-kpoint-stress` on
`wf/gpw-ts-kpoint-stress` @ `2d0977c8`, 1 ahead / many behind `dev`; feat tip
+ `reg_gpw-ts-kpoint-stress` + libMBD strain notes on branch only; do **not**
demote). Open CP2K/GPW/PW specs on `dev`=**0** (PySCF deferred only:
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation worktrees=**0**;
leftover loop builds=**none**; unmerged feature `wf/*`=**1** (incomplete
only); half-regs on `dev`=**0** for any pending slug (no `gpw-ts-kpoint-*` /
`gpw-xdm-kpoint-forces` dir on parent); concurrent uncommitted tutorial docs
left untouched. CP2K source build OK (`cp2k.psmp` 2026.1 `git:5e54ba2` via
`…/cp2k/install/env.sh`). No capability merges into `dev` since harness wave 4
(`d6d4d02a` = HEAD `chore(harness)`). Tree re-diff vs `docs/roadmap/dft.md`
§3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: **both prior pending re-verified as
true gaps** — multi-k force assembly already folds free-atom **TS** / **XDM**
identically to Γ (`gpw_forces.cpp` k-path ~1378–1459) but no dedicated multi-k
force reg/pin exists (Γ force + multi-k stress product; incomplete multi-k TS
stress tip separate). **Added 5** tree-declared residuals with offline CP2K /
literature twins (exclude done / blocked / in_progress):
`gpw-uks-pbesol` (Γ/k UKS XcKind allow-list + `xc_grid_pbesol_spin` product
on energy/forces/stress; only RKS `reg_gpw-pbesol-si` — same UKS-GGA expand
pattern as revPBE/RPBE/BLYP; CP2K UKS PBEsol twin),
`periodic-scs-ump2-kpoints` (`mp2_os_scale`/`mp2_ss_scale` already applied on
multi-k UMP2 path `mp2.cpp`; only Γ `reg_periodic-scs-ump2-si` — wave-4 hold
“behind Γ SCS-UMP2” cleared; CP2K 2026.1 aborts MP2_GPW under DFT KPOINTS —
algebraic dual + literature family as `periodic-scs-mp2-kpoints`),
`dfpt-gga-spin-finite-q-smearing` (PW finite-q collinear spin Sternheimer
already supports pure GGA FD kernel + metallic smear jointly in
`dfpt_q_spin_impl`; Γ joint `reg_dfpt-gga-spin-smearing` + finite-q joints
without the full four-way pin — CP2K UKS-PBE+SMEAR supercell
`VIBRATIONAL_ANALYSIS` twin),
`gpw-hirshfeld-kpoint-forces` (multi-k force assembly already folds
`ts_mbd_vol_ratio` / SCF `volume_ratios` into TS/MBD; only Γ
`reg_gpw-hirshfeld-forces` + multi-k energy Hirshfeld — libMBD volume-scaled
gradient dual),
`gpw-dftu-forces` (Γ UKS DFT+U energy + stress product; multi-k U forces
`reg_gpw-dftu-kpoint-forces`; Γ force assembly already packs `plus_u` into
the W-correction path — dedicated Γ ENERGY_FORCE + PLUS_U pin missing; CP2K
twin class as `gpw-dftu-nio` + forces).
**Not added:** multi-k UKS fixed-density + metallic smear (throw at
`gpw_solver.cpp` k-UKS `max_iter==0` — real residual, no fair CP2K twin for
ATV fixed-density electronic step); CPMD stress / variable-cell CP
(deliberate refusal, esmm-cpmd-02); OT+XL (architectural exclusion); XL /
CPMD + metallic fixed-density (same fixed-density+smear family); stock VV10
(**blocked**); BP86/B97/OLYP/HCTH (molecular PySCF path); shear/off-diagonal
stress (orthorhombic design); GAPW/PAW/periodic hybrid/meta-GGA (blocked);
GTH↔NC-UPF cross-representation windows; molecular post-HF / classical MW /
CUDA ports (other loops); items already `[~]` (`gpw-ts-kpoint-stress`).
Roadmap: **no ○/✓ table flips** (nothing capability-shaped merged;
incompletes unmerged); residual prose in `dft.md` intro/§8/§10 + `index.md`
GPW refreshed for wave-5 queue. **Pending = 7**; **in_progress = 1**;
**blocked = 6**.

**Assign wave 5 (shift) 2026-08-01 (parent `dev` @ `e8ed5b6c`):** recover —
incomplete_count=**1** (`[~] gpw-ts-kpoint-stress` unmerged
`wf/gpw-ts-kpoint-stress` @ `2d0977c8`, 1 ahead / ~221 behind `dev` — resume
staffed first, incomplete ≫ pending). Open CP2K/GPW/PW specs=**0** (PySCF
deferred only: `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation
worktrees=**0**; all 7 pending re-verified write-first (no
`.claude/specs/<slug>/`, no `wf/<slug>` tip, no regression dir on parent).
Staffed **8** (≤10): 1 resume + **all 7** pending → `## in_progress`. New
promotions: `gpw-ts-kpoint-forces`, `gpw-xdm-kpoint-forces`,
`gpw-uks-pbesol`, `periodic-scs-ump2-kpoints`,
`dfpt-gga-spin-finite-q-smearing`, `gpw-hirshfeld-kpoint-forces`,
`gpw-dftu-forces`. Surface notes for workers: resume
`gpw-ts-kpoint-stress` — rebase `wf/*` onto `dev` (behind ~221) then finish
acceptance / dual-gate (reg + libMBD pins already on tip); owns multi-k
`gpw_stress` free-atom TS pin. `gpw-ts-kpoint-forces` ∥
`gpw-xdm-kpoint-forces` ∥ `gpw-hirshfeld-kpoint-forces` share multi-k
`gpw_forces` dispersion fold — serialize or split patches carefully
(TS free-atom vs XDM free-atom vs Hirshfeld volume-scaled; Γ force + multi-k
stress product; dedicated multi-k force pins missing). `gpw-dftu-forces` is
Γ UKS PLUS_U force pin (NiO-class) — coordinate lightly with multi-k force
workers on `gpw_forces.cpp` if touching the same pack sites. `gpw-uks-pbesol`
is pin-only Γ-UKS PBEsol energy (spin XC product) — free of force/stress.
`periodic-scs-ump2-kpoints` is pin-only multi-k SCS scales on RI-GPW UMP2 —
free of all (algebraic dual + literature; CP2K aborts MP2_GPW@k).
`dfpt-gga-spin-finite-q-smearing` is PW response four-way joint pin — free of
GPW force/stress surface. No pick depends on another pick. **Held pending
(0):** staff emptied the census list. **Pending residual after assign = 0**
(not queue exhausted; GapCensus owns next discovery after harness).

**Harness wave 5 (shift) 2026-08-01 (parent `dev` @ `350fe453`):** **6/8**
staffed slugs on `dev` → `## done` (`gpw-xdm-kpoint-forces`,
`gpw-uks-pbesol`, `periodic-scs-ump2-kpoints`,
`dfpt-gga-spin-finite-q-smearing`, `gpw-hirshfeld-kpoint-forces`,
`gpw-dftu-forces` — feat merges `c8d29f73`..`350fe453`; specs closed ×3).
**Incomplete kept `[~]` ×2** (do not demote): `gpw-ts-kpoint-stress`
(`wf/gpw-ts-kpoint-stress` @ `f86aec1a` — feat tip + reg on branch only),
`gpw-ts-kpoint-forces` (`wf/gpw-ts-kpoint-forces` @ `3eb2a6ee` — feat tip +
reg on branch only). **Pending residual = 0** (census list emptied by
assign — **not** queue exhausted; GapCensus owns next discovery).
**in_progress = 2**; **blocked = 6** unchanged. Clearup already ran (no
isolation worktrees; merged `wf/*` pruned; incomplete `wf/*` kept). No
push.

**GapCensus wave 6 (shift) 2026-08-01 (parent `dev` @ `de9cda3a`):** recover —
incomplete_count=**2** (`[~] gpw-ts-kpoint-stress` @ `f86aec1a`, 1 ahead /
~131 behind `dev`; `[~] gpw-ts-kpoint-forces` @ `3eb2a6ee` — feat tips +
regs on branch only; do **not** demote). Open CP2K/GPW/PW specs on
`dev`=**0** (PySCF deferred only: `periodic-rhf-gamma` /
`rsh-eri-cam-b3lyp`); isolation worktrees=**0**; leftover loop builds=**none**;
unmerged feature `wf/*`=**2** (incomplete only); half-regs on `dev`=**0** for
any pending slug; concurrent uncommitted tutorial docs left untouched. CP2K
source build OK (`cp2k.psmp` 2026.1 `git:5e54ba2` via
`…/cp2k/install/env.sh`). No capability merges into `dev` since harness wave 5
(`de9cda3a` = HEAD `chore(harness)`). Tree re-diff vs `docs/roadmap/dft.md`
§3/§8/§10–12 + MD AIMD + `tools/cp2k_oracle` +
`include/atomiverse/cpu/{gpw,planewave}`: prior pending residual **0**
(harness emptied list — **not** capability-closed). **True residuals
re-opened (9)** with offline CP2K / literature twins (exclude done /
blocked / in_progress):
`gpw-metallic-forces` (multi-k RKS FD/MP energy product; force assembly
rebuilds \(P_k/W_k\) from integer \(n_{\mathrm{occ}}\) `C_{\mathrm{occ},k}` —
`store_k_payload` always uses fixed-occ density; free-energy Hellmann–Feynman
+ Pulay under fractional occupations unpinned; CP2K ENERGY_FORCE + SMEAR twin
on Na-class metallic cell),
`gpw-metallic-stress` (same integer-payload residual on multi-k analytic
stress; free-energy fixed-\(P_k\) stress under smear unpinned; CP2K
`STRESS_TENSOR NUMERICAL` + SMEAR twin),
`gpw-uks-metallic-forces` (multi-k UKS FD/MP energy product
`reg_gpw-uks-smearing`; force assembly same integer \(n_\alpha/n_\beta\)
rebuild; CP2K UKS ENERGY_FORCE + SMEAR twin),
`gpw-vdw-kpoint-forces` (multi-k RKS nonlocal vdW energy product
`reg_gpw-vdw-kpoints`; force path already adds \(V_{\mathrm{nl}}\) into k-path
`veff` in `gpw_forces.cpp` — dedicated multi-k ENERGY_FORCE pin missing after
Γ `reg_gpw-vdw-scf-potential`; CP2K NON_LOCAL + KPOINTS ENERGY_FORCE twin),
`gpw-optb88-forces` (OptB88+DRSLL energy product Γ/multi-k/UKS; SC nonlocal
forces product via `vdw_df` + `assemble_gpw_forces` — only energy pins; CP2K
OptB88 + DRSLL ENERGY_FORCE twin),
`gpw-hirshfeld-kpoint-stress` (multi-k stress fold already honors
`hirshfeld` + SCF `volume_ratios`; Γ stress gate inside
`reg_gpw-hirshfeld-forces`; dedicated multi-k fixed-\(v\) stress pin missing
after multi-k Hirshfeld forces ✓ — libMBD / product dual; no CP2K stock
TS/MBD),
`gpw-bands-si` (PW `cpu::pw::Bands` product; **no** GPW-grid band-structure
post-process on Bloch \(C_k/\varepsilon_k\); CP2K Quickstep
`&PRINT &BAND_STRUCTURE` twin on GTH-GPW Si),
`gpw-dos-si` (PW `cpu::pw::Dos` product; **no** GPW DOS on the multi-k mesh;
CP2K Quickstep DOS / PDOS twin class),
`gpw-bomd-smearing` (plain multi-k BOMD product fixed-occ RKS/UKS; metallic
smear BOMD / free-energy trajectory unpinned; CP2K Quickstep AIMD + SMEAR
twin).
**Not added:** multi-k UKS fixed-density + metallic smear (throw at
`gpw_solver.cpp` k-UKS `max_iter==0` — real residual, **no fair CP2K twin**
for ATV fixed-density electronic step); CPMD stress / variable-cell CP
(deliberate refusal, esmm-cpmd-02); OT+XL (architectural exclusion); stock
VV10 (**blocked**); BP86/B97/OLYP/HCTH (molecular PySCF path);
shear/off-diagonal stress (orthorhombic design); GAPW/PAW/periodic
hybrid/meta-GGA (blocked); GTH↔NC-UPF cross-representation windows;
molecular post-HF / classical MW / CUDA ports (other loops); items already
`[~]` (`gpw-ts-kpoint-stress`, `gpw-ts-kpoint-forces`); Γ closed-shell RKS
DFT+U (design reject — UKS or k-path only); periodic hybrid / meta-GGA
(blocked, no EXX / no τ); multi-k UKS metallic stress (hold behind RKS
metallic stress + UKS metallic forces depth this wave); GPW DFPT (large
new Sternheimer surface — hold behind metallic free-energy forces depth);
NPT AIMD composition (MD barostat + product stress — not a method-domain
gap). Roadmap: **no ○/✓ table flips** (nothing capability-shaped merged;
incompletes unmerged); residual prose in `dft.md` intro/§8/§10/§12 +
`index.md` GPW refreshed for wave-6 queue (metallic free-energy
forces/stress, multi-k vdW forces, optB88 forces, multi-k Hirshfeld
stress, GPW bands/DOS, metallic BOMD). **Pending = 9**; **in_progress =
2**; **blocked = 6**.

**Harness wave 6 (shift) 2026-08-01 (parent `dev` @ `8474914d`):** 8 of 9
wave-6 staff / incompletes on `dev` → `## done` (`gpw-ts-kpoint-forces`,
`gpw-metallic-forces`, `gpw-uks-metallic-forces`, `gpw-vdw-kpoint-forces`,
`gpw-optb88-forces`, `gpw-hirshfeld-kpoint-stress`, `gpw-bands-si`,
`gpw-dos-si`). Specs deleted on close. **Pending residual = 2**
(`gpw-metallic-stress`, `gpw-bomd-smearing` — **not** queue exhausted;
GapCensus owns discovery). **in_progress = 1** unmerged staff `[~]`
(`gpw-ts-kpoint-stress` on `wf/gpw-ts-kpoint-stress` @ `d6cab501` — do not
demote). Roadmap: **no ○/✓ table flips** (capability rows already ✓);
residual prose closed multi-k free-atom TS forces, multi-k RKS/UKS free-
energy metallic forces, multi-k nonlocal vdW forces, optB88 ENERGY_FORCE,
multi-k Hirshfeld stress, GPW bands/DOS; open → multi-k free-atom **TS
stress** (`[~]`) + multi-k free-energy **metallic stress** + metallic
**BOMD**. **Blocked = 6** unchanged. Clearup already ran. No push.

**GapCensus wave 7 (shift) 2026-08-01 (parent `dev` @ `cf23bbb1`):** recover —
incomplete_count=**2** after promote: `[~] gpw-ts-kpoint-stress` on
`wf/gpw-ts-kpoint-stress` @ `d6cab501` (1 ahead / ~18 behind `dev`; feat tip +
reg on branch only; do **not** demote); **`[~] gpw-metallic-stress` promoted
from pending** — unmerged `wf/gpw-metallic-stress` @ `d697f63d` (feat + reg +
CP2K Na BCC SMEAR stress inputs + dual-gate REFERENCE; 1 ahead / ~18 behind;
incomplete ≫ pending; do **not** demote / do not leave half-landed `wf/*` as
pending). Open CP2K/GPW/PW specs on `dev`=**0** (PySCF deferred only:
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); isolation worktrees=**0**; leftover
loop builds=**none**; unmerged feature `wf/*`=**2** (incomplete only); half-regs
on `dev`=**0** for any pending slug. CP2K source build available (`cp2k.psmp`
2026.1 `git:5e54ba2` via `…/cp2k/install/env.sh`). HEAD is harness wave-6
close (`cf23bbb1`); no further capability merges into `dev` after wave-6 ships.
Tree re-diff vs `docs/roadmap/dft.md` §3/§8/§10–12 + MD AIMD +
`tools/cp2k_oracle` + `include/atomiverse/cpu/{gpw,planewave}`: prior pending
**2** re-verified — metallic free-energy stress remains unpinned on parent
(fractional \(P_k/W_k\) force path product; dedicated stress pin only on
`wf/gpw-metallic-stress`); metallic **BOMD** still unpinned (fixed-occ multi-k
RKS/UKS BOMD product). **True residuals re-opened / added (7)** with offline
CP2K twins (exclude done / blocked / in_progress):
`gpw-bomd-smearing` (prior residual — multi-k free-energy / Mermin AIMD under
FD/MP smear; CP2K Quickstep AIMD + SMEAR k-mesh),
`gpw-bands-uks` (`gpw::Bands` throws collinear UKS — “not product yet”; RKS
`reg_gpw-bands-si` ✓ + PW spin bands ✓; CP2K Quickstep UKS
`&PRINT &BAND_STRUCTURE` twin),
`gpw-bands-metallic` (`gpw::Bands` throws metallic smear — “not product yet”;
PW `reg_bands-metallic-si` pattern; CP2K SMEAR + BAND_STRUCTURE twin on
GTH-GPW metal),
`gpw-dos-uks` (`Dos::compute` already dual-spin on multi-k UKS payload; only
RKS `reg_gpw-dos-si` pin — dedicated UKS DOS pin after foothold; CP2K UKS DOS
twin class),
`gpw-dos-metallic` (`Dos::compute` already uses \(E_{\mathrm{F}}\) under smear
for RKS integrate; dedicated metallic DOS pin missing after PW
`reg_dos-metallic-si` / RKS GPW DOS foothold; CP2K SMEAR + DOS twin),
`gpw-vdw-stress` (analytic fixed-\(P\) stress Lagrangian folds semi-local XC +
geometry D3/D4/TS/XDM/MBD but **omits nonlocal** \(E_{\mathrm{c}}^{\mathrm{nl}}\)
vdW-DF/DF2/rVV10 under strain — forces already fold \(V_{\mathrm{nl}}\); CP2K
`STRESS_TENSOR` + `NON_LOCAL` twin),
`gpw-uks-metallic-stress` (multi-k UKS free-energy stress under FD/MP after
UKS metallic forces ✓; dedicated UKS stress pin; CP2K UKS
`STRESS_TENSOR NUMERICAL` + SMEAR — dep incomplete RKS
`gpw-metallic-stress`).
**Not added:** multi-k UKS fixed-density + metallic smear (throw at
`gpw_solver.cpp` k-UKS `max_iter==0` — real residual, **no fair CP2K twin**
for ATV fixed-density electronic step); CPMD stress / variable-cell CP
(deliberate refusal, esmm-cpmd-02); OT+XL (architectural exclusion); stock
VV10 (**blocked**); BP86/B97/OLYP/HCTH (molecular PySCF path);
shear/off-diagonal stress (orthorhombic design); GAPW/PAW/periodic
hybrid/meta-GGA (**blocked**); GTH↔NC-UPF cross-representation windows;
molecular post-HF / classical MW / CUDA ports (other loops); Bands multigrid
/ DFT+U / dispersion path flags (thin hold behind UKS + metallic band pins);
joint spin×metallic GPW bands (hold behind `gpw-bands-uks` +
`gpw-bands-metallic`); optB88 dedicated stress pin (semi-local OptB88 already
in stress XC; nonlocal fold is `gpw-vdw-stress`); items already `[~]`
(`gpw-ts-kpoint-stress`, `gpw-metallic-stress`). Roadmap: **no ○/✓ table
flips** (nothing capability-shaped merged; incompletes unmerged); residual
prose in `dft.md` intro/§10/§12 + `index.md` GPW + MD AIMD refreshed for
wave-7 queue (promote metallic stress → incomplete; open bands UKS/metallic,
DOS UKS/metallic, nonlocal vdW stress, UKS metallic stress, metallic BOMD).
**Pending = 7**; **in_progress = 2**; **blocked = 6**.

## pending

(empty — wave-8 closed the residual staff from wave-7 cancel; GapCensus owns next discovery)

## in_progress

(empty)

## done (wave 8 residual close 2026-08-04 — post wave-7 cancel)

Implemented from live specs left after wave-7 cancel; dual-gate regs + oracle
inputs under `tools/cp2k_oracle/inputs/<slug>/`. Specs deleted on close.

- [x] `gpw-bands-uks` | Collinear UKS band structure NSCF on frozen dual \(V_{\mathrm{eff}}^\sigma\) | gpw | `regressions/gpw/gpw-bands-uks` (CP2K UKS BAND_STRUCTURE class; α≡β; gap 0.89 eV; wall ~7.6 s) | ratio **TBD offline** (ATV 7.56 s) | 2026-08-04
- [x] `gpw-bands-metallic` | Multi-k RKS metallic FD SCF + NSCF band path | gpw | `regressions/gpw/gpw-bands-metallic` (BCC Na; \(E_F\) class; spectrum straddles \(E_F\); wall ~1.8 s) | ratio **TBD offline** | 2026-08-04
- [x] `gpw-vdw-stress` | Analytic fixed-\(P\) stress folds nonlocal vdW-DF \(E_{\mathrm{c}}^{\mathrm{nl}}\) | gpw | `regressions/gpw/gpw-vdw-stress` (water dimer DRSLL; max\|an−FD\| 7.2e-7; bare diff 1.5e-6) | ratio **TBD offline** (ATV SCF 10.5 s) | 2026-08-04
- [x] `gpw-uks-metallic-stress` | Multi-k UKS free-energy analytic cell stress under FD | gpw | `regressions/gpw/gpw-uks-metallic-stress` (max\|an−FD\| 2.4e-8; UKS≡RKS 3e-14; mean σ pin) | ratio **~0.16 class** (ATV SCF 0.78 s) | 2026-08-04
- [x] `gpw-bomd-smearing` | Multi-k BOMD under FD smear with fractional \(P_k\) warm start | dynamics | `regressions/gpw/gpw-bomd-smearing` (H₂ NVE; \(f_k\) each step; drift 4.3e-3 ≤ 0.10) | ratio **TBD offline** | 2026-08-04

## done (wave 7 partial land 2026-08-01 — loop cancelled before harness)

Green workers merged to `dev` before cancel (merge commits on parent); remaining
staff killed with **no unique commits**. Specs for landed slugs closed.

- [x] `gpw-ts-kpoint-stress` | Multi-k analytic cell stress with free-atom TS | gpw | `regressions/gpw/gpw-ts-kpoint-stress` (libMBD strain dual-gate; max\|an−FD\| 1.36e-9; no stock CP2K TS) | ratio **perf_exempt** (no fair CP2K twin) | 2026-08-01
- [x] `gpw-metallic-stress` | Multi-k RKS free-energy analytic cell stress under FD smear | gpw | `regressions/gpw/gpw-metallic-stress` (analytic vs multi-SCF FD 2.4e-8; CP2K NUMERICAL+SMEAR class; fair ENERGY wall) | ratio **~0.16** (0.15 / 0.92) | 2026-08-01
- [x] `gpw-dos-uks` | Multi-k collinear UKS DOS dedicated pin | gpw | `regressions/gpw/gpw-dos-uks` (CP2K 2026.1 LSD+DOS Si BCT; \|ΔE\| 2.23e-5 Ha) | ratio **2.64** (13.65 / 5.16) | 2026-08-01
- [x] `gpw-dos-metallic` | Multi-k metallic FD DOS dedicated pin (re-pin / merge on wave-7 tips) | gpw | `regressions/gpw/gpw-dos-metallic` (CP2K 2026.1 SMEAR+DOS BCC Na; \(E_F\) 1.35e-3 Ha) | ratio **5.48** (2.74 / 0.50) | 2026-08-01

## done (harness wave 6 (shift) 2026-08-01)

- [x] `gpw-dos-metallic` | Density of states under metallic FD/MP smearing on GPW multi-k mesh (dedicated pin) | gpw | `regressions/gpw/gpw-dos-metallic` (CP2K 2026.1 SMEAR+DOS BCC Na; \(E_F\) 1.35e-3 Ha; \(n_e\) 17.999) | ratio **5.48** (2.736 / 0.499) | 2026-08-01

- [x] `gpw-ts-kpoint-forces` | Multi-k analytic ionic forces with free-atom TS (dedicated pin) | gpw | `regressions/gpw/gpw-ts-kpoint-forces` (libMBD / free-atom dual-gate + multi-k product wiring; CP2K has no stock TS) | ratio **perf_exempt** (no fair CP2K twin) | 2026-08-01
- [x] `gpw-metallic-forces` | Multi-k RKS free-energy analytic ionic forces under FD/MP metallic smearing | gpw | `regressions/gpw/gpw-metallic-forces` (CP2K 2026.1 ENERGY_FORCE + SMEAR H₂ twin; fractional \(P_k/W_k\) payload) | ratio **0.077** (ATV ~0.11 s / CP2K 1.43 s) | 2026-08-01
- [x] `gpw-uks-metallic-forces` | Multi-k UKS free-energy analytic ionic forces under FD/MP metallic smearing | gpw | `regressions/gpw/gpw-uks-metallic-forces` (CP2K 2026.1 UKS ENERGY_FORCE + SMEAR H₂ twin) | ratio **0.049** (0.097 / 2.00) | 2026-08-01
- [x] `gpw-vdw-kpoint-forces` | Multi-k analytic ionic forces with nonlocal vdW-DF (SC \(V_{\mathrm{nl}}\) on k-path) | gpw | `regressions/gpw/gpw-vdw-kpoint-forces` (CP2K 2026.1 NON_LOCAL + KPOINTS ENERGY_FORCE water dimer) | ratio **1.6–3.3** (host-load band; ≤ 10) | 2026-08-01
- [x] `gpw-optb88-forces` | Analytic ionic forces with OptB88 exchange + DRSLL nonlocal | gpw | `regressions/gpw/gpw-optb88-forces` (CP2K 2026.1 OptB88 + DRSLL ENERGY_FORCE; E_disp 3.77e-5 Ha) | ratio **1.108** (19.403 / 17.509) | 2026-08-01
- [x] `gpw-hirshfeld-kpoint-stress` | Multi-k analytic cell stress with Hirshfeld-scaled TS/MBD volumes | gpw | `regressions/gpw/gpw-hirshfeld-kpoint-stress` (fixed-\(v\) dual-gate + multi-k product wiring; no CP2K stock TS/MBD) | ratio **perf_exempt** (no fair CP2K twin) | 2026-08-01
- [x] `gpw-bands-si` | Band structure NSCF on frozen Veff (GPW Bloch path) | gpw | `regressions/gpw/gpw-bands-si` (CP2K 2026.1 Quickstep `&PRINT &BAND_STRUCTURE` GTH-GPW Si) | ratio **3.51** (6.66 / 1.90) | 2026-08-01
- [x] `gpw-dos-si` | Density of states on GPW multi-k mesh | gpw | `regressions/gpw/gpw-dos-si` (CP2K 2026.1 Quickstep DOS on GTH-GPW Si; \|ΔE\| 2.23e-5 Ha) | ratio **1.62** (4.209 / 2.595) | 2026-08-01

## done (harness wave 5 2026-08-01)

- [x] `gpw-xdm-kpoint-forces` | Multi-k analytic ionic forces with free-atom XDM (dedicated pin) | gpw | `regressions/gpw/gpw-xdm-kpoint-forces` (dual pure-Python energy/force dual-gate + multi-k product wiring; CP2K has no stock XDM) | ratio **perf_exempt** (no fair CP2K twin) | 2026-08-01
- [x] `gpw-uks-pbesol` | Γ-UKS PBEsol GGA (spin XC product; dedicated energy pin) | gpw | `regressions/gpw/gpw-uks-pbesol` (CP2K 2026.1 UKS + PBEsol O₂; \|ΔE\| 3.38e-5 Ha) | ratio **3.37** (62.33 / 18.52) | 2026-08-01
- [x] `periodic-scs-ump2-kpoints` | Multi-k SCS scales on RI-GPW UMP2@UKS (`mp2_os_scale`/`mp2_ss_scale`) | gpw | `regressions/gpw/periodic-scs-ump2-kpoints` (Grimme algebraic dual on product multi-k UMP2; CP2K offline twin aborts MP2_GPW under DFT KPOINTS — `oracle: literature`) | ratio **perf_exempt** (no multi-k MP2_GPW in CP2K 2026.1) | 2026-08-01
- [x] `dfpt-gga-spin-finite-q-smearing` | DFPT phonons with joint finite-q × collinear spin × pure GGA (PBE) × metallic smear (PW Sternheimer) | response | `regressions/response/dfpt-gga-spin-finite-q-smearing` (CP2K 2026.1 UKS-PBE+SMEAR supercell `VIBRATIONAL_ANALYSIS` + free-energy FD dual) | ratio **0.25** (148.1 / 585.9) | 2026-08-01
- [x] `gpw-hirshfeld-kpoint-forces` | Multi-k analytic ionic forces with Hirshfeld-scaled TS/MBD volumes (dedicated pin) | gpw | `regressions/gpw/gpw-hirshfeld-kpoint-forces` (fixed-\(v\) dual-gate + multi-k product wiring; no CP2K stock TS/MBD) | ratio **perf_exempt** (no fair CP2K twin) | 2026-08-01
- [x] `gpw-dftu-forces` | Γ-UKS analytic ionic forces with Dudarev DFT+U (dedicated pin) | gpw | `regressions/gpw/gpw-dftu-forces` (full-6N FD dual-gate + CP2K ENERGY_FORCE + PLUS_U NiO energy pin) | ratio **2.483** (NiO SCF+force vs CP2K 13.24 s) | 2026-08-01

## done (harness wave 4 (shift) 2026-08-01)

- [x] `gpw-d3-atm-kpoint-forces` | Multi-k analytic ionic forces with D3 three-body ATM (C9) dedicated pin | gpw | `regressions/gpw/gpw-d3-atm-kpoint-forces` (CP2K 2026.1 multi-k ENERGY_FORCE C9 on−off ΔF; max\|dF_ATM−CP2K\| 2.62e-10 Ha/Bohr) | ratio **9.22** (13.28 / 1.44) | 2026-08-01
- [x] `gpw-hirshfeld-forces` | Analytic ionic forces (and stress fold) with Hirshfeld-scaled TS/MBD volumes | gpw | `regressions/gpw/gpw-hirshfeld-forces` (fixed-\(v\) force dual-gate + product wiring; found rot: energy used ratios while force/stress called free-atom kernels) | ratio **perf_exempt** (no CP2K stock TS/MBD) | 2026-08-01
- [x] `gpw-d3-kpoint-forces` | Multi-k analytic ionic forces with DFT-D3 zero damping (dedicated pin) | gpw | `regressions/gpw/gpw-d3-kpoint-forces` (CP2K 2026.1 multi-k ENERGY_FORCE D3 on−off ΔF; max\|ΔF\| 7.86e-7 Ha/Bohr) | ratio **4.353** (16.284 / 3.741) | 2026-08-01
- [x] `gpw-d3bj-kpoint-forces` | Multi-k analytic ionic forces with DFT-D3(BJ) (dedicated pin) | gpw | `regressions/gpw/gpw-d3bj-kpoint-forces` (CP2K 2026.1 multi-k ENERGY_FORCE BJ ΔF) | ratio **3.544** (13.933 / 3.931) | 2026-08-01
- [x] `gpw-d4-kpoint-forces` | Multi-k analytic ionic forces with DFT-D4 two-body BJ (dedicated pin) | gpw | `regressions/gpw/gpw-d4-kpoint-forces` (CP2K 2026.1 libdftd4 multi-k ENERGY_FORCE D4 ΔF) | ratio **2.426** (9.987 / 4.117) | 2026-08-01
- [x] `gpw-d4-atm-kpoint-stress` | Multi-k analytic cell stress with D4 three-body ATM (\(s_9\)) dedicated pin | gpw | `regressions/gpw/gpw-d4-atm-kpoint-stress` (CP2K 2026.1 libdftd4 multi-k NUMERICAL ATM−2b Δσ + dftd4 3.7.0 strained-cell; max\|σ_ATM-cp2k\| 2.0e-11; dual gate 1.4e-9; wiring 4.0e-12) | ratio **0.077** (10.58 / 137.945) | 2026-08-01
- [x] `periodic-berry-gpw-uks` | King–Smith–Vanderbilt Berry-phase polarization on GPW collinear UKS (dual-channel string phases) | response | `regressions/gpw/periodic-berry-gpw-uks` (CP2K 2026.1 Quickstep `&MOMENTS PERIODIC` UKS fair twin) | ratio **0.214** (21.09 / 98.68) | 2026-08-01
- [x] `periodic-scs-ump2-si` | Γ SCS-MP2 scales on RI-GPW UMP2@UKS (`mp2_os_scale`/`mp2_ss_scale`) | gpw | `regressions/gpw/periodic-scs-ump2-si` (algebraic dual on product UMP2 + CP2K UKS SCALE wall twin) | ratio **4.12** (15.79 / 3.83) | 2026-08-01
- [x] `dfpt-gga-spin-smearing` | DFPT phonons with joint collinear spin × pure GGA (PBE) × metallic occupation smearing (PW Γ Sternheimer) | response | `regressions/response/dfpt-gga-spin-smearing` | ratio **0.21** (CP2K 2026.1 UKS-PBE+SMEAR k-point `VIBRATIONAL_ANALYSIS` 232.5 s / ATV DFPT 48.3 s, serial; CP2K optical anchor 464–535 cm⁻¹, free-energy FD pin 17.3454 eV/Å², DFPT 17.3711) | 2026-08-01

## done (harness wave 3 (shift) 2026-08-01)

- [x] `gpw-d4-atm-kpoint-forces` | Multi-k analytic ionic forces with D4 three-body ATM (\(s_9\)) dedicated pin | gpw | `regressions/gpw/gpw-d4-atm-kpoint-forces` (CP2K 2026.1 libdftd4 multi-k ENERGY_FORCE ATM ΔF + dftd4 3.7.0 dual; max\|dF_ATM−CP2K\| 1.4e-8 Ha/Bohr) | ratio **0.613** (9.682 / 15.806) | 2026-08-01
- [x] `gpw-optb88-uks-kpoints` | Multi-k UKS optB88-vdW (OptB88 + DRSLL on k-UKS path) dedicated pin | gpw | `regressions/gpw/gpw-optb88-uks-kpoints` (CP2K 2026.1 KPOINTS + UKS + GGA_X_OPTB88_VDW + DRSLL; E_disp parity 3.77e-5 Ha; multi-k UKS≡RKS) | ratio **0.73** (G2 9.8 s / CP2K 13.29 s OMP=8) | 2026-08-01
- [x] `periodic-scs-mp2-kpoints` | Multi-k SCS-MP2 scales on RI-GPW MP2 (`mp2_os_scale`/`mp2_ss_scale`) | gpw | `regressions/gpw/periodic-scs-mp2-kpoints` (Grimme algebraic dual on product multi-k RI-GPW; CP2K offline twin aborts MP2_GPW under DFT KPOINTS — `oracle: literature`) | ratio **perf_exempt** (no multi-k MP2_GPW in CP2K 2026.1) | 2026-08-01
- [x] `periodic-berry-gpw` | King–Smith–Vanderbilt Berry-phase polarization on GPW (GpwDft Bloch) | response | `regressions/gpw/periodic-berry-gpw` (symmetry-exact Si formal-polarization + Born-charge pins; CP2K Quickstep `&MOMENTS PERIODIC` fair twin) | ratio **9.6** (11.3 / 1.18) | 2026-08-01

## done (harness wave 2 (shift) 2026-08-01)

- [x] `gpw-d4-kpoint-stress` | Multi-k analytic cell stress with DFT-D4 (BJ two-body) | gpw | `regressions/gpw/gpw-d4-kpoint-stress` (CP2K 2026.1 libdftd4 multi-k NUMERICAL stress-difference; Δσ parity 2.46e-10 Ha/Bohr³; dftd4 E_disp dual) | ratio **0.139** (13.11 / 94.525) | 2026-08-01
- [x] `gpw-xdm-kpoint-stress` | Multi-k analytic cell stress with free-atom XDM | gpw | `regressions/gpw/gpw-xdm-kpoint-stress` (dual pure-Python strain pins + multi-SCF FD dual; σ shift parity 1.1e-15 Ha/Bohr³) | ratio **perf_exempt** (no CP2K XDM pair potential) | 2026-08-01
- [x] `gpw-hirshfeld-uks-kpoints` | Multi-k UKS Hirshfeld volume ratios for TS/MBD | gpw | `regressions/gpw/gpw-hirshfeld-uks-kpoints` (O₂ multi-k UKS MBD@rsSCS E_disp pin −1.0427e-4 Ha; free-atom dual) | ratio **perf_exempt** (no CP2K TS/MBD) | 2026-08-01
- [x] `cpmd-uks-thermostat-kpoints` | Multi-k UKS CPMD electronic Nosé–Hoover (combo pin: one thermostatted `CpmdCplx` per (k, σ); k-weighted Te/Eth; E_ext envelope) | dynamics | `regressions/dynamics/cpmd-uks-thermostat-kpoints` (periodic H₂ NVE 1×1×2 unreduced, n_α=n_β=1, K0=1e-5 Ha, τ_e=50 fs; E_ext drift ≤0.10 + Te ceiling 0.05 eV + UKS-vs-RKS closed-shell identity ≤3e-4 eV) | ratio **perf_exempt** (oracle family lock: CP2K 2026.1 has no fictitious-mass CP propagator) | 2026-08-01
- [x] `periodic-wannier-gpw-uks` | GPW-grid Marzari–Vanderbilt MLWF on collinear UKS | response | `regressions/gpw/periodic-wannier-gpw-uks` (CP2K 2026.1 Quickstep UKS multi-k ENERGY twin + bond-midpoint centers; E pin 2e-3 Ha) | ratio **9.57** (183.1 / 19.129; within_10x) | 2026-08-01
- [x] `periodic-scs-mp2-si` | SCS-MP2 scales on Γ RI-GPW MP2 (`mp2_os_scale`/`mp2_ss_scale`) | gpw | `regressions/gpw/periodic-scs-mp2-si` (CP2K 2026.1 SCALE_S/SCALE_T Si₂ twin; algebraic plain-MP2 dual) | ratio **0.424** (7.55 / 17.81) | 2026-08-01
- [x] `gpw-optb88-uks` | Γ-UKS optB88-vdW (OptB88 exchange + DRSLL nonlocal) | gpw | `regressions/gpw/gpw-optb88-uks` (CP2K 2026.1 UKS O₂ optB88+DRSLL; E_disp parity 1.5e-5 Ha; closed-shell UKS≡RKS) | ratio **3.35** (within_10x) | 2026-08-01
- [x] `gpw-vdw-uks-kpoint-multigrid` | Multi-k UKS nonlocal vdW on REL_CUTOFF multigrid (triple combo pin) | gpw | `regressions/gpw/gpw-vdw-uks-kpoint-multigrid` (CP2K 2026.1 NGRIDS 4 + DRSLL + UKS + MP 2×1×1; E_disp parity 4.0e-5 Ha; multi-k UKS sg↔mg) | ratio **0.85** (G2 11.6 s / CP2K 13.703 s OMP=8) | 2026-08-01
- [x] `gpw-optb88-kpoints` | Multi-k RKS optB88-vdW (OptB88 + DRSLL on k path) | gpw | `regressions/gpw/gpw-optb88-kpoints` (CP2K 2026.1 KPOINTS + GGA_X_OPTB88_VDW + DRSLL; E_disp parity 3.77e-5 Ha) | ratio **0.67** (G2 8.4 s / CP2K ~12.6 s OMP=8) | 2026-08-01

## done (harness wave 1 (shift) 2026-08-01)

- [x] `gpw-vdw-uks-kpoints` | Multi-k UKS nonlocal vdW-DF/DF2/rVV10 (dual-spin V_nl fold) | gpw | `regressions/gpw/gpw-vdw-uks-kpoints` (CP2K 2026.1 NON_LOCAL + UKS + MP 2×1×1; E_disp parity 4.0e-5 Ha) | ratio **0.35** (G2 twin vs CP2K 60.2 s OMP=8) | 2026-08-01
- [x] `gpw-d3-kpoint-stress` | Multi-k analytic cell stress with DFT-D3 zero damping (dedicated pin) | gpw | `regressions/gpw/gpw-d3-kpoint-stress` (CP2K NUMERICAL stress-difference; Δσ parity 7.5e-11 Ha/Bohr³) | ratio **0.185** (21.84 / 118.34) | 2026-08-01
- [x] `gpw-d3bj-kpoint-stress` | Multi-k analytic cell stress with DFT-D3(BJ) (dedicated pin) | gpw | `regressions/gpw/gpw-d3bj-kpoint-stress` (CP2K NUMERICAL BJ stress-difference) | ratio **0.382** (23.0 / 60.111) | 2026-08-01
- [x] `gpw-d3-atm-kpoint-stress` | Multi-k analytic cell stress with D3 three-body ATM (C9) | gpw | `regressions/gpw/gpw-d3-atm-kpoint-stress` (CP2K NUMERICAL C9 on−off; Δσ parity 8.1e-14 Ha/Bohr³) | ratio **0.285** (5.34 / 18.74) | 2026-08-01
- [x] `gpw-bomd-uks-kpoints` | Multi-k UKS plain BOMD (previous-step dual P_k^σ warm start) | dynamics | `regressions/gpw/gpw-bomd-uks-kpoints` (CP2K Quickstep UKS AIMD k-mesh pin) | ratio **1.18** | 2026-08-01
- [x] `gpw-vdw-uks-multigrid` | Γ-UKS nonlocal vdW on REL_CUTOFF multigrid | gpw | `regressions/gpw/gpw-vdw-uks-multigrid` (CP2K NGRIDS 4 + DRSLL + UKS; E_disp 1.3e-5 Ha) | ratio **4.44** (within_10x; SC RPS FFT cost) | 2026-08-01

## done (harness wave 3 (shift) 2026-07-31)

- [x] `gpw-ts-stress` | libMBD strained-cell TS pins + analytic/FD dual gate | gpw | `regressions/gpw/gpw-ts-stress` | ratio **perf_exempt** | 2026-07-31
- [x] `gpw-xdm-stress` | dual pure-Python XDM strain pins + analytic/FD dual gate | gpw | `regressions/gpw/gpw-xdm-stress` | ratio **perf_exempt** | 2026-07-31
- [x] `gpw-vdw-kpoints` | multi-k RKS nonlocal vdW-DF (Γ-limit + CP2K pin) | gpw | `regressions/gpw/gpw-vdw-kpoints` | ratio **0.74** | 2026-07-31
- [x] `gpw-vdw-uks` | Γ UKS nonlocal vdW-DF (O₂ CP2K pin + closed-shell UKS≡RKS) | gpw | `regressions/gpw/gpw-vdw-uks` | ratio **1.23** | 2026-07-31
- [x] `gpw-uks-multigrid-kpoints` | multi-k UKS REL_CUTOFF multigrid (CP2K pin; H₂ twin ratio ~14) | gpw | `regressions/gpw/gpw-uks-multigrid-kpoints` | ratio **perf_exempt** | 2026-07-31
- [x] `cpmd-thermostat-kpoints` | multi-k CPMD electronic Nosé–Hoover | dynamics | `regressions/dynamics/cpmd-thermostat-kpoints` | ratio **perf_exempt** | 2026-07-31
- [x] `second-gen-cpmd-ot-uks-kpoints` | multi-k UKS SGCP + OT corrector | dynamics | `regressions/dynamics/second-gen-cpmd-ot-uks-kpoints` | ratio **perf_exempt** | 2026-07-31
- [x] `xl-bomd-linearized-stress` | linearized XL shadow-density stress | dynamics | `regressions/dynamics/xl-bomd-linearized-stress` | ratio **2.4** | 2026-07-31
- [x] `dfpt-gga-spin-finite-q-si` | finite-q DFPT collinear spin × PBE | response | `regressions/response/dfpt-gga-spin-finite-q-si` | ratio **0.19** | 2026-07-31


## done (harness wave 2 (shift) 2026-07-31)

- [x] `gpw-d4-atm-stress` | Analytic fixed-P cell stress with DFT-D4 three-body ATM (\(s_9\)) — the s9 fold in the strained-cell rebuild now has its dedicated pin | gpw | `regressions/gpw/gpw-d4-atm-stress` (dftd4 3.7.0 strained-cell energy pins ±2e-3 × 3 axes × {s9=1, 2b}: parity 6.3e-11 Ha; derived σ^ATM parity 3.7e-16 Ha/Bohr³ vs signal ~7.5e-10; `gpw_stress_fd` dual gate 7.4e-7; product ATM-fold wiring 7.0e-11) | ratio **perf_exempt** (no CP2K DFTD4 build — same policy as `gpw-d4-atm-forces`; wall ~36 s) | done wave 2 2026-07-31 on `wf/gpw-d4-atm-stress`
- [x] `gpw-d3-atm-stress` | Analytic fixed-P cell stress with DFT-D3 three-body ATM (C9) folded; adds configurable `d3_rcut_bohr` (CP2K `R_CUTOFF`, doubled convention) on `GpwDft`/`GPWEvaluator` so the product folds the oracle's exact functional | gpw | `regressions/gpw/gpw-d3-atm-stress` (CP2K 2026.1 `STRESS_TENSOR NUMERICAL` C9 on−off difference pins: Δσ parity 1.1e−13 Ha/Bohr³ at CP2K's `DX 0.001` step — the hard-gated triple sum is step-locked, unlike the smooth two-body fold; E_disp 2.8e−10; analytic vs multi-SCF FD 2.4e−7) | ratio **0.75** (ATV product 40.8 s / CP2K 54.5 s) | 2026-07-31
- [x] `second-gen-cpmd-ot-kpoints` | Multi-k SGCP with **OT partial-SCF corrector** — the per-k OT manifold warm-starts from natural orbitals of the ASPC-predicted \(P_k\) (complex mirror of the Γ `guess_P` purification; multi-k **UKS** + OT still refused with a precise throw) | dynamics | `regressions/dynamics/second-gen-cpmd-ot-kpoints` (corrector A/B identity multi-k OT vs diagonalization \|ΔE\| 3.0e-12 Ha at the H₂ 1×1×2 input; NVE drift 5.1e-4 ≤ 0.10; production OT iters ≤ budget 5; wall ~2 s at 8t) | ratio **perf_exempt** (CP2K's OT is Γ-only — `&OT` + `&KPOINTS` is refused by CP2K itself, so no oracle twin can exist; literature SGCP envelope per family, unexecutable sketch at `tools/cp2k_oracle/inputs/second-gen-cpmd-ot-kpoints/`) | 2026-07-31
- [x] `cpmd-uks-kpoints` | Multi-k UKS Car–Parrinello MD (dual-spin Bloch \(C_k^\sigma\), one `CpmdCplx` per (k, spin), occ-1 spin-diagonal EOM; new `GpwResult::H_ks_k_beta` + multi-k UKS fixed-density `max_iter==0` branch; multi-k electronic thermostat ships separately as `cpmd-thermostat-kpoints`) | dynamics | `regressions/dynamics/cpmd-uks-kpoints` (periodic H₂ NVE 1×1×2 unreduced, n_α=n_β=1; E_cons drift envelope ≤0.10 + UKS-vs-RKS closed-shell single-point identity ≤3e-4 eV at tol 1e-7; Γ-limit solver identity unit-pinned) | ratio **perf_exempt** (oracle family lock: CP2K 2026.1 present but has no fictitious-mass CP propagator — the CP conserved quantity is not a number it can print; BO surface CP2K-anchored via `gpw-kpoints-uks` / `gpw-kpoint-uks-forces` / `gpw-bomd-kpoints`) | 2026-07-31
- [x] `xl-bomd-linearized-kpoints` | Zero-SCF linearized XL-BOMD forces on multi-k RKS/UKS (k-weighted complex shadow \(D_k^\sigma\); new k-UKS fixed-density eval + `GpwResult::H_ks_k_beta`; linearized+stress now refused at `build()`) | dynamics | `regressions/dynamics/xl-bomd-linearized-kpoints` (CP2K 2026.1 `.ener`-anchored converged-BOMD k trace + SCF-XL vs linearized A/B + UKS spin-degeneracy identity; dual gate measured on branch) | ratio **perf_exempt** (NVE drift envelope — XL/BOMD family policy) | 2026-07-31
- [x] `gpw-vdw-multigrid` | Nonlocal vdW-DF/DF2/rVV10 SCF on the REL_CUTOFF multigrid product path (`ngrids>1`; fine-grid V_nl added to veff before per-level restriction — variational; final E_c^nl re-collocated through products + spectral prolongation; found rot fixed: Γ UKS + vdW silently dropped E_c^nl — now fail-fast at `GpwDft::build`) | gpw | `regressions/gpw/gpw-vdw-multigrid` (CP2K 2026.1 NGRIDS 4 / REL_CUTOFF 40 DRSLL pins: E_disp parity 4.0e-5 Ha, ATV mg↔sg 4.28e-6 Ha vs CP2K's own 6.0e-7; E_tot inside the pre-existing 7.9e-3 Γ-dimer-twin systematic, identical on both grid paths) | ratio **0.83** (ATV 20.6 s both solves / CP2K 24.7 s NGRIDS 4, same host, OMP=8) | 2026-07-31
- [x] `dfpt-gga-finite-q-si` | Finite-q DFPT phonons with the pure GGA (PBE) kernel (PW commensurate X → 2×1×1 supercell Γ-Sternheimer + IFC Fourier) | response | `regressions/response/dfpt-gga-finite-q-si` | ratio **0.17** (CP2K 2026.1 supercell RKS-PBE `VIBRATIONAL_ANALYSIS` 288.7 s / ATV DFPT 49.8 s, serial; CP2K X-branch anchor 98–459 cm⁻¹ by eigenvector phase, supercell PBE energy-FD sharp pin 21.2006 eV/Å², DFPT 21.1974) | 2026-07-31
- [x] `gpw-d3-stress` | Analytic fixed-P cell stress with DFT-D3 zero damping (geometry-only fold rebuilt at each strained cell; CP2K NUMERICAL-stress **difference** pin — two full analytic products σ_an(d3)−σ_an(plain) gate the product fold end-to-end) | gpw | `regressions/gpw/gpw-d3-stress` (CP2K 2026.1 pins: E_disp parity 5.5e-11 Ha; Δσ parity 2.3e-10 Ha/Bohr³ — SCF-noise floor of the two-product isolation, signal 1.7e-8 = 75×; dual gate 7.4e-7; wall 47.7 s) | ratio **0.55** (ATV product 18.5 s / CP2K NUMERICAL twin 33.6 s) | 2026-07-31

## done (harness wave 1 (shift) 2026-07-31)

- [x] `gpw-mbd-kpoint-stress` | Multi-k analytic cell stress folds periodic MBD@rsSCS (k-summed fixed-\(P_k\) Lagrangian; mbd joins the k-path dispersion mutual-exclusion) | gpw | `regressions/gpw/gpw-mbd-kpoint-stress` (pymbd/libmbd `88069c6` strain pins to 1.8e-15 Ha; unstrained multi-k reproduces the Γ `gpw-mbd-periodic` pin to 1.1e-11 Ha; analytic vs multi-SCF strain-FD 1.4e-9 Ha/Bohr³) | ratio **perf_exempt** (CP2K 2026.1 stock has no MBD pair potential; wall 13.6 s) | 2026-07-31
- [x] `gpw-d3-atm-forces` | Analytic ionic forces with DFT-D3 three-body ATM (C9); fixed CASE-3 periodic image-triple double count (8× ATM energy) + pbc anchoring in `accumulate_atm_c9` | gpw | `regressions/gpw/gpw-d3-atm-forces` (CP2K 2026.1 ENERGY_FORCE pins: iso 5.0e-10 / periodic 9.8e-10 Ha/Bohr; FD 4.1e-14 / 2.2e-13) | ratio **3.3** (ATV 34.5 s / CP2K 10.5 s) | 2026-07-31
- [x] `gpw-bomd-kpoints` | Plain multi-k Born-Oppenheimer MD with previous-step \(\{P_k\}\) warm start (mirrors CP2K EXTRAPOLATION USE_PREV_P; UKS seeds both spins) | dynamics | `regressions/gpw/gpw-bomd-kpoints` (CP2K 2026.1 Quickstep AIMD k-mesh pins: per-step E_pot trace worst 2.33e-4 Ha; NVE drift 4.25e-3 vs CP2K 4.4e-3) | ratio **0.064** (ATV 2.14 s / CP2K 33.4 s, same host, OMP=8) | 2026-07-31
- [x] `gpw-multigrid-kpoints` | Multi-k RKS REL_CUTOFF multigrid product collocation (Bloch-phased per-cell density/potential stacks; fixed whole-product mass screen `kProductMassMargin`; nonlocal vdW × k now fail-fast) | gpw | `regressions/gpw/gpw-multigrid-kpoints` (CP2K 2026.1 pin −15.802416412503845 Ha, NGRIDS 4 / REL_CUTOFF 60 / MP 2×2×2; \|ΔE\| 4.7e-7 Ha; TR/Γ-limit identities ≤2.8e-14) — also replaced the `gpw-kpoints-si` ATV-frozen A7 self-golden pin with the genuine CP2K twin | ratio **1.6** (wall 55 s) | 2026-07-31
- [x] `xl-bomd-linearized-uks` | Zero-SCF dual-spin shadow XL-BOMD forces on Γ-UKS (warm iters 0; shadow \(E=\sum_\sigma[\mathrm{Tr}(H^\sigma D^\sigma)-\mathrm{Tr}(H^\sigma n^\sigma)]+E_{KS}[n]\); ⟨S²⟩ 2.0047) | dynamics | `regressions/dynamics/xl-bomd-linearized-uks` (O₂ triplet NVE, SCF-XL vs linearized A/B; drift 1.8e-7) | ratio **perf_exempt** (no cp2k binary at pin host; literature shadow-XL/NVE envelope — same policy as the XL/BOMD family) | 2026-07-31

- [x] `gpw-dftu-kpoint-forces` | Multi-k analytic ionic forces with Dudarev DFT+U | gpw | twin: CP2K 2026.1 PLUS_U + ENERGY_FORCE on periodic Γ NiO (E asserted; force twin diagnostic — ATV gross-Mulliken vs CP2K net-Mulliken projector, see `.claude/notes/gpw-dftu-projector-divergence.md`); decisive gate = full-6N FD dual gate on the k path (6.4e-6 Ha/Bohr) | done wave 1 2026-07-31 on `wf/gpw-dftu-kpoint-forces` — also fixed the +U scatter to variational ½(V̂S+SV̂) (CP2K-precedented) and re-pinned 6 ATV-self +U pins
- [x] `gpw-d3bj-stress` | Analytic fixed-P cell stress with DFT-D3(BJ) (CP2K NUMERICAL-stress difference pin: Δσ parity 3.9e-14 Ha/Bohr³; dual gate 7.4e-7) | gpw | `regressions/gpw/gpw-d3bj-stress` | ratio **0.37** (CP2K 83.8 s / ATV 31.1 s product) | 2026-07-31
- [x] `gpw-d4-atm-forces` | Analytic ionic forces with DFT-D4 three-body ATM (\(s_9\)): closed-form triple derivatives + C9→neutral-C6→CN chain replace the FD ATM leg in `cpu::scf::dftd4` | gpw | `regressions/gpw/gpw-d4-atm-forces` (dftd4 3.7.0 `--grad` pins: isolated 1e-9; periodic direct 5e-8 — the oracle's periodic ATM gradient is non-conservative by ~2.4e-8 vs its own energy, FD-verified — plus 1e-9 FD-parity over its printed displaced energies + ATM-increment gate) | ratio **perf_exempt** (no CP2K DFTD4 build; same policy as `gpw-d4-forces`) | 2026-07-31
- [x] `dfpt-gga-spin-si` | DFPT phonons with joint collinear spin **and** pure GGA (PBE) kernel (PW Γ Sternheimer) | response | `regressions/response/dfpt-gga-spin-si` | ratio **0.20** (CP2K 2026.1 UKS-PBE k-point `VIBRATIONAL_ANALYSIS` 221.8 s / ATV DFPT 44.7 s, serial; CP2K optical anchor 463–552 cm⁻¹, spin-PBE energy-FD sharp pin 17.3454 eV/Å², DFPT 17.3711) | 2026-07-31

## done (harness wave 14 2026-07-26)

- [x] `gpw-fd-phonon-si` | Finite-displacement Γ phonons with GPW analytic forces (Si₂ UKS triplet, 1D-curvature dual-gate; UKS density warm-start) | response | `regressions/gpw/gpw-fd-phonon-si` | ratio **perf_exempt** (~65 s wall; no fair CP2K twin window) | 2026-07-26
- [x] `cpmd-kpoints` | Multi-k Car-Parrinello MD (Bloch \(C_k\)) | dynamics | `regressions/dynamics/cpmd-kpoints` | ratio **perf_exempt** (NVE drift) | 2026-07-26
- [x] `periodic-wannier-gpw` | Maximally localized Wannier functions on GPW (multi-k Marzari-Vanderbilt) | response | `regressions/gpw/periodic-wannier-gpw` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-d3-forces` | Analytic ionic forces with DFT-D3 zero damping | gpw | `regressions/gpw/gpw-d3-forces` | ratio **1.4** (CP2K 12.3 s / ATV 17.8 s) | 2026-07-26
- [x] `gpw-d3bj-forces` | Analytic ionic forces with DFT-D3(BJ) | gpw | `regressions/gpw/gpw-d3bj-forces` | ratio **2.7** (CP2K 11.9 s / ATV 32.2 s) | 2026-07-26
- [x] `gpw-mbd-stress` | Analytic cell stress with periodic MBD@rsSCS (Gamma) | gpw | `regressions/gpw/gpw-mbd-stress` | ratio **perf_exempt** (no MBD-stress oracle) | 2026-07-26

Defects found and fixed while gating these (each named in its commit):

- D3-zero pair cutoff was hard-coded 40 Bohr in five sites instead of CP2K's
  `2 x R_CUTOFF`; settled against a dedicated periodic D3-zero CP2K run
  (75.589 Bohr reproduces its E_disp to 6.7e-11 Ha, 40 Bohr misses by 3.9e-7).
  Unified behind `kD3Cp2kPairCutoffBohr` so energy, force and stress share one
  convention.
- Martyna-Tuckerman placement was a silent-wrong-answer path (31.9 Ha/Bohr force
  error for an off-centre water dimer, with a plausible-looking energy).
  `GpwDft::build` now throws with the fix named.
- Multi-k analytic stress silently dropped `E_disp^MBD`; it now throws and points
  at the Gamma path / FD oracle. The fold itself remains `gpw-mbd-kpoint-stress`.
- Four architecture scrub gates were already red on `dev` before this wave
  (`tests/cpu/md/` abbreviated dir, a banned token in CLAUDE.md, two
  catch-all-numeric-type prose hits); all repaired.

## done (harness wave 12 2026-07-26)

- [x] `xl-bomd-linearized-forces` | Zero-SCF linearized-functional XL-BOMD forces | dynamics | `regressions/dynamics/xl-bomd-linearized-forces` | ratio **perf_exempt** (NVE drift) | 2026-07-26
- [x] `second-gen-cpmd-kpoints` | Multi-k second-generation CP (ASPC on \(P_k\)) | dynamics | `regressions/dynamics/second-gen-cpmd-kpoints` | ratio **perf_exempt** (NVE drift) | 2026-07-26
- [x] `gpw-d3-atm` | DFT-D3 three-body ATM (C9) on GPW | gpw | `regressions/gpw/gpw-d3-atm` | ratio **perf_exempt** (E_disp pin) | 2026-07-26
- [x] `gpw-d4-atm` | DFT-D4 three-body ATM on GPW | gpw | `regressions/gpw/gpw-d4-atm` | ratio **perf_exempt** (E_disp pin) | 2026-07-26
- [x] `bands-spin-metallic-si` | PW band structure with collinear spin **and** metallic occupation smearing | pw | `regressions/pw/bands-spin-metallic-si` | ratio **perf_exempt** | 2026-07-26
- [x] `dos-spin-metallic-si` | PW DOS with collinear spin **and** metallic occupation smearing | pw | `regressions/pw/dos-spin-metallic-si` | ratio **perf_exempt** | 2026-07-26

## done (harness wave 11 2026-07-26)

- [x] `ugw-uks-rpbe` | UGW@UKS with RPBE spin XC | gpw | `regressions/gpw/ugw-uks-rpbe` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-ts-forces` | Analytic ionic forces with free-atom TS | gpw | `regressions/gpw/gpw-ts-forces` | ratio **perf_exempt** (E_disp / force pin) | 2026-07-26
- [x] `gpw-xdm-forces` | Analytic ionic forces with free-atom XDM | gpw | `regressions/gpw/gpw-xdm-forces` | ratio **perf_exempt** (E_disp / force pin) | 2026-07-26
- [x] `gpw-kpoint-uks-stress-gga` | Multi-k UKS analytic stress for RevPBE/RPBE/BLYP | gpw | `regressions/gpw/gpw-kpoint-uks-stress-gga` | ratio **perf_exempt** | 2026-07-26
- [x] `cpmd-electronic-thermostat` | Electronic thermostat on CPMD (GPW) | dynamics | `regressions/dynamics/cpmd-electronic-thermostat` | ratio **perf_exempt** (NVE / thermostat envelope) | 2026-07-26
- [x] `xl-bomd-uks-kpoints` | Multi-k UKS XL-BOMD (dual-spin \(P_k^\sigma\)) | dynamics | `regressions/dynamics/xl-bomd-uks-kpoints` | ratio **perf_exempt** (NVE drift) | 2026-07-26
- [x] `bands-metallic-si` | Metallic occupation band structure (PW) | pw | `regressions/pw/bands-metallic-si` | ratio **perf_exempt** | 2026-07-26
- [x] `dos-spin-si` | Collinear-spin DOS (PW) | pw | `regressions/pw/dos-spin-si` | ratio **perf_exempt** | 2026-07-26
- [x] `dfpt-finite-q-spin-smearing` | Finite-q DFPT collinear spin + metallic smear (PW) | response | `regressions/response/dfpt-finite-q-spin-smearing` | ratio **perf_exempt** | 2026-07-26

## done (harness wave 10 2026-07-26)

- [x] `dfpt-spin-smearing` | DFPT collinear spin + metallic occupation smearing (PW) | response | `regressions/response/dfpt-spin-smearing` | ratio **perf_exempt** | 2026-07-26
- [x] `ot-uks-dftu-smearing` | Multi-k UKS OT + Dudarev DFT+U + metallic smear | gpw | `regressions/gpw/ot-uks-dftu-smearing` | ratio **perf_exempt** | 2026-07-26
- [x] `second-gen-cpmd-uks-ot` | SGCP ASPC + OT corrector on Γ-UKS | dynamics | `regressions/dynamics/second-gen-cpmd-uks-ot` | ratio **perf_exempt** (NVE drift) | 2026-07-26
- [x] `xl-bomd-kpoints` | Multi-k XL-BOMD auxiliary-density AIMD | dynamics | `regressions/dynamics/xl-bomd-kpoints` | ratio **perf_exempt** (NVE drift) | 2026-07-26
- [x] `bands-spin-si` | Spin-polarized band structure (PW) | pw | `regressions/pw/bands-spin-si` | ratio **perf_exempt** | 2026-07-26
- [x] `dos-metallic-si` | Metallic occupation DOS (PW) | pw | `regressions/pw/dos-metallic-si` | ratio **perf_exempt** | 2026-07-26
- [x] `periodic-wannier-uks` | Spin-polarized periodic MLWF (PW) | response | `regressions/pw/periodic-wannier-uks` | ratio **perf_exempt** | 2026-07-26
- [x] `berry-spin-si` | Spin-polarized Berry-phase polarization (PW) | response | `regressions/pw/berry-spin-si` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-mbd-kpoint-forces` | Multi-k analytic ionic forces with MBD@rsSCS | gpw | `regressions/gpw/gpw-mbd-kpoint-forces` | ratio **perf_exempt** (E_disp / force pin) | 2026-07-26

## done (harness wave 9 2026-07-26)

- [x] `ot-uks-smearing` | UKS OT + metallic occupation smearing (FD/MP) | gpw | `regressions/gpw/ot-uks-smearing` | ratio **~0.12** | 2026-07-26
- [x] `ot-uks-dftu` | Multi-k UKS OT + Dudarev DFT+U | gpw | `regressions/gpw/ot-uks-dftu` | ratio **perf_exempt** | 2026-07-26
- [x] `ot-dftu-smearing` | OT + DFT+U + metallic smearing (RKS multi-k) | gpw | `regressions/gpw/ot-dftu-smearing` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-d4-forces` | Analytic ionic forces with DFT-D4 | gpw | `regressions/gpw/gpw-d4-forces` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-mbd-forces` | Analytic ionic forces with MBD@rsSCS | gpw | `regressions/gpw/gpw-mbd-forces` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-uks-multigrid` | UKS multigrid product collocation (`ngrids>1`) | gpw | `regressions/gpw/gpw-uks-multigrid` | ratio **~1.54** | 2026-07-26
- [x] `dfpt-finite-q-smearing` | Finite-q DFPT with metallic smearing (PW) | response | `regressions/response/dfpt-finite-q-smearing` | ratio **perf_exempt** | 2026-07-26
- [x] `second-gen-cpmd-ot` | Second-gen CP with OT partial-SCF corrector | dynamics | `regressions/dynamics/second-gen-cpmd-ot` | ratio **perf_exempt** (NVE drift) | 2026-07-26

## done (harness wave 8 2026-07-26)

- [x] `periodic-urpa-kpoints` | Multi-k unrestricted direct RPA residual (URPA@UKS + k-mesh) | gpw | `regressions/gpw/periodic-urpa-kpoints` | ratio **perf_exempt** | 2026-07-26
- [x] `ot-smearing` | OT orbital minimizer + metallic occupation smearing (FD/MP) | gpw | `regressions/gpw/ot-smearing` | ratio **~0.20** | 2026-07-26
- [x] `ot-uks-kpoints` | Multi-k UKS OT minimizer | gpw | `regressions/gpw/ot-uks-kpoints` | ratio **perf_exempt** | 2026-07-26
- [x] `ot-dftu-kpoints` | Multi-k OT + Dudarev DFT+U | gpw | `regressions/gpw/ot-dftu-kpoints` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-d4-stress` | Analytic fixed-P cell stress with DFT-D4 | gpw | `regressions/gpw/gpw-d4-stress` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-vdw-scf-potential` | Nonlocal vdW-DF/DF2/rVV10 SCF potential + forces | gpw | `regressions/gpw/gpw-vdw-scf-potential` | ratio **perf_exempt** | 2026-07-26
- [x] `dfpt-metallic-si` | DFPT phonons with metallic occupation smearing (PW) | response | `regressions/response/dfpt-metallic-si` | ratio **perf_exempt** | 2026-07-26
- [x] `dfpt-spin-incommensurate` | Collinear-spin DFPT at incommensurate q (PW) | response | `regressions/response/dfpt-spin-incommensurate` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-optb88-vdw` | optB88 exchange + nonlocal vdW correlation | gpw | `regressions/gpw/gpw-optb88-vdw` | ratio **perf_exempt** (E_disp / XC combo) | 2026-07-26

## done (harness wave 7 2026-07-26)

- [x] `periodic-urpa-si` | Unrestricted direct RPA (URPA@UKS) | gpw | `regressions/gpw/periodic-urpa-si` | ratio **perf_exempt** | 2026-07-26
- [x] `periodic-ugw-kpoints` | Multi-k unrestricted spectral G0W0 residual | gpw | `regressions/gpw/periodic-ugw-kpoints` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-uks-smearing` | Metallic occupation smearing on GPW UKS (FD/MP) | gpw | `regressions/gpw/gpw-uks-smearing` | ratio **~0.017** | 2026-07-26
- [x] `gpw-hirshfeld-uks` | Hirshfeld volume ratios for UKS TS/MBD | gpw | `regressions/gpw/gpw-hirshfeld-uks` | ratio **perf_exempt** (E_disp pin) | 2026-07-26
- [x] `cpmd-uks` | Car–Parrinello MD on Γ-UKS GPW | dynamics | `regressions/dynamics/cpmd-uks` | ratio **perf_exempt** (NVE drift) | 2026-07-26
- [x] `second-gen-cpmd-uks` | Second-gen CP (ASPC + partial SCF) on Γ-UKS | dynamics | `regressions/dynamics/second-gen-cpmd-uks` | ratio **perf_exempt** (NVE drift) | 2026-07-26
- [x] `ot-kpoints` | OT orbital minimizer with Monkhorst–Pack k-mesh | gpw | `regressions/gpw/ot-kpoints` | ratio **perf_exempt** | 2026-07-26
- [x] `ot-dftu` | OT + Dudarev DFT+U (Γ) | gpw | `regressions/gpw/ot-dftu` | ratio **~0.26** | 2026-07-26
- [x] `dfpt-spin-si` | DFPT phonons with collinear spin (PW) | response | `regressions/response/dfpt-spin-si` | ratio **perf_exempt** | 2026-07-26

## done (harness wave 6 2026-07-26)

- [x] `gpw-metallic-smearing` | Metallic occupation smearing on GPW (FD/MP multi-k RKS) | gpw | `regressions/gpw/gpw-metallic-smearing` | ratio **~0.011** (ATV ~1.0 s / CP2K 89.5 s; dual-gate free energy) | 2026-07-26
- [x] `gpw-xdm-ice` | Exchange-hole dipole moment (XDM) free-atom dispersion | gpw | `regressions/gpw/gpw-xdm-ice` | ratio **perf_exempt** (E_disp pin) | 2026-07-26
- [x] `gpw-uks-rpbe` | Γ-UKS RPBE spin-GGA (`XcKind::RPBE`) | gpw | `regressions/gpw/gpw-uks-rpbe` | ratio **~2.70** | 2026-07-26
- [x] `gpw-dftu-kpoint-stress` | Multi-k Dudarev DFT+U analytic cell stress | gpw | `regressions/gpw/gpw-dftu-kpoint-stress` | ratio **perf_exempt** | 2026-07-26
- [x] `periodic-ump2-kpoints` | Multi-k unrestricted GPW MP2 residual | gpw | `regressions/gpw/periodic-ump2-kpoints` | ratio **perf_exempt** | 2026-07-26
- [x] `periodic-ugw-si` | Unrestricted spectral G0W0 (UGW@UKS) | gpw | `regressions/gpw/periodic-ugw-si` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-hirshfeld-kpoints` | Hirshfeld volume ratios on multi-k TS/MBD | gpw | `regressions/gpw/gpw-hirshfeld-kpoints` | ratio **perf_exempt** (E_disp pin) | 2026-07-26
- [x] `dfpt-gga-si` | DFPT phonons with GGA XC kernel (PW) | response | `regressions/response/dfpt-gga-si` | ratio **perf_exempt** | 2026-07-26
- [x] `xl-bomd-uks` | Niklasson XL-BOMD on Γ-UKS GPW | dynamics | `regressions/dynamics/xl-bomd-uks` | ratio **perf_exempt** (NVE drift envelope) | 2026-07-26

## done (harness wave 5 2026-07-26)

- [x] `gpw-dftu-kpoints` | Multi-k Dudarev DFT+U SCF energy (k-mesh + PLUS_U) | gpw | `regressions/gpw/gpw-dftu-kpoints` | ratio **perf_exempt** | 2026-07-26
- [x] `periodic-gw-finite-q` | Finite-q inter-k \(W\) spectral G0W0 residual | gpw | `regressions/gpw/periodic-gw-finite-q` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-uks-revpbe` | Γ-UKS revPBE spin-GGA (`XcKind::RevPBE`) | gpw | `regressions/gpw/gpw-uks-revpbe` | ratio **~3.22** | 2026-07-26
- [x] `cpmd-rattle-water` | CPMD electronic RATTLE (constraint-preserving) | dynamics | `regressions/dynamics/cpmd-rattle-water` | ratio **perf_exempt** | 2026-07-26
- [x] `periodic-ump2-si` | Unrestricted GPW MP2 (UKS reference) | gpw | `regressions/gpw/periodic-ump2-si` | ratio **perf_exempt** | 2026-07-26
- [x] `ot-uks-o2` | OT orbital minimizer for Γ-UKS | gpw | `regressions/gpw/ot-uks-o2` | ratio **~1.22** | 2026-08-05 (was ~8.50)

## done (harness wave 4 2026-07-26)

- [x] `gpw-kpoint-uks-stress` | Multi-k UKS analytic cell stress | gpw | `regressions/gpw/gpw-kpoint-uks-stress` | ratio **perf_exempt** | 2026-07-26
- [x] `periodic-gw-kpoints` | Multi-k k-diagonal spectral G0W0 | gpw | `regressions/gpw/periodic-gw-kpoints` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-vdw-df2-ice` | Nonlocal vdW-DF2 (LMKLL) correlation | gpw | `regressions/gpw/gpw-vdw-df2-ice` | ratio **perf_exempt** (E_disp pin) | 2026-07-26
- [x] `gpw-rvv10-ice` | Nonlocal rVV10 correlation | gpw | `regressions/gpw/gpw-rvv10-ice` | ratio **perf_exempt** (E_disp pin) | 2026-07-26
- [x] `gpw-mbd-periodic` | Periodic Ewald + k-mesh MBD@rsSCS | gpw | `regressions/gpw/gpw-mbd-periodic` | ratio **perf_exempt** (E_disp pin) | 2026-07-26
- [x] `pw-spin-gga` | True collinear spin-GGA on plane-wave grid | pw | `regressions/pw/pw-spin-gga` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-uks-blyp` | Γ-UKS BLYP spin-GGA | gpw | `regressions/gpw/gpw-uks-blyp` | ratio **~2.48** | 2026-07-26
- [x] `periodic-rpa-finite-q` | Finite-q inter-k RI-GPW dRPA | gpw | `regressions/gpw/periodic-rpa-finite-q` | ratio **perf_exempt** | 2026-07-26
- [x] `gpw-dftu-stress` | Analytic stress with Dudarev DFT+U | gpw | `regressions/gpw/gpw-dftu-stress` | ratio **perf_exempt** | 2026-07-26

## done (harness wave 3 2026-07-25)

- [x] `gpw-rpbe-water` | GPW RPBE GGA (`XcKind::RPBE`) | gpw | `regressions/gpw/gpw-rpbe-water` | ratio **perf_exempt** (dual-gate \|E_RPBE−E_PBE\|) | 2026-07-25
- [x] `gpw-kpoint-uks-forces` | Multi-k UKS analytic ionic forces | gpw | `regressions/gpw/gpw-kpoint-uks-forces` | ratio **perf_exempt** | 2026-07-25
- [x] `gpw-uks-stress` | Γ-UKS analytic fixed-P cell stress | gpw | `regressions/gpw/gpw-uks-stress` | ratio **perf_exempt** (multi-k UKS stress residual) | 2026-07-25
- [x] `gpw-vdw-df-ice` | Nonlocal vdW-DF (DRSLL) correlation | gpw | `regressions/gpw/gpw-vdw-df-ice` | ratio **perf_exempt** (E_disp pin) | 2026-07-25
- [x] `gpw-mbd-hirshfeld` | Hirshfeld volume ratios for TS/MBD | gpw | `regressions/gpw/gpw-mbd-hirshfeld` | ratio **perf_exempt** (E_disp + V/V_free) | 2026-07-25
- [x] `periodic-rpa-kpoints` | Multi-k RI-GPW dRPA residual | gpw | `regressions/gpw/periodic-rpa-kpoints` | ratio **perf_exempt** | 2026-07-25
- [x] `periodic-gw-si` | Γ-point spectral G0W0 (periodic Si) | gpw | `regressions/gpw/periodic-gw-si` | ratio **perf_exempt** | 2026-07-25
- [x] `dfpt-incommensurate-si` | Incommensurate-q DFPT phonons (PW) | response | `regressions/response/dfpt-incommensurate-si` | ratio **perf_exempt** | 2026-07-25

## done (harness wave 2 2026-07-25)

- [x] `gpw-kpoint-forces` | GPW analytic forces at Monkhorst–Pack k-points | gpw | `regressions/gpw/gpw-kpoint-forces` | ratio **perf_exempt** | 2026-07-25
- [x] `gpw-revpbe-si` | GPW revPBE GGA (`XcKind::RevPBE`) | gpw | `regressions/gpw/gpw-revpbe-si` | ratio **perf_exempt** | 2026-07-25
- [x] `gpw-kpoints-uks` | GPW collinear multi-k UKS SCF | gpw | `regressions/gpw/gpw-kpoints-uks` | ratio **perf_exempt** | 2026-07-25
- [x] `gpw-uks-forces` | Γ-UKS analytic ionic forces (O₂ PBE) | gpw | `regressions/gpw/gpw-uks-forces` | ratio **~2.52** | 2026-07-25
- [x] `gpw-kpoint-stress` | GPW analytic cell stress at k-points | gpw | `regressions/gpw/gpw-kpoint-stress` | ratio **perf_exempt** | 2026-07-25
- [x] `gpw-mbd-ice` | Free-atom MBD@rsSCS dispersion | gpw | `regressions/gpw/gpw-mbd-ice` | ratio **perf_exempt** (libMBD E_disp) | 2026-07-25
- [x] `periodic-mp2-kpoints` | Multi-k RI-GPW MP2 residual | gpw | `regressions/gpw/periodic-mp2-kpoints` | ratio **perf_exempt** | 2026-07-25
- [x] `periodic-rpa-si` | Γ-point GPW direct RPA (dRPA@PBE) | gpw | `regressions/gpw/periodic-rpa-si` | ratio **perf_exempt** | 2026-07-25
- [x] `dfpt-finite-q-si` | Finite-q DFPT phonons at X (PW) | response | `regressions/response/dfpt-finite-q-si` | ratio **perf_exempt** | 2026-07-25
- [x] `gpw-variable-cell-molopt` | GPW CELL_OPT MOLOPT re-twin vs CP2K | gpw | `regressions/gpw/gpw-variable-cell-molopt` | ratio **~7.5** | 2026-07-25

## done (harness wave 1 2026-07-25)

- [x] `gpw-blyp-water` | GPW BLYP GGA (`XcKind::BLYP`) | gpw | `regressions/gpw/gpw-blyp-water` | ratio **perf_exempt** (no CP2K wall; dual-gate \|E_BLYP−E_PBE\|) | 2026-07-25
- [x] `gpw-uks-pbe` | Γ-UKS PBE-GGA O₂ | gpw | `regressions/gpw/gpw-uks-pbe` | ratio **0.85** | 2026-07-25
- [x] `pw-uks-fe-smearing` | bulk Fe UKS + metallic smearing | pw | `regressions/pw/pw-uks-fe-smearing` | ratio **perf_exempt** (no fair NC-UPF twin) | 2026-07-25
- [x] `pw-variable-cell-si` | PW isotropic CELL_OPT (Nielsen–Martin stress) | pw | `regressions/pw/pw-variable-cell-si` | ratio **perf_exempt** | 2026-07-25
- [x] `gpw-ts-ice` | Tkatchenko–Scheffler free-atom TS dispersion | gpw | `regressions/gpw/gpw-ts-ice` | ratio **perf_exempt** (E_disp pin; libMBD) | 2026-07-25
- [x] `phonon-dispersion-si` | Phonon dispersion ω(q) IFC Fourier (SW Si) | response | `regressions/response/phonon-dispersion-si` | ratio **perf_exempt** | 2026-07-25
- [x] `periodic-wannier-si` | Periodic multi-k MLWF (Marzari–Vanderbilt PW) | response | `regressions/pw/periodic-wannier-si` | ratio **perf_exempt** | 2026-07-25
- [x] `second-gen-cpmd-water` | Second-generation CP (ASPC + partial SCF) | dynamics | `regressions/dynamics/second-gen-cpmd-water` | ratio **perf_exempt** | 2026-07-25
- [x] `periodic-ri-mp2-si` | Γ RI-GPW MP2 product (`mp2_ri`; multi-k residual) | gpw | `regressions/gpw/periodic-ri-mp2-si` | ratio **~0.09** | 2026-07-25

## done (wave parallel land 2026-07-25)

- [x] `mpi-pw-scaling-4rank` | MPI PW k-mesh RR gate (4 ranks) | pw | `regressions/pw/mpi-pw-scaling-4rank` | ratio **perf_exempt** (speedup 2.54×; no CP2K) | 2026-07-25
- [x] `gpw-analytic-stress` | Analytic GPW fixed-P+Pulay stress | gpw | `regressions/gpw/gpw-analytic-stress` | dual-gate vs FD ~3e-10 Ha/Bohr³ | 2026-07-25
- [x] `wannier-centers` | Foster–Boys MLWF foothold (molecular RHF) | scf | `regressions/scf/wannier-centers` | ratio **perf_exempt** | 2026-07-25
- [x] `mpi-gpw-scaling-4rank` | MPI GPW k-point RR industrial gate | gpw | `regressions/gpw/mpi-gpw-scaling-4rank` | ratio **perf_exempt** (energy rank-identical; wall≈1 until grid DD) | 2026-07-25
- [x] `cpmd-water` | Car–Parrinello MD foothold (`Cpmd` + `GPWEvaluator::cpmd`) | dynamics | `regressions/dynamics/cpmd-water` | ratio **perf_exempt** (E_cons drift 2e-3) | 2026-07-25

## blocked

- [!] `gpw-vv10-ice` | blocked — stock VV10 (≠ rVV10): CP2K 2026.1 has no `NON_LOCAL TYPE VV10` (enum DRSLL|LMKLL|RVV10 only; no VV10 kernel table); reconfirmed waves 6–10 on `wf/gpw-vv10-ice` @ `a60c73d5`; product VV-family remains rVV10; no offline CP2K twin — scope_out of pending
- [!] `periodic-hybrid-gpw` | blocked — no periodic exact exchange (same as periodic HF); HSE/PBE0 periodic ○
- [!] `gapw-dft` | blocked — GAPW not in tree; long-horizon
- [!] `ultrasoft-paw` | blocked — UPF NC + GTH only; PAW/USPP ○
- [!] `cuda-gpw` | blocked — GPW CPU-only; CUDA plane-wave / Kokkos are separate tracks (not this loop)
- [!] `periodic-meta-gga` | blocked — no kinetic-energy density (τ) channel in GPW/PW grid engines

## done

- [x] `gpw-water-pbe-mt` | Γ-RKS GPW energy (MT water) | gpw | `regressions/gpw/gpw-water-pbe-mt` | ratio **perf_exempt** (no CP2K binary at pin host) | 2026-07-23 wave 1
- [x] `gpw-forces-water` | GPW analytic forces | gpw | `regressions/gpw/gpw-forces-water` | ratio **3.56** | 2026-07-23 wave 1
- [x] `gpw-uks-o2-gamma` | Γ-point UKS GPW | gpw | `regressions/gpw/gpw-uks-o2-gamma` | ratio **0.98** | 2026-07-23 wave 1
- [x] `gpw-pbe-d3bj-ice` | GPW PBE-D3(BJ) periodic | gpw | `regressions/gpw/gpw-pbe-d3bj-ice` | ratio **perf_exempt** (E_disp pin; full QS wall unfair) | 2026-07-23 wave 1
- [x] `pw-lda-si` | PW LDA Si k-mesh energy | pw | `regressions/pw/pw-lda-si` | ratio **perf_exempt** (ABINIT-class energy band; no fair CP2K PW twin) | 2026-07-23 wave 1
- [x] `pw-pbe-gga-si` | PW PBE (true GGA) | pw | `regressions/pw/pw-pbe-gga-si` | ratio **perf_exempt** (symmetry+GGA-activation pins; no QE/CP2K binary) | 2026-07-23 wave 1
- [x] `pw-smearing-al` | metallic smearing PW | pw | `regressions/pw/pw-smearing-al` | ratio **0.008** | 2026-07-23 wave 1
- [x] `pw-spin-polarized-o2` | PW spin-polarized KS | pw | `regressions/pw/pw-spin-polarized-o2` | ratio **perf_exempt** (mag=2 + frozen e_tot pin) | 2026-07-23 wave 1
- [x] `pw-bomd-h` | PW BOMD foothold | dynamics | `regressions/pw/pw-bomd-h` | ratio **perf_exempt** (NVE drift envelope; no CP2K twin wall) | 2026-07-23 wave 1
- [x] `ot-minimizer-water` | OT vs DIIS large cell | gpw | `regressions/gpw/ot-minimizer-water` | ratio **~9.0** (fair OT-only, OMP=4) | 2026-07-23 wave 1
- [x] `gpw-stress-water` | GPW cell stress (strain-FD) | gpw | `regressions/gpw/gpw-stress-water` | ratio **4.75** | 2026-07-24 wave 2
- [x] `gpw-bomd-water` | GPW BOMD short traj | dynamics | `regressions/gpw/gpw-bomd-water` | ratio **perf_exempt** (NVE drift envelope; no CP2K AIMD twin wall) | 2026-07-24 wave 2
- [x] `pw-analytic-stress-si` | PW Nielsen–Martin analytic stress | pw | `regressions/pw/pw-analytic-stress-si` | ratio **perf_exempt** (no fair NC-UPF PW twin wall) | 2026-07-24 wave 2
- [x] `fd-phonon-supercell` | FD phonons (supercell Γ) | response | `regressions/pw/fd-phonon-supercell` | ratio **perf_exempt** (SW 64-atom Γ; no CP2K VIBRATIONAL_ANALYSIS wall) | 2026-07-24 wave 2
- [x] `gpw-dftu-nio` | Periodic DFT+U (Dudarev) | gpw | `regressions/gpw/gpw-dftu-nio` | ratio **0.21** | 2026-07-24 wave 2
- [x] `bands-dos-si` | Band structure / DOS | pw | `regressions/pw/bands-dos-si` | ratio **perf_exempt** (literature Γ→X window; no fair GTH twin wall) | 2026-07-24 wave 2
- [x] `pw-uks-kmesh-fe` | k-point UKS plane-wave | pw | `regressions/pw/pw-uks-kmesh-fe` | ratio **perf_exempt** (O₂ multi-k mag=2; no fair NC-UPF twin) | 2026-07-24 wave 2
- [x] `gpw-pbesol-si` | GPW PBEsol GGA | gpw | `regressions/gpw/gpw-pbesol-si` | ratio **perf_exempt** (dual-gate PBE vs PBEsol; no CP2K binary) | 2026-07-24 wave 2
- [x] `gpw-d4-water` | GPW DFT-D4 periodic | gpw | `regressions/gpw/gpw-d4-water` | ratio **perf_exempt** (dftd4 3.7.0 E_disp pin; CP2K Arrhenius build lacks DFTD4) | 2026-07-24 wave 2
- [x] `gpw-kpoints-si` | Monkhorst–Pack k-point GPW RKS (phased lattice sums, KB nonlocal, per-k DIIS; diamond-Si vs CP2K Quickstep) | gpw | `regressions/gpw/gpw-kpoints-si` | ratio **perf_exempt** (no CP2K binary at pin host) | 2026-07-25 claude wave 1
- [x] `gpw-variable-cell-ice` | GPW variable-cell relax (evaluator re-target, stress axes mask, `StressConverged`; a\* = 3.2111 Å vs CP2K CELL_OPT 2026.1) | gpw | `regressions/gpw/gpw-variable-cell-ice` | walls ATV 1.9 s / CP2K 3.9 s (GTH chain; MOLOPT re-twin queued) | 2026-07-25 claude wave 1
- [x] `dfpt-phonon-si` | Γ-point DFPT phonons for PW (Sternheimer PCG, Ewald Hessian, LDA kernel; energy-FD twin 0.07 %) | response | `regressions/response/dfpt-phonon-si` | ratio **perf_exempt** (≈24 s wall; no CP2K DFPT wall recorded) | 2026-07-25 claude wave 1
- [x] `berry-polarization` | KSV Berry-phase polarization on PW (+ Hermitian Davidson/g-vector fixes; symmetry-exact Si pins) | response | `regressions/pw/berry-polarization` | ratio **perf_exempt** (no CP2K binary at pin host; ATV ~18 s) | 2026-07-25 claude wave 1
- [x] `periodic-mp2-si` | Γ-point canonical GPW MP2 **foothold** (`GpwDft::mp2`, Poisson-per-pair (ia\|jb); sc-Si2) — capability row stays ○, RI/k-point + RPA/GW remain | gpw | `regressions/gpw/periodic-mp2-si` | ratio **perf_exempt** (no CP2K binary at pin host) | 2026-07-25 claude wave 1
- [x] `xl-bomd-water` | Niklasson dissipative XL-BOMD on GPW RKS (`XlBomd` K∈[3,9], `GpwDft::run(guess_P)`, `GPWEvaluator::xl_bomd`); H₂O NVE drift envelope | dynamics | `regressions/dynamics/xl-bomd-water` | ratio **perf_exempt** (no CP2K binary; drift 1.2e-5 / 33 s) | 2026-07-25
