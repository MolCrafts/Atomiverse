# Method handbook pages

**Style (current generation): technical report.**  
Dense, sectioned, validation-first. **Not** the narrative Theory→Practice chapters
in `docs/tutorial/ch*.md`. A house narrative style will be applied in a later
rewrite pass.

Produced/updated by workflow **Tutorial** phases after each successful merge
wave:

- `pyscf-method-loop` — molecular post-HF / response (arg `write_tutorials`)
- `cp2k-method-loop` — GPW / PW / periodic DFT vs offline CP2K pins
- `metalwalls-method-loop` — classical electrochemical MD + LAMMPS-overlap MD
  (arg `write_tutorials`; LAMMPS oracle when feature overlaps; MW for MW-only)
- `tb-method-loop` — NDDO / DFTB / GFN-xTB vs offline OpenMOPAC | DFTB+ | xtb
  pins (family-locked; arg `write_tutorials`)

## Required sections

| Section | Content |
| ------- | ------- |
| Summary | What / when / API / regression path |
| Method | Equations, assumptions, exclusions |
| Implementation | Minimal public C++; how to run `reg_<slug>` |
| Validation | Hard-coded pins from `REFERENCE.md` (PySCF, CP2K, LAMMPS/MetalWalls, or OpenMOPAC/DFTB+/xtb) |
| Performance | Fair wall \(T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) (or `perf_exempt`) |
| Limitations | Scope + known debt |
| References | Paths + literature |

Template: [`_TEMPLATE.md`](_TEMPLATE.md).

## Naming

```text
docs/tutorial/methods/<slug>.md
```

`slug` = queue / regression directory name.

## Data sources (do not invent)

**PySCF campaign (molecular):**

- Accuracy: `regressions/scf/<slug>/REFERENCE.md`
- Speed: `docs/developer/pyscf-method-perf.md` or `tools/pyscf_oracle/fair_bench.py`
- Oracle rules: `docs/tutorial/appendix-pyscf-oracle.md` (PySCF never in ctest)

**CP2K capability (GPW / PW / periodic):**

- Accuracy: `regressions/{gpw,pw}/<slug>/REFERENCE.md`
- Speed / ratios: `docs/developer/cp2k-method-handoff.md` (fair wall or `perf_exempt`)
- Oracle: `tools/cp2k_oracle/README.md` (offline pins only; CP2K never in ctest)
- Queue / design: `docs/developer/cp2k-method-queue.md`, `cp2k-method-loop.md`

**MetalWalls / classical electrochem + LAMMPS-overlap MD:**

- Accuracy: `regressions/md/<slug>/REFERENCE.md` or
  `regressions/forcefield/<slug>/REFERENCE.md` (`oracle: lammps|metalwalls|literature`)
- Speed / ratios: `docs/developer/metalwalls-method-handoff.md` (fair wall or `perf_exempt`)
- Oracle: `tools/metalwalls_oracle/README.md` (offline pins only; never live in ctest)
- Queue / design: `docs/developer/metalwalls-method-queue.md`, `metalwalls-method-loop.md`
- **Oracle priority:** if the feature also exists in LAMMPS, pin and fair-bench
  vs **LAMMPS**. MetalWalls only for MW-only physics.

**TB / SE (NDDO / DFTB / GFN-xTB):**

- Accuracy: `regressions/semiempirical/<slug>/REFERENCE.md`
  (`oracle: mopac|dftb+|xtb|literature` + `oracle_version`)
- Speed / ratios: `docs/developer/tb-method-handoff.md` (fair wall or `perf_exempt`)
- Oracle twins: `tools/tb_oracle/README.md` (offline only; never live in ctest)
- Queue / design: `docs/developer/tb-method-queue.md`, `tb-method-loop.md`
- **Family lock:** NDDO → OpenMOPAC; DFTB* → DFTB+; GFN-xTB → xtb. Forbidden:
  cross-family energy pins; treating GFN as DFTB3 successor.

## Index — TB / SE capability wave 1 (2026-07-27)

Offline family-locked pins only — see
[tb-method-handoff](../../developer/tb-method-handoff.md) and
`tools/tb_oracle/`. Family lock: NDDO → OpenMOPAC; DFTB\* → DFTB+;
GFN-xTB → xtb. Narrative: [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md).

| Method | Page | Regression | Oracle |
| ------ | ---- | ---------- | ------ |
| SCC-DFTB2 CH₄ 3ob-3-1 organic SK | [dftb2-3ob-organic](dftb2-3ob-organic.md) | `regressions/semiempirical/dftb2-3ob-organic` | dftb+ 24.1 (`perf_exempt`) |
| SCC-DFTB2 analytic nuclear forces H₂O | [dftb2-analytic-forces](dftb2-analytic-forces.md) | `regressions/semiempirical/dftb2-analytic-forces` | dftb+ formula (`perf_exempt`) |
| Collinear spin-DFTB2 O₂ triplet | [spin-dftb2-o2](spin-dftb2-o2.md) | `regressions/semiempirical/spin-dftb2-o2` | literature + family dftb+ (`perf_exempt`) |

## Index — TB / SE capability wave 2 (2026-07-27)

Offline family-locked pins only — see
[tb-method-handoff](../../developer/tb-method-handoff.md) and
`tools/tb_oracle/`. Oracle = **DFTB+** for DFTB family rows (`oracle: dftb+`
or `family_oracle: dftb+` with `oracle: literature` when absolute SK digits
are ATV-mio-style). Fair walls from REFERENCE (`ratio` or `perf_exempt`).

| Method | Page | Regression | Oracle | ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) |
| ------ | ---- | ---------- | ------ | --------------------------------------------- |
| SCC-DFTB2 H₂ total \(E\) vs mio-1-1 | [dftb2-dftbplus-energy-pin](dftb2-dftbplus-energy-pin.md) | `regressions/semiempirical/dftb2-dftbplus-energy-pin` | dftb+ 24.1 | **≈ 0.4** |
| Γ-point periodic SCC-DFTB2 (1-D C chain) | [dftb2-periodic-gamma](dftb2-periodic-gamma.md) | `regressions/semiempirical/dftb2-periodic-gamma` | literature (+ family dftb+) | **perf_exempt** |

## Index — TB / SE capability wave 3 (2026-07-29)

Offline family-locked pins only — see
[tb-method-handoff](../../developer/tb-method-handoff.md) and
`tools/tb_oracle/`. Final harness-wave-3 batch: lattice-image Γ-PBC forces,
multi-k energy, DFTB3 energy foothold. Fair walls deferred when no DFTB+
binary on host (`perf_exempt`).

| Method | Page | Regression | Oracle | ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) |
| ------ | ---- | ---------- | ------ | --------------------------------------------- |
| Γ-PBC analytic nuclear forces (off-eq C chain) | [dftb2-periodic-forces](dftb2-periodic-forces.md) | `regressions/semiempirical/dftb2-periodic-forces` | literature (+ family dftb+) | **perf_exempt** |
| Multi-k SCC-DFTB2 (4×1×1 C chain) | [dftb2-kpoints](dftb2-kpoints.md) | `regressions/semiempirical/dftb2-kpoints` | dftb+ family (+ ATV mio-style pin) | **perf_exempt** |
| SCC-DFTB3 H₂O (mio-style; third-order \(E\)) | [dftb3-water-mio](dftb3-water-mio.md) | `regressions/semiempirical/dftb3-water-mio` | dftb+ formula-Gaus2011 (Hamiltonian twin) | **perf_exempt** |
## Index — TB / SE capability wave 4 (2026-07-29)

Offline family-locked pins only. Oracle = **DFTB+** for DFTB family rows; a
row may carry `oracle: literature` (formalism) when no DFTB+ binary and no
published value for the exact geometry + SK set exist — the REFERENCE `Gaps`
section must then say so explicitly and the case ships structural gates only.

| Method | Page | Regression | Oracle |
| ------ | ---- | ---------- | ------ |
| Linear-response TD-DFTB (Casida) H₂ singlet/triplet | [td-dftb-h2](td-dftb-h2.md) | `regressions/semiempirical/td-dftb-h2` | literature (formalism; **no absolute pin** — `perf_exempt`) |
| DFTB3 closed-form nuclear gradient (H₂O off-eq) | — | `regressions/semiempirical/dftb3-analytic-forces` | dftb+ formula-Gaus2011 + h² FD identity (`perf_exempt`) |
| Multi-k Bloch analytic F **and** σ (off-eq C chain) | — | `regressions/semiempirical/dftb2-multi-k-forces` | dftb+ formula + FD/strain-FD identity (`perf_exempt`) |
| Γ-PBC lattice-sum D3(BJ) (water dimer, 12 Å cell) | — | `regressions/semiempirical/dftb2-dispersion-d3-periodic` | **s-dftd3 1.2.1 offline, periodic** (`perf_exempt`) |
| ThreeOb closed-form cell stress (asymmetric CH₄) | — | `regressions/semiempirical/dftb2-analytic-stress-threeob` | dftb+ formula + strain-FD (`perf_exempt`) |
| Collinear open-shell cell stress (O₂ triplet, NO doublet) | — | `regressions/semiempirical/dftb2-spin-analytic-stress` | dftb+ formula + strain-FD (`perf_exempt`) |
| AM1 H₂O ΔHf / E / charges | — | `regressions/semiempirical/am1-water-mopac-energy` | **OpenMOPAC** transcript (Dewar 1985) (`perf_exempt`) |
| RM1 H₂O ΔHf at the RM1 optimum | — | `regressions/semiempirical/rm1-water-mopac-energy` | **OpenMOPAC** PARAM survey + Scientia Plena 2016 (`perf_exempt`) |

## Index — TB / SE capability wave 8 (2026-08-25)

Offline family-locked pins only — see
[tb-method-handoff](../../developer/tb-method-handoff.md) and
`tools/tb_oracle/`. Oracle = **xtb** for GFN-xTB rows (never OpenMOPAC /
DFTB+; GFN is not a DFTB3 successor). Fair walls from REFERENCE (`ratio`
or `perf_exempt`).

| Method | Page | Regression | Oracle | ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) |
| ------ | ---- | ---------- | ------ | --------------------------------------------- |
| GFN0-xTB analytic nuclear forces (off-eq H₂O) | [gfn0-xtb-forces](gfn0-xtb-forces.md) | `regressions/semiempirical/gfn0-xtb-forces` | xtb 6.7.1 | **≈ 0.092** |
| GFN1-xTB organic energy (CH₄, H/C/N/O) | [gfn1-xtb-organic](gfn1-xtb-organic.md) | `regressions/semiempirical/gfn1-xtb-organic` | xtb 6.7.1 | **0.5** |
| GFN2-xTB charge–quadrupole AES + \(q_{\mathrm{kernel}}\) AXC (H₂O) | [gfn2-quadrupole-aes](gfn2-quadrupole-aes.md) | `regressions/semiempirical/gfn2-quadrupole-aes` | xtb 6.7.1 | **perf_exempt** |

## Index — TB / SE capability wave 9 (2026-08-25)

Offline family-locked pins only — see
[tb-method-handoff](../../developer/tb-method-handoff.md) and
`tools/tb_oracle/`. Oracle = **xtb** for GFN-xTB rows (never OpenMOPAC /
DFTB+; GFN is not a DFTB3 successor). Fair walls from REFERENCE (`ratio`
or `perf_exempt`). Harness wave 9 closed GFN1/GFN2 analytic \(F\) and GFN0
FD Hessian (`-DATV_BUILD_VIB=ON` exception). Family GFN1/2 Hessian is
wave 10 [gfn-xtb-analytic-hessian](gfn-xtb-analytic-hessian.md). GFN2 AES
remains post-SCC (angular \(F\) hole).

| Method | Page | Regression | Oracle | ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) |
| ------ | ---- | ---------- | ------ | --------------------------------------------- |
| GFN1-xTB analytic nuclear forces (off-eq H₂O) | [gfn1-xtb-forces](gfn1-xtb-forces.md) | `regressions/semiempirical/gfn1-xtb-forces` | xtb 6.7.1 | **≈ 0.16** |
| GFN2-xTB analytic nuclear forces (off-eq H₂O) | [gfn2-xtb-forces](gfn2-xtb-forces.md) | `regressions/semiempirical/gfn2-xtb-forces` | xtb 6.7.1 | **asserted ≤ 10** (xtb wall 0.018 s) |
| GFN0-xTB FD Hessian of analytic \(G\) (off-eq H₂O) | [gfn0-xtb-analytic-hessian](gfn0-xtb-analytic-hessian.md) | `regressions/semiempirical/gfn0-xtb-analytic-hessian` | xtb 6.7.1 | **≈ 0.06** |

## Index — TB / SE capability wave 10 (2026-08-25)

Offline family-locked pins only — see
[tb-method-handoff](../../developer/tb-method-handoff.md) and
`tools/tb_oracle/`. Oracle = **xtb** for GFN-xTB / IPEA rows (never
OpenMOPAC / DFTB+; GFN is not a DFTB3 successor). Fair walls from
REFERENCE (`ratio` or `perf_exempt`). Harness wave 10 closed IPEA-xTB
energy (`reg_ipea-xtb-small`) and the family GFN1/2 FD Hessian
(`reg_gfn-xtb-analytic-hessian`; `-DATV_BUILD_VIB=ON` exception). GFN2 AES
remains post-SCC (angular \(F\) hole — recover `gfn2-aes-fock`).

| Method | Page | Regression | Oracle | ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) |
| ------ | ---- | ---------- | ------ | --------------------------------------------- |
| IPEA-xTB organic energy (CH₄, extra \((n+1)s\)) | [ipea-xtb-small](ipea-xtb-small.md) | `regressions/semiempirical/ipea-xtb-small` | xtb 6.7.1 | **2.0** |
| GFN1/GFN2-xTB FD Hessian of analytic \(G\) (off-eq H₂O) | [gfn-xtb-analytic-hessian](gfn-xtb-analytic-hessian.md) | `regressions/semiempirical/gfn-xtb-analytic-hessian` | xtb 6.7.1 | **≈ 0.11** |

## Index — MetalWalls capability wave 1 (2026-07-23)

Classical electrochemical MD + LAMMPS-overlap MD (map E1–E3, E8–E9). Offline
pins only — see [metalwalls-method-handoff](../../developer/metalwalls-method-handoff.md)
and `tools/metalwalls_oracle/`. **Oracle priority:** LAMMPS when the feature
overlaps; MetalWalls / literature for MW-only physics.

| Method | Page | Regression | Oracle |
| ------ | ---- | ---------- | ------ |
| Const-pot BO parallel plate (madelung, E1) | [const-pot-bo-parallel-plate](const-pot-bo-parallel-plate.md) | `regressions/md/const-pot-bo-parallel-plate` | lammps ELECTRODE (`perf_exempt`) |
| Freeze electrode atoms (`setforce 0`) | [freeze-electrode-atoms](freeze-electrode-atoms.md) | `regressions/md/freeze-electrode-atoms` | lammps (ratio **3.98**) |
| Electrode total-charge observer | [const-pot-charge-observer](const-pot-charge-observer.md) | `regressions/md/const-pot-charge-observer` | lammps (`perf_exempt`) |
| Charging work \(-\sum_i \Psi_i Q_i\) | [charging-work-energy](charging-work-energy.md) | `regressions/md/charging-work-energy` | literature MW/Scalfi (`perf_exempt`) |
| Const-pot MD NVT short trajectory | [const-pot-md-nvt-short](const-pot-md-nvt-short.md) | `regressions/md/const-pot-md-nvt-short` | metalwalls (`perf_exempt`) |
| Gaussian electrode charge kernels (E2) | [gaussian-electrode-charge](gaussian-electrode-charge.md) | `regressions/md/gaussian-electrode-charge` | literature Reed/MW (`perf_exempt`) |
| Slab 2D Ewald foothold (Yeh–Berkowitz) | [slab-2d-ewald-foothold](slab-2d-ewald-foothold.md) | `regressions/md/slab-2d-ewald-foothold` | lammps (ratio **6.49**) |
| External uniform E-field MD | [external-efield-md](external-efield-md.md) | `regressions/md/external-efield-md` | lammps (ratio **0.60**) |
| Buckingham pair (LAMMPS `buck`) | [buckingham-pair-lammps](buckingham-pair-lammps.md) | `regressions/forcefield/buckingham-pair-lammps` | lammps (`perf_exempt`) |
| Four-site TIP4P foothold | [four-site-tip4p-foothold](four-site-tip4p-foothold.md) | `regressions/md/four-site-tip4p-foothold` | lammps (ratio **0.338**) |

## Index — MetalWalls capability wave 2 (2026-07-24)

Classical electrochemical MD + LAMMPS-overlap force fields (map E2–E4, E6–E8,
E10). Offline pins only — see
[metalwalls-method-handoff](../../developer/metalwalls-method-handoff.md) and
`tools/metalwalls_oracle/`. **Oracle priority:** LAMMPS when the feature
overlaps; MetalWalls / literature for MW-only physics. Fair walls from
REFERENCE / handoff (`ratio` or `perf_exempt`).

| Method | Page | Regression | Oracle |
| ------ | ---- | ---------- | ------ |
| Const-pot CG large multi-site electrode (E2) | [const-pot-cg-large-electrode](const-pot-cg-large-electrode.md) | `regressions/md/const-pot-cg-large-electrode` | metalwalls / dense BO dual (`perf_exempt`) |
| Electrode charge predictor (Kolafa ASPC, E2) | [electrode-charge-predictor](electrode-charge-predictor.md) | `regressions/md/electrode-charge-predictor` | metalwalls (`perf_exempt`) |
| Finite-field + single electrode (Dufils 3D, E3) | [finite-field-single-electrode](finite-field-single-electrode.md) | `regressions/md/finite-field-single-electrode` | metalwalls / literature (`perf_exempt`) |
| Bulk PIM self-consistent induced dipoles (E4) | [pim-induced-dipole-bulk](pim-induced-dipole-bulk.md) | `regressions/md/pim-induced-dipole-bulk` | metalwalls (`perf_exempt`) |
| Rigid piston electrode constant \(P_z\) (E6) | [piston-electrode-pz](piston-electrode-pz.md) | `regressions/md/piston-electrode-pz` | metalwalls (`perf_exempt`) |
| Thomas–Fermi finite metallicity electrode (E7) | [thomas-fermi-electrode](thomas-fermi-electrode.md) | `regressions/md/thomas-fermi-electrode` | metalwalls (`perf_exempt`) |
| Born–Mayer–Huggins pair foothold (E8) | [born-repulsion-foothold](born-repulsion-foothold.md) | `regressions/forcefield/born-repulsion-foothold` | lammps `pair_style born` (`perf_exempt`) |
| Fumi–Tosi multi-type Born + Coulomb NaCl (E8) | [fumi-tosi-nacl](fumi-tosi-nacl.md) | `regressions/forcefield/fumi-tosi-nacl` | lammps `hybrid/overlay born coul/cut` (`perf_exempt`) |
| Steele 10-4-3 wall (Ar single-point, E8) | [steele-wall-argon](steele-wall-argon.md) | `regressions/forcefield/steele-wall-argon` | lammps `wall/lj1043` (`perf_exempt`) |
| Capacitance from electrode charge fluctuations (E10) | [capacitance-charge-fluctuation](capacitance-charge-fluctuation.md) | `regressions/md/capacitance-charge-fluctuation` | literature (`perf_exempt`) |

## Index — MetalWalls capability wave 3 (2026-07-24)

Classical electrochemical MD + LAMMPS-overlap MD (map E3, E5, E9–E10 + shared
MD). Offline pins only — see
[metalwalls-method-handoff](../../developer/metalwalls-method-handoff.md) and
`tools/metalwalls_oracle/`. **Oracle priority:** LAMMPS when the feature
overlaps; MetalWalls / literature for MW-only physics. Fair walls from
REFERENCE / handoff (`ratio` or `perf_exempt`).

| Method | Page | Regression | Oracle |
| ------ | ---- | ---------- | ------ |
| Constant-force ion pull (`fix addforce`) | [constant-force-pull-ion](constant-force-pull-ion.md) | `regressions/md/constant-force-pull-ion` | lammps (ratio **0.12**) |
| Oscillating external E-field MD (E3) | [oscillating-efield-md](oscillating-efield-md.md) | `regressions/md/oscillating-efield-md` | lammps (ratio **2.12**) |
| Slab \(\Psi(z)\) planar Poisson observer (E10) | [potential-profile-observer](potential-profile-observer.md) | `regressions/md/potential-profile-observer` | literature (`perf_exempt`) |
| Surface-tension / NPγT coupling | [surface-tension-coupling](surface-tension-coupling.md) | `regressions/md/surface-tension-coupling` | lammps (ratio **2.88**) |
| Multi-topology virtual sites TIP4P+TIP5P (E9) | [virtual-site-multi-topology](virtual-site-multi-topology.md) | `regressions/md/virtual-site-multi-topology` | lammps (ratio **0.084**) |
| Coupled PIM dipoles ⊕ electrode \(Q\) (E5) | [dipole-electrode-joint](dipole-electrode-joint.md) | `regressions/md/dipole-electrode-joint` | metalwalls (`perf_exempt`) |

## Index — MetalWalls capability wave 5 (2026-07-27)

Electrochem production depth + polarization dynamics (map E1–E4, E7, E9).
Wave 4 (2026-07-24) landed production depth on the existing footholds with no
new method pages, so there is no wave-4 section. Offline pins only — see
[metalwalls-method-handoff](../../developer/metalwalls-method-handoff.md) and
`tools/metalwalls_oracle/`. **Oracle priority:** LAMMPS when the feature
overlaps; MetalWalls / literature for MW-only physics. Fair walls from
REFERENCE / handoff (`ratio` or `perf_exempt`).

| Method | Page | Regression | Oracle |
| ------ | ---- | ---------- | ------ |
| Per-electrode Gaussian width \(\eta_e\) on `ConstPotBO` (E2) | [per-electrode-eta](per-electrode-eta.md) | `regressions/md/per-electrode-eta` | literature Reed/Scalfi/MW (`perf_exempt`) |
| Constant-charge electrode control (conq, E1) | [const-charge-electrode](const-charge-electrode.md) | `regressions/md/const-charge-electrode` | lammps (`perf_exempt`) |
| Post-solve ghost-charge halo refresh (MPI DD, E1) | [const-pot-charge-halo-mpi](const-pot-charge-halo-mpi.md) | `regressions/md/const-pot-charge-halo-mpi` | literature (`perf_exempt`) |
| True mass-zero constrained MaZe charge dynamics (E1) | [maze-symplectic-charge-dynamics](maze-symplectic-charge-dynamics.md) | `regressions/md/maze-symplectic-charge-dynamics` | metalwalls (`perf_exempt`) |
| Constant-\(D\) Zhang–Sprik polarization feedback (E3) | [constant-d-polarization-feedback](constant-d-polarization-feedback.md) | `regressions/md/constant-d-polarization-feedback` | metalwalls (`perf_exempt`) |
| Extended-Lagrangian PIM dipole dynamics (E4) | [extended-lagrangian-dipoles](extended-lagrangian-dipoles.md) | `regressions/md/extended-lagrangian-dipoles` | metalwalls (`perf_exempt`) |
| QEq fluctuating-charge equilibration (E4) | [qeq-fluctuating-charge](qeq-fluctuating-charge.md) | `regressions/md/qeq-fluctuating-charge` | lammps `fix qeq/point` (ratio **0.45**) |
| Bulk polarizable IL NPT — PIM virial → barostat (E9) | [pim-npt-polarizable-il](pim-npt-polarizable-il.md) | `regressions/md/pim-npt-polarizable-il` | metalwalls offline Thole-PIM pin (`perf_exempt`) |
| Voronoi site-volume companion for Thomas–Fermi (E7) | [voronoi-volume-tf](voronoi-volume-tf.md) | `regressions/md/voronoi-volume-tf` | metalwalls (`perf_exempt`) |
| Const-pot ionic-liquid EDL production box (E9) | [const-pot-il-edl-box](const-pot-il-edl-box.md) | `regressions/md/const-pot-il-edl-box` | metalwalls (`perf_exempt`) |

## Index — CP2K capability wave 1 (2026-07-23)

| Method | Page | Regression |
| ------ | ---- | ---------- |
| GPW Γ-RKS H₂O PBE (MT) | [gpw-water-pbe-mt](gpw-water-pbe-mt.md) | `regressions/gpw/gpw-water-pbe-mt` |
| GPW analytic forces H₂O | [gpw-forces-water](gpw-forces-water.md) | `regressions/gpw/gpw-forces-water` |
| Γ-UKS GPW O₂ | [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md) | `regressions/gpw/gpw-uks-o2-gamma` |
| GPW PBE-D3(BJ) ice dimer | [gpw-pbe-d3bj-ice](gpw-pbe-d3bj-ice.md) | `regressions/gpw/gpw-pbe-d3bj-ice` |
| PW LDA Si (k-mesh) | [pw-lda-si](pw-lda-si.md) | `regressions/pw/pw-lda-si` |
| PW true PBE GGA Si | [pw-pbe-gga-si](pw-pbe-gga-si.md) | `regressions/pw/pw-pbe-gga-si` |
| Metallic Fermi–Dirac Al | [pw-smearing-al](pw-smearing-al.md) | `regressions/pw/pw-smearing-al` |
| PW collinear spin O₂ | [pw-spin-polarized-o2](pw-spin-polarized-o2.md) | `regressions/pw/pw-spin-polarized-o2` |
| PW BOMD H (NVE foothold) | [pw-bomd-h](pw-bomd-h.md) | `regressions/pw/pw-bomd-h` |
| OT SCF water (GPW PBE) | [ot-minimizer-water](ot-minimizer-water.md) | `regressions/gpw/ot-minimizer-water` |

## Index — CP2K capability wave 2 (2026-07-24)

Offline CP2K pins only — see [cp2k-method-handoff](../../developer/cp2k-method-handoff.md)
and `tools/cp2k_oracle/`. Fair walls from harness (`ratio` or `perf_exempt`).

| Method | Page | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ------ | ---- | ---------- | --------------------------------------------- |
| GPW cell stress strain-FD H₂O (NUMERICAL pin) | [gpw-stress-water](gpw-stress-water.md) | `regressions/gpw/gpw-stress-water` | **4.75** |
| GPW analytic fixed-\(P\) stress H₂O | [gpw-analytic-stress](gpw-analytic-stress.md) | `regressions/gpw/gpw-analytic-stress` | dual-gate vs FD |
| GPW BOMD H₂O NVE energy-drift foothold | [gpw-bomd-water](gpw-bomd-water.md) | `regressions/gpw/gpw-bomd-water` | **perf_exempt** |
| PW Nielsen–Martin analytic stress Si | [pw-analytic-stress-si](pw-analytic-stress-si.md) | `regressions/pw/pw-analytic-stress-si` | **perf_exempt** |
| FD supercell phonons diamond Si (Γ) | [fd-phonon-supercell](fd-phonon-supercell.md) | `regressions/pw/fd-phonon-supercell` | **perf_exempt** |
| Dudarev DFT+U NiO mol (Γ-UKS GPW) | [gpw-dftu-nio](gpw-dftu-nio.md) | `regressions/gpw/gpw-dftu-nio` | **0.21** |
| PW bands + DOS bulk Si | [bands-dos-si](bands-dos-si.md) | `regressions/pw/bands-dos-si` | **perf_exempt** |
| PW k-point collinear UKS O₂ | [pw-uks-kmesh-fe](pw-uks-kmesh-fe.md) | `regressions/pw/pw-uks-kmesh-fe` | **perf_exempt** |
| GPW Γ-RKS PBEsol Si₂ (MT) | [gpw-pbesol-si](gpw-pbesol-si.md) | `regressions/gpw/gpw-pbesol-si` | **perf_exempt** |
| GPW PBE-D4 periodic water dimer | [gpw-d4-water](gpw-d4-water.md) | `regressions/gpw/gpw-d4-water` | **perf_exempt** |

## Index — CP2K capability wave 11 (2026-07-26)

Offline CP2K pins only — see [cp2k-method-handoff](../../developer/cp2k-method-handoff.md)
wave 11 and `tools/cp2k_oracle/`. Fair walls from harness (`ratio` or
`perf_exempt`). All nine staffed slugs closed 2026-07-26 (pending residual
emptied — not queue exhausted).

| Method | Page | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ------ | ---- | ---------- | --------------------------------------------- |
| Γ-UGW@UKS with RPBE spin XC (sc-Si₂) | [ugw-uks-rpbe](ugw-uks-rpbe.md) | `regressions/gpw/ugw-uks-rpbe` | **perf_exempt** |
| Free-atom TS analytic ionic forces (water dimer) | [gpw-ts-forces](gpw-ts-forces.md) | `regressions/gpw/gpw-ts-forces` | **perf_exempt** |
| Multi-k free-atom TS analytic ionic forces (H₂) | [gpw-ts-kpoint-forces](gpw-ts-kpoint-forces.md) | `regressions/gpw/gpw-ts-kpoint-forces` | **perf_exempt** |
| Free-atom XDM analytic ionic forces (water dimer) | [gpw-xdm-forces](gpw-xdm-forces.md) | `regressions/gpw/gpw-xdm-forces` | **perf_exempt** |
| Multi-k UKS analytic stress RevPBE/RPBE/BLYP | [gpw-kpoint-uks-stress-gga](gpw-kpoint-uks-stress-gga.md) | `regressions/gpw/gpw-kpoint-uks-stress-gga` | **perf_exempt** |
| CPMD + Nosé–Hoover electronic thermostat (H₂O GPW) | [cpmd-electronic-thermostat](cpmd-electronic-thermostat.md) | `regressions/dynamics/cpmd-electronic-thermostat` | **perf_exempt** |
| Multi-k UKS XL-BOMD dual-spin \(P_k^\sigma\) (H₂) | [xl-bomd-uks-kpoints](xl-bomd-uks-kpoints.md) | `regressions/dynamics/xl-bomd-uks-kpoints` | **perf_exempt** |
| PW metallic-occupation bands Si | [bands-metallic-si](bands-metallic-si.md) | `regressions/pw/bands-metallic-si` | **perf_exempt** |
| PW collinear-spin DOS Si | [dos-spin-si](dos-spin-si.md) | `regressions/pw/dos-spin-si` | **perf_exempt** |
| Finite-q DFPT collinear spin + FD smear (Si) | [dfpt-finite-q-spin-smearing](dfpt-finite-q-spin-smearing.md) | `regressions/response/dfpt-finite-q-spin-smearing` | **perf_exempt** |

## Index — CP2K capability wave 12 (2026-07-26)

Offline CP2K pins only — see [cp2k-method-handoff](../../developer/cp2k-method-handoff.md)
wave 12 and `tools/cp2k_oracle/`. Fair walls from harness (`ratio` or
`perf_exempt`). Six staffed slugs closed 2026-07-26 (pending residual not
queue exhausted).

| Method | Page | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ------ | ---- | ---------- | --------------------------------------------- |
| Zero-SCF linearized XL-BOMD forces H₂O NVE | [xl-bomd-linearized-forces](xl-bomd-linearized-forces.md) | `regressions/dynamics/xl-bomd-linearized-forces` | **perf_exempt** |
| Multi-k second-gen CP (ASPC on \(P_k\)) H₂ NVE | [second-gen-cpmd-kpoints](second-gen-cpmd-kpoints.md) | `regressions/dynamics/second-gen-cpmd-kpoints` | **perf_exempt** |
| Multi-k second-gen CP + OT corrector H₂ NVE | [second-gen-cpmd-ot-kpoints](second-gen-cpmd-ot-kpoints.md) | `regressions/dynamics/second-gen-cpmd-ot-kpoints` | **perf_exempt** |
| GPW PBE-D3 + ATM (C9) isolated water dimer | [gpw-d3-atm](gpw-d3-atm.md) | `regressions/gpw/gpw-d3-atm` | **perf_exempt** |
| GPW PBE-D4 + ATM (\(s_9=1\)) periodic water dimer | [gpw-d4-atm](gpw-d4-atm.md) | `regressions/gpw/gpw-d4-atm` | **perf_exempt** |
| PW collinear-spin + metallic-occupation bands Si | [bands-spin-metallic-si](bands-spin-metallic-si.md) | `regressions/pw/bands-spin-metallic-si` | **perf_exempt** |
| PW collinear-spin + metallic FD DOS Si | [dos-spin-metallic-si](dos-spin-metallic-si.md) | `regressions/pw/dos-spin-metallic-si` | **perf_exempt** |

## Index — CP2K capability wave 14 (2026-07-26)

Offline CP2K 2026.1 / pymbd pins only — see
[cp2k-method-queue](../../developer/cp2k-method-queue.md) wave 14 and
`tools/cp2k_oracle/`. Five slugs closed; `gpw-fd-phonon-si` stays open on SCF
conditioning (not force correctness — its earlier block reason was retracted).

| Method | Page | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ------ | ---- | ---------- | --------------------------------------------- |
| Multi-k Car–Parrinello MD on Bloch \(C_k\) (H₂ NVE) | [cpmd-kpoints](cpmd-kpoints.md) | `regressions/dynamics/cpmd-kpoints` | **perf_exempt** |
| Multi-k Marzari–Vanderbilt MLWF on GPW | [periodic-wannier-gpw](periodic-wannier-gpw.md) | `regressions/gpw/periodic-wannier-gpw` | **perf_exempt** |
| GPW PBE-D3 analytic ionic forces (isolated dimer) | [gpw-d3-forces](gpw-d3-forces.md) | `regressions/gpw/gpw-d3-forces` | **1.4** |
| GPW PBE-D3(BJ) analytic ionic forces (periodic dimer) | [gpw-d3bj-forces](gpw-d3bj-forces.md) | `regressions/gpw/gpw-d3bj-forces` | **2.7** |
| GPW Γ analytic cell stress with periodic MBD@rsSCS | [gpw-mbd-stress](gpw-mbd-stress.md) | `regressions/gpw/gpw-mbd-stress` | **perf_exempt** |

## Index — CP2K capability shift wave (2026-07-31)

Offline CP2K 2026.1 pins only — see
[cp2k-method-queue](../../developer/cp2k-method-queue.md) and
`tools/cp2k_oracle/`. Fair walls from each page's Performance section /
`REFERENCE.md` (`ratio` or `perf_exempt`). The ninth slug of this batch,
[dfpt-gga-spin-si](dfpt-gga-spin-si.md) (fair ratio **0.20**), is indexed in
the Response / phonons table below.

| Method | Page | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ------ | ---- | ---------- | --------------------------------------------- |
| GPW PBE-D3 zero-damping analytic cell stress (periodic dimer) | [gpw-d3-stress](gpw-d3-stress.md) | `regressions/gpw/gpw-d3-stress` | **0.55** |
| GPW PBE-D3(BJ) analytic cell stress (periodic dimer) | [gpw-d3bj-stress](gpw-d3bj-stress.md) | `regressions/gpw/gpw-d3bj-stress` | **0.37** |
| Multi-k analytic cell stress + periodic MBD@rsSCS | [gpw-mbd-kpoint-stress](gpw-mbd-kpoint-stress.md) | `regressions/gpw/gpw-mbd-kpoint-stress` | **perf_exempt** |
| GPW DFT-D3 three-body ATM (C9) analytic ionic forces | [gpw-d3-atm-forces](gpw-d3-atm-forces.md) | `regressions/gpw/gpw-d3-atm-forces` | **3.3** |
| GPW DFT-D3 three-body ATM (C9) analytic cell stress | [gpw-d3-atm-stress](gpw-d3-atm-stress.md) | `regressions/gpw/gpw-d3-atm-stress` | **0.75** |
| GPW DFT-D4 three-body ATM (\(s_9=1\)) analytic ionic forces | [gpw-d4-atm-forces](gpw-d4-atm-forces.md) | `regressions/gpw/gpw-d4-atm-forces` | **perf_exempt** |
| Multi-k Dudarev DFT+U analytic ionic forces (H₂ + NiO Γ) | [gpw-dftu-kpoint-forces](gpw-dftu-kpoint-forces.md) | `regressions/gpw/gpw-dftu-kpoint-forces` | **≈ 1.0** |
| Plain multi-k BOMD on GPW (periodic H₂ NVE) | [gpw-bomd-kpoints](gpw-bomd-kpoints.md) | `regressions/gpw/gpw-bomd-kpoints` | **0.064** |
| Multi-k GPW RKS REL_CUTOFF multigrid (diamond Si) | [gpw-multigrid-kpoints](gpw-multigrid-kpoints.md) | `regressions/gpw/gpw-multigrid-kpoints` | **≈ 1.6** |
| Zero-SCF linearized XL-BOMD on Γ-UKS (O₂ NVE) | [xl-bomd-linearized-uks](xl-bomd-linearized-uks.md) | `regressions/dynamics/xl-bomd-linearized-uks` | **perf_exempt** |

## Index — CP2K capability wave 2 (shift) (2026-07-31)

Offline CP2K 2026.1 pins only — see
[cp2k-method-queue](../../developer/cp2k-method-queue.md) wave 2 (shift) and
`tools/cp2k_oracle/`. Fair walls from each page's Performance section /
`REFERENCE.md` (`ratio` or `perf_exempt`). Eight slugs merged; three of them
([gpw-d3-stress](gpw-d3-stress.md) **0.55**,
[gpw-d3-atm-stress](gpw-d3-atm-stress.md) **0.75**,
[second-gen-cpmd-ot-kpoints](second-gen-cpmd-ot-kpoints.md) **perf_exempt**)
already carry index rows in the sections above and are not repeated here.

| Method | Page | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ------ | ---- | ---------- | --------------------------------------------- |
| Finite-\(q\) DFPT phonons at X, pure GGA kernel (diamond Si) | [dfpt-gga-finite-q-si](dfpt-gga-finite-q-si.md) | `regressions/response/dfpt-gga-finite-q-si` | **0.17** |
| Zero-SCF linearized XL-BOMD on the k path (periodic H₂ NVE) | [xl-bomd-linearized-kpoints](xl-bomd-linearized-kpoints.md) | `regressions/dynamics/xl-bomd-linearized-kpoints` | **1.3** |
| GPW nonlocal vdW on the REL_CUTOFF multigrid (NGRIDS 4) | [gpw-vdw-multigrid](gpw-vdw-multigrid.md) | `regressions/gpw/gpw-vdw-multigrid` | **≈ 0.83** |
| Multi-k UKS Car–Parrinello MD on dual-spin Bloch \(C_k^\sigma\) (H₂ NVE) | [cpmd-uks-kpoints](cpmd-uks-kpoints.md) | `regressions/dynamics/cpmd-uks-kpoints` | **perf_exempt** |
| Multi-k UKS CPMD + electronic Nosé–Hoover per (k, σ) (H₂ NVE) | [cpmd-uks-thermostat-kpoints](cpmd-uks-thermostat-kpoints.md) | `regressions/dynamics/cpmd-uks-thermostat-kpoints` | **perf_exempt** |
| GPW DFT-D4 three-body ATM (\(s_9=1\)) analytic cell stress | [gpw-d4-atm-stress](gpw-d4-atm-stress.md) | `regressions/gpw/gpw-d4-atm-stress` | **perf_exempt** |

## Index — CP2K capability wave 3 (shift) (2026-07-31)

Offline CP2K 2026.1 pins only — see
[cp2k-method-queue](../../developer/cp2k-method-queue.md) wave 3 (shift) and
`tools/cp2k_oracle/`. Fair walls from each page's Performance section /
`REFERENCE.md`.

| Method | Page | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ------ | ---- | ---------- | --------------------------------------------- |
| Finite-\(q\) DFPT at X, joint collinear spin × pure GGA (diamond Si) | [dfpt-gga-spin-finite-q-si](dfpt-gga-spin-finite-q-si.md) | `regressions/response/dfpt-gga-spin-finite-q-si` | **0.19** |
| Finite-\(q\) DFPT at X, joint collinear spin × pure GGA × metallic smear (diamond Si) | [dfpt-gga-spin-finite-q-smearing](dfpt-gga-spin-finite-q-smearing.md) | `regressions/response/dfpt-gga-spin-finite-q-smearing` | **0.25** |

## Index — restored capability pages (indexed 2026-07-28)

These pages were written by the Tutorial phase of earlier
`cp2k-method-loop` waves (plus the AO AIMD spec chain `esmm-cpmd-04/05`) but
their index rows never landed: each wave stashed its tutorial pages before the
merge. The pages were recovered on 2026-07-28 and their code samples refreshed
to the current public API (`frame.box`, `set_evaluator` / `set_evaluator_ref`,
and the `cpu::electronic::{CarParrinello,XlBomdForce,SecondGenCp}` /
`{CarParrinelloMethod,XlBomdMethod}` AIMD wrappers).

Every slug below maps 1:1 to a regression directory that exists in the tree.
Fair walls (`ratio` or `perf_exempt`) live in each page's own Performance
section and the matching `REFERENCE.md`; they are not repeated here. Rows are
grouped by regression domain, alphabetical by slug.

### GPW (`regressions/gpw/`)

| Method | Page | Regression |
| ------ | ---- | ---------- |
| GPW Γ-RKS BLYP, isolated H₂O (MT) | [gpw-blyp-water](gpw-blyp-water.md) | `regressions/gpw/gpw-blyp-water` |
| GPW PBE-D4 analytic ionic forces (periodic dimer) | [gpw-d4-forces](gpw-d4-forces.md) | `regressions/gpw/gpw-d4-forces` |
| GPW PBE-D4 analytic cell stress (periodic dimer) | [gpw-d4-stress](gpw-d4-stress.md) | `regressions/gpw/gpw-d4-stress` |
| Multi-k Dudarev DFT+U analytic cell stress (H₂ UKS) | [gpw-dftu-kpoint-stress](gpw-dftu-kpoint-stress.md) | `regressions/gpw/gpw-dftu-kpoint-stress` |
| Multi-k Dudarev DFT+U, GPW UKS (H₂ toy) | [gpw-dftu-kpoints](gpw-dftu-kpoints.md) | `regressions/gpw/gpw-dftu-kpoints` |
| GPW analytic cell stress + Dudarev DFT+U (NiO) | [gpw-dftu-stress](gpw-dftu-stress.md) | `regressions/gpw/gpw-dftu-stress` |
| GPW multi-k Hirshfeld volumes (water dimer) | [gpw-hirshfeld-kpoints](gpw-hirshfeld-kpoints.md) | `regressions/gpw/gpw-hirshfeld-kpoints` |
| Γ-UKS Hirshfeld-scaled MBD@rsSCS (O₂) | [gpw-hirshfeld-uks](gpw-hirshfeld-uks.md) | `regressions/gpw/gpw-hirshfeld-uks` |
| Multi-k UKS Hirshfeld-scaled MBD@rsSCS (O₂) | [gpw-hirshfeld-uks-kpoints](gpw-hirshfeld-uks-kpoints.md) | `regressions/gpw/gpw-hirshfeld-uks-kpoints` |
| GPW analytic forces at k-points (H₂) | [gpw-kpoint-forces](gpw-kpoint-forces.md) | `regressions/gpw/gpw-kpoint-forces` |
| GPW analytic cell stress at k-points (diamond Si) | [gpw-kpoint-stress](gpw-kpoint-stress.md) | `regressions/gpw/gpw-kpoint-stress` |
| Multi-k UKS GPW analytic forces (high-spin H₂) | [gpw-kpoint-uks-forces](gpw-kpoint-uks-forces.md) | `regressions/gpw/gpw-kpoint-uks-forces` |
| Multi-k UKS GPW analytic cell stress (high-spin H₂) | [gpw-kpoint-uks-stress](gpw-kpoint-uks-stress.md) | `regressions/gpw/gpw-kpoint-uks-stress` |
| GPW collinear multi-k UKS (H₂ identity + O₂ twin) | [gpw-kpoints-uks](gpw-kpoints-uks.md) | `regressions/gpw/gpw-kpoints-uks` |
| GPW MBD@rsSCS ionic forces (water dimer) | [gpw-mbd-forces](gpw-mbd-forces.md) | `regressions/gpw/gpw-mbd-forces` |
| GPW PBE-MBD@rsSCS + Hirshfeld volumes (water dimer) | [gpw-mbd-hirshfeld](gpw-mbd-hirshfeld.md) | `regressions/gpw/gpw-mbd-hirshfeld` |
| GPW PBE-MBD@rsSCS, free-atom water dimer | [gpw-mbd-ice](gpw-mbd-ice.md) | `regressions/gpw/gpw-mbd-ice` |
| GPW multi-k analytic forces + MBD@rsSCS | [gpw-mbd-kpoint-forces](gpw-mbd-kpoint-forces.md) | `regressions/gpw/gpw-mbd-kpoint-forces` |
| GPW PBE-MBD@rsSCS, periodic Ewald + k-mesh | [gpw-mbd-periodic](gpw-mbd-periodic.md) | `regressions/gpw/gpw-mbd-periodic` |
| Metallic Fermi–Dirac smearing (BCC Na, multi-k) | [gpw-metallic-smearing](gpw-metallic-smearing.md) | `regressions/gpw/gpw-metallic-smearing` |
| GPW optB88-vdW (periodic water dimer) | [gpw-optb88-vdw](gpw-optb88-vdw.md) | `regressions/gpw/gpw-optb88-vdw` |
| GPW Γ-RKS revPBE (Si₂ fragment, MT) | [gpw-revpbe-si](gpw-revpbe-si.md) | `regressions/gpw/gpw-revpbe-si` |
| GPW Γ-RKS RPBE, isolated H₂O (MT) | [gpw-rpbe-water](gpw-rpbe-water.md) | `regressions/gpw/gpw-rpbe-water` |
| GPW PBE + rVV10 (periodic water dimer) | [gpw-rvv10-ice](gpw-rvv10-ice.md) | `regressions/gpw/gpw-rvv10-ice` |
| GPW PBE-TS (periodic water dimer) | [gpw-ts-ice](gpw-ts-ice.md) | `regressions/gpw/gpw-ts-ice` |
| Multi-k free-atom TS analytic ionic forces (H₂) | [gpw-ts-kpoint-forces](gpw-ts-kpoint-forces.md) | `regressions/gpw/gpw-ts-kpoint-forces` |
| Γ-UKS GPW BLYP (O₂) | [gpw-uks-blyp](gpw-uks-blyp.md) | `regressions/gpw/gpw-uks-blyp` |
| Γ-UKS GPW analytic forces (O₂ / PBE, MT) | [gpw-uks-forces](gpw-uks-forces.md) | `regressions/gpw/gpw-uks-forces` |
| Γ-UKS GPW multigrid (O₂ / PADE) | [gpw-uks-multigrid](gpw-uks-multigrid.md) | `regressions/gpw/gpw-uks-multigrid` |
| Γ-UKS GPW PBE (O₂) | [gpw-uks-pbe](gpw-uks-pbe.md) | `regressions/gpw/gpw-uks-pbe` |
| Γ-UKS GPW revPBE (O₂) | [gpw-uks-revpbe](gpw-uks-revpbe.md) | `regressions/gpw/gpw-uks-revpbe` |
| Γ-UKS GPW RPBE (O₂) | [gpw-uks-rpbe](gpw-uks-rpbe.md) | `regressions/gpw/gpw-uks-rpbe` |
| Collinear UKS + FD smearing (BCC Na, multi-k) | [gpw-uks-smearing](gpw-uks-smearing.md) | `regressions/gpw/gpw-uks-smearing` |
| Γ-UKS GPW analytic cell stress (O₂ / PBE, MT) | [gpw-uks-stress](gpw-uks-stress.md) | `regressions/gpw/gpw-uks-stress` |
| GPW `CELL_OPT` MOLOPT re-twin (water chain) | [gpw-variable-cell-molopt](gpw-variable-cell-molopt.md) | `regressions/gpw/gpw-variable-cell-molopt` |
| GPW PBE + vdW-DF (DRSLL), periodic dimer | [gpw-vdw-df-ice](gpw-vdw-df-ice.md) | `regressions/gpw/gpw-vdw-df-ice` |
| GPW PBE + vdW-DF2 (LMKLL), periodic dimer | [gpw-vdw-df2-ice](gpw-vdw-df2-ice.md) | `regressions/gpw/gpw-vdw-df2-ice` |
| GPW nonlocal vdW: self-consistent \(V_{\mathrm{nl}}\) + forces | [gpw-vdw-scf-potential](gpw-vdw-scf-potential.md) | `regressions/gpw/gpw-vdw-scf-potential` |
| GPW nonlocal vdW on the REL_CUTOFF multigrid (NGRIDS 4) | [gpw-vdw-multigrid](gpw-vdw-multigrid.md) | `regressions/gpw/gpw-vdw-multigrid` |
| GPW free-atom XDM (periodic water dimer) | [gpw-xdm-ice](gpw-xdm-ice.md) | `regressions/gpw/gpw-xdm-ice` |
| MPI GPW k-point scaling (4-rank gate) | [mpi-gpw-scaling-4rank](mpi-gpw-scaling-4rank.md) | `regressions/gpw/mpi-gpw-scaling-4rank` |
| OT + Dudarev DFT+U, Γ-UKS GPW (NiO) | [ot-dftu](ot-dftu.md) | `regressions/gpw/ot-dftu` |
| Multi-k OT + Dudarev DFT+U (H₂) | [ot-dftu-kpoints](ot-dftu-kpoints.md) | `regressions/gpw/ot-dftu-kpoints` |
| Multi-k OT + DFT+U + Fermi–Dirac smearing (H₂) | [ot-dftu-smearing](ot-dftu-smearing.md) | `regressions/gpw/ot-dftu-smearing` |
| Multi-k Orbital Transformation SCF (compact H₂) | [ot-kpoints](ot-kpoints.md) | `regressions/gpw/ot-kpoints` |
| OT + metallic Fermi–Dirac smearing (BCC Na) | [ot-smearing](ot-smearing.md) | `regressions/gpw/ot-smearing` |
| Multi-k UKS OT + Dudarev DFT+U (high-spin H₂) | [ot-uks-dftu](ot-uks-dftu.md) | `regressions/gpw/ot-uks-dftu` |
| Multi-k UKS OT + DFT+U + metallic smearing (H₂) | [ot-uks-dftu-smearing](ot-uks-dftu-smearing.md) | `regressions/gpw/ot-uks-dftu-smearing` |
| Multi-k collinear UKS OT (high-spin H₂) | [ot-uks-kpoints](ot-uks-kpoints.md) | `regressions/gpw/ot-uks-kpoints` |
| OT orbital minimizer for Γ-UKS (O₂) | [ot-uks-o2](ot-uks-o2.md) | `regressions/gpw/ot-uks-o2` |
| UKS OT + metallic Fermi–Dirac smearing (BCC Na) | [ot-uks-smearing](ot-uks-smearing.md) | `regressions/gpw/ot-uks-smearing` |
| Periodic finite-\(q\) inter-\(k\) spectral G0W0 (Si) | [periodic-gw-finite-q](periodic-gw-finite-q.md) | `regressions/gpw/periodic-gw-finite-q` |
| Multi-k spectral G0W0 (diamond Si, bct) | [periodic-gw-kpoints](periodic-gw-kpoints.md) | `regressions/gpw/periodic-gw-kpoints` |
| Periodic Γ-point spectral G0W0 (sc-Si₂) | [periodic-gw-si](periodic-gw-si.md) | `regressions/gpw/periodic-gw-si` |
| Multi-k RI-GPW MP2 (diamond Si, bct) | [periodic-mp2-kpoints](periodic-mp2-kpoints.md) | `regressions/gpw/periodic-mp2-kpoints` |
| Periodic Γ-point GPW MP2, canonical (sc-Si₂) | [periodic-mp2-si](periodic-mp2-si.md) | `regressions/gpw/periodic-mp2-si` |
| Periodic Γ-point RI-GPW MP2 (sc-Si₂) | [periodic-ri-mp2-si](periodic-ri-mp2-si.md) | `regressions/gpw/periodic-ri-mp2-si` |
| Finite-\(q\) inter-\(k\) RI-GPW dRPA (diamond Si) | [periodic-rpa-finite-q](periodic-rpa-finite-q.md) | `regressions/gpw/periodic-rpa-finite-q` |
| Multi-k RI-GPW direct RPA (diamond Si₄, bct) | [periodic-rpa-kpoints](periodic-rpa-kpoints.md) | `regressions/gpw/periodic-rpa-kpoints` |
| Periodic Γ-point GPW direct RPA (sc-Si₂) | [periodic-rpa-si](periodic-rpa-si.md) | `regressions/gpw/periodic-rpa-si` |
| Multi-k unrestricted spectral G0W0 (UGW) | [periodic-ugw-kpoints](periodic-ugw-kpoints.md) | `regressions/gpw/periodic-ugw-kpoints` |
| Periodic Γ-point UGW@UKS (sc-Si₂) | [periodic-ugw-si](periodic-ugw-si.md) | `regressions/gpw/periodic-ugw-si` |
| Multi-k unrestricted GPW MP2 (UMP2) | [periodic-ump2-kpoints](periodic-ump2-kpoints.md) | `regressions/gpw/periodic-ump2-kpoints` |
| Periodic Γ-point GPW UMP2 (sc-Si₂) | [periodic-ump2-si](periodic-ump2-si.md) | `regressions/gpw/periodic-ump2-si` |
| Multi-k unrestricted direct RPA (URPA@UKS) | [periodic-urpa-kpoints](periodic-urpa-kpoints.md) | `regressions/gpw/periodic-urpa-kpoints` |
| Periodic Γ-point URPA@UKS (sc-Si₂) | [periodic-urpa-si](periodic-urpa-si.md) | `regressions/gpw/periodic-urpa-si` |

### Plane wave (`regressions/pw/`)

| Method | Page | Regression |
| ------ | ---- | ---------- |
| PW collinear spin band structure (bulk Si) | [bands-spin-si](bands-spin-si.md) | `regressions/pw/bands-spin-si` |
| Collinear-spin Berry-phase polarization (bulk Si) | [berry-spin-si](berry-spin-si.md) | `regressions/pw/berry-spin-si` |
| Metallic-occupation DOS (bulk Si, Fermi–Dirac) | [dos-metallic-si](dos-metallic-si.md) | `regressions/pw/dos-metallic-si` |
| MPI plane-wave k-mesh scaling (4 ranks) | [mpi-pw-scaling-4rank](mpi-pw-scaling-4rank.md) | `regressions/pw/mpi-pw-scaling-4rank` |
| Periodic multi-k MLWF (bulk Si, Marzari–Vanderbilt) | [periodic-wannier-si](periodic-wannier-si.md) | `regressions/pw/periodic-wannier-si` |
| Collinear spin-polarized multi-k MLWF (bulk Si, UKS) | [periodic-wannier-uks](periodic-wannier-uks.md) | `regressions/pw/periodic-wannier-uks` |
| PW collinear spin-GGA PBE (O₂ triplet, Γ) | [pw-spin-gga](pw-spin-gga.md) | `regressions/pw/pw-spin-gga` |
| PW collinear UKS + metallic smearing (BCC Fe) | [pw-uks-fe-smearing](pw-uks-fe-smearing.md) | `regressions/pw/pw-uks-fe-smearing` |
| PW isotropic variable-cell relax (diamond Si) | [pw-variable-cell-si](pw-variable-cell-si.md) | `regressions/pw/pw-variable-cell-si` |

### Response / phonons (`regressions/response/`)

| Method | Page | Regression |
| ------ | ---- | ---------- |
| Finite-\(q\) DFPT phonons at X (diamond Si) | [dfpt-finite-q-si](dfpt-finite-q-si.md) | `regressions/response/dfpt-finite-q-si` |
| Finite-\(q\) DFPT phonons + metallic smearing (Si at X) | [dfpt-finite-q-smearing](dfpt-finite-q-smearing.md) | `regressions/response/dfpt-finite-q-smearing` |
| Finite-\(q\) DFPT phonons at X, pure GGA kernel (diamond Si) | [dfpt-gga-finite-q-si](dfpt-gga-finite-q-si.md) | `regressions/response/dfpt-gga-finite-q-si` |
| DFPT phonons at Γ, pure GGA kernel (diamond Si) | [dfpt-gga-si](dfpt-gga-si.md) | `regressions/response/dfpt-gga-si` |
| Finite-\(q\) DFPT at X, joint collinear spin × pure GGA (diamond Si) | [dfpt-gga-spin-finite-q-si](dfpt-gga-spin-finite-q-si.md) | `regressions/response/dfpt-gga-spin-finite-q-si` |
| Finite-\(q\) DFPT at X, joint collinear spin × pure GGA × metallic smear (diamond Si) | [dfpt-gga-spin-finite-q-smearing](dfpt-gga-spin-finite-q-smearing.md) | `regressions/response/dfpt-gga-spin-finite-q-smearing` |
| DFPT phonons at Γ, joint collinear spin × pure GGA (diamond Si) | [dfpt-gga-spin-si](dfpt-gga-spin-si.md) | `regressions/response/dfpt-gga-spin-si` |
| DFPT phonons at Γ, joint collinear spin × pure GGA × metallic smear (diamond Si) | [dfpt-gga-spin-smearing](dfpt-gga-spin-smearing.md) | `regressions/response/dfpt-gga-spin-smearing` |
| Incommensurate-\(q\) DFPT phonons (diamond Si) | [dfpt-incommensurate-si](dfpt-incommensurate-si.md) | `regressions/response/dfpt-incommensurate-si` |
| DFPT phonons + metallic smearing (diamond Si) | [dfpt-metallic-si](dfpt-metallic-si.md) | `regressions/response/dfpt-metallic-si` |
| DFPT phonons at Γ (diamond Si, PW LSDA) | [dfpt-phonon-si](dfpt-phonon-si.md) | `regressions/response/dfpt-phonon-si` |
| Collinear-spin DFPT at incommensurate \(q\) (Si) | [dfpt-spin-incommensurate](dfpt-spin-incommensurate.md) | `regressions/response/dfpt-spin-incommensurate` |
| DFPT phonons at Γ, collinear spin (diamond Si) | [dfpt-spin-si](dfpt-spin-si.md) | `regressions/response/dfpt-spin-si` |
| DFPT phonons: collinear spin + metallic smearing (Si) | [dfpt-spin-smearing](dfpt-spin-smearing.md) | `regressions/response/dfpt-spin-smearing` |
| Phonon dispersion \(\omega(q)\) (diamond Si, SW IFC) | [phonon-dispersion-si](phonon-dispersion-si.md) | `regressions/response/phonon-dispersion-si` |

### Ab-initio MD (`regressions/dynamics/`)

| Method | Page | Regression |
| ------ | ---- | ---------- |
| Car–Parrinello MD + electronic RATTLE (H₂O NVE, GPW) | [cpmd-rattle-water](cpmd-rattle-water.md) | `regressions/dynamics/cpmd-rattle-water` |
| AO-basis CPMD on the SCF methods (RHF / RKS / UHF) | [cpmd-rhf-h2](cpmd-rhf-h2.md) | `regressions/dynamics/cpmd-{rhf-h2,rks-water,uhf-o2}` |
| Car–Parrinello MD on Γ-UKS (O₂ triplet NVE, GPW) | [cpmd-uks](cpmd-uks.md) | `regressions/dynamics/cpmd-uks` |
| Car–Parrinello MD (H₂O NVE, GPW fictitious e⁻ DOF) | [cpmd-water](cpmd-water.md) | `regressions/dynamics/cpmd-water` |
| Second-generation CP with OT corrector (H₂O NVE) | [second-gen-cpmd-ot](second-gen-cpmd-ot.md) | `regressions/dynamics/second-gen-cpmd-ot` |
| Second-generation CP on Γ-UKS (O₂ triplet NVE) | [second-gen-cpmd-uks](second-gen-cpmd-uks.md) | `regressions/dynamics/second-gen-cpmd-uks` |
| Second-generation CP + OT corrector on Γ-UKS (O₂) | [second-gen-cpmd-uks-ot](second-gen-cpmd-uks-ot.md) | `regressions/dynamics/second-gen-cpmd-uks-ot` |
| Second-generation CP: ASPC + partial SCF (H₂O NVE) | [second-gen-cpmd-water](second-gen-cpmd-water.md) | `regressions/dynamics/second-gen-cpmd-water` |
| Multi-k XL-BOMD (H₂ NVE, GPW + k-mesh) | [xl-bomd-kpoints](xl-bomd-kpoints.md) | `regressions/dynamics/xl-bomd-kpoints` |
| XL-BOMD on the Gaussian SCF methods (RHF H₂O NVE) | [xl-bomd-rhf-water](xl-bomd-rhf-water.md) | `regressions/dynamics/xl-bomd-rhf-water` |
| XL-BOMD on Γ-UKS (O₂ triplet NVE, GPW) | [xl-bomd-uks](xl-bomd-uks.md) | `regressions/dynamics/xl-bomd-uks` |
| XL-BOMD, extended-Lagrangian BOMD (H₂O NVE, GPW) | [xl-bomd-water](xl-bomd-water.md) | `regressions/dynamics/xl-bomd-water` |

### Gaussian SCF (`regressions/scf/`)

| Method | Page | Regression |
| ------ | ---- | ---------- |
| Maximally localized Wannier functions (Foster–Boys) | [wannier-centers](wannier-centers.md) | `regressions/scf/wannier-centers` |

## Index — PySCF campaign methods

| Method | Page | Regression |
| ------ | ---- | ---------- |
| UMP2 | [ump2-o2-sto3g](ump2-o2-sto3g.md) | `regressions/scf/ump2-o2-sto3g` |
| Hybrid TDDFT-TDA (B3LYP) | [h2-b3lyp-hybrid-tda](h2-b3lyp-hybrid-tda.md) | `…/h2-b3lyp-hybrid-tda` |
| Frozen-core RMP2 | [frozen-core-mp2-h2o](frozen-core-mp2-h2o.md) | `…/frozen-core-mp2-h2o` |
| CIS(D) | [cisd-h2co-sto3g](cisd-h2co-sto3g.md) | `…/cisd-h2co-sto3g` |
| BLYP GGA | [blyp-water-sto3g](blyp-water-sto3g.md) | `…/blyp-water-sto3g` |
| GHF analytic gradient | [ghf-grad-h3-sto3g](ghf-grad-h3-sto3g.md) | `…/ghf-grad-h3-sto3g` |
| C-PCM free-energy forces | [pcm-grad-h2o](pcm-grad-h2o.md) | `…/pcm-grad-h2o` |
| SCS-MP2 | [scs-mp2-h2o-sto3g](scs-mp2-h2o-sto3g.md) | `…/scs-mp2-h2o-sto3g` |
| Full LR-TDDFT (RKS) | [full-lr-tddft-h2-pbe](full-lr-tddft-h2-pbe.md) | `…/full-lr-tddft-h2-pbe` |
| ROKS | [roks-o2-sto3g](roks-o2-sto3g.md) | `…/roks-o2-sto3g` |
| RHF static polarizability | [static-polariz-h2o](static-polariz-h2o.md) | `…/static-polariz-h2o` |
| IEF-PCM | [ief-pcm-h2o](ief-pcm-h2o.md) | `…/ief-pcm-h2o` |
| ROMP2 | [romp2-o2-sto3g](romp2-o2-sto3g.md) | `…/romp2-o2-sto3g` |
| ROKS analytic gradient | [roks-grad-o2-sto3g](roks-grad-o2-sto3g.md) | `…/roks-grad-o2-sto3g` |
| Hybrid full-LR TDDFT (B3LYP RPA) | [h2-b3lyp-rpa-tddft](h2-b3lyp-rpa-tddft.md) | `…/h2-b3lyp-rpa-tddft` |
| Spin-flip TDA (SF-CIS) | [spin-flip-tda-o2](spin-flip-tda-o2.md) | `…/spin-flip-tda-o2` |
| Spin-flip full TDDFT (SF-RPA) | [sf-tddft-o2](sf-tddft-o2.md) | `…/sf-tddft-o2` |
| ROHF nuclear Hessian | [rohf-hessian-o2](rohf-hessian-o2.md) | `…/rohf-hessian-o2` |
| ADC(2) | [adc2-h2co-sto3g](adc2-h2co-sto3g.md) | `…/adc2-h2co-sto3g` |
| ADC(2)-x | [adc2x-h2co-sto3g](adc2x-h2co-sto3g.md) | `…/adc2x-h2co-sto3g` |
| IP-ADC(2) | [ip-adc2-h2o-sto3g](ip-adc2-h2o-sto3g.md) | `…/ip-adc2-h2o-sto3g` |
| EA-ADC(2) | [ea-adc2-h2-sto3g](ea-adc2-h2-sto3g.md) | `…/ea-adc2-h2-sto3g` |
| RMP3 | [mp3-h2-sto3g](mp3-h2-sto3g.md) | `…/mp3-h2-sto3g` |
| RMP4(SDTQ) | [mp4-h2-sto3g](mp4-h2-sto3g.md) | `…/mp4-h2-sto3g` |
| CCSD | [ccsd-h2-sto3g](ccsd-h2-sto3g.md) | `…/ccsd-h2-sto3g` |
| CCSD nuclear gradient | [ccsd-grad-h2o-sto3g](ccsd-grad-h2o-sto3g.md) | `…/ccsd-grad-h2o-sto3g` |
| CC2 | [cc2-h2-sto3g](cc2-h2-sto3g.md) | `…/cc2-h2-sto3g` |
| CCSD(T) | [ccsd-t-h2o-sto3g](ccsd-t-h2o-sto3g.md) | `…/ccsd-t-h2o-sto3g` |
| G0W0 | [g0w0-h2-sto3g](g0w0-h2-sto3g.md) | `…/g0w0-h2-sto3g` |
| BSE@G0W0 | [bse-h2-sto3g](bse-h2-sto3g.md) | `…/bse-h2-sto3g` |
| B2PLYP double hybrid | [b2plyp-h2o](b2plyp-h2o.md) | `…/b2plyp-h2o` |
| CASCI | [casci-h2-sto3g](casci-h2-sto3g.md) | `…/casci-h2-sto3g` |
| CASSCF | [casscf-h2-sto3g](casscf-h2-sto3g.md) | `…/casscf-h2-sto3g` |
| SC-NEVPT2 | [nevpt2-h2-sto3g](nevpt2-h2-sto3g.md) | `…/nevpt2-h2-sto3g` |
| ΔSCF (IMOM) | [delta-scf-n2](delta-scf-n2.md) | `…/delta-scf-n2` |
| EOM-EE-CCSD | [eom-ee-ccsd](eom-ee-ccsd.md) | `…/eom-ee-ccsd` |
| EOM-SF-CCSD | [eom-sf-ccsd-o2](eom-sf-ccsd-o2.md) | `…/eom-sf-ccsd-o2` |
| EOM-CC2 | [eom-cc2-h2](eom-cc2-h2.md) | `…/eom-cc2-h2` |
| DFT-D4 (PBE params) | [dft-d4-h2o-pbe](dft-d4-h2o-pbe.md) | `…/dft-d4-h2o-pbe` |
| EOM-EA-CCSD | [eom-ea-ccsd](eom-ea-ccsd.md) | `…/eom-ea-ccsd` |
| EOM-IP-CCSD | [eom-ip-ccsd](eom-ip-ccsd.md) | `…/eom-ip-ccsd` |
| One-shot GF2 (G0F2) | [gf2-h2-sto3g](gf2-h2-sto3g.md) | `…/gf2-h2-sto3g` |
| Direct RPA | [direct-rpa-h2-sto3g](direct-rpa-h2-sto3g.md) | `…/direct-rpa-h2-sto3g` |
| RKS nuclear Hessian (FD of forces) | [rks-hessian-h2o-sto3g](rks-hessian-h2o-sto3g.md) | `…/rks-hessian-h2o-sto3g` |
| UKS nuclear Hessian (FD of forces) | [uks-hessian-o2-sto3g](uks-hessian-o2-sto3g.md) | `…/uks-hessian-o2-sto3g` |
| UKS full-LR TDDFT (LSDA O₂) | [uks-full-lr-o2-lsda](uks-full-lr-o2-lsda.md) | `…/uks-full-lr-o2-lsda` |
| UMP2 analytic gradient | [ump2-grad-o2-sto3g](ump2-grad-o2-sto3g.md) | `…/ump2-grad-o2-sto3g` |

Existing pages may still use an older Theory/Practice layout; next workflow run
or a bulk rewrite should move them to the technical-report template above.
