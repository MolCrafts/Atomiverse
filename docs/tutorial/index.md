# Tutorials

Each chapter teaches one scientific method. Install and project layout live in
the [User Guide](../user-guide/). Every chapter uses `cpu::Frame`, the
Driver/Component pattern, and units (eV, Å, fs, amu).

## How every chapter is organized

| Block | What it is |
|-------|------------|
| **Theory** | Why the method exists, equations, assumptions, limitations |
| **Practice** | Minimal C++ (members, `run`, outputs) |
| **Results** | Engine numbers vs literature / PySCF / other reference |

Read Theory with a pen; type or adapt the Practice snippet; check Results
against the table. Figures appear only for computed fields or quantitative
trends—not bare toy geometries.

Suggested paths:

- Electronic structure: [Ch. 2](ch02-rhf.md) → [3](ch03-open-shell.md) →
  [4](ch04-basis-sets.md) → [18](ch18-scf-techniques.md)–[20](ch20-hf-properties-environment.md)
- Classical MD: [Ch. 10](ch10-classical-md.md)
- Architecture: [Ch. 0](ch00-architecture.md)

## I. Foundations

| Ch | Title |
|----|-------|
| [0](ch00-architecture.md) | CPU Backend |
| [1](ch01-foundations.md) | Systems and Units |

## II. Hartree–Fock core

| Ch | Title |
|----|-------|
| [2](ch02-rhf.md) | Restricted Hartree–Fock |
| [3](ch03-open-shell.md) | Unrestricted Hartree–Fock |
| [4](ch04-basis-sets.md) | Basis Sets |

## III. DFT

| Ch | Title |
|----|-------|
| [5](ch05-kohn-sham-dft.md) | Kohn–Sham DFT |
| [6](ch06-hybrid-dft.md) | Hybrid Functionals |

## IV. Correlation

| Ch | Title |
|----|-------|
| [7](ch07-correlation.md) | Correlation Methods |

## V. Periodic electronic structure

| Ch | Title |
|----|-------|
| [8](ch08-plane-wave.md) | Plane-Wave DFT |
| [9](ch09-gpw.md) | Gaussian and Plane Waves |

## VI. Classical simulation

| Ch | Title |
|----|-------|
| [10](ch10-classical-md.md) | Classical Molecular Dynamics |
| [11](ch11-ensembles.md) | Ensembles |
| [12](ch12-geometry-optimization.md) | Geometry Optimization |

## VII. Analysis and response

| Ch | Title |
|----|-------|
| [13](ch13-observables.md) | Observables |
| [14](ch14-vibrations.md) | Vibrations |
| [15](ch15-cross-method.md) | Cross-Method Workflows |

## VIII–IX. Beyond pairs and semiempirical

| Ch | Title |
|----|-------|
| [16](ch16-many-body-potentials.md) | Many-Body Potentials |
| [17](ch17-semiempirical-am1.md) | Semiempirical AM1 |

## X. Hartree–Fock advanced

| Ch | Title |
|----|-------|
| [18](ch18-scf-techniques.md) | SCF Convergence |
| [19](ch19-hf-large-systems.md) | Large Molecules |
| [20](ch20-hf-properties-environment.md) | Properties and Environment |
| [Appendix](appendix-pyscf-oracle.md) | External References |

---

## Method tutorials

Campaign handbook pages (technical-report style, one per regression slug).
Full index and template: [methods/README](methods/README.md).

### TB / SE capability wave 1 (semiempirical + tight-binding)

Offline OpenMOPAC / DFTB+ / xtb pins only — see
[tb-method-handoff](../developer/tb-method-handoff.md) and `tools/tb_oracle/`.
Family-locked oracles: NDDO → OpenMOPAC; DFTB* → DFTB+; GFN-xTB → xtb.
Narrative: [Ch. 17 Semiempirical AM1](ch17-semiempirical-am1.md).

| Method | Page | Regression | Oracle |
|--------|------|------------|--------|
| SCC-DFTB2 CH₄ 3ob-3-1 organic SK | [dftb2-3ob-organic](methods/dftb2-3ob-organic.md) | `regressions/semiempirical/dftb2-3ob-organic` | dftb+ 24.1 (`perf_exempt`) |
| SCC-DFTB2 analytic nuclear forces H₂O | [dftb2-analytic-forces](methods/dftb2-analytic-forces.md) | `regressions/semiempirical/dftb2-analytic-forces` | dftb+ formula (`perf_exempt`) |
| Collinear spin-DFTB2 O₂ triplet | [spin-dftb2-o2](methods/spin-dftb2-o2.md) | `regressions/semiempirical/spin-dftb2-o2` | literature + family dftb+ (`perf_exempt`) |

### TB / SE capability wave 2 (semiempirical + tight-binding)

Offline OpenMOPAC / DFTB+ / xtb pins only — see
[tb-method-handoff](../developer/tb-method-handoff.md) and `tools/tb_oracle/`.
Family-locked oracles: NDDO → OpenMOPAC; DFTB* → DFTB+; GFN-xTB → xtb.
Full index: [methods/README](methods/README.md). Narrative:
[Ch. 17 Semiempirical AM1](ch17-semiempirical-am1.md).

| Method | Page | Regression | Oracle |
|--------|------|------------|--------|
| SCC-DFTB2 H₂ total \(E\) vs mio-1-1 | [dftb2-dftbplus-energy-pin](methods/dftb2-dftbplus-energy-pin.md) | `regressions/semiempirical/dftb2-dftbplus-energy-pin` | dftb+ 24.1 (ratio **≈ 0.4**) |
| Γ-point periodic SCC-DFTB2 (1-D C chain) | [dftb2-periodic-gamma](methods/dftb2-periodic-gamma.md) | `regressions/semiempirical/dftb2-periodic-gamma` | literature + family dftb+ (`perf_exempt`) |

### TB / SE capability wave 3 (semiempirical + tight-binding)

Offline OpenMOPAC / DFTB+ / xtb pins only — see
[tb-method-handoff](../developer/tb-method-handoff.md) and `tools/tb_oracle/`.
Family-locked oracles: NDDO → OpenMOPAC; DFTB* → DFTB+; GFN-xTB → xtb.
Full index: [methods/README](methods/README.md). Final harness-wave-3 batch
(lattice-image Γ-PBC \(F\), multi-k energy, DFTB3 energy).

| Method | Page | Regression | Oracle |
|--------|------|------------|--------|
| Γ-PBC analytic nuclear forces (off-eq C chain) | [dftb2-periodic-forces](methods/dftb2-periodic-forces.md) | `regressions/semiempirical/dftb2-periodic-forces` | literature + family dftb+ (`perf_exempt`) |
| Multi-k SCC-DFTB2 (4×1×1 C chain) | [dftb2-kpoints](methods/dftb2-kpoints.md) | `regressions/semiempirical/dftb2-kpoints` | dftb+ family (`perf_exempt`) |
| SCC-DFTB3 H₂O (mio-style; third-order \(E\)) | [dftb3-water-mio](methods/dftb3-water-mio.md) | `regressions/semiempirical/dftb3-water-mio` | dftb+ formula-Gaus2011 (`perf_exempt`) |
### TB / SE capability wave 4 (semiempirical + tight-binding)

Offline OpenMOPAC / DFTB+ / xtb pins only — see
[tb-method-handoff](../developer/tb-method-handoff.md) and `tools/tb_oracle/`.
Family-locked oracles: NDDO → OpenMOPAC; DFTB* → DFTB+; GFN-xTB → xtb. A row
may carry `oracle: literature` (formalism only) when neither an oracle binary
nor a published value for the exact geometry + SK set exists; the case then
ships structural gates and says so in its REFERENCE `Gaps`.
Full index: [methods/README](methods/README.md).

| Method | Page | Regression | Oracle |
|--------|------|------------|--------|
| Linear-response TD-DFTB (Casida) H₂ singlet/triplet | [td-dftb-h2](methods/td-dftb-h2.md) | `regressions/semiempirical/td-dftb-h2` | literature (formalism; **no absolute pin**, `perf_exempt`) |
| DFTB3 closed-form nuclear gradient (H₂O off-eq) | — | `regressions/semiempirical/dftb3-analytic-forces` | dftb+ formula-Gaus2011 + h² FD identity (`perf_exempt`) |
| Multi-k Bloch analytic F **and** σ (off-eq C chain) | — | `regressions/semiempirical/dftb2-multi-k-forces` | dftb+ formula + FD/strain-FD identity (`perf_exempt`) |
| Γ-PBC lattice-sum D3(BJ) (water dimer, 12 Å cell) | — | `regressions/semiempirical/dftb2-dispersion-d3-periodic` | **s-dftd3 1.2.1 offline, periodic** (`perf_exempt`) |
| ThreeOb closed-form cell stress (asymmetric CH₄) | — | `regressions/semiempirical/dftb2-analytic-stress-threeob` | dftb+ formula + strain-FD (`perf_exempt`) |
| Collinear open-shell cell stress (O₂ triplet, NO doublet) | — | `regressions/semiempirical/dftb2-spin-analytic-stress` | dftb+ formula + strain-FD (`perf_exempt`) |
| AM1 H₂O ΔHf / E / charges | — | `regressions/semiempirical/am1-water-mopac-energy` | **OpenMOPAC** transcript (Dewar 1985) (`perf_exempt`) |
| RM1 H₂O ΔHf at the RM1 optimum | — | `regressions/semiempirical/rm1-water-mopac-energy` | **OpenMOPAC** PARAM survey + Scientia Plena 2016 (`perf_exempt`) |

### TB / SE capability wave 8 (semiempirical + tight-binding)

Offline OpenMOPAC / DFTB+ / xtb pins only — see
[tb-method-handoff](../developer/tb-method-handoff.md) and `tools/tb_oracle/`.
Family-locked oracles: NDDO → OpenMOPAC; DFTB* → DFTB+; GFN-xTB → xtb.
Harness wave 8 (2026-08-25) closed GFN1 organic energy, GFN0 analytic \(F\),
and GFN2 quadrupole AES. Full index: [methods/README](methods/README.md).

| Method | Page | Regression | Oracle |
|--------|------|------------|--------|
| GFN0-xTB analytic nuclear forces (off-eq H₂O) | [gfn0-xtb-forces](methods/gfn0-xtb-forces.md) | `regressions/semiempirical/gfn0-xtb-forces` | xtb 6.7.1 (ratio **≈ 0.092**) |
| GFN1-xTB organic energy (CH₄, H/C/N/O) | [gfn1-xtb-organic](methods/gfn1-xtb-organic.md) | `regressions/semiempirical/gfn1-xtb-organic` | xtb 6.7.1 (ratio **0.5**) |
| GFN2-xTB charge–quadrupole AES + \(q_{\mathrm{kernel}}\) AXC (H₂O) | [gfn2-quadrupole-aes](methods/gfn2-quadrupole-aes.md) | `regressions/semiempirical/gfn2-quadrupole-aes` | xtb 6.7.1 (`perf_exempt`) |

### TB / SE capability wave 9 (semiempirical + tight-binding)

Offline OpenMOPAC / DFTB+ / xtb pins only — see
[tb-method-handoff](../developer/tb-method-handoff.md) and `tools/tb_oracle/`.
Family-locked oracles: NDDO → OpenMOPAC; DFTB* → DFTB+; GFN-xTB → xtb.
Harness wave 9 (2026-08-25) closed GFN1/GFN2 analytic \(F\) and GFN0 FD
Hessian (`-DATV_BUILD_VIB=ON` exception). Family GFN1/2 Hessian is
wave 10 [`gfn-xtb-analytic-hessian`](methods/gfn-xtb-analytic-hessian.md).
GFN2 AES remains post-SCC (angular \(F\) hole).
Full index: [methods/README](methods/README.md).

| Method | Page | Regression | Oracle |
|--------|------|------------|--------|
| GFN1-xTB analytic nuclear forces (off-eq H₂O) | [gfn1-xtb-forces](methods/gfn1-xtb-forces.md) | `regressions/semiempirical/gfn1-xtb-forces` | xtb 6.7.1 (ratio **≈ 0.16**) |
| GFN2-xTB analytic nuclear forces (off-eq H₂O) | [gfn2-xtb-forces](methods/gfn2-xtb-forces.md) | `regressions/semiempirical/gfn2-xtb-forces` | xtb 6.7.1 (**asserted ≤ 10**, xtb wall 0.018 s) |
| GFN0-xTB FD Hessian of analytic \(G\) (off-eq H₂O) | [gfn0-xtb-analytic-hessian](methods/gfn0-xtb-analytic-hessian.md) | `regressions/semiempirical/gfn0-xtb-analytic-hessian` | xtb 6.7.1 (ratio **≈ 0.06**) |

### TB / SE capability wave 10 (semiempirical + tight-binding)

Offline OpenMOPAC / DFTB+ / xtb pins only — see
[tb-method-handoff](../developer/tb-method-handoff.md) and `tools/tb_oracle/`.
Family-locked oracles: NDDO → OpenMOPAC; DFTB* → DFTB+; GFN-xTB / IPEA → xtb.
Harness wave 10 (2026-08-25) closed IPEA-xTB energy and the family GFN1/2
FD Hessian (`-DATV_BUILD_VIB=ON` exception). GFN2 AES remains post-SCC
(angular \(F\) hole).
Full index: [methods/README](methods/README.md).

| Method | Page | Regression | Oracle |
|--------|------|------------|--------|
| IPEA-xTB organic energy (CH₄, extra \((n+1)s\)) | [ipea-xtb-small](methods/ipea-xtb-small.md) | `regressions/semiempirical/ipea-xtb-small` | xtb 6.7.1 (ratio **2.0**) |
| GFN1/GFN2-xTB FD Hessian of analytic \(G\) (off-eq H₂O) | [gfn-xtb-analytic-hessian](methods/gfn-xtb-analytic-hessian.md) | `regressions/semiempirical/gfn-xtb-analytic-hessian` | xtb 6.7.1 (ratio **≈ 0.11**) |

### MetalWalls capability wave 1 (classical electrochem + LAMMPS-overlap MD)

Offline LAMMPS / MetalWalls / literature pins only — see
[metalwalls-method-handoff](../developer/metalwalls-method-handoff.md) and
`tools/metalwalls_oracle/`. Narrative: [Ch. 10 Classical MD](ch10-classical-md.md).
Oracle priority: **LAMMPS** when the feature overlaps; MetalWalls for MW-only.

| Method | Page | Regression | Oracle |
|--------|------|------------|--------|
| Const-pot BO parallel plate (madelung, E1) | [const-pot-bo-parallel-plate](methods/const-pot-bo-parallel-plate.md) | `regressions/md/const-pot-bo-parallel-plate` | lammps (`perf_exempt`) |
| Constant-charge electrode control (conq) | [const-charge-electrode](methods/const-charge-electrode.md) | `regressions/md/const-charge-electrode` | lammps (`perf_exempt`) |
| Freeze electrode atoms (`setforce 0`) | [freeze-electrode-atoms](methods/freeze-electrode-atoms.md) | `regressions/md/freeze-electrode-atoms` | lammps (ratio **3.98**) |
| Electrode total-charge observer | [const-pot-charge-observer](methods/const-pot-charge-observer.md) | `regressions/md/const-pot-charge-observer` | lammps (`perf_exempt`) |
| Charging work \(-\sum_i \Psi_i Q_i\) | [charging-work-energy](methods/charging-work-energy.md) | `regressions/md/charging-work-energy` | literature (`perf_exempt`) |
| Const-pot MD NVT short trajectory | [const-pot-md-nvt-short](methods/const-pot-md-nvt-short.md) | `regressions/md/const-pot-md-nvt-short` | metalwalls (`perf_exempt`) |
| Gaussian electrode charge kernels (E2) | [gaussian-electrode-charge](methods/gaussian-electrode-charge.md) | `regressions/md/gaussian-electrode-charge` | literature (`perf_exempt`) |
| Slab 2D Ewald foothold (Yeh–Berkowitz) | [slab-2d-ewald-foothold](methods/slab-2d-ewald-foothold.md) | `regressions/md/slab-2d-ewald-foothold` | lammps (ratio **6.49**) |
| External uniform E-field MD | [external-efield-md](methods/external-efield-md.md) | `regressions/md/external-efield-md` | lammps (ratio **0.60**) |
| Buckingham pair (LAMMPS `buck`) | [buckingham-pair-lammps](methods/buckingham-pair-lammps.md) | `regressions/forcefield/buckingham-pair-lammps` | lammps (`perf_exempt`) |
| Four-site TIP4P foothold | [four-site-tip4p-foothold](methods/four-site-tip4p-foothold.md) | `regressions/md/four-site-tip4p-foothold` | lammps (ratio **0.338**) |

### MetalWalls capability wave 2 (classical electrochem + LAMMPS-overlap MD)

Offline LAMMPS / MetalWalls / literature pins only — see
[metalwalls-method-handoff](../developer/metalwalls-method-handoff.md) and
`tools/metalwalls_oracle/`. Narrative: [Ch. 10 Classical MD](ch10-classical-md.md).
Oracle priority: **LAMMPS** when the feature overlaps; MetalWalls for MW-only.
Full index: [methods/README](methods/README.md).

| Method | Page | Regression | Oracle |
|--------|------|------------|--------|
| Const-pot CG large multi-site electrode (E2) | [const-pot-cg-large-electrode](methods/const-pot-cg-large-electrode.md) | `regressions/md/const-pot-cg-large-electrode` | metalwalls (`perf_exempt`) |
| Electrode charge predictor (Kolafa ASPC, E2) | [electrode-charge-predictor](methods/electrode-charge-predictor.md) | `regressions/md/electrode-charge-predictor` | metalwalls (`perf_exempt`) |
| Finite-field + single electrode (Dufils 3D, E3) | [finite-field-single-electrode](methods/finite-field-single-electrode.md) | `regressions/md/finite-field-single-electrode` | metalwalls (`perf_exempt`) |
| Bulk PIM self-consistent induced dipoles (E4) | [pim-induced-dipole-bulk](methods/pim-induced-dipole-bulk.md) | `regressions/md/pim-induced-dipole-bulk` | metalwalls (`perf_exempt`) |
| Rigid piston electrode constant \(P_z\) (E6) | [piston-electrode-pz](methods/piston-electrode-pz.md) | `regressions/md/piston-electrode-pz` | metalwalls (`perf_exempt`) |
| Thomas–Fermi finite metallicity electrode (E7) | [thomas-fermi-electrode](methods/thomas-fermi-electrode.md) | `regressions/md/thomas-fermi-electrode` | metalwalls (`perf_exempt`) |
| Born–Mayer–Huggins pair foothold (E8) | [born-repulsion-foothold](methods/born-repulsion-foothold.md) | `regressions/forcefield/born-repulsion-foothold` | lammps (`perf_exempt`) |
| Fumi–Tosi multi-type Born + Coulomb NaCl (E8) | [fumi-tosi-nacl](methods/fumi-tosi-nacl.md) | `regressions/forcefield/fumi-tosi-nacl` | lammps (`perf_exempt`) |
| Steele 10-4-3 wall (Ar single-point, E8) | [steele-wall-argon](methods/steele-wall-argon.md) | `regressions/forcefield/steele-wall-argon` | lammps (`perf_exempt`) |
| Capacitance from electrode charge fluctuations (E10) | [capacitance-charge-fluctuation](methods/capacitance-charge-fluctuation.md) | `regressions/md/capacitance-charge-fluctuation` | literature (`perf_exempt`) |

### MetalWalls capability wave 3 (classical electrochem + LAMMPS-overlap MD)

Offline LAMMPS / MetalWalls / literature pins only — see
[metalwalls-method-handoff](../developer/metalwalls-method-handoff.md) and
`tools/metalwalls_oracle/`. Narrative: [Ch. 10 Classical MD](ch10-classical-md.md).
Oracle priority: **LAMMPS** when the feature overlaps; MetalWalls for MW-only.
Full index: [methods/README](methods/README.md).

| Method | Page | Regression | Oracle |
|--------|------|------------|--------|
| Constant-force ion pull (`fix addforce`) | [constant-force-pull-ion](methods/constant-force-pull-ion.md) | `regressions/md/constant-force-pull-ion` | lammps (ratio **0.12**) |
| Oscillating external E-field MD (E3) | [oscillating-efield-md](methods/oscillating-efield-md.md) | `regressions/md/oscillating-efield-md` | lammps (ratio **2.12**) |
| Slab \(\Psi(z)\) planar Poisson observer (E10) | [potential-profile-observer](methods/potential-profile-observer.md) | `regressions/md/potential-profile-observer` | literature (`perf_exempt`) |
| Surface-tension / NPγT coupling | [surface-tension-coupling](methods/surface-tension-coupling.md) | `regressions/md/surface-tension-coupling` | lammps (ratio **2.88**) |
| Multi-topology virtual sites TIP4P+TIP5P (E9) | [virtual-site-multi-topology](methods/virtual-site-multi-topology.md) | `regressions/md/virtual-site-multi-topology` | lammps (ratio **0.084**) |
| Coupled PIM dipoles ⊕ electrode \(Q\) (E5) | [dipole-electrode-joint](methods/dipole-electrode-joint.md) | `regressions/md/dipole-electrode-joint` | metalwalls (`perf_exempt`) |

### CP2K capability wave 1 (GPW / PW / periodic)

Offline CP2K pins only — see [cp2k-method-handoff](../developer/cp2k-method-handoff.md)
and `tools/cp2k_oracle/`. Narrative chapters: [Ch. 8 PW](ch08-plane-wave.md),
[Ch. 9 GPW](ch09-gpw.md).

| Method | Page | Regression |
|--------|------|------------|
| GPW Γ-RKS H₂O PBE (MT) | [gpw-water-pbe-mt](methods/gpw-water-pbe-mt.md) | `regressions/gpw/gpw-water-pbe-mt` |
| GPW analytic forces H₂O | [gpw-forces-water](methods/gpw-forces-water.md) | `regressions/gpw/gpw-forces-water` |
| Γ-UKS GPW O₂ | [gpw-uks-o2-gamma](methods/gpw-uks-o2-gamma.md) | `regressions/gpw/gpw-uks-o2-gamma` |
| GPW PBE-D3(BJ) ice dimer | [gpw-pbe-d3bj-ice](methods/gpw-pbe-d3bj-ice.md) | `regressions/gpw/gpw-pbe-d3bj-ice` |
| GPW PBE-D3 + ATM (C9) isolated dimer | [gpw-d3-atm](methods/gpw-d3-atm.md) | `regressions/gpw/gpw-d3-atm` |
| GPW PBE-D3 analytic ionic forces | [gpw-d3-forces](methods/gpw-d3-forces.md) | `regressions/gpw/gpw-d3-forces` |
| GPW PBE-D3(BJ) analytic ionic forces | [gpw-d3bj-forces](methods/gpw-d3bj-forces.md) | `regressions/gpw/gpw-d3bj-forces` |
| GPW Γ cell stress with periodic MBD@rsSCS | [gpw-mbd-stress](methods/gpw-mbd-stress.md) | `regressions/gpw/gpw-mbd-stress` |
| Multi-k MLWF on GPW (Marzari–Vanderbilt) | [periodic-wannier-gpw](methods/periodic-wannier-gpw.md) | `regressions/gpw/periodic-wannier-gpw` |
| PW LDA Si (k-mesh) | [pw-lda-si](methods/pw-lda-si.md) | `regressions/pw/pw-lda-si` |
| PW true PBE GGA Si | [pw-pbe-gga-si](methods/pw-pbe-gga-si.md) | `regressions/pw/pw-pbe-gga-si` |
| Metallic Fermi–Dirac Al | [pw-smearing-al](methods/pw-smearing-al.md) | `regressions/pw/pw-smearing-al` |
| PW collinear spin O₂ | [pw-spin-polarized-o2](methods/pw-spin-polarized-o2.md) | `regressions/pw/pw-spin-polarized-o2` |
| PW BOMD H (NVE foothold) | [pw-bomd-h](methods/pw-bomd-h.md) | `regressions/pw/pw-bomd-h` |
| OT SCF water (GPW PBE) | [ot-minimizer-water](methods/ot-minimizer-water.md) | `regressions/gpw/ot-minimizer-water` |

### CP2K capability wave 2 (GPW / PW / periodic)

Offline CP2K pins only — see [cp2k-method-handoff](../developer/cp2k-method-handoff.md)
and `tools/cp2k_oracle/`. Narrative chapters: [Ch. 8 PW](ch08-plane-wave.md),
[Ch. 9 GPW](ch09-gpw.md). Full index: [methods/README](methods/README.md).

| Method | Page | Regression |
|--------|------|------------|
| GPW cell stress strain-FD H₂O (NUMERICAL pin) | [gpw-stress-water](methods/gpw-stress-water.md) | `regressions/gpw/gpw-stress-water` |
| GPW analytic fixed-\(P\) stress H₂O | [gpw-analytic-stress](methods/gpw-analytic-stress.md) | `regressions/gpw/gpw-analytic-stress` |
| GPW BOMD H₂O NVE energy-drift foothold | [gpw-bomd-water](methods/gpw-bomd-water.md) | `regressions/gpw/gpw-bomd-water` |
| Car–Parrinello MD H₂O NVE (fictitious e⁻ DOF) | [cpmd-water](methods/cpmd-water.md) | `regressions/dynamics/cpmd-water` |
| Car–Parrinello MD + electronic RATTLE H₂O NVE | [cpmd-rattle-water](methods/cpmd-rattle-water.md) | `regressions/dynamics/cpmd-rattle-water` |
| AO-basis CPMD on the SCF methods (RHF / RKS / UHF) | [cpmd-rhf-h2](methods/cpmd-rhf-h2.md) | `regressions/dynamics/cpmd-{rhf-h2,rks-water,uhf-o2}` |
| XL-BOMD on the Gaussian SCF methods (RHF H₂O NVE) | [xl-bomd-rhf-water](methods/xl-bomd-rhf-water.md) | `regressions/dynamics/xl-bomd-rhf-water` |
| Multi-k second-gen CP (ASPC on \(P_k\)) H₂ NVE | [second-gen-cpmd-kpoints](methods/second-gen-cpmd-kpoints.md) | `regressions/dynamics/second-gen-cpmd-kpoints` |
| Multi-k second-gen CP + OT corrector H₂ NVE | [second-gen-cpmd-ot-kpoints](methods/second-gen-cpmd-ot-kpoints.md) | `regressions/dynamics/second-gen-cpmd-ot-kpoints` |
| Multi-k UKS second-gen CP + OT corrector H₂ NVE | [second-gen-cpmd-ot-uks-kpoints](methods/second-gen-cpmd-ot-uks-kpoints.md) | `regressions/dynamics/second-gen-cpmd-ot-uks-kpoints` |
| Multi-k Car–Parrinello MD on Bloch \(C_k\) H₂ NVE | [cpmd-kpoints](methods/cpmd-kpoints.md) | `regressions/dynamics/cpmd-kpoints` |
| PW Nielsen–Martin analytic stress Si | [pw-analytic-stress-si](methods/pw-analytic-stress-si.md) | `regressions/pw/pw-analytic-stress-si` |
| FD supercell phonons diamond Si (Γ) | [fd-phonon-supercell](methods/fd-phonon-supercell.md) | `regressions/pw/fd-phonon-supercell` |
| Dudarev DFT+U NiO mol (Γ-UKS GPW) | [gpw-dftu-nio](methods/gpw-dftu-nio.md) | `regressions/gpw/gpw-dftu-nio` |
| PW bands + DOS bulk Si | [bands-dos-si](methods/bands-dos-si.md) | `regressions/pw/bands-dos-si` |
| PW metallic-occupation bands Si | [bands-metallic-si](methods/bands-metallic-si.md) | `regressions/pw/bands-metallic-si` |
| PW k-point collinear UKS O₂ | [pw-uks-kmesh-fe](methods/pw-uks-kmesh-fe.md) | `regressions/pw/pw-uks-kmesh-fe` |
| GPW Γ-RKS PBEsol Si₂ (MT) | [gpw-pbesol-si](methods/gpw-pbesol-si.md) | `regressions/gpw/gpw-pbesol-si` |
| GPW PBE-D4 periodic water dimer | [gpw-d4-water](methods/gpw-d4-water.md) | `regressions/gpw/gpw-d4-water` |

### CP2K industrial infra (MPI k-point foothold)

| Method | Page | Regression |
|--------|------|------------|
| MPI GPW k-point RR (n=1 vs n=4) | [mpi-gpw-scaling-4rank](methods/mpi-gpw-scaling-4rank.md) | `regressions/gpw/mpi-gpw-scaling-4rank` (`perf_exempt`) |

### Restored capability pages (indexed 2026-07-28)

Pages written by the Tutorial phase of earlier `cp2k-method-loop` waves (plus
the AO AIMD spec chain `esmm-cpmd-04/05`) whose index rows never landed — each
wave stashed its tutorial pages before merging. Recovered 2026-07-28 with code
samples refreshed to the current public API. Every slug maps 1:1 to a
regression directory in the tree; fair walls live in each page's Performance
section. Full index: [methods/README](methods/README.md). Grouped by regression
domain, alphabetical by slug.

| Method | Page | Regression |
|--------|------|------------|
| GPW Γ-RKS BLYP, isolated H₂O (MT) | [gpw-blyp-water](methods/gpw-blyp-water.md) | `regressions/gpw/gpw-blyp-water` |
| GPW PBE-D3 (zero damping) analytic cell stress | [gpw-d3-stress](methods/gpw-d3-stress.md) | `regressions/gpw/gpw-d3-stress` |
| GPW DFT-D4 three-body ATM (\(s_9=1\)) analytic cell stress | [gpw-d4-atm-stress](methods/gpw-d4-atm-stress.md) | `regressions/gpw/gpw-d4-atm-stress` |
| GPW PBE-D3(BJ) analytic cell stress | [gpw-d3bj-stress](methods/gpw-d3bj-stress.md) | `regressions/gpw/gpw-d3bj-stress` |
| GPW PBE-D4 analytic ionic forces | [gpw-d4-forces](methods/gpw-d4-forces.md) | `regressions/gpw/gpw-d4-forces` |
| GPW PBE-D4 analytic cell stress | [gpw-d4-stress](methods/gpw-d4-stress.md) | `regressions/gpw/gpw-d4-stress` |
| Multi-k DFT+U analytic cell stress (H₂ UKS) | [gpw-dftu-kpoint-stress](methods/gpw-dftu-kpoint-stress.md) | `regressions/gpw/gpw-dftu-kpoint-stress` |
| Multi-k Dudarev DFT+U (GPW UKS) | [gpw-dftu-kpoints](methods/gpw-dftu-kpoints.md) | `regressions/gpw/gpw-dftu-kpoints` |
| GPW analytic cell stress + DFT+U (NiO) | [gpw-dftu-stress](methods/gpw-dftu-stress.md) | `regressions/gpw/gpw-dftu-stress` |
| GPW multi-k Hirshfeld volumes | [gpw-hirshfeld-kpoints](methods/gpw-hirshfeld-kpoints.md) | `regressions/gpw/gpw-hirshfeld-kpoints` |
| Γ-UKS Hirshfeld-scaled MBD@rsSCS (O₂) | [gpw-hirshfeld-uks](methods/gpw-hirshfeld-uks.md) | `regressions/gpw/gpw-hirshfeld-uks` |
| Multi-k UKS Hirshfeld-scaled MBD@rsSCS (O₂) | [gpw-hirshfeld-uks-kpoints](methods/gpw-hirshfeld-uks-kpoints.md) | `regressions/gpw/gpw-hirshfeld-uks-kpoints` |
| GPW analytic forces at k-points (H₂) | [gpw-kpoint-forces](methods/gpw-kpoint-forces.md) | `regressions/gpw/gpw-kpoint-forces` |
| GPW analytic cell stress at k-points (Si) | [gpw-kpoint-stress](methods/gpw-kpoint-stress.md) | `regressions/gpw/gpw-kpoint-stress` |
| Multi-k UKS GPW analytic forces | [gpw-kpoint-uks-forces](methods/gpw-kpoint-uks-forces.md) | `regressions/gpw/gpw-kpoint-uks-forces` |
| Multi-k UKS GPW analytic cell stress | [gpw-kpoint-uks-stress](methods/gpw-kpoint-uks-stress.md) | `regressions/gpw/gpw-kpoint-uks-stress` |
| GPW collinear multi-k UKS (H₂ + O₂ twin) | [gpw-kpoints-uks](methods/gpw-kpoints-uks.md) | `regressions/gpw/gpw-kpoints-uks` |
| GPW MBD@rsSCS ionic forces | [gpw-mbd-forces](methods/gpw-mbd-forces.md) | `regressions/gpw/gpw-mbd-forces` |
| GPW PBE-MBD@rsSCS + Hirshfeld volumes | [gpw-mbd-hirshfeld](methods/gpw-mbd-hirshfeld.md) | `regressions/gpw/gpw-mbd-hirshfeld` |
| GPW PBE-MBD@rsSCS, free-atom water dimer | [gpw-mbd-ice](methods/gpw-mbd-ice.md) | `regressions/gpw/gpw-mbd-ice` |
| GPW multi-k analytic forces + MBD@rsSCS | [gpw-mbd-kpoint-forces](methods/gpw-mbd-kpoint-forces.md) | `regressions/gpw/gpw-mbd-kpoint-forces` |
| GPW PBE-MBD@rsSCS, periodic Ewald + k-mesh | [gpw-mbd-periodic](methods/gpw-mbd-periodic.md) | `regressions/gpw/gpw-mbd-periodic` |
| Metallic FD smearing (BCC Na, GPW multi-k) | [gpw-metallic-smearing](methods/gpw-metallic-smearing.md) | `regressions/gpw/gpw-metallic-smearing` |
| GPW optB88-vdW (periodic water dimer) | [gpw-optb88-vdw](methods/gpw-optb88-vdw.md) | `regressions/gpw/gpw-optb88-vdw` |
| GPW Γ-RKS revPBE (Si₂ fragment, MT) | [gpw-revpbe-si](methods/gpw-revpbe-si.md) | `regressions/gpw/gpw-revpbe-si` |
| GPW Γ-RKS RPBE, isolated H₂O (MT) | [gpw-rpbe-water](methods/gpw-rpbe-water.md) | `regressions/gpw/gpw-rpbe-water` |
| GPW PBE + rVV10 (periodic water dimer) | [gpw-rvv10-ice](methods/gpw-rvv10-ice.md) | `regressions/gpw/gpw-rvv10-ice` |
| GPW PBE-TS (periodic water dimer) | [gpw-ts-ice](methods/gpw-ts-ice.md) | `regressions/gpw/gpw-ts-ice` |
| Multi-k free-atom TS analytic ionic forces (H₂) | [gpw-ts-kpoint-forces](methods/gpw-ts-kpoint-forces.md) | `regressions/gpw/gpw-ts-kpoint-forces` |
| Γ-UKS GPW BLYP (O₂) | [gpw-uks-blyp](methods/gpw-uks-blyp.md) | `regressions/gpw/gpw-uks-blyp` |
| Γ-UKS GPW analytic forces (O₂ / PBE) | [gpw-uks-forces](methods/gpw-uks-forces.md) | `regressions/gpw/gpw-uks-forces` |
| Γ-UKS GPW multigrid (O₂ / PADE) | [gpw-uks-multigrid](methods/gpw-uks-multigrid.md) | `regressions/gpw/gpw-uks-multigrid` |
| Γ-UKS GPW PBE (O₂) | [gpw-uks-pbe](methods/gpw-uks-pbe.md) | `regressions/gpw/gpw-uks-pbe` |
| Γ-UKS GPW revPBE (O₂) | [gpw-uks-revpbe](methods/gpw-uks-revpbe.md) | `regressions/gpw/gpw-uks-revpbe` |
| Γ-UKS GPW RPBE (O₂) | [gpw-uks-rpbe](methods/gpw-uks-rpbe.md) | `regressions/gpw/gpw-uks-rpbe` |
| Collinear UKS + FD smearing (BCC Na) | [gpw-uks-smearing](methods/gpw-uks-smearing.md) | `regressions/gpw/gpw-uks-smearing` |
| Γ-UKS GPW analytic cell stress (O₂ / PBE) | [gpw-uks-stress](methods/gpw-uks-stress.md) | `regressions/gpw/gpw-uks-stress` |
| GPW `CELL_OPT` MOLOPT re-twin (water chain) | [gpw-variable-cell-molopt](methods/gpw-variable-cell-molopt.md) | `regressions/gpw/gpw-variable-cell-molopt` |
| GPW PBE + vdW-DF (DRSLL) | [gpw-vdw-df-ice](methods/gpw-vdw-df-ice.md) | `regressions/gpw/gpw-vdw-df-ice` |
| GPW PBE + vdW-DF2 (LMKLL) | [gpw-vdw-df2-ice](methods/gpw-vdw-df2-ice.md) | `regressions/gpw/gpw-vdw-df2-ice` |
| GPW nonlocal vdW: SC \(V_{\mathrm{nl}}\) + forces | [gpw-vdw-scf-potential](methods/gpw-vdw-scf-potential.md) | `regressions/gpw/gpw-vdw-scf-potential` |
| GPW nonlocal vdW on the REL_CUTOFF multigrid | [gpw-vdw-multigrid](methods/gpw-vdw-multigrid.md) | `regressions/gpw/gpw-vdw-multigrid` |
| GPW free-atom XDM (periodic water dimer) | [gpw-xdm-ice](methods/gpw-xdm-ice.md) | `regressions/gpw/gpw-xdm-ice` |
| OT + Dudarev DFT+U, Γ-UKS GPW (NiO) | [ot-dftu](methods/ot-dftu.md) | `regressions/gpw/ot-dftu` |
| Multi-k OT + Dudarev DFT+U (H₂) | [ot-dftu-kpoints](methods/ot-dftu-kpoints.md) | `regressions/gpw/ot-dftu-kpoints` |
| Multi-k OT + DFT+U + FD smearing (H₂) | [ot-dftu-smearing](methods/ot-dftu-smearing.md) | `regressions/gpw/ot-dftu-smearing` |
| Multi-k Orbital Transformation SCF (H₂) | [ot-kpoints](methods/ot-kpoints.md) | `regressions/gpw/ot-kpoints` |
| OT + metallic FD smearing (BCC Na) | [ot-smearing](methods/ot-smearing.md) | `regressions/gpw/ot-smearing` |
| Multi-k UKS OT + Dudarev DFT+U (H₂) | [ot-uks-dftu](methods/ot-uks-dftu.md) | `regressions/gpw/ot-uks-dftu` |
| Multi-k UKS OT + DFT+U + smearing (H₂) | [ot-uks-dftu-smearing](methods/ot-uks-dftu-smearing.md) | `regressions/gpw/ot-uks-dftu-smearing` |
| Multi-k collinear UKS OT (high-spin H₂) | [ot-uks-kpoints](methods/ot-uks-kpoints.md) | `regressions/gpw/ot-uks-kpoints` |
| OT orbital minimizer for Γ-UKS (O₂) | [ot-uks-o2](methods/ot-uks-o2.md) | `regressions/gpw/ot-uks-o2` |
| UKS OT + metallic FD smearing (BCC Na) | [ot-uks-smearing](methods/ot-uks-smearing.md) | `regressions/gpw/ot-uks-smearing` |
| Periodic finite-\(q\) inter-\(k\) G0W0 (Si) | [periodic-gw-finite-q](methods/periodic-gw-finite-q.md) | `regressions/gpw/periodic-gw-finite-q` |
| Multi-k spectral G0W0 (diamond Si, bct) | [periodic-gw-kpoints](methods/periodic-gw-kpoints.md) | `regressions/gpw/periodic-gw-kpoints` |
| Periodic Γ-point spectral G0W0 (sc-Si₂) | [periodic-gw-si](methods/periodic-gw-si.md) | `regressions/gpw/periodic-gw-si` |
| Multi-k RI-GPW MP2 (diamond Si, bct) | [periodic-mp2-kpoints](methods/periodic-mp2-kpoints.md) | `regressions/gpw/periodic-mp2-kpoints` |
| Periodic Γ-point GPW MP2, canonical (sc-Si₂) | [periodic-mp2-si](methods/periodic-mp2-si.md) | `regressions/gpw/periodic-mp2-si` |
| Periodic Γ-point RI-GPW MP2 (sc-Si₂) | [periodic-ri-mp2-si](methods/periodic-ri-mp2-si.md) | `regressions/gpw/periodic-ri-mp2-si` |
| Finite-\(q\) inter-\(k\) RI-GPW dRPA (Si) | [periodic-rpa-finite-q](methods/periodic-rpa-finite-q.md) | `regressions/gpw/periodic-rpa-finite-q` |
| Multi-k RI-GPW direct RPA (Si₄, bct) | [periodic-rpa-kpoints](methods/periodic-rpa-kpoints.md) | `regressions/gpw/periodic-rpa-kpoints` |
| Periodic Γ-point GPW direct RPA (sc-Si₂) | [periodic-rpa-si](methods/periodic-rpa-si.md) | `regressions/gpw/periodic-rpa-si` |
| Multi-k unrestricted spectral G0W0 (UGW) | [periodic-ugw-kpoints](methods/periodic-ugw-kpoints.md) | `regressions/gpw/periodic-ugw-kpoints` |
| Periodic Γ-point UGW@UKS (sc-Si₂) | [periodic-ugw-si](methods/periodic-ugw-si.md) | `regressions/gpw/periodic-ugw-si` |
| Multi-k unrestricted GPW MP2 (UMP2) | [periodic-ump2-kpoints](methods/periodic-ump2-kpoints.md) | `regressions/gpw/periodic-ump2-kpoints` |
| Periodic Γ-point GPW UMP2 (sc-Si₂) | [periodic-ump2-si](methods/periodic-ump2-si.md) | `regressions/gpw/periodic-ump2-si` |
| Multi-k unrestricted direct RPA (URPA@UKS) | [periodic-urpa-kpoints](methods/periodic-urpa-kpoints.md) | `regressions/gpw/periodic-urpa-kpoints` |
| Periodic Γ-point URPA@UKS (sc-Si₂) | [periodic-urpa-si](methods/periodic-urpa-si.md) | `regressions/gpw/periodic-urpa-si` |
| PW collinear spin band structure (bulk Si) | [bands-spin-si](methods/bands-spin-si.md) | `regressions/pw/bands-spin-si` |
| Collinear-spin Berry-phase polarization (Si) | [berry-spin-si](methods/berry-spin-si.md) | `regressions/pw/berry-spin-si` |
| Metallic-occupation DOS (bulk Si) | [dos-metallic-si](methods/dos-metallic-si.md) | `regressions/pw/dos-metallic-si` |
| MPI plane-wave k-mesh scaling (4 ranks) | [mpi-pw-scaling-4rank](methods/mpi-pw-scaling-4rank.md) | `regressions/pw/mpi-pw-scaling-4rank` |
| Periodic multi-k MLWF (bulk Si) | [periodic-wannier-si](methods/periodic-wannier-si.md) | `regressions/pw/periodic-wannier-si` |
| Collinear spin-polarized multi-k MLWF (Si, UKS) | [periodic-wannier-uks](methods/periodic-wannier-uks.md) | `regressions/pw/periodic-wannier-uks` |
| PW collinear spin-GGA PBE (O₂ triplet, Γ) | [pw-spin-gga](methods/pw-spin-gga.md) | `regressions/pw/pw-spin-gga` |
| PW collinear UKS + metallic smearing (BCC Fe) | [pw-uks-fe-smearing](methods/pw-uks-fe-smearing.md) | `regressions/pw/pw-uks-fe-smearing` |
| PW isotropic variable-cell relax (diamond Si) | [pw-variable-cell-si](methods/pw-variable-cell-si.md) | `regressions/pw/pw-variable-cell-si` |
| Finite-\(q\) DFPT phonons at X (diamond Si) | [dfpt-finite-q-si](methods/dfpt-finite-q-si.md) | `regressions/response/dfpt-finite-q-si` |
| Finite-\(q\) DFPT + metallic smearing (Si at X) | [dfpt-finite-q-smearing](methods/dfpt-finite-q-smearing.md) | `regressions/response/dfpt-finite-q-smearing` |
| Finite-\(q\) DFPT phonons at X, pure GGA kernel (Si) | [dfpt-gga-finite-q-si](methods/dfpt-gga-finite-q-si.md) | `regressions/response/dfpt-gga-finite-q-si` |
| DFPT phonons at Γ, pure GGA kernel (Si) | [dfpt-gga-si](methods/dfpt-gga-si.md) | `regressions/response/dfpt-gga-si` |
| Finite-\(q\) DFPT at X, joint spin × GGA (Si) | [dfpt-gga-spin-finite-q-si](methods/dfpt-gga-spin-finite-q-si.md) | `regressions/response/dfpt-gga-spin-finite-q-si` |
| Finite-\(q\) DFPT at X, joint spin × GGA × smear (Si) | [dfpt-gga-spin-finite-q-smearing](methods/dfpt-gga-spin-finite-q-smearing.md) | `regressions/response/dfpt-gga-spin-finite-q-smearing` |
| DFPT phonons at Γ, joint spin × GGA (Si) | [dfpt-gga-spin-si](methods/dfpt-gga-spin-si.md) | `regressions/response/dfpt-gga-spin-si` |
| DFPT phonons at Γ, joint spin × GGA × smear (Si) | [dfpt-gga-spin-smearing](methods/dfpt-gga-spin-smearing.md) | `regressions/response/dfpt-gga-spin-smearing` |
| Incommensurate-\(q\) DFPT phonons (Si) | [dfpt-incommensurate-si](methods/dfpt-incommensurate-si.md) | `regressions/response/dfpt-incommensurate-si` |
| DFPT phonons + metallic smearing (Si) | [dfpt-metallic-si](methods/dfpt-metallic-si.md) | `regressions/response/dfpt-metallic-si` |
| DFPT phonons at Γ (diamond Si, PW LSDA) | [dfpt-phonon-si](methods/dfpt-phonon-si.md) | `regressions/response/dfpt-phonon-si` |
| Collinear-spin DFPT at incommensurate \(q\) (Si) | [dfpt-spin-incommensurate](methods/dfpt-spin-incommensurate.md) | `regressions/response/dfpt-spin-incommensurate` |
| DFPT phonons at Γ, collinear spin (Si) | [dfpt-spin-si](methods/dfpt-spin-si.md) | `regressions/response/dfpt-spin-si` |
| DFPT: collinear spin + metallic smearing (Si) | [dfpt-spin-smearing](methods/dfpt-spin-smearing.md) | `regressions/response/dfpt-spin-smearing` |
| Phonon dispersion \(\omega(q)\) (diamond Si) | [phonon-dispersion-si](methods/phonon-dispersion-si.md) | `regressions/response/phonon-dispersion-si` |
| Car–Parrinello MD on Γ-UKS (O₂ triplet NVE) | [cpmd-uks](methods/cpmd-uks.md) | `regressions/dynamics/cpmd-uks` |
| Multi-k UKS CPMD on dual-spin Bloch \(C_k^\sigma\) (H₂ NVE) | [cpmd-uks-kpoints](methods/cpmd-uks-kpoints.md) | `regressions/dynamics/cpmd-uks-kpoints` |
| Multi-k UKS CPMD + electronic Nosé–Hoover per (k, σ) (H₂ NVE) | [cpmd-uks-thermostat-kpoints](methods/cpmd-uks-thermostat-kpoints.md) | `regressions/dynamics/cpmd-uks-thermostat-kpoints` |
| Second-generation CP with OT corrector (H₂O) | [second-gen-cpmd-ot](methods/second-gen-cpmd-ot.md) | `regressions/dynamics/second-gen-cpmd-ot` |
| Second-generation CP on Γ-UKS (O₂ triplet) | [second-gen-cpmd-uks](methods/second-gen-cpmd-uks.md) | `regressions/dynamics/second-gen-cpmd-uks` |
| Second-generation CP + OT on Γ-UKS (O₂) | [second-gen-cpmd-uks-ot](methods/second-gen-cpmd-uks-ot.md) | `regressions/dynamics/second-gen-cpmd-uks-ot` |
| Second-generation CP: ASPC + partial SCF (H₂O) | [second-gen-cpmd-water](methods/second-gen-cpmd-water.md) | `regressions/dynamics/second-gen-cpmd-water` |
| Multi-k XL-BOMD (H₂ NVE, GPW + k-mesh) | [xl-bomd-kpoints](methods/xl-bomd-kpoints.md) | `regressions/dynamics/xl-bomd-kpoints` |
| XL-BOMD on Γ-UKS (O₂ triplet NVE, GPW) | [xl-bomd-uks](methods/xl-bomd-uks.md) | `regressions/dynamics/xl-bomd-uks` |
| XL-BOMD, extended-Lagrangian BOMD (H₂O NVE) | [xl-bomd-water](methods/xl-bomd-water.md) | `regressions/dynamics/xl-bomd-water` |
| Maximally localized Wannier functions (Foster–Boys) | [wannier-centers](methods/wannier-centers.md) | `regressions/scf/wannier-centers` |

### PySCF campaign (molecular post-HF / response)

| Method | Page | Regression |
|--------|------|------------|
| UMP2 | [ump2-o2-sto3g](methods/ump2-o2-sto3g.md) | `regressions/scf/ump2-o2-sto3g` |
| Hybrid TDDFT-TDA (B3LYP) | [h2-b3lyp-hybrid-tda](methods/h2-b3lyp-hybrid-tda.md) | `…/h2-b3lyp-hybrid-tda` |
| Frozen-core RMP2 | [frozen-core-mp2-h2o](methods/frozen-core-mp2-h2o.md) | `…/frozen-core-mp2-h2o` |
| CIS(D) | [cisd-h2co-sto3g](methods/cisd-h2co-sto3g.md) | `…/cisd-h2co-sto3g` |
| BLYP GGA | [blyp-water-sto3g](methods/blyp-water-sto3g.md) | `…/blyp-water-sto3g` |
| GHF analytic gradient | [ghf-grad-h3-sto3g](methods/ghf-grad-h3-sto3g.md) | `…/ghf-grad-h3-sto3g` |
| C-PCM free-energy forces | [pcm-grad-h2o](methods/pcm-grad-h2o.md) | `…/pcm-grad-h2o` |
| SCS-MP2 | [scs-mp2-h2o-sto3g](methods/scs-mp2-h2o-sto3g.md) | `…/scs-mp2-h2o-sto3g` |
| Full LR-TDDFT (RKS) | [full-lr-tddft-h2-pbe](methods/full-lr-tddft-h2-pbe.md) | `…/full-lr-tddft-h2-pbe` |
| ROKS | [roks-o2-sto3g](methods/roks-o2-sto3g.md) | `…/roks-o2-sto3g` |
| RHF static polarizability | [static-polariz-h2o](methods/static-polariz-h2o.md) | `…/static-polariz-h2o` |
| IEF-PCM | [ief-pcm-h2o](methods/ief-pcm-h2o.md) | `…/ief-pcm-h2o` |
| ROMP2 | [romp2-o2-sto3g](methods/romp2-o2-sto3g.md) | `…/romp2-o2-sto3g` |
| ROKS analytic gradient | [roks-grad-o2-sto3g](methods/roks-grad-o2-sto3g.md) | `…/roks-grad-o2-sto3g` |
| Hybrid full-LR TDDFT (B3LYP RPA) | [h2-b3lyp-rpa-tddft](methods/h2-b3lyp-rpa-tddft.md) | `…/h2-b3lyp-rpa-tddft` |
| Spin-flip TDA (SF-CIS) | [spin-flip-tda-o2](methods/spin-flip-tda-o2.md) | `…/spin-flip-tda-o2` |
| Spin-flip full TDDFT (SF-RPA) | [sf-tddft-o2](methods/sf-tddft-o2.md) | `…/sf-tddft-o2` |
| ROHF nuclear Hessian | [rohf-hessian-o2](methods/rohf-hessian-o2.md) | `…/rohf-hessian-o2` |
| ADC(2) | [adc2-h2co-sto3g](methods/adc2-h2co-sto3g.md) | `…/adc2-h2co-sto3g` |
| ADC(2)-x | [adc2x-h2co-sto3g](methods/adc2x-h2co-sto3g.md) | `…/adc2x-h2co-sto3g` |
| IP-ADC(2) | [ip-adc2-h2o-sto3g](methods/ip-adc2-h2o-sto3g.md) | `…/ip-adc2-h2o-sto3g` |
| EA-ADC(2) | [ea-adc2-h2-sto3g](methods/ea-adc2-h2-sto3g.md) | `…/ea-adc2-h2-sto3g` |
| RMP3 | [mp3-h2-sto3g](methods/mp3-h2-sto3g.md) | `…/mp3-h2-sto3g` |
| RMP4(SDTQ) | [mp4-h2-sto3g](methods/mp4-h2-sto3g.md) | `…/mp4-h2-sto3g` |
| CCSD | [ccsd-h2-sto3g](methods/ccsd-h2-sto3g.md) | `…/ccsd-h2-sto3g` |
| CCSD nuclear gradient | [ccsd-grad-h2o-sto3g](methods/ccsd-grad-h2o-sto3g.md) | `…/ccsd-grad-h2o-sto3g` |
| CC2 | [cc2-h2-sto3g](methods/cc2-h2-sto3g.md) | `…/cc2-h2-sto3g` |
| CCSD(T) | [ccsd-t-h2o-sto3g](methods/ccsd-t-h2o-sto3g.md) | `…/ccsd-t-h2o-sto3g` |
| G0W0 | [g0w0-h2-sto3g](methods/g0w0-h2-sto3g.md) | `…/g0w0-h2-sto3g` |
| BSE@G0W0 | [bse-h2-sto3g](methods/bse-h2-sto3g.md) | `…/bse-h2-sto3g` |
| Direct RPA | [direct-rpa-h2-sto3g](methods/direct-rpa-h2-sto3g.md) | `…/direct-rpa-h2-sto3g` |
| One-shot GF2 (G0F2) | [gf2-h2-sto3g](methods/gf2-h2-sto3g.md) | `…/gf2-h2-sto3g` |
| B2PLYP double hybrid | [b2plyp-h2o](methods/b2plyp-h2o.md) | `…/b2plyp-h2o` |
| DFT-D4 (PBE params) | [dft-d4-h2o-pbe](methods/dft-d4-h2o-pbe.md) | `…/dft-d4-h2o-pbe` |
| CASCI | [casci-h2-sto3g](methods/casci-h2-sto3g.md) | `…/casci-h2-sto3g` |
| CASSCF | [casscf-h2-sto3g](methods/casscf-h2-sto3g.md) | `…/casscf-h2-sto3g` |
| SC-NEVPT2 | [nevpt2-h2-sto3g](methods/nevpt2-h2-sto3g.md) | `…/nevpt2-h2-sto3g` |
| ΔSCF (IMOM) | [delta-scf-n2](methods/delta-scf-n2.md) | `…/delta-scf-n2` |
| EOM-EE-CCSD | [eom-ee-ccsd](methods/eom-ee-ccsd.md) | `…/eom-ee-ccsd` |
| EOM-EA-CCSD | [eom-ea-ccsd](methods/eom-ea-ccsd.md) | `…/eom-ea-ccsd` |
| EOM-IP-CCSD | [eom-ip-ccsd](methods/eom-ip-ccsd.md) | `…/eom-ip-ccsd` |
| EOM-SF-CCSD | [eom-sf-ccsd-o2](methods/eom-sf-ccsd-o2.md) | `…/eom-sf-ccsd-o2` |
| EOM-CC2 | [eom-cc2-h2](methods/eom-cc2-h2.md) | `…/eom-cc2-h2` |
| RKS nuclear Hessian (FD of forces) | [rks-hessian-h2o-sto3g](methods/rks-hessian-h2o-sto3g.md) | `…/rks-hessian-h2o-sto3g` |
| UKS nuclear Hessian (FD of forces) | [uks-hessian-o2-sto3g](methods/uks-hessian-o2-sto3g.md) | `…/uks-hessian-o2-sto3g` |
| UKS full-LR TDDFT (LSDA O₂) | [uks-full-lr-o2-lsda](methods/uks-full-lr-o2-lsda.md) | `…/uks-full-lr-o2-lsda` |
| UMP2 analytic gradient | [ump2-grad-o2-sto3g](methods/ump2-grad-o2-sto3g.md) | `…/ump2-grad-o2-sto3g` |

SCF reading map: [User Guide → SCF Learning Path](../user-guide/scf-roadmap.md).
