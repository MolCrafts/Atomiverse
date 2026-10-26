# Density-Functional Theory

Tutorial: [KS](../tutorial/ch05-kohn-sham-dft.md) · [hybrids](../tutorial/ch06-hybrid-dft.md) · [PW](../tutorial/ch08-plane-wave.md) · [GPW](../tutorial/ch09-gpw.md).

Molecular KS shares the Gaussian SCF solver with
[Hartree–Fock](hartree-fock.md). Periodic DFT (plane-wave + GPW) lives in
§4–6. Legend: [Roadmap index](index.md).

This inventory follows **`dev`**. CPU validation on 2026-10-25 passed 1785
unit tests and all 68 enabled regression cases (301 sections at their declared
thread widths), plus the relocated installed XC extension project. Affected
PW cases were rechecked after the final repair; unchanged cases retain their
accepted results. CUDA was not rerun. Energy, forces and stress have separate
coverage; a ✓ remains limited to the systems, quantities and settings in its
cited case. Passing an existing check does not close the qualification gaps below.

**Blocked on periodic engines** (no CP2K-comparable twin in tree): hybrids,
meta-GGA, GAPW/PAW, stock VV10. rVV10 ships.

## 1. Molecular Kohn–Sham

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| RKS / UKS | ✓ | ○ | [scf/rks][scf-rks], [scf/uks][scf-uks]; no Gaussian KS on device |
| ROKS | ✓ | ○ | [scf/roks][scf-roks]: energy + analytic gradient |
| RKS / UKS analytic gradient | ✓ | ○ | [scf/rks][scf-rks] `custom-xc-water`, [scf/uks][scf-uks] `custom-xc-h2plus`: scaled Slater exchange, STO-3G, PySCF full force vectors; broader functional/basis coverage remains open |
| External CPU XC provider | ✓ | ○ | [Independent installed project](../../examples/xc-extension/README.md); batched rho/sigma contract on existing RKS/UKS, energy + first derivatives; the cited force pins exercise rho-only exchange. No injected PW/GPW, ROKS, meta-GGA, hybrid or response kernel support |
| ROKS analytic gradient | ✓ | ○ | [scf/roks][scf-roks], `roks-grad-o2-sto3g`: PySCF full gradient |
| RKS / UKS Hessian | ✓ | ○ | [scf/rks][scf-rks], [scf/uks][scf-uks]: `Hessian<Method>::run`, FD of analytic forces, PySCF FD-gradient pins (not full CPKS) |
| Molecular DFT+U / ΔSCF-DFT / GKS | ○ | ○ | Periodic DFT+U is GPW (§4) |

## 2. Functionals (molecular + shared)

Hand-coded; libxc is an optional test oracle only (`ATV_BUILD_XC_ORACLE`).

| Rung / functional | CPU | CUDA | Notes |
| ----------------- | :-: | :--: | ----- |
| LSDA (Slater+VWN5) | ✓ | ✓ | CPU molecular: [scf/roks][scf-roks]; CUDA PW LDA/spin: [pw/rks][pw-rks], [pw/collinear-spin][pw-spin]; PW LDA has a broad ABINIT-class energy band |
| PW92 correlation base | ◐ | ◐ | Used inside PBE correlation on both backends; no standalone PW92 `XCType` or registered PW92-functional energy pin; `XCType::LSDA` selects VWN5 |
| PBE | ✓ | ✓ | CPU molecular energy: [scf/tddft][scf-td]; GPW: [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]; CUDA PW: [pw/rks][pw-rks], [pw/collinear-spin][pw-spin]; see PW limits below |
| BLYP | ✓ | ○ | [scf/rks][scf-rks], `blyp-water-sto3g`; GPW reachable, not separately pinned |
| PBEsol / revPBE / RPBE | ◐ | ○ | Molecular + GPW reachable; libxc unit parity only, no end-to-end pin (GPW functional-swap cases removed) |
| optB88 | ◐ | ○ | GPW (with vdW-DF); no end-to-end pin (cases removed) |
| BP86 / other GGA catalog | ○ | ○ | |
| Meta-GGA (TPSS, r2SCAN, …) | ○ | ○ | No τ channel in grid engine |
| B3LYP | ✓ | ○ | Gaussian only; [scf/rks][scf-rks], `b3lyp-fused-jk-h2o`: energy |
| PBE0 | ◐ | ○ | Gaussian implementation; no registered PBE0 end-to-end section |
| CAM-B3LYP | ✓ | ○ | [scf/rks][scf-rks], `cam-b3lyp-h2-sto3g`: RKS energy only |
| Other RSH / screened hybrids | ○ | ○ | |
| B2PLYP | ✓ | ○ | [scf/rks][scf-rks], `b2plyp-h2o`: hybrid KS + scaled RMP2 energy |
| Other double hybrids | ○ | ○ | |
| HF-3c / PBEh-3c / … | ○ | ○ | |

## 3. Dispersion and nonlocal correlation

| Method | Molecular | GPW | CUDA | Notes |
| ------ | :-------: | :-: | :--: | ----- |
| D3 zero | ○ | ✓ | ○ | [gpw/d3][gpw-d3]: E, stress and ATM pins; force/ATM sections still include private headers. GPW ATM stress uses the complete energy strain derivative; the standalone internal D3 virial lacks the direct ATM term |
| D3-BJ | ✓ | ✓ | ○ | Molecular E: [scf/dispersion][scf-disp]; GPW E: [gpw/d3][gpw-d3]; GPW force/stress sections still include private headers |
| D4 | ✓ | ✓ | ○ | Molecular E: [scf/dispersion][scf-disp]; GPW E/F/σ + ATM: [gpw/d4][gpw-d4], dftd4 pins |
| TS | ○ | ✓ | ○ | [gpw/tkatchenko-scheffler][gpw-ts], [gpw/hirshfeld][gpw-hirshfeld]; free-atom periodic E pinned to libMBD; force/stress public-consumer coverage incomplete |
| MBD@rsSCS | ○ | ✓ | ○ | [gpw/mbd][gpw-mbd], [gpw/hirshfeld][gpw-hirshfeld]; libMBD E pins; force/stress public-consumer coverage incomplete |
| XDM | ○ | ◐ | ○ | [gpw/xdm][gpw-xdm]: free-atom path; numerical pins are independent Python reimplementations, not an accepted external oracle; private-header leaks also remain |
| vdW-DF / rVV10 | ○ | ✓ | ○ | [gpw/vdw-df][gpw-vdw]: SC nonlocal potential + F/σ in the recorded envelope |
| vdW-DF2 / optB88-vdW | ○ | ◐ | ○ | Same kernel path; no end-to-end pin (cases removed) |
| VV10 (stock) | ○ | ○ | ○ | **Blocked** — no CP2K `TYPE VV10` |

GPW dispersion has Γ and multi-k, RKS and UKS paths, with uneven evidence.
An energy pin does not close force/stress coverage, and an independent Python
implementation does not qualify as a numerical oracle. See the cases above and
[tutorial/methods catalog](../tutorial/methods/catalog.md).

## 4. Periodic Kohn–Sham

Two engines: **plane-wave** (`cpu::pw::PW` / `cuda::pw::PW`) and **GPW**
(`cpu::gpw::GpwDft` / `GPWEvaluator`). CUDA GPW is a partial product slice
(◐), not full CPU parity.

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Plane-wave DFT | ✓ | ✓ | [pw/rks][pw-rks]; NC-UPF or in-tree GTH; CUDA ⊂ CPU; broad energy/symmetry gates, not a matched absolute-energy qualification |
| GPW DFT | ✓ | ◐ | [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]; CUDA first product slices |
| Γ-point RKS | ✓ | ◐ | [gpw/rks][gpw-rks]; CUDA CPU-twin slice |
| k-point RKS | ✓ | ✓ | [pw/rks][pw-rks] on both; [gpw/rks][gpw-rks] CPU; CUDA GPW remains ◐ |
| Γ / k-point UKS | ✓ | ◐ | CPU: [pw/collinear-spin][pw-spin], [gpw/uks][gpw-uks]; CUDA PW collinear subset, GPW Γ energy/force slices; GPW multi-k UKS ○ |
| Periodic LDA / GGA | ✓ | ✓ | Cases above; no periodic hybrid / meta-GGA |
| Periodic DFT+U | ◐ | ○ | [gpw/dft-plus-u][gpw-u]: Γ/multi-k E/F/σ paths; ATV gross-Mulliken and CP2K net-Mulliken differ, so the recorded loose comparisons do not qualify a matched-model pin |
| Periodic dispersion | ◐ | ○ | See §3: method-dependent E/F/σ evidence; XDM oracle and public-consumer gaps remain |
| Periodic RI-MP2 / SCS-MP2 (Γ) | ✓ | ○ | [gpw/periodic-mp2][gpw-mp2]: CP2K correlation-energy pins; declared CPU sections rechecked |
| Periodic UMP2 / multi-k MP2 | ◐ | ○ | GPW; no external pin, self-pinned cases removed |
| Periodic dRPA / URPA | ◐ | ○ | Γ, multi-k, finite-q reachable; no external pin, self-pinned cases removed |
| Periodic G0W0 / UGW | ◐ | ○ | Γ, multi-k, finite-q reachable; no external pin, self-pinned cases removed |
| GAPW / PAW / USPP | ○ | ○ | NC-UPF or GTH only |

### Plane-wave depth (CPU)

| Capability | Status | Notes |
| ---------- | :----: | ----- |
| Analytic ionic forces | ✓ | [pw/rks][pw-rks]: local HF + Ewald exact-zero symmetry gate; general nonzero forces and KB lack a registered external-vector pin |
| Cell stress (one-SCF frozen-density strain FD) | ✓ | [pw/rks][pw-rks]: numerical Nielsen–Martin HF derivative, fixed strain step 2e-3; symmetry + ATV full-SCF FD checks, not a closed-form tensor |
| Closed-form analytic cell stress | ○ | Current `pw_stress_analytic_impl` differentiates NSCF strained energies numerically; spin×smearing stress rejects |
| True GGA PBE + collinear spin-GGA | ✓ | [pw/rks][pw-rks]: symmetry/activation gates; [pw/collinear-spin][pw-spin]: O₂ absolute energy pinned to an actual QE 7.5 run, alongside magnetization and GGA activation |
| Metallic smearing | ✓ | [pw/smearing][pw-smear]: electron-count/entropy identities; FD / MP implemented, recorded gates have a restricted envelope |
| Bands / DOS | ✓ | [pw/bands-dos][pw-bands]; includes spin + metallic sections |
| CELL_OPT | ✓ | [pw/variable-cell][pw-cell]; reference states the geometry/energy comparison limits |
| BOMD | ✓ | [dynamics/bomd][bomd], `pw-bomd-h`: foothold |
| Berry / MLWF | ✓ | [pw/berry-phase][pw-berry], [pw/wannier][pw-wannier]; includes spin |
| DFPT phonons | ✓ | [response/dfpt][dfpt]: Γ / finite-q / incommensurate × GGA × spin × smear; per-section oracle limits apply |

### Plane-wave depth (CUDA)

| Capability | Status | Notes |
| ---------- | :----: | ----- |
| LSDA + PBE energy | ✓ | [pw/rks][pw-rks]: device XC, CPU-twin/symmetry envelope |
| Analytic forces + cell stress (closed-shell) | ✓ | [pw/rks][pw-rks]: LSDA F/σ + PBE σ CPU twins; stress is one-SCF NSCF strain FD, not a closed-form tensor; closed-shell PBE forces not separately pinned |
| Collinear spin forces / cell stress | ✓ | [pw/collinear-spin][pw-spin]; includes spin-PBE CPU twins; stress uses NSCF strain FD; general external-vector qualification remains open |
| Closed-form analytic cell stress | ○ | Current device stress implementation also uses strained NSCF energies at step 2e-3 |
| Metallic smear energy + F/σ | ◐ | [pw/smearing][pw-smear]: restricted path; multi-atom stress residual; spin×smear stress ○ |
| Host Ritz + SCF control | — | Hot path (H·ψ, FFT, Poisson, V_eff) on device |

### GPW depth (CPU)

RKS/UKS, Γ/Monkhorst–Pack and DIIS/OT paths exist. The registered sections
validate specific combinations; they do not establish every Cartesian product
of solver, functional, spin, grid, dispersion and k-mesh.

| Area | Status | Notes |
| ---- | :----: | ----- |
| Functionals | ✓ | [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]: PADE/LSD, PBE; other reachable functionals lack GPW pins (§2) |
| DFT+U | ◐ | [gpw/dft-plus-u][gpw-u], [gpw/orbital-transformation][gpw-ot]: projector mismatch; OT reachability does not close external validation |
| Smearing / multigrid / OT | ✓ | [gpw/rks][gpw-rks], [gpw/uks][gpw-uks], [gpw/orbital-transformation][gpw-ot]; only the recorded combinations |
| Analytic F + analytic stress | ✓ | [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]; Γ water external pins; broader k/spin/metallic depth has FD-only gaps |
| Dispersion / vdW | ◐ | §3: energy pins for several methods; XDM oracle and public force/stress coverage remain open |
| Bands / DOS / Berry / MLWF | ✓ | [gpw/bands-dos][gpw-bands], [gpw/berry-phase][gpw-berry], [gpw/wannier][gpw-wannier]; recorded spin/metallic sections |
| BOMD / CPMD / SGCP / XL-BOMD | ✓ | [dynamics/bomd][bomd], [dynamics/car-parrinello][cpmd], [dynamics/second-gen-cpmd][sgcp], [dynamics/xl-bomd][xlbomd]; see [MD § AIMD](molecular-dynamics.md#5-ab-initio-md) |
| Periodic correlated | ◐ | RI-MP2 / SCS-MP2 at Γ pinned; RPA, GW, UMP2, multi-k not (§4 table) |

**Public shape:** `GpwDft` is a `Driver<gpw::Phase>`, with kinds/cutoff/XC as
constructor identity and Frame `run` / `forces` / `stress`. Born–Oppenheimer
work mounts the method directly on the parent Driver; forces/stress follow
the parent's requests and use the converged state. `GPWEvaluator` remains for
electronic propagators and unmigrated callers. OT still has a private free
loop. Keep `local_pp_on_grid` consistent between energy and gradient calls.

### CUDA GPW

| Capability | Status | Notes |
| ---------- | :----: | ----- |
| Γ RKS DIIS energy + F/σ slice | ◐ | [gpw/rks][gpw-rks]: CPU twins; host AO Pulay / stress residual; no independent CP2K qualification |
| Γ UKS energy / force slice | ◐ | [gpw/uks][gpw-uks]: Padé LSD CPU twins; full F/σ and PBE depth open |
| Multi-k RKS energy | ◐ | [gpw/rks][gpw-rks]: H₂ foothold + device mid-SCF pieces; setup still hybrid |
| Full method parity with CPU | ○ | Table stays ◐ until production depth + external pins |

## 5. Periodic response and analysis

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Periodic analytic forces | ✓ | ◐ | [pw/rks][pw-rks], [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]; quantity limits in §4 |
| Periodic analytic stress | ✓ | ◐ | GPW CPU fixed-P stress: [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]; PW CPU/CUDA stress is numerical NSCF strain FD (§4); CUDA GPW host-assembly slice |
| DFPT | ✓ | ○ | [response/dfpt][dfpt], PW only |
| FD phonons / ω(q) | ✓ | ○ | [response/phonons][phonons]: classical SW supercell/dispersion pins; isolated GPW Si₂ energy/full forces, stationary geometry and stretch pinned to actual CP2K 2026.2 outputs, with five unprojected near-zero modes |
| Berry polarization | ✓ | ○ | [pw/berry-phase][pw-berry], [gpw/berry-phase][gpw-berry]; spin |
| MLWF | ✓ | ○ | [pw/wannier][pw-wannier], [gpw/wannier][gpw-wannier]; spin |

## 6. Periodic infrastructure

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Monkhorst–Pack k-mesh | ✓ | ✓ | [pw/rks][pw-rks], [gpw/rks][gpw-rks] |
| Block Davidson | ✓ | ✓ | [pw/rks][pw-rks]; exercised inside the method, no standalone product claim |
| NC-UPF (KB) | ✓ | ✓ | [pw/rks][pw-rks]; energy path; KB nonzero-force external pin open |
| GTH PP | ✓ | ◐ | CPU GPW [gpw/rks][gpw-rks]; CUDA PW H local-only symmetry gates and GPW partial slices |
| FFT Poisson | ✓ | ✓ | [pw/rks][pw-rks], [gpw/rks][gpw-rks] |
| Martyna–Tuckerman Poisson | ✓ | ○ | [gpw/rks][gpw-rks], `gpw-water-pbe-mt` |
| OT minimizer | ✓ | ○ | [gpw/orbital-transformation][gpw-ot]; private-loop architecture debt remains |
| Multigrid collocation | ✓ | ◐ | [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]; CUDA MG energy CPU-twin slice exists |
| Metallic smearing | ✓ | ◐ | [pw/smearing][pw-smear], [gpw/rks][gpw-rks]; CUDA restricted PW (+ spin×smear F; stress open) |
| Bands / DOS post-process | ✓ | ○ | [pw/bands-dos][pw-bands], [gpw/bands-dos][gpw-bands] |

## 7. Validation gaps

The entries distinguish observed numerical failures from qualification and
coverage gaps.

| State | Scope | Evidence / remaining check |
| ----- | ----- | -------------------------- |
| Verified CPU baseline | Registered Gaussian, PW, GPW and composed workflows | 1785 units and all 68 enabled cases pass; all 301 reported section times are ≤60 s (maximum 52.4 s). The independent installed XC project passes RKS/UKS energies, full forces and lifecycle checks after relocation. CPU only; CUDA still requires GH200 validation. |
| Pending external qualification | Molecular PBE0; broader RKS/UKS functional/basis force vectors; GPW functional variants | The custom scaled-Slater water/H₂⁺ force pins cover their declared systems only. Reachable implementation or unit parity is insufficient for a ✓ on another quantity or combination. |
| Pending matched-model qualification | DFT+U | Gross-Mulliken ATV occupation differs from CP2K's net-Mulliken block. Loose NiO comparisons and ATV finite differences do not close this gap. |
| Pending oracle replacement | XDM | Literature supplies formulas; the numerical pins are Python reimplementations. Obtain numbers printed by an independent established engine for the same free-atom model. |
| Missing algorithm | Closed-form PW cell stress, CPU and CUDA | Functions named `analytic` currently use one reference SCF + frozen-density NSCF strain FD at step 2e-3. Existing cell-stress gates do not establish a closed-form implementation. |
| Pending public-consumer coverage | GPW dispersion forces/stress | 17 CPU section sources across D3, TS, MBD, Hirshfeld and XDM still include `internal/`. Their energy pins do not certify the whole F/σ surface. |
| Verified repaired slice | GPW finite-displacement Si₂ phonons | [response/phonons][phonons], `gpw-fd-phonon-si`: actual MT Poisson and matching Gaussian-product density/force contractions; CP2K stationary input replaces the nonstationary frequency fixture. Original energy/full-force, five raw modes below 80 cm⁻¹, stretch and 60 s bounds retained. |
| Verified repaired slice | GPW UKS forces / O₂ stress | Coupled-spin DIIS, energy-only XC for strained energies and reuse of the accepted SCF state remove redundant work. Both existing sections pass their numerical gates within 60 s at the declared 8 threads; this is section-runtime evidence, not a matched-engine speedup claim. |
| Rechecked declared CPU envelope | Γ RI/SCS-MP2, UKS+U/OT repeatability, second-gen CP OT | Surviving registered sections pass their original numerical and runtime gates. This does not close the DFT+U projector-definition mismatch or qualify unregistered settings. |
| Verified PW XC/response repair | GGA quadrature, joint-spin SCF and finite-q C(q) | [Native validation](../../tools/diagnostics/pw-xc-response/README.md): one Pulay implementation, private 3/2 XC quadrature, true Fourier C(q), and exact-zero empty-channel density. The Γ PBE ASR residual is 0.0201 eV/Å² below the unchanged 0.05 bound; single-H forces satisfy the original 1e-9 eV/Å identity without force/ASR projection. |
| Partial device product | CUDA GPW; CUDA PW metallic/spin joints | Public atom-list/cell APIs and host force/stress work remain in GPW; CPU-twin sections do not close device residency or every production combination. |
| Blocked by missing engine work | Periodic hybrids, meta-GGA, GAPW/PAW | No matched implemented periodic path; meta-GGA also lacks a τ channel. Stock VV10 has no CP2K `TYPE VV10` twin. |

The PW references also contain matched Quantum ESPRESSO timing/output records.
Those output values are not automatically correctness gates: e.g.
`pw-pbe-gga-si` still asserts symmetry and GGA activation. GPW timing records
affected by OpenMP binding require an unbound re-measurement before quoting a
current performance gap. No new performance qualification is implied here.

[scf-rks]: ../../regressions/scf/rks/REFERENCE.md
[scf-uks]: ../../regressions/scf/uks/REFERENCE.md
[scf-roks]: ../../regressions/scf/roks/REFERENCE.md
[scf-td]: ../../regressions/scf/tddft/REFERENCE.md
[scf-disp]: ../../regressions/scf/dispersion/REFERENCE.md
[pw-rks]: ../../regressions/pw/rks/REFERENCE.md
[pw-spin]: ../../regressions/pw/collinear-spin/REFERENCE.md
[pw-smear]: ../../regressions/pw/smearing/REFERENCE.md
[pw-bands]: ../../regressions/pw/bands-dos/REFERENCE.md
[pw-cell]: ../../regressions/pw/variable-cell/REFERENCE.md
[pw-berry]: ../../regressions/pw/berry-phase/REFERENCE.md
[pw-wannier]: ../../regressions/pw/wannier/REFERENCE.md
[gpw-rks]: ../../regressions/gpw/rks/REFERENCE.md
[gpw-uks]: ../../regressions/gpw/uks/REFERENCE.md
[gpw-u]: ../../regressions/gpw/dft-plus-u/REFERENCE.md
[gpw-ot]: ../../regressions/gpw/orbital-transformation/REFERENCE.md
[gpw-d3]: ../../regressions/gpw/d3/REFERENCE.md
[gpw-d4]: ../../regressions/gpw/d4/REFERENCE.md
[gpw-ts]: ../../regressions/gpw/tkatchenko-scheffler/REFERENCE.md
[gpw-mbd]: ../../regressions/gpw/mbd/REFERENCE.md
[gpw-hirshfeld]: ../../regressions/gpw/hirshfeld/REFERENCE.md
[gpw-xdm]: ../../regressions/gpw/xdm/REFERENCE.md
[gpw-vdw]: ../../regressions/gpw/vdw-df/REFERENCE.md
[gpw-mp2]: ../../regressions/gpw/periodic-mp2/REFERENCE.md
[gpw-bands]: ../../regressions/gpw/bands-dos/REFERENCE.md
[gpw-berry]: ../../regressions/gpw/berry-phase/REFERENCE.md
[gpw-wannier]: ../../regressions/gpw/wannier/REFERENCE.md
[bomd]: ../../regressions/dynamics/bomd/REFERENCE.md
[cpmd]: ../../regressions/dynamics/car-parrinello/REFERENCE.md
[sgcp]: ../../regressions/dynamics/second-gen-cpmd/REFERENCE.md
[xlbomd]: ../../regressions/dynamics/xl-bomd/REFERENCE.md
[dfpt]: ../../regressions/response/dfpt/REFERENCE.md
[phonons]: ../../regressions/response/phonons/REFERENCE.md
