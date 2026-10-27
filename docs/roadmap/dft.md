# Density-Functional Theory

Tutorial: [KS](../tutorial/ch05-kohn-sham-dft.md) · [hybrids](../tutorial/ch06-hybrid-dft.md) · [PW](../tutorial/ch08-plane-wave.md) · [GPW](../tutorial/ch09-gpw.md).

Molecular KS shares the Gaussian SCF solver with
[Hartree–Fock](hartree-fock.md). Periodic DFT (plane-wave + GPW) lives in
§4–6. Legend: [Roadmap index](index.md).

The tables retain the historical **`dev`** inventory and its stated scope.
The [scientific regression runner was removed](../../regressions/README.md);
the linked case sources and reference outputs are now archives, not executable
checks of the current checkout. Current molecular acceptance runs through
[`benchmarks/scf/bench_rks.py`](../../benchmarks/scf/bench_rks.py) and
[`benchmarks/scf/bench_uks.py`](../../benchmarks/scf/bench_uks.py), using live
PySCF and the public Python methods. Historical marks below do not substitute
for current benchmark results.

Historical CPU validation on 2026-10-25 passed 1785
unit tests and all 68 enabled regression cases (301 sections at their declared
thread widths), plus the relocated installed XC extension project. Affected
PW cases were rechecked after the final repair; unchanged cases retain their
accepted results. CUDA was not rerun. Energy, forces and stress have separate
coverage; a ✓ remains limited to the systems, quantities and settings in its
cited case. Passing an existing check does not close the qualification gaps below.

The CPU NC-extension repair was rechecked on 2026-10-26: 1817 unit tests and
14 affected regression cases (63 sections, maximum 26.5 s per section) passed.
The 50 Python PW tests and all five modes of the relocated independent NC
project passed, along with its three required modes on a fresh minimal
PW+Evaluate installation. [Native outputs and hashes](../../tools/diagnostics/nc-extension/native-validation.json)
record the scope. This is a scoped recheck, not a new run of all 68 cases.
CUDA was not rerun.

## Current executable evidence

The [follow-up public molecular matrix](../../benchmarks/scf/results/2026-10-26-dft-wave2/README.md)
passed 67 workloads / 134 required CPU-oracle cells, retaining the preceding
35-workload envelope and adding standard CAM-B3LYP, PBEsol, revPBE and RPBE.
Maximum errors were 1.287e-9 Ha and 1.208e-9 Ha/Bohr. The
[standard CAM record](../../tools/diagnostics/molecular-dft/cam-standard-20261026/README.md)
also passed 62 native units and six additional live molecular energy/full-force
comparisons; three extreme scalar spin/density points retain explicitly
recorded B88/LYP/VWN numerical differences. [Earlier molecular diagnostics](../../tools/diagnostics/molecular-dft/README.md)
retain their 28 affected native units, 24 comparisons and initial CAM failures.
The [GH200 record](../../tools/diagnostics/cuda-pw-public/README.md) separately
passed 37 actual units, public PW result/reuse checks, and a matched builtin
GTH Si₂ PBE/Γ/smearing QE energy/full-force pin. These are scoped numerical
results; dirty-source/build-attestation and broader capability gaps remain.

The [shared smearing repair](../../tools/diagnostics/pw-smearing/README.md)
passes 65 local points and 20 spectra from actually compiled QE 7.6 routines,
plus nine native units. The archived production implementation retains 61
failed comparisons, including one NaN. MP quasi-occupations are now signed,
the generalized entropy has the correct degeneracy, and chemical-potential
search stops on electron count instead of drifting to a gap edge. Six public
DFPT admission tests separately verify that MP response is rejected before
SCF at all three q paths in restricted and collinear settings; response
projectors still require a signed-weight implementation.

The [CPU PW follow-up](../../tools/diagnostics/pw-spin-smear-stress/README.md)
records LSDA/PBE × FD/Gaussian/MP1 at Γ on skew H₃. All six reference-cell
energies and six-component stress tensors pass matched actual QE 7.6 pins:
maximum errors 1.193e-6 Ha and 1.984e-5 eV/Å³. Five independent full-SCF
simple-deformation checks pass within 4.162e-7 eV/Å³, and both FD rotation
checks within 3.365e-7 eV/Å³. The LSDA-MP compressed sample does not converge;
the complete matrix remains explicitly failed. Stress now retains the
reference integer basis and FFT shape, accounts for both symmetric-shear
lanes, and refills the two spin spectra at one chemical potential. This is
still a numerical frozen-density strain derivative, not closed-form stress.
The same CPU record compares one and two MPI ranks on a 2×2×2 k mesh:
energy and stress differences are 3.047e-12 Ha and 2.039e-9 eV/Å³. Its original
source and cutoff-reselection negative controls retain their expected failures.

<!-- benchmark-evidence:start -->

Campaign `9d6af9a65069414b83d739d432387d79`; profile `linux-x86_64-cpu1-acceptance`; tier `acceptance`.

Numerical acceptance and roadmap qualification are separate. A local dirty or unverified build cannot promote ✓/⚡. Unmigrated capabilities retain explicit gaps.

| Capability ID | Capability | CPU | CUDA | Evidence |
| --- | --- | :---: | :---: | --- |
| `dft.rks.energy` | RKS molecular energy | ◐ | ○ | CPU numerical pass; CUDA numerical coverage pending; official source/build qualification incomplete |
| `dft.rks.gradient` | RKS molecular gradient | ◐ | ○ | CPU numerical pass; CUDA numerical coverage pending; official source/build qualification incomplete |
| `dft.uks.energy` | UKS molecular energy | ◐ | ○ | CPU numerical pass; CUDA numerical coverage pending; official source/build qualification incomplete |
| `dft.uks.gradient` | UKS molecular gradient | ◐ | ○ | CPU numerical pass; CUDA numerical coverage pending; official source/build qualification incomplete |

<!-- benchmark-evidence:end -->

**Blocked on periodic engines** (no CP2K-comparable twin in tree): hybrids,
meta-GGA, GAPW/PAW, stock VV10. rVV10 ships.

## 1. Molecular Kohn–Sham

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| RKS / UKS | ✓ | ○ | [scf/rks][scf-rks], [scf/uks][scf-uks]; no Gaussian KS on device |
| ROKS | ✓ | ○ | [scf/roks][scf-roks]: energy + analytic gradient |
| RKS / UKS analytic gradient | ✓ | ○ | [Current live matrix](../../benchmarks/scf/results/2026-10-26-dft-wave2/README.md): standard CAM and PBE-family full forces, plus the previous LSDA/PBE/PBE0 envelope; spherical def2-SVP for PBE/PBE0/CAM, including non-symmetric water and empty-spin H₂⁺. The archived external-provider pins cover scaled Slater only; other models remain open |
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
| PBEsol / revPBE / RPBE | ◐ | ○ | [Current live matrix](../../benchmarks/scf/results/2026-10-26-dft-wave2/README.md): RKS/UKS STO-3G energies/full forces against live PySCF; RKS asymmetric water and UKS H₂⁺/O₂. GPW and CUDA variants do not inherit these passes |
| optB88 | ◐ | ○ | GPW (with vdW-DF); no end-to-end pin (cases removed) |
| BP86 / other GGA catalog | ○ | ○ | |
| Meta-GGA (TPSS, r2SCAN, …) | ○ | ○ | No τ channel in grid engine |
| B3LYP | ✓ | ○ | Gaussian only; [scf/rks][scf-rks], `b3lyp-fused-jk-h2o`: energy |
| PBE0 | ◐ | ○ | [Current live matrix](../../benchmarks/scf/results/2026-10-26-dft/README.md): public RKS/UKS energies and full analytic forces, STO-3G/def2-SVP. Scoped numerical acceptance passes; formal source/build qualification remains open |
| CAM-B3LYP | ◐ | ○ | [Current standard-model evidence](../../tools/diagnostics/molecular-dft/cam-standard-20261026/README.md): short-range ITYH-B88 and VWN5 replace the old B3LYP semi-local mixture; RKS/UKS energy/full-force pins include spherical def2-SVP. Extreme scalar spin/density parity limitations remain recorded; no periodic or device CAM path |
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
| Plane-wave DFT | ✓ | ✓ | [pw/rks][pw-rks]; CPU GTH and scalar NC-UPF share one operator path; the independent NC project below has a matched Si₂ energy/full-force pin. CUDA ⊂ CPU |
| External scalar NC provider | ✓ | ○ | [Independent installed project](../../examples/nc-extension/README.md): immutable radial data/local callback on existing PW; published Si GTH-PBE-q4, absolute Mermin energy and all six nonzero forces against actual QE. Single species, real scalar l≤2; no USPP/PAW/NLCC/spin-orbit |
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
| Analytic ionic forces | ✓ | [pw/rks][pw-rks]: local HF + Ewald symmetry gate; [external NC project](../../examples/nc-extension/README.md): local + full off-diagonal KB, all six nonzero Si₂ force components against QE at 1e-4 Ha/Bohr. Broader elements/potentials remain unqualified |
| Cell stress (one-SCF frozen-density strain FD) | ✓ | [Current public diagnostic](../../tools/diagnostics/pw-spin-smear-stress/README.md): fixed reference basis, physical shear and shared-μ spin/smearing free energy at strain step 2e-3. Six Γ reference-cell E/σ QE pins pass; five independent full-SCF deformation checks pass, with the LSDA-MP compressed sample explicitly failed. Not a closed-form tensor |
| Closed-form analytic cell stress | ○ | Current `pw_stress_analytic_impl` differentiates NSCF strained energies numerically; spin×smearing uses fractional occupations and the matching free-energy correction |
| True GGA PBE + collinear spin-GGA | ✓ | [pw/rks][pw-rks]: symmetry/activation gates; [pw/collinear-spin][pw-spin]: O₂ absolute energy pinned to an actual QE 7.5 run, alongside magnetization and GGA activation |
| Metallic smearing | ✓ | [Current shared implementation](../../tools/diagnostics/pw-smearing/README.md): FD/Gaussian/MP occupation, entropy and chemical-potential pins from real QE 7.6; nine variational/endpoint units. Full-SCF stress and response have separate qualification limits |
| Bands / DOS | ✓ | [pw/bands-dos][pw-bands]; includes spin + metallic sections |
| CELL_OPT | ✓ | [pw/variable-cell][pw-cell]; reference states the geometry/energy comparison limits |
| BOMD | ✓ | [dynamics/bomd][bomd], `pw-bomd-h`: foothold |
| Berry / MLWF | ✓ | [pw/berry-phase][pw-berry], [pw/wannier][pw-wannier]; includes spin |
| DFPT phonons | ◐ | [response/dfpt][dfpt]: historical Γ / finite-q / incommensurate × GGA × spin × smear. The public interface now rejects MP response before SCF: signed quasi-occupations are not supported by its occupation thresholds/projectors. Fixed occupations, FD and Gaussian retain their existing scope; historical MP checks do not qualify the corrected model |

### Plane-wave depth (CUDA)

| Capability | Status | Notes |
| ---------- | :----: | ----- |
| LSDA + PBE energy | ✓ | [Current GH200 diagnostic](../../tools/diagnostics/cuda-pw-public/README.md): shared CPU/device padded XC quadrature; builtin GTH Si₂ PBE/Γ/FD-smearing Mermin energy pinned to actual QE 7.5. Broader LSDA/PBE evidence remains the archived CPU-twin/symmetry envelope |
| Analytic forces + cell stress (closed-shell) | ✓ | [Current GH200 diagnostic](../../tools/diagnostics/cuda-pw-public/README.md): builtin GTH Si₂ PBE/Γ/FD-smearing all six nonzero forces against QE, plus stretched-Si CPU twin. Stress remains one-SCF NSCF strain FD; its archived checks do not qualify a closed-form tensor or external stress |
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
| GTH PP | ✓ | ◐ | CPU PW uses the public NC protocol and canonical radial operators; [external Si₂ project](../../examples/nc-extension/README.md) pins E/full F. [CUDA builtin Si₂](../../tools/diagnostics/cuda-pw-public/README.md) now has a matched QE PBE/Γ/smearing E/full-F pin. CPU GPW retains its Gaussian-projector path; CUDA GPW remains partial |
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
| Historical CPU baseline (2026-10-25) | Registered Gaussian, PW, GPW and composed workflows | 1785 units and all 68 enabled cases passed; all 301 reported section times were ≤60 s (maximum 52.4 s). The independent installed XC project passed RKS/UKS energies, full forces and lifecycle checks after relocation. CPU only; CUDA still requires GH200 validation. |
| Pending broader qualification | Molecular functional/basis/system range; GPW functional variants | The current live matrix closes specific RKS/UKS energy/full-force combinations, including PBE0 and spherical def2-SVP. Other systems, functionals, providers and GPW variants do not inherit those passes. |
| Pending matched-model qualification | DFT+U | Gross-Mulliken ATV occupation differs from CP2K's net-Mulliken block. Loose NiO comparisons and ATV finite differences do not close this gap. |
| Verified model repair, limited scalar boundary parity | Standard CAM-B3LYP | [New standard-model evidence](../../tools/diagnostics/molecular-dft/cam-standard-20261026/README.md) passes molecular E/full F and normal-domain scalar/ITYH checks. The original four failed standard-CAM comparisons remain historical records. Three extreme scalar points still differ in underlying B88/LYP/VWN components; they are not marked passed. |
| Pending oracle replacement | XDM | Literature supplies formulas; the numerical pins are Python reimplementations. Obtain numbers printed by an independent established engine for the same free-atom model. |
| Missing algorithm | Closed-form PW cell stress, CPU and CUDA | Functions named `analytic` currently use one reference SCF + frozen-density NSCF strain FD at step 2e-3. Existing cell-stress gates do not establish a closed-form implementation. |
| Missing response algorithm | Methfessel–Paxton DFPT | Correct MP ground-state quasi-occupations can be negative or exceed their spin degeneracy. DFPT occupation thresholds and occupied-space projectors assume probability weights; the public phonon interface rejects MP at all q and spin settings before SCF. Ground-state E/F/σ qualification does not extend to response. |
| Unclosed convergence boundary | LSDA × MP signed density near zero | The H₃ MP1/.04-Ry simple −xx strain sample fails joint density tol 1e-9 at both 250 and 1000 iterations. A quadrature spin density crosses zero and its clipped-functional potential jumps between about −.0444 Ha and zero. Inner Ritz residuals meet 1e-10; [716 exact-spectrum QE replays](../../tools/diagnostics/pw-smearing/mp-root-review-20261026/summary.json) and tighter-root checks exclude a chemical-potential branch/precision defect. This independent full-SCF FD row remains failed. Removing the potential clip would break the existing energy derivative; a consistent XC extension or constrained solver is still required. |
| Verified repaired slice | Molecular spherical XC forces and range-separated integrals | [CPU evidence](../../tools/diagnostics/molecular-dft/README.md): correct Cartesian density dual, erf Rys-root mapping and analytic derivatives, and the missing UKS long-range term. The follow-up standard-CAM repair has its own separate evidence; the earlier native-composition FD checks do not substitute for standard-functional pins. |
| Verified repaired slice | CUDA PW XC quadrature and derivative results | [GH200 evidence](../../tools/diagnostics/cuda-pw-public/results/2026-10-26-gh200/validation.json): 37 actual units, public result/reuse checks, builtin Si₂ QE E/full-F pin, original-source negative control and zero-error padded-stage memcheck. Private LSDA/PBE quadrature now matches CPU; force/stress results carry their own SCF status. Spin×smearing stress rejects the missing occupation derivative path. No GPW or general device-residency qualification follows from these checks. |
| Pending public-consumer coverage | GPW dispersion forces/stress | 17 CPU section sources across D3, TS, MBD, Hirshfeld and XDM still include `internal/`. Their energy pins do not certify the whole F/σ surface. |
| Verified repaired slice | GPW finite-displacement Si₂ phonons | [response/phonons][phonons], `gpw-fd-phonon-si`: actual MT Poisson and matching Gaussian-product density/force contractions; CP2K stationary input replaces the nonstationary frequency fixture. Original energy/full-force, five raw modes below 80 cm⁻¹, stretch and 60 s bounds retained. |
| Verified repaired slice | GPW UKS forces / O₂ stress | Coupled-spin DIIS, energy-only XC for strained energies and reuse of the accepted SCF state remove redundant work. Both existing sections pass their numerical gates within 60 s at the declared 8 threads; this is section-runtime evidence, not a matched-engine speedup claim. |
| Rechecked declared CPU envelope | Γ RI/SCS-MP2, UKS+U/OT repeatability, second-gen CP OT | Surviving registered sections pass their original numerical and runtime gates. This does not close the DFT+U projector-definition mismatch or qualify unregistered settings. |
| Verified PW XC/response repair | GGA quadrature, joint-spin SCF and finite-q C(q) | [Historical captures and current scope](../../tools/diagnostics/pw-xc-response/README.md): fixed 3/2 LDA and 3x GGA quadrature with adjoint projection, shared Pulay mixing for ground-state/response solves, true Fourier C(q), and exact-zero empty-channel density. Current 14-case recheck retains the original science and 60 s section bounds, with no force/ASR projection. |
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
