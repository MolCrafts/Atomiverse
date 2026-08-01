# Scientific Methods Roadmap

This roadmap is the honest inventory of what Atomiverse can actually compute
today, method by method, on each execution backend. It is written against the
code on `dev`, not against intent: a row is only marked implemented if the
method is reachable from the public API and validated.

## Backends

| Backend | Role |
| ------- | ---- |
| **CPU** | Primary portable path (OpenMP / MPI; xsimd where used). |
| **CUDA** | Device path for selected classical MD and SCF/PW kernels (GH200). |
| **Kokkos** | Planned portable performance backend (host + device via Kokkos); **○ everywhere today**. |

Status tables use three columns: `CPU | CUDA | Kokkos`. A CUDA or Kokkos mark
requires the **principal** work of that method to run on that backend — a CPU
fallback is not counted as CUDA/Kokkos support.

## Status

| Symbol | Meaning                                 |
| ------ | --------------------------------------- |
| ○      | Not implemented                         |
| ✓      | Implemented and numerically correct     |
| ⚡      | Numerically correct and fully optimized |

### What this roadmap does not list

A method that a **strictly better algorithm dominates** — same purpose, same or lower
cost, no regime where the older one wins — is simply **not in these tables**. It is not
marked, not tracked, and not planned. Listing it, even as "will not implement", would
suggest the roadmap has an opinion worth reading about it; it does not.

So there is no BFGS row (L-BFGS matches its convergence at O(mN) memory instead of
O(N²)), no Fletcher–Reeves row (Polak–Ribière⁺ restarts strictly better at the same
cost), no Berendsen-thermostat row (CSVR samples the canonical ensemble; Berendsen does
not sample it at all).

This is only about *dominance*, never about our schedule. A method that survives because
it occupies a genuinely different cost/accuracy niche stays ○ and stays listed — steepest
descent is dominated by CG for minimization proper, but it is the robust first resort on a
badly broken start geometry, so it keeps its row. Brownian dynamics keeps its row too: it
is different physics from Langevin, not a worse integrator.

### How the symbols are assigned here

**✓** means the method runs from the public API and its numbers are pinned
against an external reference or an analytic result — a validation case under
`benchmarks/<domain>/<case>/validation/` (RHF, MP2, FCI, GPW, vibrations,
plane-wave Si, LJ MD, bonded MD vs LAMMPS, LJ MC vs NIST), or a unit test that
checks the physics rather than the plumbing.

**⚡** is deliberately scarce. It is reserved for kernels that are both correct
*and* carry a committed performance suite measured against a production engine.
Today that is exactly one method family: the Lennard–Jones 12-6 pair style on
CPU (xsimd, `-march=native`) and CUDA, historically benchmarked against
LAMMPS/Kokkos (`/* benchmarks removed; re-add later */ lj/performance/`).
Everything else that is correct but has no standing performance gate is a ✓,
including code that is fast in practice. No Kokkos ⚡ yet (backend ○).

**○** means not implemented. It does not distinguish "planned" from "not
planned"; the sections below are a method census, not a commitment schedule.

## Chapters

Each chapter is one **method family**. Periodicity, gradients, and excited-state
response are *axes*, not families — a periodic calculation is still DFT — so
those rows live inside the method that owns them: periodic DFT (plane-wave, GPW)
is a section of the DFT chapter, periodic HF a section of the Hartree–Fock one.

| Chapter | Covers |
| ------- | ------ |
| [Hartree–Fock](hartree-fock.md) | RHF/UHF/ROHF/GHF, HF gradients and Hessians, periodic HF, SCF infrastructure |
| [Density-Functional Theory](dft.md) | Kohn–Sham methods, functionals (LDA → double hybrid), dispersion, **periodic DFT: plane-wave, GPW, forces, stress, phonons** |
| [Post-Hartree–Fock](post-hf.md) | Møller–Plesset, configuration interaction, coupled cluster, multireference |
| [Excited States and Many-Body](excited-states.md) | CIS/TDHF/TDDFT, ADC, EOM-CC, GW/BSE/RPA |
| [Semiempirical and Tight Binding](semiempirical.md) | NDDO/AM1, DFTB, extended tight binding (xTB) |
| [Force Fields](force-fields.md) | Bonded, nonbonded, electrostatics, polarizable, many-body/reactive, coarse-grained, machine-learning potentials |
| [Molecular Dynamics](molecular-dynamics.md) | Integrators, thermostats, barostats, ensembles, constraints, nonequilibrium MD, AIMD, nuclear quantum, nonadiabatic, vibrations |
| [Geometry Optimization](optimization.md) | Minimizers, transition states, reaction paths |
| [Sampling and Free Energy](sampling.md) | Monte Carlo, enhanced sampling, free-energy methods |
| [Multiscale and Properties](multiscale.md) | Atomic charges, implicit solvent, QM/MM, embedding, relativistic, intermolecular interactions |

## Campaign principle (CPU method-first)

As of **2026-07-23** on **`dev`**. Closed architecture: **backend-mirror**,
**cuda-resurrect**, **cuda-pme-accuracy**. Closed method campaign: **CPU next
A/B/C** (parallel worktrees `feat/cpu-next-{md,tb-arch,msamp,response}`,
merged and deleted). Active track remains **CPU-only method breadth** —
textbook cases that prove Driver / Component / Frame can host **electronic
structure through mesoscale**, not product depth in one family and not CUDA
ports.

Each shipped method is a public API + `regressions/` gate (≤ 60 s, literature
or analytic oracle). Prefer a **new family or scale** over another variant of
a proven shape.

### Already framework-proven

| Scale | Methods (do not re-prove) |
| ----- | ------------------------- |
| Electronic mean-field | RHF/UHF/ROHF/GHF, RKS/UKS/**ROKS** (+ **ROKS forces**); concrete AM1/RM1 / PW / GPW; **RHF/UHF CPHF Hessian** + **ROHF/RKS/UKS FD Hessian**; RHF CPHF polarizability |
| Electronic TB / response | RHF TDA/TDHF; `RKS::tda` (+ full MO-product LSDA ALDA + hybrid B3LYP/PBE0 exact-exchange kernel); **`UKS::tda`** (LSDA); **`UHF::sf_tda`**; **`RKS::tddft` full Casida A/B** (pure KS + hybrid B3LYP); **ADC(2)**; **G0W0**; **EOM-EE-CCSD**; **ΔSCF/MOM/IMOM**; analytic CIS/TDA forces; true NAC vectors; DFTB2 H/C/N/O SK; GFN0 toy; canonical + RI + SCS/SOS + **ROMP2** + **RMP3** + **CCSD** + **CCSD(T)** + **CASCI** + **RMP2 grad** `RHF::mp2` / `mp3` / `ccsd` / `ccsd_t` / `casci` / `ROHF::mp2` / `mp2_forces`; **B2PLYP** double hybrid |
| Classical FF / MD | LJ+bonded, FENE, WCA, Coul/PME, EAM/SW; NVE/NVT/NPT (MTK diag + **PR full-cell**); **NPH**; FIRE/LBFGS/CG; RATTLE/SETTLE; Brownian |
| Soft / meso | soft LJ/Coul, Yukawa, DPD (+ multi-type \(a_{ij}\)); λ TI; Widom; H-REMD λ |
| Paths / replicas | NEB, dimer, string, IRC, T-REMD, H-REMD, PIMD+Rg² |
| Sampling | MC NVT/NPT/μVT; Distance/Angle/Dihedral CV + umbrella; 1D + **multi-CV** well-tempered metaD; multi-window WHAM (molrs) |
| Cell | isotropic + joint + anisotropic + triclinic stress relax; PW Nielsen–Martin + GPW analytic fixed-\(P\)+Pulay stress (RKS Γ/k + Γ-UKS + multi-k UKS + Γ DFT+U + multi-k DFT+U SCF) |
| Multiscale | mechanical + electrostatic + **polarizable** QM/MM; **H link atoms**; C-PCM/COSMO/**IEF-PCM** |

### Closed queues (2026-07-23)

| Queue | Items |
| ----- | ----- |
| breadth-first 1–6 | FIRE, soft LJ, Widom, PIMD, isotropic cell, dimer |
| breadth-next 1–9 | soft Coulomb, AngleCV, FEP/TI, joint cell, string, IRC, RPMD Rg², SETTLE, Brownian |
| method ladder 1–8 | RKS-TDA, SCC-DFTB2, PW-cell, DihedralCV, FENE+WCA, DPD, Yukawa, H-REMD λ |
| open scales 1–8 | ALDA flag, GFN0, aniso cell, mech. QM/MM, metaD, multi-DPD, TDA forces, MP2 |
| depth-next 1–4 | ALDA diagonal, ES QM/MM, TDA NAC strength, triclinic cell |
| **cpu-next A (md)** | PR full-cell NPT, isotropic NPH |
| **cpu-next B (tb-arch)** | multi-element DFTB2 SK; concrete PW/GPW/AM1–RM1 (no public Config bags) |
| **cpu-next C (msamp)** | H link atoms, polarizable QM/MM, multi-CV metaD, multi-window WHAM |
| **cpu-next D (response)** | full MO-product ALDA, analytic TDA forces, true NAC vectors, RI-MP2 |

### Recommended next (still method-first, CPU)

| Theme | Candidates |
| ----- | ---------- |
| Response | Surface-hopping consumer of NAC; DIP/DEA/CVS-EOM (EOM-IP/EE/EA/SF + EOM-CC2 + UKS full-LR + UKS Hessian + ADC(2)-x/IP/EA + SF-TDDFT + dRPA/BSE/GF2 ✓) |
| TB product | Production GFN2 / 3ob tables (not toys); DFTB3; analytic DFTB forces |
| Cell / MD | Semi-isotropic / surface-tension PR; NPH anisotropic; CUDA PR/NPH parity |
| Multiscale | ONIOM / subtractive; adaptive QM/MM; long-range PME QM/MM |
| Sampling | OPES / multi-walker metaD; MBAR/PMF production curve; REST2 |
| Post-HF | PC/QD-NEVPT2 / CASPT2; density-based CC analytic grad; SA-CASSCF (SC-NEVPT2 + CASSCF + CCSD forces + UMP2 grad + RKS Hessian + MP4/CC2/CCSD(T)/CASCI/MP3/CCSD/ROMP2/RMP2 grad/SCS/SOS/frozen-core ✓) |

**Parked until a consumer asks:** CUDA method ports, HF/KS product depth
(hf-parity parking lot), ReaxFF/ML potentials, full GAFF polymer protocols,
Kokkos backend (WIP archived on `archive/impl-kokkos-wip`, not on `dev`).

## Work in progress — known gaps and caveats

The tables tell you *whether* a method exists. This section tells you what to
watch out for in the ones that do. Snapshot: **2026-07-23** on `dev`.

### Gaussian SCF

- **Cartesian is the default engine space**; a **spherical (pure) path exists**
  for sets defined that way (def2-SVP spherical regression
  `regressions/scf/h2o_rhf_def2svp_sph`). Native Rys ERI covers `l ≤ 3`
  (s/p/d/f).
- **RHF / UHF analytic nuclear Hessians** via CPHF (`RHF::hessian`,
  `UHF::hessian`; `reg_h2o_rhf_freq`, `reg_uhf-hessian-o2`). **ROHF nuclear
  Hessian** via FD of analytic forces (`ROHF::hessian`; `reg_rohf-hessian-o2`;
  not full CPHF). **RKS nuclear Hessian** via FD of analytic forces
  (`RKS::hessian`; `reg_rks-hessian-h2o-sto3g`; not full CPKS). Generic vib /
  thermochemistry still defaults to central FD of forces (`cpu::vib::Hessian`).
  **RHF static polarizability** via CPHF ships (`RHF::polarizability`;
  `reg_static-polariz-h2o`); open-shell / UKS polarizabilities and UKS
  analytic Hessian remain ○.
- **Open shell:** UHF is the default workhorse. **ROHF** is energy + **analytic
  gradient** + ROMP2 + FD Hessian (O₂ vs PySCF). **GHF** energy + analytic
  gradient (real two-component; `reg_h3_ghf_sto3g`, `reg_ghf-grad-h3-sto3g`).
  **ROKS** energy + **analytic gradient** (`reg_roks-o2-sto3g`,
  `reg_roks-grad-o2-sto3g`).
- **Shipped infra (CPU):** DIIS, level shift, SOSCF (hard cases), stability
  analysis, RI-J/RI-JK, CFMM-J, C-PCM / COSMO / **IEF-PCM** energy + C-PCM
  free-energy forces (`reg_h2o_rhf_631gs_cpcm`, `reg_ief-pcm-h2o`,
  `reg_pcm-grad-h2o`), point-charge embedding + external field forces, ECP
  (semi-numerical forces), x2c-1e energy (forces throw), counterpoise, RHF
  TDA/TDHF (`RHF::tda` / `RHF::tdhf`), APT/IR dipole derivatives. Hard gates /
  docs honesty for some of these are still in the `hf-parity` parking lot.
- **CUDA SCF is still a restricted-HF energy engine.** Device Fock/density/
  diagonalization; no DIIS, no UHF/KS, **no gradients** → no GPU AIMD.
  Device-native integrals are s-type only.
- **MINAO guess covers H, C, N, O, F only**; other elements fall back to Hcore.

### Post-Hartree–Fock

- **MP2** — closed-shell RMP2 on RHF; **UMP2** on UHF (`UHF::mp2`,
  `reg_ump2-o2-sto3g`); **ROMP2** on ROHF (`ROHF::mp2`, `reg_romp2-o2-sto3g`).
  Canonical path: full O(N⁵) AO→MO transform.
  **RI-MP2** when `ri = RI::{J,JK}` + aux basis on RHF only
  (`reg_h2o_ri_mp2`, `MP2Result::ri`). Frozen-core via `rhf.frozen=N`
  (`reg_frozen-core-mp2-h2o`). **SCS/SOS-MP2** via `mp2_os_scale` /
  `mp2_ss_scale` (`reg_scs-mp2-h2o-sto3g`, `reg_sos-mp2-h2o-sto3g`).
  **RMP2 analytic gradient** (`RHF::mp2_forces`; `reg_mp2-grad-h2o-sto3g`).
  **UMP2 analytic gradient** (`UHF::mp2_forces`; `reg_ump2-grad-o2-sto3g`).
  **RMP3 / RMP4(SDTQ)** (`RHF::mp3` / `mp4`; `reg_mp3-h2-sto3g`,
  `reg_mp4-h2-sto3g`). **CC2 / CCSD / CCSD(T)** (`RHF::cc2` / `ccsd` /
  `ccsd_t`). **CCSD nuclear forces** (`RHF::ccsd_forces`;
  `reg_ccsd-grad-h2o-sto3g` — FD of CCSD energy). **CASSCF**
  (`RHF::casscf`; `reg_casscf-h2-sto3g`). No ROMP2 gradients.
- **FCI is a reference tool**, not production: dense CI matrix, hard cap
  (`n_mo > 14` or `N_det > 2000`).

### Excited states

- **RHF TDA/TDHF** (`reg_ch2o_tdhf`); **CIS(D)** on RHF TDA (`reg_cisd-h2co-sto3g`);
  **`RKS::tda`** with **full MO-product LSDA ALDA** (`reg_h2_full_grid_alda`) and
  **hybrid TDDFT-TDA** (B3LYP exact-exchange kernel; `reg_h2-b3lyp-hybrid-tda`).
  **`UKS::tda`** spin-conserving LSDA (`reg_uks-tda-o2-lsda`). **`UHF::sf_tda`**
  collinear spin-flip TDA (`reg_spin-flip-tda-o2`); **`UHF::sf_tddft`** SF-RPA
  (`reg_sf-tddft-o2`). **Full LR-TDDFT** Casida A/B pure KS + hybrid B3LYP
  (`RKS::tddft`; `reg_full-lr-tddft-h2-pbe`, `reg_h2-b3lyp-rpa-tddft`) and
  **UKS full-LR** pure LSDA (`UKS::tddft`; `reg_uks-full-lr-o2-lsda`).
- **ADC(2)/ADC(2)-x / IP-ADC / EA-ADC** (`RHF::adc2` / `adc2x` / `ip_adc2` /
  `ea_adc2`). **EOM-EE/IP/EA-CCSD** + **EOM-CC2** on RHF; **EOM-SF-CCSD** on
  UHF. **G0W0 / dRPA / BSE / GF2** on RHF (`reg_g0w0-h2-sto3g`,
  `reg_direct-rpa-h2-sto3g`, `reg_bse-h2-sto3g`, `reg_gf2-h2-sto3g`).
- **Analytic CIS/TDA forces** (default) and **true NAC vectors**
  (`RHF::tda_forces` / `tda_nac_vectors` / `tda_nac_strength`;
  `reg_h2_analytic_tda_force`, `reg_h2_nac_vectors`). FD fallback remains.
  Surface hopping still ○ (no dynamics consumer of NAC).

### Exchange-correlation

- Molecular selectors: `{LSDA, PBE, PBEsol, BLYP, B3LYP, PBE0, B2PLYP}`. BP86 still ○.
- Molecular dispersion: **D3 zero** (GPW), **D3-BJ** (Gaussian + GPW) +
  **D4** (Gaussian + **GPW**) + **TS** free-atom on GPW (`reg_d3bj-molecular-pbe`,
  `reg_gpw-pbe-d3bj-ice`, `reg_dft-d4-h2o-pbe`, `reg_gpw-d4-water`,
  `reg_gpw-ts-ice`).
- libxc is an optional *test oracle* only (`ATV_BUILD_XC_ORACLE`).
- GPU XC outside plane waves: none (PW: Slater + PW92 LDA on device).

### Plane-wave DFT

- Analytic local + Ewald + KB forces; FD oracle remains.
  **Cell stress is Nielsen–Martin analytic** (`pw_stress`; FD oracle
  `pw_stress_fd`; `reg_pw-analytic-stress-si`).
- CPU: **true GGA PBE** on the FFT grid (`reg_pw-pbe-gga-si`), **collinear
  spin** incl. **true spin-GGA** (`reg_pw-spin-gga`) and **k-mesh UKS**
  (`reg_pw-spin-polarized-o2`, `reg_pw-uks-kmesh-fe`), **metallic smearing**
  (`reg_pw-smearing-al`, bulk Fe UKS+smear `reg_pw-uks-fe-smearing`),
  **bands/DOS** (`reg_bands-dos-si`) + **spin bands** (`reg_bands-spin-si`)
  + **metallic DOS** (`reg_dos-metallic-si`), **BOMD foothold**
  (`reg_pw-bomd-h`), **PW CELL_OPT** (`reg_pw-variable-cell-si`),
  **periodic MLWF** (`reg_periodic-wannier-si` + spin
  `reg_periodic-wannier-uks`), **Berry** (`reg_berry-polarization` + spin
  `reg_berry-spin-si`), **DFPT** Γ / finite-q / incommensurate /
  **GGA kernel** / **collinear spin** / **joint spin×GGA** /
  **finite-q×GGA** / **finite-q×spin×GGA** /
  **metallic smear** / spin-incommensurate / **finite-q+smear** /
  **spin+smear**
  (`reg_dfpt-phonon-si` / `reg_dfpt-finite-q-si` /
  `reg_dfpt-incommensurate-si` / `reg_dfpt-gga-si` / `reg_dfpt-spin-si` /
  `reg_dfpt-gga-spin-si` / `reg_dfpt-gga-finite-q-si` /
  `reg_dfpt-gga-spin-finite-q-si` /
  `reg_dfpt-metallic-si` / `reg_dfpt-spin-incommensurate` /
  `reg_dfpt-finite-q-smearing` / `reg_dfpt-spin-smearing`). CUDA PW remains
  LDA / spin-unpolarized.
- Norm-conserving UPF only (no USPP/PAW). Empty UPF = toy V_loc.
- CUDA: H·ψ / FFT / Poisson / V_eff / LDA on device; Ritz + SCF control on host.

### GPW

- Γ-point RKS + **Γ-point UKS** (PADE LSD `reg_gpw-uks-o2-gamma` + **PBE-GGA**
  `reg_gpw-uks-pbe` + **BLYP** `reg_gpw-uks-blyp` + **forces**
  `reg_gpw-uks-forces`) + **Monkhorst–Pack k-point RKS** and **k-point UKS**
  (phased lattice sums / KB / per-k DIIS; `reg_gpw-kpoints-si` /
  `reg_gpw-kpoints-uks`; **k-forces** RKS `reg_gpw-kpoint-forces` + UKS
  `reg_gpw-kpoint-uks-forces`). No hybrids / meta-GGA / GAPW. **Analytic
  cell stress** at Γ RKS/UKS, k-points RKS, **multi-k UKS**
  (`reg_gpw-kpoint-uks-stress`), and **DFT+U** (`reg_gpw-dftu-stress`)
  (fixed-\(P\)+Pulay product; multi-SCF FD oracle; `reg_gpw-analytic-stress` /
  `reg_gpw-uks-stress` / `reg_gpw-kpoint-stress` / `reg_gpw-stress-water`)
  and **GPW CELL_OPT** (`reg_gpw-variable-cell-ice` + MOLOPT
  `reg_gpw-variable-cell-molopt`). After harness wave 12–14 residual prose
  closes zero-SCF linearized XL (Γ RKS), multi-k SGCP, D3/D4 ATM energy,
  collinear spin+metallic bands/DOS, multi-k CPMD, GPW MLWF,
  **GPW FD phonons** (`reg_gpw-fd-phonon-si`), D3/D3(BJ) **forces**
  (`reg_gpw-d3-forces` / `reg_gpw-d3bj-forces`), and Γ MBD **stress**
  (`reg_gpw-mbd-stress`); **harness wave 1 (2026-07-31)** ships D3(BJ) +
  multi-k MBD stress (`reg_gpw-d3bj-stress` /
  `reg_gpw-mbd-kpoint-stress`), D3/D4 ATM forces
  (`reg_gpw-d3-atm-forces` / `reg_gpw-d4-atm-forces`), multi-k DFT+U
  forces (`reg_gpw-dftu-kpoint-forces`), joint DFPT GGA×spin
  (`reg_dfpt-gga-spin-si`), multi-k BOMD (`reg_gpw-bomd-kpoints`),
  multi-k multigrid (`reg_gpw-multigrid-kpoints`), and Γ-UKS linearized
  XL (`reg_xl-bomd-linearized-uks`); **shift wave 2 (2026-07-31)** ships
  D3-zero stress (`reg_gpw-d3-stress`), finite-q DFPT GGA
  (`reg_dfpt-gga-finite-q-si`), vdW-DF multigrid
  (`reg_gpw-vdw-multigrid`), multi-k linearized XL —
  RKS + UKS, CP2K-anchored (`reg_xl-bomd-linearized-kpoints`),
  multi-k UKS CPMD (`reg_cpmd-uks-kpoints`), multi-k SGCP OT
  (`reg_second-gen-cpmd-ot-kpoints`), and D3/D4 ATM stress
  (`reg_gpw-d3-atm-stress` / `reg_gpw-d4-atm-stress`); **wave 3
  (shift, 2026-07-31)** ships TS/XDM stress, multi-k + Γ-UKS nonlocal
  vdW, multi-k UKS multigrid, multi-k CPMD eNH, multi-k UKS SGCP OT,
  linearized-XL stress, and finite-q spin×GGA DFPT
  (`reg_gpw-ts-stress` / `reg_gpw-xdm-stress` / `reg_gpw-vdw-kpoints` /
  `reg_gpw-vdw-uks` / `reg_gpw-uks-multigrid-kpoints` /
  `reg_cpmd-thermostat-kpoints` /
  `reg_second-gen-cpmd-ot-uks-kpoints` /
  `reg_xl-bomd-linearized-stress` /
  `reg_dfpt-gga-spin-finite-q-si`);
  product after green wave-5 merges: multi-k UKS nonlocal vdW
  (`reg_gpw-vdw-uks-kpoints`), multi-k D3/D3(BJ)/ATM stress pins,
  multi-k UKS BOMD (`reg_gpw-bomd-uks-kpoints`), Γ-UKS vdW multigrid
  (`reg_gpw-vdw-uks-multigrid`); residual pending / incomplete: multi-k
  D4/TS/XDM stress dedicated pins (staff `[~]`), multi-k UKS Hirshfeld
  (`[~]`), multi-k UKS CPMD eNH combo pin, GPW spin MLWF, SCS-MP2 scale
  pin, Γ-UKS optB88-vdW;
  stock VV10 remains **blocked** (no CP2K twin); hybrids / meta-GGA /
  GAPW blocked — see `cp2k-method-queue.md` (queue not exhausted —
  GapCensus owns next discovery).
- Concrete method `GpwDft` (parameters as members; no public Config bag).
  Analytic Γ forces → AIMD (`reg_gpw-bomd-water`); keep member
  `local_pp_on_grid` consistent between energy and gradient
  (`reg_gpw-forces-water`, `reg_gpw-water-pbe-mt`). Product **OT** via
  `use_ot` (RKS `reg_ot-minimizer-water` + Γ-UKS `reg_ot-uks-o2` + multi-k
  RKS `reg_ot-kpoints` + multi-k UKS `reg_ot-uks-kpoints` + OT+DFT+U
  `reg_ot-dftu` / `reg_ot-dftu-kpoints` + OT+smear RKS `reg_ot-smearing` +
  UKS OT+smear `reg_ot-uks-smearing` + multi-k UKS OT+U
  `reg_ot-uks-dftu` + OT+U+smear `reg_ot-dftu-smearing` + multi-k UKS
  OT+U+smear `reg_ot-uks-dftu-smearing`). **UKS multigrid**
  (`reg_gpw-uks-multigrid`). **PBEsol** (`reg_gpw-pbesol-si`); **revPBE**
  RKS + UKS (`reg_gpw-revpbe-si` / `reg_gpw-uks-revpbe`); **RPBE** RKS +
  UKS (`reg_gpw-rpbe-water` / `reg_gpw-uks-rpbe`); **BLYP**
  (`reg_gpw-blyp-water`); **Dudarev DFT+U** (Γ energy/stress + multi-k
  SCF/stress: `reg_gpw-dftu-nio` / `reg_gpw-dftu-stress` /
  `reg_gpw-dftu-kpoints` / `reg_gpw-dftu-kpoint-stress`); **GPW metallic
  smear** RKS + UKS (`reg_gpw-metallic-smearing` / `reg_gpw-uks-smearing`).
  Γ + multi-k **RI-GPW MP2** + **UMP2@UKS** (Γ + multi-k:
  `reg_periodic-ri-mp2-si` / `reg_periodic-mp2-kpoints` /
  `reg_periodic-ump2-si` / `reg_periodic-ump2-kpoints`; Poisson foothold
  `reg_periodic-mp2-si`); Γ + multi-k + **finite-q** **dRPA** + **URPA@UKS**
  (Γ + multi-k: `reg_periodic-rpa-si` / `reg_periodic-rpa-kpoints` /
  `reg_periodic-rpa-finite-q` / `reg_periodic-urpa-si` /
  `reg_periodic-urpa-kpoints`); Γ + multi-k + **finite-q** spectral G0W0 +
  **UGW@UKS** (Γ + multi-k + RPBE: `reg_periodic-gw-si` /
  `reg_periodic-gw-kpoints` / `reg_periodic-gw-finite-q` /
  `reg_periodic-ugw-si` / `reg_periodic-ugw-kpoints` /
  `reg_ugw-uks-rpbe`).
- DFT-D3 zero-damping, **D3(BJ)**, **D4** (+ analytic stress + ionic
  forces), **TS**, free-atom **MBD** (+ Γ and multi-k ionic forces),
  **Hirshfeld-scaled MBD** (Γ + multi-k RKS + Γ-UKS), **periodic Ewald
  MBD**, **XDM**, **vdW-DF DRSLL**, **vdW-DF2 LMKLL**, **rVV10**,
  **optB88-vdW**, and **nonlocal SCF vdW pot/forces** on GPW (PBE params;
  `reg_gpw-pbe-d3bj-ice`, `reg_gpw-d4-water`, `reg_gpw-d4-stress`,
  `reg_gpw-d4-forces`, `reg_gpw-ts-ice`, `reg_gpw-ts-forces`,
  `reg_gpw-mbd-ice`, `reg_gpw-mbd-forces`, `reg_gpw-mbd-kpoint-forces`,
  `reg_gpw-mbd-hirshfeld`, `reg_gpw-hirshfeld-kpoints`,
  `reg_gpw-hirshfeld-uks`, `reg_gpw-mbd-periodic`, `reg_gpw-xdm-ice`,
  `reg_gpw-xdm-forces`, `reg_gpw-vdw-df-ice`, `reg_gpw-vdw-df2-ice`,
  `reg_gpw-rvv10-ice`, `reg_gpw-optb88-vdw`, `reg_gpw-vdw-scf-potential`).
  Molecular Gaussian path also has **DFT-D3(BJ)** / **D4**.

### Semiempirical

- **Concrete AM1 / RM1** (`cpu::semiempirical::AM1` / `RM1`) on the shared
  NDDO path with analytic forces → Evaluate / MD / Minimize. Hessian remains
  FD. AM1-BCC charges via molrs typing. `NDDOSolver` is internal.
- **SCC-DFTB2** multi-element H/C/N/O static SK (`DFTB2`,
  `reg_h2o_dftb2_mio`); forces are **central FD of E**, not analytic.
  **GFN0-xTB toy** (`XtbGfn0`) remains a separate TB family. Production
  GFN2 / 3ob / DFTB3 remain ○.

### Classical force fields

- **GAFF-capable on CPU**: multi-type LJ, CoulCut, CoulLong+PME, Amber
  specials (incl. 1–4 = 0.5), harmonic bond/angle, CHARMM + multi/harmonic
  dihedrals, harmonic improper. Gaps: multi-term Fourier dihedral as a
  first-class style, `.ff` text parser, CUDA Amber parity, paper-length
  polymer protocols.
- **Many-body:** EAM + Stillinger–Weber (CPU, MPI, ghost-of-ghost halo).
  No Tersoff/MEAM/ReaxFF/CUDA many-body.
- **Open-boundary FMM tree math** is validated (`reg_fmm_point_charges`) —
  not yet a MD `KSpace` / pair-style drop-in under PBC.
- **Constraints:** RATTLE + **SETTLE** (CPU; `reg_settle_tip3p`). No LINCS /
  general rigid bodies. **TIP4P M-site** virtual charges ship
  (`reg_four-site-tip4p-foothold`); general multi-topology virtual sites ○.
- **No machine-learning potentials** in-tree (see force-fields ML table).

### Molecular dynamics

- NVE / NVT / NPT on CPU and CUDA. Standalone `NoseHoover` is a **single**
  chain; M-chain NHC lives inside MTK NPT only. Bussi/CSVR is CPU-only.
- **Two NPT integrators on CPU:** diagonal MTK (`NoseHooverNPT`) and
  **full-cell Parrinello–Rahman** (`ParrinelloRahmanNPT`, `reg_pr_npt_lj`).
  **Isotropic NPH** ships as thermostat-free `NphIntegrator` (`reg_nph_lj`).
  **0 K cell optimizers:** isotropic, joint, anisotropic, **triclinic**.
- `Deform` uniaxial only. CUDA reciprocal space is single-GPU; CPU PME
  replicates the grid rather than distributing the FFT. CUDA PR/NPH not
  ported.
- **CUDA PME Madelung residual** closed 2026-07-23 (`cuda-pme-accuracy`).

### Sampling and paths

- MC: NVT / NPT / μVT GCMC on CPU (single rank; O(N) trial energy by design).
  No CBMC, Gibbs. **Widom** (`cpu::Widom`). Soft-core **λ TI** (`dU/dλ`).
- **Umbrella:** Distance / Angle / Dihedral CV + `HarmonicBias`. **1D
  well-tempered metadynamics** (`Metadynamics1D`) and **multi-CV**
  (`Metadynamics`, `reg_metad_2d`). **Multi-window WHAM** via molrs
  (`cpu::Wham`, `reg_wham_double_well`). OPES / multi-walker still ○.
- **REMD** (T ladder) + **H-REMD** (λ ladder). REST2 / multidimensional RE
  absent.
- **CI-NEB**, **dimer**, **string**, **IRC** ship
  (`reg_{neb,dimer,string,irc}_*`).
- **PIMD skeleton** ships (`cpu::path::Pimd` + ring springs +
  `IdentityReplication`, `reg_pimd_free_particle`). No RPMD / PILE /
  ring-polymer contraction product path.

### Optimization

- CPU: L-BFGS + Polak–Ribière⁺ CG + **FIRE** (`cpu::minimizer::FIRE`).
  CUDA: L-BFGS / CG / steepest descent (no FIRE).
- **Variable-cell:** isotropic, joint atom+cell, anisotropic, and
  triclinic (`*CellRelax` + `PressureConverged`).
- Cartesian only; no internal coordinates.
