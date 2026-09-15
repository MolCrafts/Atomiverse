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
| ◐      | Partial on that backend                 |
| ✓      | Implemented and numerically correct     |
| ⚡      | Numerically correct and fully optimized |

### What this roadmap does not list

A method that a **strictly better algorithm dominates** — same purpose, same or lower
cost, no regime where the older one wins — is simply **not in these tables**. It is not
marked, not tracked, and not planned. Listing it, even as "will not implement", would
suggest the roadmap has an opinion worth reading about it; it does not.

So there is no BFGS row (L-BFGS matches its convergence at O(mN) memory instead of
O(N²)) and no Fletcher–Reeves row (Polak–Ribière⁺ restarts strictly better at the same
cost).

Dominance is about what the roadmap *recommends*, not about what the code contains.
`cpu::md::Berendsen` ships and has no row: CSVR is the same weak-coupling idea done
correctly, so nothing here will ever suggest Berendsen for sampling — but a published
protocol that specifies Berendsen has to be reproduced with Berendsen, and that is a
different job from sampling. Where a dominated method exists for exactly that reason,
its docstring says so.

This is only about *dominance*, never about our schedule. A method that survives because
it occupies a genuinely different cost/accuracy niche stays ○ and stays listed — steepest
descent is dominated by CG for minimization proper, but it is the robust first resort on a
badly broken start geometry, so it keeps its row. Brownian dynamics keeps its row too: it
is different physics from Langevin, not a worse integrator.

### How the symbols are assigned here

**✓** means the method runs from the public API and its numbers are pinned by
an end-to-end case under `regressions/<domain>/<case>/`, checked against an
external reference (PySCF, CP2K, DFTB+, xtb, Quantum ESPRESSO, LAMMPS, MetalWalls,
NIST) or against an analytic result. Each case is an out-of-library consumer that
links only the public headers, runs in ≤ 60 s, and cites the number its oracle
printed. A method that only has unit tests behind it is not ✓.

**⚡** is deliberately scarce. It is reserved for kernels that are both correct
*and* carry a standing performance harness measured against a production engine
on named hardware. Today that is exactly one method family: the Lennard–Jones
12-6 pair style — CPU (xsimd, `-march=native`) and the CUDA cluster/tile engine,
benchmarked against LAMMPS/Kokkos (`benchmarks/lj/`, `benchmarks/cuda/`).
Everything else that is correct but has no standing performance gate is a ✓,
including code that is fast in practice. No Kokkos ⚡ yet (backend ○).

**◐** means partial: an implementation exists on that backend, but the
domain's product depth or independent anchor is incomplete — a foothold, a
restricted sub-case, or a pin that is still self-consistency rather than an
external number. A row never jumps ○→✓ while so marked; the chapter prose
states what is missing.

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
| [Electrodes and Electrochemistry](electrodes.md) | Constant-potential / constant-charge electrodes, MaZe, Thomas–Fermi metallicity, piston, polarizable coupling, electrode observers |
| [Geometry Optimization](optimization.md) | Minimizers, transition states, reaction paths |
| [Sampling and Free Energy](sampling.md) | Monte Carlo, enhanced sampling, free-energy methods |
| [Multiscale and Properties](multiscale.md) | Atomic charges, implicit solvent, QM/MM, embedding, relativistic, intermolecular interactions |
| [Observables and Analysis](observables.md) | The `cpu/observe/` census: trajectory metrics, densities, stress, population/ESP charges, electrode observers, molrs bridges |

## Work in progress — known gaps and caveats

The tables tell you *whether* a method exists. This section tells you what to
watch out for in the ones that do — restricted sub-cases, pins that are analytic
rather than external, and the places where one backend is behind the other.
Snapshot: **2026-09-15** on `dev`.

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
- **CUDA SCF is still a restricted-HF engine.** Device Fock/density/
  diagonalization and CDIIS; no UHF/KS. Device-native integrals **and
  analytic forces** cover STO-3G s/p (l ≤ 1, H…O). GPU BOMD exists as
  `cuda::md::MD` + `cuda::scf::RHF` (`reg_h2_rhf_sto3g_aimd`, GH200
  **1332540**). Out-of-coverage `forces()` throws — use `cpu::scf::RHF`.
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
  **spin+smear** / **finite-q×spin×GGA×smear**
  (`reg_dfpt-phonon-si` / `reg_dfpt-finite-q-si` /
  `reg_dfpt-incommensurate-si` / `reg_dfpt-gga-si` / `reg_dfpt-spin-si` /
  `reg_dfpt-gga-spin-si` / `reg_dfpt-gga-spin-smearing` /
  `reg_dfpt-gga-finite-q-si` / `reg_dfpt-gga-spin-finite-q-si` /
  `reg_dfpt-gga-spin-finite-q-smearing` /
  `reg_dfpt-metallic-si` / `reg_dfpt-spin-incommensurate` /
  `reg_dfpt-finite-q-smearing` / `reg_dfpt-spin-smearing`). CUDA PW remains
  LDA / spin-unpolarized.
- Norm-conserving UPF only (no USPP/PAW), or the in-tree analytic GTH tables
  (`gth_name`). Exactly one of the two, single-species; neither/both throws —
  there is no toy V_loc fallback any more.
- CUDA: H·ψ / FFT / Poisson / V_eff / LDA on device; Ritz + SCF control on host.

### GPW

- **SCF flavours.** Restricted (RKS) and collinear-spin (UKS) Kohn–Sham, at Γ
  and on a Monkhorst–Pack k-mesh (phased lattice sums, phased KB projectors,
  per-k DIIS). Diagonalization or the **orbital-transformation (OT)** minimizer,
  each combinable with DFT+U and with metallic smearing, on either spin channel
  and at either k-sampling (`reg_gpw-kpoints-si`, `reg_gpw-kpoints-uks`,
  `reg_ot-minimizer-water`, `reg_ot-uks-kpoints`, `reg_ot-uks-dftu-smearing`).
- **Functionals.** PADE/LSD, PBE, PBEsol, revPBE, RPBE, BLYP, optB88 — on both
  spin channels. **No hybrids, no meta-GGA, no GAPW**: those are blocked, not
  scheduled, because there is no CP2K-comparable path in tree to anchor them.
- **Derivatives.** Analytic ionic forces and analytic cell stress
  (fixed-\(P\) + Pulay product, multi-SCF FD oracle) across the whole
  RKS/UKS × Γ/multi-k × DFT+U × smearing grid, plus **CELL_OPT**
  (`reg_gpw-variable-cell-ice`, MOLOPT `reg_gpw-variable-cell-molopt`) and **FD
  phonons** (`reg_gpw-fd-phonon-si`).
- **Dispersion and vdW.** D3-zero, D3(BJ), D4 — each with ATM three-body,
  forces and stress, at Γ and multi-k; TS and free-atom **MBD** including
  Hirshfeld-scaled and periodic-Ewald variants; **XDM**; the nonlocal
  functionals **vdW-DF (DRSLL)**, **vdW-DF2 (LMKLL)**, **rVV10** and
  **optB88-vdW**, with a self-consistent nonlocal potential and its forces and
  stress. Stock **VV10 is blocked** — no CP2K twin to pin it against.
- **Electronic structure output.** Bands and DOS including the spin and
  metallic cases (`reg_gpw-bands-si` / `-uks` / `-metallic`, `reg_gpw-dos-*`),
  Berry-phase polarization (RKS and UKS), and maximally localized Wannier
  functions (RKS and UKS).
- **Ab-initio dynamics.** Born–Oppenheimer MD including the multi-k, UKS and
  metallic-smearing cases; Car–Parrinello MD with an electronic Nosé–Hoover
  thermostat and at multiple k-points; second-generation CPMD (SGCP) including
  the OT and UKS variants; XL-BOMD including its linearized form with forces
  and stress.
- **Periodic correlated methods.** RI-MP2 and UMP2, SCS-MP2 and SCS-UMP2,
  direct RPA and URPA, and spectral G0W0 and UGW — each at Γ, on a k-mesh, and
  (for RPA and GW) at finite q.

- **Public shape.** One concrete class, `cpu::gpw::GpwDft`, carrying its
  parameters as members — no Config bag, no factory. Everything above is
  selected on that object (`use_ot`, DFT+U, smearing, XC, `k_mesh`, the
  dispersion selector). `GpwDft::run` takes a `std::vector<GpwAtom>` in **Bohr**
  with the basis and GTH pseudopotential already resolved per atom, so the
  Frame-shaped entry point users should bind is `cpu::gpw::GPWEvaluator`, which
  owns the solver and does that translation. (That `GpwDft` does not take a
  Frame is standing debt against the Frame-only rule, not a design intent.)
  One operational trap: keep the member `local_pp_on_grid` consistent between
  an energy call and a gradient call, or forces will not match the energy
  surface they claim to differentiate (`reg_gpw-forces-water`,
  `reg_gpw-water-pbe-mt`).
- **CUDA GPW** is kernel footholds only — device collocation, integration,
  H assembly, DIIS/eigen and forces exist and are gated (`reg_cuda-gpw-*`),
  but there is no CUDA GPW *method* product: the public `GpwDft` is CPU.

### Semiempirical

- **NDDO.** Concrete `cpu::semiempirical::AM1` / `RM1` on the shared NDDO path,
  with analytic nuclear forces, so they plug into Evaluate / MD / Minimize like
  any other energy source. Hessian is FD of those forces. AM1-BCC charges via
  molrs typing. `NDDOSolver` stays internal.
- **DFTB.** `DFTB0` (non-SCC), `DFTB2` (SCC) and `DFTB3` (third-order), on
  static in-tree Slater–Koster tables for H/C/N/O — **mio-1-1** and
  **3ob-3-1**, both transcribed with provenance. DFTB2 carries analytic
  nuclear forces *and* analytic intensive cell stress for closed shell and
  collinear open shell, at Γ and on a Monkhorst–Pack k-mesh (the 3ob path is
  molecular/Γ only — it throws on `k_mesh ≠ {1,1,1}` rather than silently
  falling back). Also: Ewald-summed SCC under PBC, D3 dispersion (molecular and
  periodic), DFTB+U, SCC charge guessing via ASPC, and linear-response
  **TD-DFTB** (Casida) on a closed-shell DFTB2 reference.
- **Extended tight binding.** `GFN0`, `GFN1`, `GFN2` and `IPEA`,
  each with analytic nuclear forces and an FD Hessian, pinned against **xtb
  6.7.1**. GFN2 includes the charge–quadrupole anisotropic electrostatics
  (AES) and the charge-kernel anisotropic XC term; IPEA-xTB adds ΔSCC vertical
  IP/EA.
- **What is missing.** Analytic Hessians anywhere in this chapter (all FD);
  elements beyond H/C/N/O in the shipped SK and xTB parameter tables; any CUDA
  path — the whole semiempirical domain is CPU-only.

### Classical force fields

- **GAFF-capable on CPU**: multi-type LJ, CoulCut, CoulLong+PME, Amber
  specials (incl. 1–4 = 0.5), harmonic bond/angle, CHARMM + multi/harmonic
  dihedrals, harmonic improper. Gaps: multi-term Fourier dihedral as a
  first-class style, `.ff` text parser, CUDA Amber parity, paper-length
  polymer protocols.
- **Many-body:** EAM + Stillinger–Weber (CPU, MPI, ghost-of-ghost halo).
  No Tersoff/MEAM/ReaxFF/CUDA many-body.
- **Beyond LJ:** Morse, Buckingham, Born–Mayer–Huggins (incl. Fumi–Tosi molten
  salt), Yukawa, WCA, FENE, Gaussian-charge Coulomb, DPD with multi-type
  \(a_{ij}\), soft-core LJ/Coulomb, and a Steele 10-4-3 wall.
- **Polarizable:** point-induced dipoles (PIM) with an extended-Lagrangian
  propagator, and QEq fluctuating charges (`reg_pim-induced-dipole-bulk`,
  `reg_pim-npt-polarizable-il`, `reg_qeq-fluctuating-charge`).
- **Open-boundary FMM tree math** is validated (`reg_fmm_point_charges`) —
  not yet a MD `KSpace` / pair-style drop-in under PBC.
- **Constraints:** RATTLE + **SETTLE** (CPU; `reg_settle_tip3p`). No LINCS /
  general rigid bodies. **Virtual sites** are general, not a TIP4P special case:
  `cpu::md::VirtualSites` reconstructs massless sites from parents and
  redistributes their forces (Feenstra/GROMACS-style), covering TIP4P
  `Weighted3` and TIP5P `OutOfPlane3` with ordinary pair styles
  (`reg_four-site-tip4p-foothold`, `reg_virtual-site-multi-topology`).
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
- **Thermostats:** Nosé–Hoover, Langevin, Bussi/CSVR, and **Berendsen** —
  the last for reproducing protocols that name it, never for sampling (see
  above). **Surface-tension NPT** couples the lateral and normal cell axes
  independently (`reg_surface-tension-coupling`).
- **Non-equilibrium drives:** constant force, constant-velocity SMD, static and
  oscillating external E-fields, D-field control, moving and frozen groups.

### Electrodes and electrochemistry

- **Constant potential** via Born–Oppenheimer charge solve or conjugate
  gradient, **constant charge**, and the **MaZe** mass-zero constrained
  formulation — with per-electrode η, Thomas–Fermi metallicity, global
  electroneutrality, finite-field single-electrode, and a piston electrode.
- **Observers** for charge autocorrelation, capacitance from charge
  fluctuation, elastance, potential profiles, charging work, and Voronoi
  volumes. Pinned against **MetalWalls**.
- CPU only, and the constant-potential charge halo is MPI-parallel
  (`reg_const-pot-charge-halo-mpi`).

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
