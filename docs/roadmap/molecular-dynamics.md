# Molecular Dynamics

Legend and caveats: [Roadmap index](index.md).

The Dynamics driver (`cpu::md::MD`, `cuda::md::MD`) is the most complete part of
the engine: NVE/NVT/NPT all run on both backends, with MPI domain decomposition,
halo exchange, and migration on CPU, and multi-GPU support on CUDA. Constraints
(RATTLE) run on CPU under MPI. On **CPU**, full-cell **Parrinello–Rahman NPT**
and isotropic **NPH** also ship. Remaining gaps are specific rather than
structural — uniaxial-only `Deform`, no general rigid bodies, no CUDA PR/NPH.

## 1. Molecular-Dynamics Integrators

| Method                            | CPU | CUDA | Kokkos |
| --------------------------------- | :-: | :--: | :---: |
| Velocity Verlet                   |  ✓  |   ✓  |   ○  |
| Forest–Ruth integrator            |  ○  |   ○  |   ○  |
| Reversible RESPA                  |  ○  |   ○  |   ○  |
| Langevin dynamics                 |  ✓  |   ✓  |   ○  |
| BAOAB Langevin integrator         |  ○  |   ○  |   ○  |
| GJF Langevin integrator           |  ○  |   ○  |   ○  |
| Langevin-middle integrator        |  ○  |   ○  |   ○  |
| Brownian dynamics                 |  ✓  |   ○  |   ○  |
| Generalized Langevin equation     |  ○  |   ○  |   ○  |
| Rigid-body quaternion integration |  ○  |   ○  |   ○  |

Langevin dynamics is the Schneider–Stoll additive-force form (`FixLangevin`),
not one of the named splitting schemes. `NVELimit` (LAMMPS `nve/limit`, a
per-step displacement cap) is also available on both backends for relaxing
overlapping start configurations.

## 2. Thermostats

| Method                                   | CPU | CUDA | Kokkos |
| ---------------------------------------- | :-: | :--: | :---: |
| Langevin thermostat                      |  ✓  |   ✓  |   ○  |
| Nosé–Hoover thermostat                   |  ✓  |   ✓  |   ○  |
| Nosé–Hoover chain thermostat             |  ✓  |   ✓  |   ○  |
| Bussi stochastic velocity rescaling      |  ✓  |   ○  |   ○  |
| Gaussian isokinetic thermostat           |  ○  |   ○  |   ○  |
| Massive Nosé–Hoover chains               |  ○  |   ○  |   ○  |
| Dissipative particle dynamics thermostat |  ✓  |   ○  |   ○  |
| Dual thermostat for Drude particles      |  ○  |   ○  |   ○  |
| PILE thermostat                          |  ○  |   ○  |   ○  |
| PIGLET thermostat                        |  ○  |   ○  |   ○  |

Two caveats behind those ✓ marks:

- The standalone `NoseHoover` component is a **single-chain** thermostat (one
  scalar ξ) — a documented divergence from LAMMPS, which chains by default.
- The real **M-chain NHC** (default M = 3, with `drag`) exists only *inside* the
  MTK NPT integrator, not as a standalone NVT thermostat. Choosing NVT with
  chains today means running the NPT integrator, or accepting the single chain.

Bussi/CSVR (`FixCSVR`) is CPU-only.

## 3. Barostats

| Method                              | CPU | CUDA | Kokkos |
| ----------------------------------- | :-: | :--: | :---: |
| Berendsen barostat                  |  ✓  |   ✓  |   ○  |
| Parrinello–Rahman barostat          |  ✓  |   ○  |   ○  |
| Martyna–Tobias–Klein barostat       |  ✓  |   ✓  |   ○  |
| Langevin piston                     |  ○  |   ○  |   ○  |
| Monte Carlo isotropic barostat      |  ○  |   ○  |   ○  |
| Monte Carlo anisotropic barostat    |  ○  |   ○  |   ○  |
| Isotropic pressure coupling         |  ✓  |   ✓  |   ○  |
| Semi-isotropic pressure coupling    |  ✓  |   ✓  |   ○  |
| Fully anisotropic pressure coupling |  ✓  |   ○  |   ○  |
| Constant-surface-tension coupling   |  ✓  |   ○  |   ○  |
| Constant-stress dynamics            |  ✓  |   ○  |   ○  |

Two NPT integrators on CPU:

- **`NoseHooverNPT`** (MTK) couples the **cell diagonal** only: isotropic, or
  per-axis via `couple_axis[3]` (an axis left off is held rigid). No
  off-diagonal / shear DOFs. Also on CUDA.
- **`ParrinelloRahmanNPT`** is MTK with the **full cell matrix** (9 DOFs with
  constraints for volume/shape as configured) — the PR constant-stress path
  (`reg_pr_npt_lj`). CPU-only.

Berendsen remains isotropic only. **Constant surface tension** ships as
`cpu::md::SurfaceTensionNPT` (MTK semi-isotropic, LAMMPS `fix npt … couple xy`
lockstep + NPγT \(P_\perp = P_n - n_{\mathrm{surf}}\gamma/L_z\`); pin
`reg_surface-tension-coupling`.

## 4. Molecular-Dynamics Ensembles

| Method                              | CPU | CUDA | Kokkos |
| ----------------------------------- | :-: | :--: | :---: |
| Microcanonical dynamics, NVE        |  ✓  |   ✓  |   ○  |
| Canonical dynamics, NVT             |  ✓  |   ✓  |   ○  |
| Isothermal–isobaric dynamics, NPT   |  ✓  |   ✓  |   ○  |
| Isoenthalpic–isobaric dynamics, NPH |  ✓  |   ○  |   ○  |
| Constant surface-tension dynamics   |  ✓  |   ○  |   ○  |
| Constant stress dynamics            |  ✓  |   ○  |   ○  |
| Simulated annealing                 |  ○  |   ○  |   ○  |
| Temperature-ramp dynamics           |  ✓  |   ✓  |   ○  |
| Pressure-ramp dynamics              |  ○  |   ○  |   ○  |

**Isotropic NPH** ships as `NphIntegrator` — a thermostat-free barostat
integrator (`reg_nph_lj`). Anisotropic NPH and CUDA NPH remain ○. Constant-stress
dynamics is the PR-NPT path above.

Temperature ramps are supported by both the Langevin thermostat and the MTK NPT
integrator (`T_target` → `T_stop`). Simulated annealing is marked ○ because it
is a *protocol* over those ramps, not a component — you can express it today by
driving the ramp yourself, but the engine ships no annealing schedule.

## 5. Constraint and Rigid-Body Methods

| Method                       | CPU | CUDA | Kokkos |
| ---------------------------- | :-: | :--: | :---: |
| RATTLE                       |  ✓  |   ○  |   ○  |
| P-LINCS                      |  ○  |   ○  |   ○  |
| SETTLE                       |  ✓  |   ○  |   ○  |
| CCMA                         |  ○  |   ○  |   ○  |
| Distance constraints         |  ✓  |   ○  |   ○  |
| Angle constraints            |  ✓  |   ○  |   ○  |
| Rigid-water dynamics         |  ✓  |   ○  |   ○  |
| General rigid-body dynamics  |  ○  |   ○  |   ○  |
| Virtual-site dynamics        |  ✓  |   ○  |   ○  |
| Hydrogen-mass repartitioning |  ○  |   ○  |   ○  |

`cpu::md::FixRattle` (Andersen 1983) holds distance and 1–3 (angle) constraints, runs
under MPI, and throws rather than return a molecule that is merely *nearly* rigid. It
holds TIP3P geometry to 5 × 10⁻¹¹ relative over 5 ps at `dt = 2 fs`
(`regressions/md/rigid_water`) — that is the **rigid-water** path today (general
RATTLE, not a water-only closed form).

**SETTLE** ships as `cpu::md::FixSettle` (`reg_settle_tip3p`) — closed-form
three-site rigid water; general molecules still use RATTLE.

**Virtual sites:** (1) TIP4P foothold via `cpu::PairLJCutTip4pCut` (in-pair M-site
reconstruct + Feenstra partition; `reg_four-site-tip4p-foothold`); (2) multi-topology
engine `cpu::md::VirtualSites` with **Weighted3** (TIP4P M-site as Frame atoms) and
**OutOfPlane3** (TIP5P lone-pair geometry) + force redistribute
(`reg_virtual-site-multi-topology` vs LAMMPS tip4p path). Further Gromacs-style
`virtual_sitesn` / tabulated constructions remain open.

Before RATTLE, every molecule was fully flexible: water models had to be flexible
variants and the timestep was bounded by the fastest X–H stretch (~9 fs period for O–H,
so ~0.5 fs steps). Constrained water runs at `dt = 2 fs`. What is still missing for
biomolecular work is **general rigid bodies** (and remaining vsite constructors).

## 6. Nonequilibrium Molecular Dynamics

| Method                              | CPU | CUDA | Kokkos |
| ----------------------------------- | :-: | :--: | :---: |
| Constant electric-field MD          |  ✓  |   ○  |   ○  |
| Oscillating electric-field MD       |  ✓  |   ○  |   ○  |
| Constant-force pulling              |  ✓  |   ○  |   ○  |
| Constant-velocity pulling           |  ✓  |   ○  |   ○  |
| Steered molecular dynamics          |  ○  |   ○  |   ○  |
| SLLOD shear dynamics                |  ○  |   ○  |   ○  |
| Lees–Edwards shear dynamics         |  ○  |   ○  |   ○  |
| Uniaxial-deformation MD             |  ✓  |   ✓  |   ○  |
| Biaxial-deformation MD              |  ○  |   ○  |   ○  |
| Triaxial-deformation MD             |  ○  |   ○  |   ○  |
| Oscillatory-shear MD                |  ○  |   ○  |   ○  |
| Reverse nonequilibrium MD           |  ○  |   ○  |   ○  |
| Müller–Plathe heat-flux method      |  ○  |   ○  |   ○  |
| Homogeneous nonequilibrium MD       |  ○  |   ○  |   ○  |
| Temperature-gradient MD             |  ○  |   ○  |   ○  |
| Pressure-driven flow                |  ○  |   ○  |   ○  |
| Poiseuille-flow MD                  |  ○  |   ○  |   ○  |
| Shock-wave MD                       |  ○  |   ○  |   ○  |

`Deform` drives one axis at a time, with affine coordinate remap, in either
engineering strain (`erate`, `L = L₀(1 + Rt)`) or true strain (`trate`,
`L = L₀·exp(Rt)`). The CUDA port supports `erate` only. This is enough for a
uniaxial tension/compression run (see `/* benchmarks removed; re-add later */ pe/`, polyethylene under
tension) and nothing more — no shear, no flow, no heat flux.

**Constant electric field** ships as `cpu::md::FixEfield` (LAMMPS `fix efield`
constant path): \(\mathbf{F}_i = q_i\mathbf{E}\), energy
\(U=-\sum_i q_i\mathbf{E}\cdot\mathbf{r}_i^{\mathrm{u}}\) (`reg_external-efield-md`).
**Oscillating / AC field** is `cpu::md::FixOscillatingEfield` (cosine
\(E(t)\); `reg_oscillating-efield-md`). **Constant-force pull** is
`cpu::md::FixAddForce` (LAMMPS `fix addforce`; `reg_constant-force-pull-ion`).
**Constant-velocity SMD** is `cpu::md::FixMove` (LAMMPS `fix move`;
`reg_constant-velocity-smd`). **Displacement \(\mathbf{D}\)-field** is
`cpu::md::FixDfield` (`reg_displacement-dfield-md`). Classical electrochemistry
Components on `cpu::md::MD` (MetalWalls map E1–E10; waves 1–5 done — see
`docs/developer/metalwalls-method-queue.md`; residual pending wave-6+:
nanoporous carbon product, molten-salt+Al+MaZe scenario, TI capacitance):

- **E1:** `ConstPotBO`, `FixFreeze`, charge observer, charging work, short
  const-pot NVT; wave-4: `ConstPotMaZe` (CG+ASPC adiabatic \(Q\)),
  `write_electrode_forces`, `global_electroneutrality`; wave-5: per-electrode
  \(\eta_e\) (`add_electrode(type, psi, eta)`, `reg_per-electrode-eta`),
  const-charge control (`add_electrode_charge`, LAMMPS `electrode/conq` twin,
  `reg_const-charge-electrode`), post-solve ghost-charge refresh under MPI DD
  (`reg_const-pot-charge-halo-mpi`), symplectic mass-zero MaZe
  (`ConstPotMaZe.symplectic`, `reg_maze-symplectic-charge-dynamics`)
- **E2:** `ConstPotCG` (large multi-site walls); Kolafa ASPC on `ConstPotBO`;
  `PairGaussianCoul` NeighborList bulk path (`reg_gaussian-coul-neighborlist`)
- **E3:** single-electrode finite-field (Dufils); AC `FixOscillatingEfield`;
  `FixDfield` displacement field, incl. self-consistent constant-\(D\)
  ensemble \(E = D - 4\pi k_e P\) (`polarization_feedback`,
  `reg_constant-d-polarization-feedback`, wave-5)
- **E5:** `DipoleElectrodeJoint` — coupled PIM \(\boldsymbol{\mu}\) ⊕ electrode
  \(Q\) SCF (`reg_dipole-electrode-joint`)
- **E6:** `PistonElectrode` constant \(P_z\) rigid wall (not Langevin piston)
- **E7:** Thomas–Fermi finite metallicity (`ConstPotBO::l_tf`, `site_density`);
  `VoronoiVolume` site-volume companion (`reg_voronoi-volume-tf`, wave-5)
- **E9:** multi-topology virtual sites (TIP4P/TIP5P); const-pot + TIP4P EDL
  foothold (`reg_const-pot-tip4p-edl`); wave-5: planar const-pot + IL EDL
  production box (`reg_const-pot-il-edl-box`), PIM-NPT polarizable IL twin
  (`reg_pim-npt-polarizable-il`). Still open: nanoporous carbon product
- **E10:** `CapacitanceChargeFluctuation`; slab \(\Psi(z)\) `PotentialProfile`;
  wave-4: `ChargeAutocorr` (\(C_{QQ}\)), `ElastanceObserver` (\(\mathrm{cond}(A)\))
- **E8 walls:** `FixWallLJ1043` Steele 10-4-3 (force-fields roadmap note)


## 7. Ab Initio Molecular Dynamics

| Method                                  | CPU | CUDA | Kokkos |
| --------------------------------------- | :-: | :--: | :---: |
| Hartree–Fock Born–Oppenheimer MD        |  ✓  |   ○  |   ○  |
| Kohn–Sham Born–Oppenheimer MD           |  ✓  |   ○  |   ○  |
| Plane-wave Born–Oppenheimer MD          |  ✓  |   ○  |   ○  |
| Semiempirical Born–Oppenheimer MD       |  ✓  |   ○  |   ○  |
| DFTB molecular dynamics                 |  ○  |   ○  |   ○  |
| xTB molecular dynamics                  |  ○  |   ○  |   ○  |
| Car–Parrinello molecular dynamics       |  ✓  |   ○  |   ○  |
| Extended-Lagrangian Born–Oppenheimer MD |  ✓  |   ○  |   ○  |
| Second-generation Car–Parrinello MD     |  ✓  |   ○  |   ○  |

AIMD works by binding a force-providing method/component to `md::Phase::force`
(energy/force kinds: `HF`, `PlaneWave`, `GPW`, `Semiempirical`, `Potential`, …):

- Concrete `RHF`/`UHF`/`RKS`/`UKS` (or nested SCF) with gradients → **HF and KS
  BOMD**, with an ASPC density predictor for warm starts.
- `cpu::gpw::GPWEvaluator` with forces on → **periodic KS BOMD** (Γ-point,
  closed-shell); NVE energy-drift pin `reg_gpw-bomd-water`. **Multi-k
  BOMD** is product (`reg_gpw-bomd-kpoints`; previous-step \(P_k\) SCF
  warm start mirroring CP2K EXTRAPOLATION USE_PREV_P; CP2K Quickstep
  AIMD k-mesh pin, fair wall ratio 0.064; multi-k XL is product too;
  multi-k UKS BOMD is product too — dual-spin previous-step
  \(P_k^\sigma\) warm start, `reg_gpw-bomd-uks-kpoints`). Multi-k free-energy
  **forces under smear** are product (`reg_gpw-metallic-forces` /
  `reg_gpw-uks-metallic-forces`). Multi-k free-energy **stress under
  smear** is incomplete on `wf/gpw-metallic-stress` (`[~]`); metallic
  occupation-smear BOMD (free-energy / Mermin trajectory) remains queued
  as `gpw-bomd-smearing`.
- `cpu::pw::PWEvaluator` with `with_gradient(true)` → **plane-wave BOMD**
  (analytic local + Ewald + KB forces; default is energy-only); foothold
  `reg_pw-bomd-h`.
- `cpu::semiempirical::AM1Evaluator` with `with_gradient(true)` (default) writes
  nuclear forces (eV/Å). Semiempirical BOMD is pinned by
  `regressions/semiempirical/am1_water_bomd`.
- CUDA SCF has no gradient kernel — **there is no GPU AIMD**.

**XL-BOMD** (Niklasson dissipative auxiliary density) ships as
`cpu::electronic::XlBomd` + `GPWEvaluator::{xl_bomd, xl_bomd_k}` on closed-shell
GPW (`reg_xl-bomd-water`) **and Γ-UKS** dual-spin auxiliary densities
(`reg_xl-bomd-uks`) **and multi-k RKS** (`reg_xl-bomd-kpoints`) **and
multi-k UKS** dual-spin \(P_k^\sigma\) (`reg_xl-bomd-uks-kpoints`). ASPC
remains a separate density *predictor* (warm start); OT+XL stays
architecturally excluded (OT parameterizes orbitals, not density);
**zero-SCF linearized-functional forces** productize on **Γ RKS**
(`xl_bomd_linearized`; `reg_xl-bomd-linearized-forces`), **Γ-UKS**
dual-spin shadow densities (`reg_xl-bomd-linearized-uks`), **and the
multi-k path** — RKS and collinear UKS, k-weighted complex shadow
\(D_k^\sigma\) with the new k-UKS fixed-density eval
(`reg_xl-bomd-linearized-kpoints`, CP2K-anchored per-step trace);
linearized XL + analytic fixed-\(P\) **stress** productizes
(`reg_xl-bomd-linearized-stress`; multi-SCF strain-FD oracle still
refused — BO stress, not the zero-SCF shadow functional).

**Car–Parrinello MD** (fictitious electronic DOF) ships as `cpu::electronic::Cpmd` +
`GPWEvaluator::{cpmd, cpmd_mu}` (the nuclear dt is pushed in from the MD
integrator through `TimestepProvider`/`TimestepConsumer`, never re-declared on
the evaluator) on closed-shell Γ GPW: cold SCF seed,
then fixed-density KS eval + electronic Verlet + analytic forces at that
density (`reg_cpmd-water`) **and Γ-UKS** (`reg_cpmd-uks`). Conserved
\(E_{\mathrm{cons}} = T_n + T_e + E_{\mathrm{KS}}\). **Electronic RATTLE**
(Andersen constraint-preserving orbitals; `reg_cpmd-rattle-water`) is product
on the closed-shell path. **Electronic thermostat** (Nosé–Hoover on fictitious
electronic DOF) productizes at Γ (`reg_cpmd-electronic-thermostat`) **and on
the multi-k path** (one thermostatted `CpmdCplx` per (k, σ) channel;
k-weighted \(T_e\) / \(E_{\mathrm{therm}}\`;
`reg_cpmd-thermostat-kpoints`) **and multi-k UKS** (dual-spin eNH combo
pin; `reg_cpmd-uks-thermostat-kpoints`). **Multi-k
CPMD** (Bloch \(C_k\), one `cpu::electronic::CpmdCplx` per sampled k) ships
for RKS (`reg_cpmd-kpoints`) **and collinear UKS** (dual-spin Bloch
\(C_k^\sigma\), one propagator per (k, spin) with occ-1 spin-diagonal EOM
on \(H_k^\sigma\) from the dual-spin fixed-density multi-k eval;
`reg_cpmd-uks-kpoints`).

**Second-generation Car–Parrinello MD** (Kühne et al. 2007; ASPC predictor +
partial-SCF corrector, no fictitious electronic mass) ships on closed-shell Γ
GPW via `GPWEvaluator::{second_gen_cp, second_gen_cp_k, second_gen_cp_max_iter}`
(`reg_second-gen-cpmd-water`) **and Γ-UKS** (`reg_second-gen-cpmd-uks`).
**RKS OT corrector** is product (`use_ot=true` with ASPC natural-orbital
warm-start; `reg_second-gen-cpmd-ot`) **and Γ-UKS OT + SGCP**
(`reg_second-gen-cpmd-uks-ot`) **and multi-k SGCP** (ASPC on complex
\(P_k\); `reg_second-gen-cpmd-kpoints`) **and multi-k SGCP + OT corrector**
(per-k OT manifold warm-started from natural orbitals of the predicted
\(P_k\); `reg_second-gen-cpmd-ot-kpoints`) **and multi-k UKS SGCP + OT**
(dual-spin per-(k, σ) OT manifolds warm-started from natural orbitals of
the predicted \(P_k^\sigma\); `reg_second-gen-cpmd-ot-uks-kpoints`).

## 8. Nuclear Quantum Dynamics

| Method                                 | CPU | CUDA | Kokkos |
| -------------------------------------- | :-: | :--: | :---: |
| Path-integral molecular dynamics       |  ✓  |   ○  |   ○  |
| Ring-polymer molecular dynamics        |  ○  |   ○  |   ○  |
| Centroid molecular dynamics            |  ○  |   ○  |   ○  |
| Thermostatted ring-polymer MD          |  ○  |   ○  |   ○  |
| Normal-mode PIMD                       |  ○  |   ○  |   ○  |
| Staging-coordinate PIMD                |  ○  |   ○  |   ○  |
| Ring-polymer contraction               |  ○  |   ○  |   ○  |
| Quantum thermal bath                   |  ○  |   ○  |   ○  |
| Path-integral free-energy perturbation |  ○  |   ○  |   ○  |

## 9. Nonadiabatic Dynamics

| Method                                | CPU | CUDA | Kokkos |
| ------------------------------------- | :-: | :--: | :---: |
| Ehrenfest dynamics                    |  ○  |   ○  |   ○  |
| Fewest-switches surface hopping       |  ○  |   ○  |   ○  |
| Landau–Zener surface hopping          |  ○  |   ○  |   ○  |
| Decoherence-corrected surface hopping |  ○  |   ○  |   ○  |
| Spin-mixed surface hopping            |  ○  |   ○  |   ○  |
| Mapping-variable dynamics             |  ○  |   ○  |   ○  |
| Ab initio multiple spawning           |  ○  |   ○  |   ○  |
| Diabatic-state dynamics               |  ○  |   ○  |   ○  |

Nonadiabatic dynamics needs a dynamics consumer of NAC / excited forces.
CIS/TDA forces and true NAC vectors now ship (see
[Excited States and Many-Body](excited-states.md)); FSSH / Ehrenfest loops
themselves remain ○.

## 10. Vibrational Analysis

Not part of the reference taxonomy, but it is what the Response driver does, and
it is what stands in for every "analytic Hessian" row in this roadmap.

| Method                                    | CPU | CUDA | Kokkos |
| ----------------------------------------- | :-: | :--: | :---: |
| Finite-difference Hessian (central, 6N)   |  ✓  |   ○  |   ○  |
| Mass-weighted normal modes / frequencies  |  ✓  |   ○  |   ○  |
| Harmonic thermochemistry (RRHO ideal gas) |  ✓  |   ○  |   ○  |
| Analytic Hessian (any method)             |  ○  |   ○  |   ○  |

`cpu::vib::Hessian` displaces each Cartesian degree of freedom and differences
the analytic forces, so it works with any force-providing evaluator — classical
or SCF. It is molecular: there is no supercell/q-point phonon machinery.
