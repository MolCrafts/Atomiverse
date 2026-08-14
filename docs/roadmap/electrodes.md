# Electrodes and Classical Electrochemistry

Legend and caveats: [Roadmap index](index.md).

Constant-potential electrodes are **Components on `cpu::md::MD`** — not a
driver category, not a `*Config` bag (MetalWalls capability map, slices
E1–E10; `.claude/notes/metalwalls-capability-map.md`). The charge solvers and
field fixes live in `cpu/dynamics/` (`const_pot_bo.h`, `const_pot_cg.h`,
`const_pot_maze.h`, `dipole_electrode_joint.h`, `piston_electrode.h`,
`fix_dfield.h`, `qeq.h`); the dedicated `electrodes/` directory holds the
support pieces (`include/atomiverse/cpu/electrodes/charge_aspc.h`,
`charging_work.h` + `src/cpu/electrodes/charge_aspc.cpp`,
`charging_work.cpp`); the analysis side sits in `cpu/observe/`
(`electrode_charge.h`, `capacitance_charge_fluctuation.h`,
`potential_profile.h`, `charge_autocorr.h`, `elastance_observer.h`,
`voronoi_volume.h`, `classical_polarization.h`). Electrode membership is
declared by atom type on the Frame (`add_electrode(type, psi)` names the
group); solved charges write `frame.span(kAtom_charge)`, with a post-solve
ghost-charge refresh under MPI domain decomposition
(`reg_const-pot-charge-halo-mpi` — serial↔MPI equivalence invariant, no
external binary in the loop).

The whole domain is CPU-only by layout (`cuda/` has no `electrodes/` and no
constant-potential dynamics) — the CUDA and Kokkos columns are ○ everywhere.
Anchors are ~35 cases under `regressions/md/`, pinned against the **real
MetalWalls binary** (git `c8ad175d`, release branch), **LAMMPS ELECTRODE**
(Ahrens-Iwers et al., JCP 157, 084801 (2022)), or a cited literature closed
form; each case names its oracle in `REFERENCE.md`.

## 1. Constant-Potential Charge Solvers

| Capability                                    | CPU | CUDA | Kokkos | Anchor (regressions/md/) |
| --------------------------------------------- | :-: | :--: | :---: | ------------------------ |
| Constant-potential BO solve (Cholesky)        |  ✓  |   ○  |   ○  | `const-pot-bo-parallel-plate` (LAMMPS ELECTRODE madelung) |
| Constant-potential CG solve                   |  ✓  |   ○  |   ○  | `const-pot-cg-large-electrode` (MetalWalls) |
| MaZe mass-zero charge dynamics, symplectic    |  ✓  |   ○  |   ○  | `maze-symplectic-charge-dynamics` (MetalWalls) |
| MaZe adiabatic CG+ASPC twin                   |  ✓  |   ○  |   ○  | `maze-charge-dynamics` (MetalWalls) |
| Per-electrode potential + Gaussian width η_e  |  ✓  |   ○  |   ○  | `per-electrode-eta` (MetalWalls) |
| Constant-charge control (conq)                |  ✓  |   ○  |   ○  | `const-charge-electrode` (LAMMPS `electrode/conq`) |
| Electrode electroneutrality Σ Q_i = 0         |  ✓  |   ○  |   ○  | `const-pot-bo-parallel-plate` (augmented system) |
| Global electroneutrality (incl. electrolyte)  |  ✓  |   ○  |   ○  | `global-electroneutrality-bo` (MetalWalls) |
| Gaussian charge on electrode atoms            |  ✓  |   ○  |   ○  | `gaussian-electrode-charge` (literature: Reed JCP 2007); pair `gaussian-coul-neighborlist` |
| Thomas–Fermi finite metallicity               |  ✓  |   ○  |   ○  | `thomas-fermi-electrode` (MetalWalls) |
| Voronoi site volume for TF                    |  ✓  |   ○  |   ○  | `voronoi-volume-tf` (MetalWalls) |
| Charge-prediction ASPC (Kolafa)               |  ✓  |   ○  |   ○  | `electrode-charge-predictor` (MetalWalls coefficients) |
| Forces on electrode atoms                     |  ✓  |   ○  |   ○  | `electrode-atom-forces` (MetalWalls) |
| Charging work −Σ Ψ_i Q_i in energy            |  ✓  |   ○  |   ○  | `charging-work-energy` (literature: JOSS eq. 6 / Scalfi PCCP 2020) |
| Frozen / immobile electrode atoms             |  ✓  |   ○  |   ○  | `freeze-electrode-atoms` (LAMMPS) |
| QEq fluctuating charge (bulk, ≠ electrode)    |  ✓  |   ○  |   ○  | `qeq-fluctuating-charge` (LAMMPS `fix qeq/point`) |

`ConstPotBO` is the workhorse: dense Cholesky over the Gaussian elastance
\(A\), `use_cg` for the iterative path, `predictor_order` for the ASPC
warm start, `l_tf` + `site_density` (or per-site `site_volume` from
`VoronoiVolume`) for Thomas–Fermi metallicity, `add_electrode_charge` for
the constant-charge inverse, `global_electroneutrality` for the extended
constraint, `write_electrode_forces` for the term breakdown.
`ConstPotMaZe.symplectic` is the true SHAKE mass-zero path (Coretti et al.,
JCP 152, 194701 (2020)); `symplectic = false` keeps the adiabatic CG+ASPC
twin.

## 2. External Fields and Electrode Mechanics

| Capability                                  | CPU | CUDA | Kokkos | Anchor (regressions/md/) |
| ------------------------------------------- | :-: | :--: | :---: | ------------------------ |
| Piston electrode, constant P_z              |  ✓  |   ○  |   ○  | `piston-electrode-pz` (MetalWalls) |
| Finite-field + single electrode (Dufils)    |  ✓  |   ○  |   ○  | `finite-field-single-electrode` (MetalWalls) |
| Displacement D-field (vacuum)               |  ✓  |   ○  |   ○  | `displacement-dfield-md` (analytic vacuum limit) |
| Constant-D polarization feedback            |  ✓  |   ○  |   ○  | `constant-d-polarization-feedback` (literature: Zhang–Sprik PRB 2016) |

Constant and oscillating E-fields (`FixEfield`, `FixOscillatingEfield`) are
general nonequilibrium MD and stay in the
[Molecular Dynamics](molecular-dynamics.md) chapter. The Steele 10-4-3 wall
(`FixWallLJ1043`) and Fumi–Tosi / Born salts live in
[Force Fields](force-fields.md).

## 3. Polarizable Coupling

| Capability                                  | CPU | CUDA | Kokkos | Anchor (regressions/md/) |
| ------------------------------------------- | :-: | :--: | :---: | ------------------------ |
| PIM induced dipoles (SCF)                   |  ✓  |   ○  |   ○  | `pim-induced-dipole-bulk` (MetalWalls) |
| Extended-Lagrangian dipoles (XL)            |  ✓  |   ○  |   ○  | `extended-lagrangian-dipoles` (MetalWalls stationary gate; dynamics gates self-consistency) |
| Coupled dipole ⊕ electrode Q solve          |  ✓  |   ○  |   ○  | `dipole-electrode-joint` (MetalWalls kernels; exact joint system has no MW twin — `mw_expressible: false` in the case REFERENCE) |

AIM/DAIM aspherical and deformable ions are **not implemented** (map slice
E11, deferred off the active queue). `DipoleElectrodeJoint` alternates the
Gaussian-elastance \(AQ\) solve with the Thole-damped PIM dipole solve to
`joint_tol`; it has no Ewald yet, so a fair large-slab wall-time comparison
against MetalWalls is still owed (`perf_exempt` in the case REFERENCE).

## 4. Electrode Observers

| Observer                                    | CPU | CUDA | Kokkos | Anchor (regressions/md/) |
| ------------------------------------------- | :-: | :--: | :---: | ------------------------ |
| Per-electrode total charge Q(t)             |  ✓  |   ○  |   ○  | `const-pot-charge-observer` (LAMMPS) |
| Capacitance from charge fluctuations        |  ✓  |   ○  |   ○  | `capacitance-charge-fluctuation` (literature: Limmer JCP 2013 closed form) |
| Slab potential profile Ψ(z)                 |  ✓  |   ○  |   ○  | `potential-profile-observer` (analytic planar Poisson); also in `const-pot-il-edl-box` |
| Charge–charge autocorrelation C_QQ(τ)       |  ◐  |   ○  |   ○  | `charge-charge-autocorr` — self-consistency pins only, owes a real MW/analytic number |
| Elastance cond(A) + empty capacitance       |  ◐  |   ○  |   ○  | `elastance-condition-observer` — self-consistency pins only, owes a real MW/analytic number |
| Voronoi volume (TF companion)               |  ✓  |   ○  |   ○  | `voronoi-volume-tf` (MetalWalls) |
| Classical polarization density P            |  ✓  |   ○  |   ○  | exercised in `finite-field-single-electrode` (MetalWalls) |

## 5. Scenario coverage

The MetalWalls `example/` scenarios that matter are landed as product boxes:
aqueous TIP4P EDL (`const-pot-tip4p-edl`, MW polarity/neutrality gates),
planar ionic-liquid EDL (`const-pot-il-edl-box`, MetalWalls), PIM-NPT
polarizable IL (`pim-npt-polarizable-il`, literature PIM/Thole closed form),
short const-pot NVT (`const-pot-md-nvt-short`, MetalWalls), molten salt +
Al + MaZe (`maze-molten-salt-al`, MetalWalls), nanoporous slit-pore carbon
(`nanoporous-carbon-const-pot` — composition self-consistency gates, the
weakest anchor in the set), and a TI capacitance workflow
(`thermo-integration-capacitance`, closed-form linear capacitor + Kirkwood
identity).

## Missing / not planned

From the capability map's pending and deferred rows:

- **Full 2D-PBC Ewald electrochemical cell** — only the Yeh–Berkowitz slab
  correction exists (`KSpaceEwald::slab_factor`); the true 2D product is out
  of the current loop's scope.
- **AIM / DAIM aspherical and deformable ions** — blocked E11; scheduled only
  with an explicit charter.
- **ML heterogeneous electrodes** (PiNNwall-class) — blocked E12; an external
  interface, not this loop.
- **XFT potential**, **linear molecule model** — MetalWalls extras with no
  ATV consumer.
- **PLUMED bridge** — ATV's in-tree `cpu/bias/` covers the constrained-ion
  scenarios instead.
- **Blue energy / CDC desalination workflows** — product compositions on top
  of landed pieces, not new physics.
