<!-- mol:note:topic:metalwalls-capability-map -->
## [2026-07-23] MetalWalls → Atomiverse capability map

**Why.** MetalWalls is a full electrochemical classical MD code, not “solve
\(AQ=b\) once.” Agents must not under-scope (claim MW = constant-potential
kernel only) or over-scope (rewrite ATV’s MD stack to clone Fortran
`system`). This page is the canonical inventory and delivery slicing.

**Rule (binding for electrochemistry / constant-potential work):**

1. **Map first.** Any plan or spec that cites MetalWalls must state which
   **slice** (E1–E12 below) it covers and which map rows it intentionally
   leaves out.
2. **Architecture.** Constant-potential electrodes are a **Component** on
   existing `cpu::md::MD` (and later CUDA MD if needed) — **not** a new
   Driver category, not a public `*Config` bag, not logic welded into
   `KSpace*::compute`. Parameters = ctor args / members; \(A\), \(Q\), CG
   workspace = private; charges write `frame.span(kAtom_charge)`.
3. **E1 is the minimum landable product**, not the full MW feature set.
4. **Do not reimplement** ATV strengths already marked ATV+ (PME/CUDA,
   GAFF bonded stack, SCF/PW/GPW, MC/path methods) “for parity.”
5. **Foothold ≠ closed.** A textbook regression + Component does **not**
   flip a row to ✓ if production depth is still missing (e.g. O(N²)
   pairs without NeighborList, Yeh–Berkowitz slab vs full 2D-PBC product).
   Use **~** until production depth lands; open a deeper slug.

**Evidence.** MW: GitLab `ampere2/metalwalls` branch `release` (src +
`example/LIST`), JOSS 2020 (10.21105/joss.02373), JCP 2022 polarizable
interfaces. ATV: tree as of **2026-07-27 wave-5 harness** on `dev`
(`ca3b511a`; waves 1–5 shipped 46 classical electrochem/MD/FF slugs; see
`docs/developer/metalwalls-method-queue.md`).

**Supersedes:** conversation-only summaries that treated MW as “just
constant-potential BO." Map ATV marks re-audited **2026-07-27** (wave-5
refresh gap-census): wave-4 landings flipped — MaZe ○→~ (`ConstPotMaZe`
CG+ASPC foothold), electrode \(Q\)-forces ○→✓
(`ConstPotBO::write_electrode_forces`), global electroneutrality ~→✓,
Gaussian electrode pair ~→✓ (NeighborList bulk), D-field ○→~ (`FixDfield`
vacuum foothold), \(C_{QQ}\) ○→✓ (`ChargeChargeAutocorr`), cond(\(A\)) /
empty-\(C\) ○→✓ (`ElastanceObserver`), Fumi–Tosi / Born ~→✓ (NL bulk),
TIP4P+const-pot ○→✓ (`reg_const-pot-tip4p-edl`), molten-salt+MaZe scenario
○→~. Pending refill adds `maze-symplectic-charge-dynamics`,
`constant-d-polarization-feedback`, `maze-molten-salt-al`.
**Wave-5 harness re-audit (2026-07-27):** per-electrode \(V,\eta\) ~→✓
(`add_electrode(type, psi, eta)`), const-charge control ~→✓
(`add_electrode_charge` conq twin), charge-halo timing designed→landed
(post-solve ghost-charge refresh, `reg_const-pot-charge-halo-mpi`), MaZe
foothold→true symplectic (`ConstPotMaZe.symplectic`), D-field ~→✓
(`FixDfield.polarization_feedback`), Voronoi TF ○→✓ (`VoronoiVolume`), XL
dipoles ○→✓ (`PimXlDipole`), FQ/QEq ○→✓ (`cpu::md::QEq`), PIM-NPT IL and
planar const-pot+IL scenarios ~→✓ (`reg_pim-npt-polarizable-il`,
`reg_const-pot-il-edl-box`). Residual open: nanoporous carbon product ✓ (slit-pore reg 2026-07-29),
molten-salt+Al+MaZe product box, TI capacitance workflow.

---

### Legend

| Mark | Meaning |
|:---:|---|
| ✓ | Present, production-usable |
| ~ | Partial / different form / not the same physics |
| ○ | Absent |
| ★ | MetalWalls specialty (electrochemical depth) |
| ATV+ | Atomiverse stronger; MW weak or absent |

**Priority** (if targeting electrochemical MD): **P0** now · **P1**
production parity · **P2** enhancement · **P3** ecosystem · **—** non-goal
or already covered better in ATV.

---

### 0. Positioning

| Dimension | MetalWalls | Atomiverse |
|---|---|---|
| Role | Electrochemical-interface classical MD | Dual-backend MD + electronic structure |
| Depth | Constant potential / PIM / AIM / TF / piston / 2D | Driver architecture, PME, force catalogue, CUDA, SCF/PW/GPW |
| Stack | Fortran + MPI + optional Python/PLUMED | C++20 + optional CUDA + molrs |
| Shape | Monolithic `system` + specialty modules | Driver / Component / Frame |

Parity ≠ cloning the program: **add electrochemical depth** on ATV’s MD
spine; keep ATV advantages.

---

### 1. Electrodes and constant potential ★

| Capability | MW | ATV | ATV home / gap | Pri |
|---|:---:|:---:|---|:---:|
| Constant-potential electrodes (fluctuating \(Q\)) | ✓★ | ✓ | `ConstPotBO` / `ConstPotCG` → `kAtom_charge` | **P0** |
| Gaussian charge on electrode atoms | ✓★ | ✓ | `PairGaussianCoul` (NeighborList bulk, `reg_gaussian-coul-neighborlist`) + Gaussian \(A\) in BO | **P0–P1** |
| Constant-charge electrodes (control) | ✓ | ✓ | `ConstPotBO::add_electrode_charge` const-\(Q\) control (LAMMPS `electrode/conq` twin; `reg_const-charge-electrode`) | P1 |
| BO matrix inverse for \(Q\) | ✓ | ✓ | Dense Cholesky path on `ConstPotBO` | **P0** |
| CG solve for \(Q\) | ✓ | ✓ | `ConstPotCG` / `ConstPotBO::use_cg` | P1 |
| MaZe mass-zero charge dynamics | ✓★ | ✓ | `ConstPotMaZe.symplectic` true SHAKE mass-zero path (wave-5, `reg_maze-symplectic-charge-dynamics`); `symplectic=false` keeps the CG+ASPC adiabatic twin (`reg_maze-charge-dynamics`) | P2 |
| Electrode charge predictor | ✓ | ✓ | `ChargeASPC` / `predictor_order` on BO | P1 |
| Electrode electroneutrality \(\sum Q_i=0\) | ✓ | ✓ | Augmented system on BO/CG | **P0** |
| Global electroneutrality (incl. electrolyte) | ✓ | ✓ | `ConstPotBO::global_electroneutrality` (`reg_global-electroneutrality-bo`) | P1 |
| Multi-electrode, per-electrode \(V,\eta\) | ✓ | ✓ | `add_electrode(type, psi, eta)` per-group widths, mixed \(\eta_{ij}\) (`reg_per-electrode-eta`, wave-5) | **P0** |
| Forces on electrode atoms (term breakdown) | ✓ | ✓ | `ConstPotBO::write_electrode_forces` (`reg_electrode-atom-forces`) | P1 |
| Freeze / immobile ions | ✓ (`mobile`) | ✓ | `FixFreeze` | **P0** |
| Piston electrode + constant \(P_z\) | ✓★ | ✓ | `PistonElectrode` | P2 |
| Thomas–Fermi finite metallicity | ✓★ | ✓ | `ConstPotBO::l_tf`, `site_density` | P2 |
| Voronoi volume (TF) | ✓ | ✓ | `VoronoiVolume` TF site-volume companion (`reg_voronoi-volume-tf`, wave-5) | P2 |
| Applied E-field (MD) | ✓★ | ✓ | `FixEfield` + `FixOscillatingEfield` | P1 |
| Displacement D-field | ✓★ | ✓ | `FixDfield` vacuum path (`reg_displacement-dfield-md`) + self-consistent constant-\(D\) `polarization_feedback` \(E = D - 4\pi k_e P\) (`reg_constant-d-polarization-feedback`, wave-5) | P2 |
| Finite-field + single electrode (3D) | ✓★ | ✓ | Dufils path (`ConstPotBO` + \(E_z\)) | P2 |
| 2D PBC electrochemical cell | ✓★ | ~ | Yeh–Berkowitz **slab** only; full 2D product **out of loop scope** | P1 |
| Charging work \(-\sum\Psi_i Q_i\) in energy | ✓ | ✓ | `ChargingWork` | **P0** |
| Capacitance / charge-fluctuation analysis | ✓ tools | ✓ | `CapacitanceChargeFluctuation` | P1 |
| Potential-profile post-processing | ✓ tools | ✓ | `PotentialProfile` \(\Psi(z)\) | P1 |
| Charge–charge autocorrelation | ✓ tools | ✓ | `ChargeChargeAutocorr` streaming \(C_{QQ}(\tau)\) (`reg_charge-charge-autocorr`) | P2 |
| Empty capacitance / matrix condition | ✓ tools | ✓ | `ElastanceObserver` \(\mathrm{cond}(A)\) + empty-\(C\) (`reg_elastance-condition-observer`) | P1 |
| ML heterogeneous electrodes (PiNNwall, …) | ecosystem ✓ | ○ | External interface — **blocked E12** | P3 |

**MD phase / halo (implementation constraint — resolved wave-5):**
`kAtom_charge` already borders (`kColComm|kColBorder|kColWriteBack`).
Updating \(Q\) at `pre_force` after `HaloSend` leaves **stale ghost charges**
under MPI; the landed design refreshes ghost charges **after** the solve
(`reg_const-pot-charge-halo-mpi`). Any new charge-writing Component must keep
that post-solve refresh.

---

### 2. Polarizable electrolyte and floating charge ★

| Capability | MW | ATV | Notes | Pri |
|---|:---:|:---:|---|:---:|
| PIM induced dipoles (SCF) | ✓★ | ✓ | `PimInducedDipole` bulk foothold | P1 |
| Extended-Lagrangian dipoles | ~ MaZe-related | ✓ | `PimXlDipole` time-reversible XL propagation (`reg_extended-lagrangian-dipoles`, wave-5) | P2 |
| **Coupled** dipole ⊕ electrode \(Q\) solve | ✓★ | ✓ | `DipoleElectrodeJoint` | **P1** |
| Thole / Tang–Toennies damping | ✓ | ✓ | Thole + optional `tt_b` on PIM | P1 |
| Per-species polarizability | ✓ | ✓ | `alpha_by_type` table on PIM | P1 |
| AIM aspherical ions | ✓★ | ○ | **blocked E11** until scheduled | P3 |
| DAIM deformable ions | ✓★ | ○ | **blocked E11** | P3 |
| Multipole force field | ✓ | ~ | `observe/multipole` ≠ multipole FF | P2 |
| Fluctuating-charge / QEq | ~ electrode only | ✓ | `cpu::md::QEq` (LAMMPS `fix qeq/point` twin; `reg_qeq-fluctuating-charge`, ratio 0.45, wave-5); ≠ const-potential electrode | P2 |
| Drude / AMOEBA | ○ | ○ | Both roadmap-level | — |
| QM/MM polarizable | ○ | ~ | `qmmm_polarizable` — ATV path, not MW | ATV+ |

---

### 3. Classical nonbonded

| Capability | MW | ATV | Notes | Pri |
|---|:---:|:---:|---|:---:|
| LJ 12-6 | ✓ | ✓⚡ CPU/CUDA | ATV stronger | — |
| LJ soft-core | ✓ | ✓ `PairLJSoft` | | — |
| Coulomb cut | ✓ | ✓ `PairCoulCut` | | — |
| Coulomb soft-core | ✓ | ✓ `PairCoulSoft` | | — |
| Ewald real + reciprocal | ✓ | ✓ `CoulLong` + `KSpaceEwald` | | — |
| PME / SPME | ○/~ | ✓ CPU/CUDA | **ATV+** | ATV+ |
| PPPM | ○ | ✓ | | ATV+ |
| 2D Ewald / 2D PBC | ✓★ | ~ | Yeh–Berkowitz slab foothold; full 2D **out of scope** this loop | P1 |
| \(k=0\) slab correction | ✓ | ✓ | `KSpaceEwald::slab_factor` | P1 |
| Fumi–Tosi | ✓★ | ✓ | `PairBorn` multi-type + Coul; NeighborList bulk (`reg_born-buck-neighborlist-bulk`) | P2 |
| XFT | ✓ | ○ | | P3 |
| Born repulsion | ✓ | ✓ | `PairBorn` NeighborList bulk (`reg_born-buck-neighborlist-bulk`; prefers `ctx.md.pair_list`) | P2 |
| Steele wall | ✓★ | ✓ | `FixWallLJ1043` (`wall/lj1043`) | P2 |
| Morse / Yukawa / WCA / DPD | ○ | ✓ | | ATV+ |
| EAM / Stillinger–Weber | ○ | ✓ | | ATV+ |
| `special_bonds` scales | ~ | ✓ | GAFF path mature | ATV+ |

---

### 4. Bonded / molecular models

| Capability | MW | ATV | Notes | Pri |
|---|:---:|:---:|---|:---:|
| Harmonic bond / angle | ✓ | ✓ CPU/CUDA | | — |
| Dihedrals | ✓ | ✓ multi/harmonic + CHARMM | ATV broader | ATV+ |
| Impropers | ✓ | ✓ harmonic | | — |
| FENE | ○ | ✓ | | ATV+ |
| Four-site water (TIP4P-class) | ✓★ | ✓ | `PairLJCutTip4pCut` + `VirtualSites` Weighted3 | P1 (aqueous electrodes) |
| Linear molecule model | ✓ | ○ | | P3 |
| Molecule-type table | ✓ | ~ | ATV per-atom topology columns | ~ |
| Virtual sites | ✓ four-site | ✓ | Weighted3 + OutOfPlane3 (TIP4P/TIP5P) | P1 |

---

### 5. Constraints / integration / ensembles

| Capability | MW | ATV | Notes | Pri |
|---|:---:|:---:|---|:---:|
| Velocity Verlet-class | ✓ | ✓ CPU/CUDA | | — |
| RATTLE | ✓ | ✓ `FixRattle` | | — |
| SETTLE | ○ | ✓ | | ATV+ |
| Nose–Hoover chain NVT | ✓ | ✓ `NoseHoover` | | — |
| Langevin / CSVR | ○ | ✓ | | ATV+ |
| NPT (MTK-class) | ✓ | ✓ `NoseHooverNPT` etc. | | — |
| NPH / PR NPT | ○ | ✓ | | ATV+ |
| Piston NPT (electrodes) | ✓★ | ✓ | `PistonElectrode` (rigid \(P_z\); ≠ Langevin piston) | P2 |
| Velocity init / scale | ✓ | ✓ | | — |
| Deform / Brownian / NVE limit | ○ | ✓ | | ATV+ |

---

### 6. Long-range and communication

| Capability | MW | ATV | Notes | Pri |
|---|:---:|:---:|---|:---:|
| 3D Ewald | ✓ | ✓ | | — |
| MPI MD + domain decomp | ✓ | ✓ | | — |
| Neighbor lists | ✓ | ✓ | | — |
| Charge halo correctness after \(Q\) update | ✓ | ✓ | Post-solve ghost-charge refresh landed wave-5 (`reg_const-pot-charge-halo-mpi`); see §1 | **P0** |
| CUDA MD + PME | ○ | ✓ | **ATV+** | ATV+ |
| FMM open-boundary | ○ | ~ operators / SCF | Not MD `KSpace` | — |

---

### 7. Driver / workflow

| Capability | MW | ATV | Notes | Pri |
|---|:---:|:---:|---|:---:|
| Dedicated MD loop | ✓ monolith | ✓ `md::MD` nestable | | ATV+ |
| File input language | ✓ runtime/data | ~ library API | | — |
| Python driving | ✓ f2py | ~ nanobind wheel | Different shape | P3 |
| Thermodynamic integration workflow | ✓★ + soft-core | ~ soft-core only | | P2 |
| PLUMED | ✓ | ○ | ATV has in-tree bias/metadynamics | ~ |
| Collective vars / metadynamics | via PLUMED | ✓ `cpu/bias/*` | | ATV+ |
| Continuable runs | ✓ | ✓ | | — |

---

### 8. Observe / I/O / post-process

| Capability | MW | ATV | Notes | Pri |
|---|:---:|:---:|---|:---:|
| Thermo / trajectories | ✓ | ✓ + molrs | | — |
| RDF / MSD / VACF | ~ | ✓ | | ATV+ |
| Virial / pressure tensor | ✓ | ✓ | | — |
| Capacitance scripts | ✓★ | ✓ | `CapacitanceChargeFluctuation` | P1 |
| Empty capacitance / matrix condition number | ✓ tools | ✓ | `ElastanceObserver` (`reg_elastance-condition-observer`) | P1 |
| Electrode charge time series | ✓ | ✓ | `ElectrodeCharge` observer | **P0** |
| Widom / WHAM | ○ | ✓ | | ATV+ |
| Electronic observers (Mulliken, …) | ○ | ✓ | | ATV+ |

---

### 9. Scenario coverage (MW `example/`)

| Scenario | MW | ATV today | Gap |
|---|:---:|---|---|
| Bulk non-polarizable IL | ✓ | ✓ LJ+Coul+PME | — |
| Bulk polarizable IL / PIM-NPT | ✓ | ✓ | `reg_pim-npt-polarizable-il` (wave-5) |
| Planar graphene + const-pot + IL | ✓ | ✓ | `reg_const-pot-il-edl-box` production EDL box (wave-5) |
| Nanoporous carbon + const-pot | ✓ | ○ | Geometry + E1 (product) |
| Blue energy / CDC desalination | ✓ | ○ | Product workflow |
| Finite-field single electrode | ✓ | ✓ | Dufils foothold |
| TF gold capacitor | ✓ | ✓ | `thomas-fermi-electrode` |
| Molten salt + Al + MaZe | ✓ | ✓ | MaZe + Fumi–Tosi NL footholds ✓; product box ✓ (`maze-molten-salt-al` 2026-07-29) |
| Piston constant \(P_z\) | ✓ | ✓ | `piston-electrode-pz` |
| PLUMED constrained ion | ✓ | ~ ATV bias | PLUMED bridge **out of scope** |
| TIP4P water | ✓ | ✓ | four-site + multi-topology vsites |
| TIP4P + const-pot electrodes | ✓ | ✓ | `reg_const-pot-tip4p-edl` (wave-4 aqueous EDL) |
| AIM tests | ✓ | ○ | **blocked E11** |
| AIMD / Gaussian SCF / PW / GPW | ○ | ✓ | **ATV+** |
| CUDA large LJ/PME | ○ | ✓ | **ATV+** |
| NEB / REMD / PIMD / MC | ○ | ✓ | **ATV+** |

---

### 10. Delivery slices (use these in specs)

| Slice | Map coverage | Goal |
|---|---|---|
| **E1 const-pot v1** | §1 P0: BO \(Q\), multi-electrode \(V\), neutrality, freeze, charging work, Observer, bind `md::MD` | Parallel-plate EDL runs |
| **E2 electrostatic rigor** | Gaussian kernels, 2D Ewald, predictor, CG | Numerical near MW small cases |
| **E3 external field** | E-field, finite-field single electrode | Dufils-class setups |
| **E4 polarization** | PIM + damping | Bulk polarizable IL |
| **E5 coupling** | Dipole ⊕ electrode joint solve | JCP 2022 main capability |
| **E6 electrode mechanics** | Piston \(P_z\), electrode forces | Piston examples |
| **E7 metallicity** | Thomas–Fermi | Imperfect metal |
| **E8 salt / walls** | Fumi–Tosi, Steele, Born | LiCl–Al, graphite wall |
| **E9 molecular water** | Four-site / virtual sites | TIP4P at electrodes |
| **E10 ecosystem** | Capacitance tools, TI workflow, optional PLUMED bridge | Paper workflows |
| **E11 AIM/DAIM** | §2 AIM | Deep molten-salt users only — **deferred off** `metalwalls-method-loop` active queue (2026-07-29); schedule only with explicit E11 charter |
| **E12 ML electrodes** | PiNNwall-class | Research frontier — **deferred off** active queue (2026-07-29); external ML interface, not this loop |

**E1 only** is the “simple” kernel discussed in chat — roughly the P0 rows of
§1, **not** the full map.

---

### 11. Forbidden when implementing from this map

- New `ElectrochemistryDriver` / category base  
- Public `ElectrodeConfig` / `ConstPotentialConfig` bags  
- Absorbing charge solve into `KSpaceEwald` / `KSpacePME`  
- Claiming MetalWalls equivalence from point-charge \(A\) without documenting
  the approximation  
- Ignoring MPI ghost charge after \(Q\) update  
- Full Fortran `system` port or dual MD architectures

---

### Delivery workflow

- Loop: `.grok/workflows/metalwalls-method-loop.rhai` (`/metalwalls-method-loop`)
- Design / queue / handoff: `docs/developer/metalwalls-method-*.md`
- Oracle: `tools/metalwalls_oracle/` — **LAMMPS first** when feature overlaps;
  MetalWalls only for MW-only physics
- Build: **MD-only** (`ATV_BUILD_MD`+`KSPACE`+tests/regressions; HF/PW/GPW/CUDA OFF)
- Default parallel workers: **staff_count=10**

### References (stable)

- Repo: https://gitlab.com/ampere2/metalwalls (`release`)  
- JOSS: Marin-Laflèche et al., JOSS 5, 2373 (2020)  
- JCP 2022: polarizable electrolyte + metallic electrodes  
- ATV roadmap polarizable/FQ rows: `docs/roadmap/force-fields.md`  

<!-- mol:note:topic:metalwalls-capability-map -->
