<!-- mol:note:topic:metalwalls-capability-map -->
## MetalWalls → Atomiverse capability map

MetalWalls is a full electrochemical classical MD code, not "solve AQ = b once".
Do not under-scope (MW = constant-potential kernel) or over-scope (clone the
Fortran `system` on ATV's MD stack). What ATV currently implements is stated in
`docs/roadmap/` (force-fields, electrodes, molecular-dynamics), not here.

### Rules (binding for electrochemistry / constant-potential work)

1. **Map first.** A plan citing MetalWalls names its slice (E1–E12) and the
   map rows it leaves out.
2. **Architecture.** Constant-potential electrodes are a **Component** on
   `cpu::md::MD` — not a Driver category, not a `*Config` bag, not logic inside
   `KSpace*::compute`. Parameters = ctor args / members; A, Q, CG workspace
   private; charges write `frame.span(kAtom_charge)`.
3. **E1 is the minimum landable product**, not the full MW feature set.
4. **Do not reimplement ATV strengths** (PME/CUDA, GAFF bonded stack,
   SCF/PW/GPW, MC/path methods) "for parity".
5. **Foothold ≠ closed.** A textbook regression + Component does not make a
   roadmap row ✓ while production depth is missing (O(N²) pairs without
   NeighborList, Yeh–Berkowitz slab vs full 2D-PBC).
6. **Charge halo.** A charge-writing Component refreshes ghost charges **after**
   its solve (`ChargePacker`); `kAtom_charge` is `read_stable`, not on the
   per-step wire (`reg_const-pot-charge-halo-mpi`).

### Delivery slices

| Slice | Coverage | Goal |
| --- | --- | --- |
| **E1 const-pot v1** | BO Q, multi-electrode V, neutrality, freeze, charging work, Observer, bind `md::MD` | Parallel-plate EDL runs |
| **E2 electrostatic rigor** | Gaussian kernels, 2D Ewald, predictor, CG | Numerical near MW small cases |
| **E3 external field** | E-field, finite-field single electrode | Dufils-class setups |
| **E4 polarization** | PIM + damping | Bulk polarizable IL |
| **E5 coupling** | Dipole ⊕ electrode joint solve | JCP 2022 main capability |
| **E6 electrode mechanics** | Piston P_z, electrode forces | Piston examples |
| **E7 metallicity** | Thomas–Fermi | Imperfect metal |
| **E8 salt / walls** | Fumi–Tosi, Steele, Born | LiCl–Al, graphite wall |
| **E9 molecular water** | Four-site / virtual sites | TIP4P at electrodes |
| **E10 ecosystem** | Capacitance tools, TI workflow, optional PLUMED bridge | Paper workflows |
| **E11 AIM/DAIM** | Aspherical / deformable ions | Deferred off the active queue; needs an explicit charter |
| **E12 ML electrodes** | PiNNwall-class | Deferred; external ML interface, not this loop |

### Known gaps vs MetalWalls (check the roadmap for current marks)

- Full 2D-PBC electrochemical cell / 2D Ewald (ATV has Yeh–Berkowitz slab
  only) — out of the current loop scope.
- AIM / DAIM ions (E11), PiNNwall-class ML electrodes (E12).
- Multipole force field (ATV `observe/multipole` is not an FF), XFT, linear
  molecule model.
- Thermodynamic-integration workflow (ATV has soft-core only); PLUMED bridge
  (out of scope; ATV has in-tree bias/metadynamics).
- Scenario products: blue energy / CDC desalination; file input language and
  f2py-style driving have different shapes in ATV (library API, nanobind).

### Forbidden

- `ElectrochemistryDriver` or any category base.
- Public `ElectrodeConfig` / `ConstPotentialConfig` bags.
- Absorbing the charge solve into `KSpaceEwald` / `KSpacePME`.
- Claiming MW equivalence from a point-charge A without documenting it.
- Ignoring MPI ghost charges after a Q update.
- A full Fortran `system` port or dual MD architectures.

### Delivery workflow

- Loop: `.grok/workflows/metalwalls-method-loop.rhai` (`/metalwalls-method-loop`);
  design/queue/handoff `docs/developer/metalwalls-method-*.md`.
- Oracle: `tools/metalwalls_oracle/` — **LAMMPS first** where the feature
  overlaps; MetalWalls only for MW-only physics.
- Build: MD-only (`ATV_BUILD_MD` + `KSPACE` + tests/regressions; HF/PW/GPW/CUDA
  OFF). Default staff count 10.

### References

- https://gitlab.com/ampere2/metalwalls (`release`)
- Marin-Laflèche et al., JOSS 5, 2373 (2020)
- JCP 2022: polarizable electrolyte + metallic electrodes
