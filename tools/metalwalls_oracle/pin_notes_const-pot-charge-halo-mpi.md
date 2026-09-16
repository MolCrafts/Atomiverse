# Pin notes — const-pot-charge-halo-mpi

- **Oracle choice:** `literature` (queue row). No oracle binary was run: the
  observable is an MPI-transport self-consistency invariant (ghost q must
  equal owner q after the constant-potential charge solve), for which no
  external engine produces a comparable number. LAMMPS is not a *numeric*
  twin here — the invariant itself is LAMMPS's documented behavior
  (`fix electrode` forward-communicates per-atom charge after every conp
  solve before force evaluation), which is what we pin as the contract.
- **Citations** (full details in the regression's REFERENCE.md):
  - Thompson et al., Comput. Phys. Commun. 271, 108171 (2022) — LAMMPS
    forward/reverse ghost comm semantics.
  - Ahrens-Iwers, Janssen, Tee, Meißner, J. Chem. Phys. 157, 084801 (2022) —
    ELECTRODE package: charge solve + forward comm of q per step.
  - Marin-Laflèche et al., J. Open Source Softw. 5, 2373 (2020) — MetalWalls.
- **Pinned values:** exact transport bound 0 (asserted ≤ 1e-15 e, memcpy
  transport); non-vacuity gates (nonzero-charge electrode ghosts seen;
  every ghost tag resolves to an owner; check runs every step).
- **perf_exempt:** yes — no fair wall-clock twin exists for an MPI halo
  self-consistency check (a LAMMPS run would measure a full conp Ewald
  pipeline, not the transport under test).
- **Inputs:** none (no oracle run); the regression geometry is fully defined
  in `regressions/md/const-pot-charge-halo-mpi/`.

Verified 2026-07-27 (metalwalls loop wave 5 worker).
