# qmmm_es_h2o_pyscf

## System

- Electrostatic QM/MM coupling (`QmmmElectrostatic`) with empty cross-LJ table
- QM region: H₂O via `kAtomQmZ` (Z = 8,1,1)
- MM: two atoms with charges ±0.5 e at z = ±3 Å
- Engine: RHF/STO-3G, `embed_self_energy = false` (engine default via subsystem)

## Oracle

Offline PySCF 2.14.0 `qmmm.mm_charge` pins reused from
`regressions/scf/embed_h2o_grad` (QM forces) and
`regressions/scf/embed_h2o_mm_forces` (MM-site forces). The coupling must add
no error over the bare embedding SCF of esmm-qmmm-01.

## Pins

- QM gradients [Ha/Bohr]: same as `embed_h2o_grad` REFERENCE
- MM gradients [Ha/Bohr]: same as `embed_h2o_mm_forces` REFERENCE
- |Σ F| ≤ 1e-9 eV/Å

## Supersedes

`regressions/multiscale/qmmm_es_h2` (toy tag-list API; deleted with this case).
