# embed_h2o_mm_forces

## System

- H₂O RHF/STO-3G (Cartesian), same geometry as `embed_h2o_grad`
- Two external charges: `q = ±0.5 e` at `(0, 0, ±3)` Å
- `embed_self_energy = false` — PySCF `qmmm.mm_charge` energy does **not**
  include classical MM–MM, so the FD pin is the QM↔MM seam only

## Oracle

- **Code:** PySCF 2.14.0 `qmmm.mm_charge` + central FD of `mf.e_tot` w.r.t. MM
  coordinates (Δ = 1e-4 Bohr), offline capture only.
- **Recipe:** same charge layout as `examples/qmmm/30-force_on_mm_particles.py`
  (force-on-MM-particles pattern); numbers below are the other code’s output.

## Pin — ∂E/∂R_MM [Ha/Bohr]

| site | x | y | z |
| ---- | - | - | - |
| +0.5 @ z=+3 | 0 | 0 | `+4.053776692103e-03` |
| −0.5 @ z=−3 | 0 | 0 | `−3.391311764744e-03` |

Tolerance: element-wise ≤ `1e-5` Ha/Bohr.

## Also checked

- With `embed_self_energy = false`, `|Σ_QM F + Σ_MM F| ≤ 1e-9` eV/Å (Newton’s
  third law on the QM↔MM seam).
