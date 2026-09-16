# MetalWalls twin — const-pot-md-nvt-short

Geometry matches the ATV regression (14×14×20 Å, 2×2 walls, 4 Na + 4 Cl).

- `runtime.inpt` has `@NUMPBC@` placeholder.
- `runtime_pbc3.inpt` is the 3D-periodic twin of ATV's PME cell (use this for pin).
- `mw_output/` archives MetalWalls c8ad175d `total_charges.out` for num_pbc=3:

| step | Q_C1 [e] | Q_C2 [e] |
|------|----------|----------|
| 0 | −9.711155787094686e-2 | +9.711155787094686e-2 |
| 1 | −9.742254974821957e-2 | +9.742254974821957e-2 |

ATV after step 1 (PME): |Q| ≈ 9.99e-2 e (~2.9 % from MW step 0).
