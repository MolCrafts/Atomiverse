# Reference — arch-cuda-scf-solver-fold

## Intent

Architecture gate: after folding free-loop `SCFSolver` under
`cuda::scf::RHF`, the public method surface still reproduces the H₂ / STO-3G
RHF literature energy.

## Reference value

**E_RHF/STO-3G(H₂) ≈ −1.1167 Hartree** at R(H–H) = 0.74 Å (textbook
minimal-basis pin; same order as `tests/cuda/scf/test_rhf.cu` and
Hehre–Stewart–Pople STO-3G family).

Tolerance: 0.01 Ha (architecture gate, not a pin-softening of an existing
regression).

## Public API only

`cuda::scf::RHF{BasisSet::STO3G}.run(frame)` — Frame in/out; no SCFSolver,
no *Config, no Context.
