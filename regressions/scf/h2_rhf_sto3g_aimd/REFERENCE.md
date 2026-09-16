# Reference — H₂ / STO-3G short RHF BOMD (multi-device)

## Method

Short NVE BOMD: concrete **RHF** (STO-3G) as the force evaluator on
`md::MD` + VelocityVerlet. **One regression case**; device selected at run
time (`--device=cpu|cuda|both`), PyTorch-style — not a separate `*_cuda` case.

## Acceptance

After 5 × 0.5 fs steps:

1. `frame.meta.get(kEnergy)` is finite and **bound** (PE < 0).
2. PE lies in a **literature-class envelope** only: about −1.117 Ha
   (≈ −30.4 eV) for H₂ RHF/STO-3G near R = 0.74 Å — **wide** band
   [−35, −25] eV so a few NVE steps off the minimum still pass.
3. **Never** an ATV self-golden PE-drift snapshot.

## Architecture checks

- No `AimdDriver` — composition only: `MD` + `set_evaluator(RHF)`.
- No public `SCFSolver` / `*Config` / user Context.
- Frame is the only I/O token.

## Provenance

H₂ RHF/STO-3G equilibrium energy class is standard textbook / CCCBDB-class
(≈ −1.117 Ha). This case does **not** pin PE drift against Atomiverse itself.
