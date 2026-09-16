# Reference — DFTB2 H₂ energy vs DFTB+ (mio-1-1)

```text
oracle: dftb+
oracle_version: 24.1
```

## Method

SCC-DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998)).
Hamiltonian / SK set: **mio-1-1** H–H (static transcription of
`H-H.skf` into `basis_data/dftb_mio_hh_sk.h`; no runtime `.skf` I/O).

## Geometry

Isolated H₂, r(H–H) = **0.74 Å**, charge 0, closed shell. Cubic box 30 Å
(isolation only).

## Offline pin (DFTB+ 24.1)

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | −0.6749509324 | Ha |
| Total energy | **−18.36635052** | eV |
| Conversion | 1 Ha = 27.211386245988 eV | — |
| Band energy | −0.6807883427 | Ha |
| Repulsive | +0.0058374104 | Ha |
| SCC energy | 0 (Δq = 0) | Ha |

Inputs: `tools/tb_oracle/inputs/dftbplus/h2_mio_scc/`.

## Gate

|E_ATV − (−18.36635052 eV)| ≤ **1×10⁻³ eV** on `frame.meta[kEnergy]` via
`cpu::Evaluate` + `cpu::semiempirical::DFTB2` (`with_forces = false`).

## Perf (optional, not ctest)

| Timer | Wall [s] |
| ----- | -------- |
| ATV `reg_dftb2-dftbplus-energy-pin` (Evaluate+DFTB2, forces off) | ~0.06 |
| DFTB+ 24.1 cold (same geometry; includes I/O) | ~0.15 |

Ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) ≈ **0.4** (≤ 10). Not asserted in ctest.
Exact |E_ATV − E_pin| on this host: ~3×10⁻⁹ eV.
