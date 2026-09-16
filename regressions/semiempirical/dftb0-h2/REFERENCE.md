# Reference — H₂ non-SCC DFTB0 energy vs DFTB+ (mio-1-1)

```text
oracle: dftb+
oracle_version: 24.1
```

## Method

Non-SCC DFTB / **DFTB0** (band structure of H⁰ + pair repulsive; no SCC):

1. Slater–Koster H⁰ / S from the transcribed mio-1-1 H–H grid
   (`basis_data/dftb_mio_hh_sk.h`, read through
   `internal/mio_hh_table.h` — the same grid, interpolation and Bohr/Hartree
   convention the SCC path uses)
2. Generalized eigen \(H_0 C = S C \varepsilon\)
3. Closed-shell occupation \(n_\mathrm{occ} = n_e/2\)
4. \(E = \sum_i 2\varepsilon_i + E_\mathrm{rep}\), with \(E_\mathrm{rep}\) the
   mio-1-1 H–H repulsive spline

Reference method: DFTB0 / non-SCC DFTB — Elstner et al., *Phys. Rev. B* **58**,
7260 (1998), DOI 10.1103/PhysRevB.58.7260 (the zeroth-order band-structure +
repulsive term of that paper, before the second-order charge-fluctuation
expansion). **Not** OpenMOPAC; **not** xtb.

## Why an SCC oracle run pins a non-SCC method

The offline oracle transcript is a **DFTB2 (SCC)** DFTB+ run. It is a valid
DFTB0 oracle *for this system only*, and by construction rather than by
assumption:

- H₂ is homonuclear, so the two atomic sites are related by symmetry and the
  Mulliken charge fluctuations vanish exactly: \(\Delta q_A = \Delta q_B = 0\).
- The DFTB2 second-order energy is
  \(E_\mathrm{2nd} = \tfrac12 \sum_{AB} \gamma_{AB}\,\Delta q_A \Delta q_B\),
  which is then identically **zero**, and the SCC shift
  \(\sum_B \gamma_{AB}\Delta q_B\) added to H⁰ is likewise zero, so the
  converged Hamiltonian *is* H⁰ and the converged eigenvalues *are* the
  non-SCC eigenvalues.
- DFTB2 therefore reduces exactly to \(E = \sum_i 2\varepsilon_i +
  E_\mathrm{rep}\) — the DFTB0 expression — and the DFTB+ total is a DFTB0
  total.

The oracle transcript states this independently: its `detailed.out`
decomposition is band \(-0.6807883427\) H + repulsive \(+0.0058374104\) H +
SCC \(0\) H (Δq = 0), summing to the pinned total. ATV's own DFTB2 path
reproduces it with converged \(|\Delta q| = 4\times10^{-16}\,e\).

This licence is **specific to homonuclear diatomics**. A heteronuclear DFTB0
pin needs its own non-SCC DFTB+ run.

## Offline oracle run

Carried by the sibling regression
`regressions/semiempirical/dftb2-dftbplus-energy-pin/` (same oracle binary,
same parameter set, same geometry); inputs at
`tools/tb_oracle/inputs/dftbplus/h2_mio_scc/`. Nothing is executed at ctest
time — the numbers below are transcribed constants.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | −0.6749509324 | Ha |
| Total energy | **−18.36635052** | eV |
| Band energy \(\sum_i 2\varepsilon_i\) | −0.6807883427 | Ha |
| Repulsive | +0.0058374104 | Ha |
| SCC (second-order) | 0 (Δq = 0) | Ha |
| Conversion | 1 Ha = 27.211386245988 eV | — |

## Data

In-tree static tables, nothing parsed at build/test/run time:
`basis_data/dftb_mio_hh_sk.h` (H–H, transcribed DFTB.org mio-1-1 `H-H.skf`)
and the H onsite \(E_s\) / Hubbard U rows in `basis_data/dftb_mio_tables.h`
(also mio-1-1). Unsupported Z throws at lookup — no fallback.

## Geometry

Gas-phase H₂, bond along \(x\), \(R = 0.74\) Å (and \(R = 3.0\) Å for
non-flatness). Cubic box 30 Å (isolation only). Charge 0, singlet.

| Atom | x    | y | z |
|------|------|---|---|
| H    | 0.00 | 0 | 0 |
| H    | 0.74 | 0 | 0 |

## Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E(R=0.74\,\text{Å})\) | **−18.36635052** | eV | \(1\times10^{-3}\) eV |
| \(|E(0.74)-E(3.0)|\) | \(> 10^{-6}\) | eV | pipeline not flat |
| Force skeleton at \(R=0.90\) Å | \(\|F\|^2 > 0\) | eV²/Å² | finite, non-zero |

**Tolerance rationale.** \(1\times10^{-3}\) eV is the sibling DFTB2 pin's
tolerance for the *same* oracle number; gating the same transcript twice at
two different precisions would claim a precision the transcript does not
carry (it is printed to \(10^{-10}\) Ha ≈ \(2.7\times10^{-9}\) eV). It also
sits ~230× below the parameter-fidelity defect this gate exists to catch —
reading mio-*style* sketch H–H parameters instead of the transcribed mio-1-1
grid moved this energy by 0.229 eV — while leaving room for host BLAS /
eigensolver variation. Observed residual on the CPU reference host:
\(|E_\mathrm{ATV} - E_\mathrm{pin}| \approx 3\times10^{-9}\) eV, bit-identical
to the DFTB2 path's own energy at this geometry (both −18.3663505230 eV),
which is the symmetry argument above holding numerically.

## Perf

Offline DFTB+ binary is not available on the build host; fair wall
\(T_\mathrm{atv}/T_\mathrm{oracle}\) not measured here → **perf_exempt**
(the sibling pin records ~0.06 s ATV vs ~0.15 s DFTB+ cold for the same
molecule).
