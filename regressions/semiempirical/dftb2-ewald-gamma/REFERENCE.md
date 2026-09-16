# dftb2-ewald-gamma — Ewald-summed periodic SCC γ vs DFTB+

```text
oracle: dftb+
oracle_version: 25.1 development (commit a23bfb2) + mio-1-1
```

Offline pin only — ctest never launches DFTB+.

## Why this case exists

Every other periodic DFTB2 case in this suite (`dftb2-periodic-gamma`,
`dftb2-periodic-forces`, `dftb2-kpoints`, `dftb2-multi-k-forces`) is the same
homonuclear 1-D carbon chain. There Δq ≡ 0 by symmetry, so
\(E^{(2)}=\tfrac12\sum\gamma\Delta q\Delta q\) vanishes **identically** and the
SCC charge kernel is never evaluated at all. The periodic γ was therefore
untested for as long as those cases were the only periodic ones, and it was
wrong: the \(1/R\) half of \(\gamma\) was truncated at the Slater–Koster
cutoff, which is not an approximation of a lattice sum but a different,
non-positive-definite operator. On this cell that gave \(E^{(2)} = -2.19\) Ha
against DFTB+'s \(+0.138\) Ha and a 37 eV error in the total.

This case is polar by construction and asserts \(\max|\Delta q| > 0.1\,e\), so
it cannot silently degenerate into the same blind spot.

## Method

SCC-DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998), DOI
[10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)),
`boundary = lattice`: image-summed \(H^0\)/\(S\)/repulsive within the SK
cutoffs, and \(\gamma = 1/R - S(\tau_A,\tau_B,R)\) with the \(1/R\) half taken
as an Ewald sum with a neutralising background
(`internal/dftb_gamma_lattice.h`) and the exponential \(S\) half truncated
where it falls below \(10^{-12}\) eV.

SK set: mio-1-1 H/O (`basis_data/dftb_mio_sk_tables.h`). No runtime `.skf`.

## Geometry

8 water molecules, r(O–H) = 0.96 Å, ∠HOH = 104.5°, oxygens on a 2×2×2
sublattice with random rigid-body orientations (seed 20260902) and a 0.15 Å
positional jitter. Cubic cell **L = 7.0 Å** (≈ 0.81 g/cm³). Minimum
inter-atomic distance under the minimum image is 0.96 Å (the intramolecular
O–H bond), i.e. no intermolecular contact is closer than a bond length.

Full coordinates are in the regression source and in
`tools/tb_oracle/inputs/dftbplus/dftb2-ewald-gamma/geo.gen`.

## DFTB+ settings

`SCC = Yes`, `SCCTolerance = 1e-10`, `MaxAngularMomentum = {O = p; H = s}`,
`KPointsAndWeights = SuperCellFolding {1 0 0  0 1 0  0 0 1  0 0 0}` (Γ),
`Analysis { CalculateForces = Yes }`, `ParserVersion = 8`.

## Pins

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Band energy | −30.0466782447 | Ha |
| **Energy SCC** | **+0.1379943876** | Ha |
| Repulsive energy | +0.6186556656 | Ha |
| **Total energy** | **−32.6355809899** | Ha |
| Total energy | **−888.0593996784** | eV |
| Energy SCC | +3.7550185808 | eV |

Conversion 1 Ha = 27.211386245988 eV. Per-atom forces [eV/Å] are transcribed
into the regression source from `detailed.out`.

## Gates

| Gate | Tolerance |
| ---- | --------- |
| Total energy vs DFTB+ | 1×10⁻³ eV |
| **SCC energy vs DFTB+** (on its own) | 1×10⁻³ eV |
| max per-component force vs DFTB+ | 1×10⁻³ eV/Å |
| max \|Δq\| (the cell is actually polar) | > 0.1 e |

Observed at the time of writing: |ΔE| = 1.9×10⁻⁶ eV, |ΔE_SCC| = 1.8×10⁻⁶ eV,
max |ΔF| = 3.4×10⁻⁶ eV/Å, max |Δq| ≈ 0.6 e.

## Re-run oracle

```bash
source tools/tb_oracle/setup_dftbplus_env.sh
tools/tb_oracle/run_dftbplus_pin.sh tools/tb_oracle/inputs/dftbplus/dftb2-ewald-gamma mio
```
