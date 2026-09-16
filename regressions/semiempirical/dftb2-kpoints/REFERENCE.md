# dftb2-kpoints — multi-k SCC-DFTB2 (4×1×1 Monkhorst–Pack)

```text
oracle: dftb+
oracle_version: DFTB+ family (offline twin; ATV mio-style SK Hamiltonian pin)
family_oracle: dftb+
```

**Note:** No offline DFTB+ binary was available on the build host. Absolute
energy is pinned to the ATV static mio-style SK Hamiltonian at the geometry
and mesh below (same hygiene as `dftb2-periodic-gamma`). Family numerical
oracle for this method remains **DFTB+** once bit-identical SK tables ship.

## Method

Self-consistent-charge DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260
(1998), DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260))
with **Monkhorst–Pack k-mesh** beyond Γ:

- Bloch-phased lattice sums of two-center \(H^0(\mathbf{k})\) and
  \(S(\mathbf{k})\) (fractional \(\kappa\), phase \(e^{2\pi i\,\kappa\cdot n}\))
- k-independent SCC shift \(\varepsilon_A=\sum_B\gamma_{AB}\Delta q_B\)
- k-weighted Mulliken Δq and band energy
  \(\sum_k w_k\,\mathrm{Re}\,\mathrm{Tr}(P^k H^{0k})+E^{(2)}+E_\mathrm{rep}\)
- Real-space screened \(\gamma\) and repulsive: lattice-summed (Γ-style),
  no Ewald
- Mesh convention matches CP2K/DFTB+: \(\kappa_d=(2r-N_d-1)/(2N_d)\),
  optional time-reversal folding of \(\pm\mathbf{k}\)

No multi-k analytic forces/stress in this foothold.

## Data

In-tree static tables `basis_data/dftb_mio_tables.h` for H/C/N/O only.
**Not** a full DFTB.org mio-1-1 `.skf` dump. SK set: `DftbSkSet::MioSketch`.

## Geometry

1-D carbon dimer crystal (two C atoms per orthorhombic cell) — same as
`dftb2-periodic-gamma`:

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C    | 0.00  | 0.00  | 0.00  |
| C    | 1.40  | 0.00  | 0.00  |

Cell: orthorhombic \(L_x = 2.8\) Å, \(L_y = L_z = 12.0\) Å.

## k-mesh

- `k_mesh = {4, 1, 1}` (sampling along the chain axis)
- `k_time_reversal = true`

Offline DFTB+ twin: `tools/tb_oracle/inputs/dftbplus/dftb2-kpoints/`
(`KPointsAndWeights = SupercellFolding { 4 0 0; 0 1 0; 0 0 1; 0 0 0 }`).

## SCC settings

- `scc_tol = 1e-10`
- `max_scc = 50` (default)
- `with_forces = false` (multi-k energy gate)

## Pin

| Quantity | Value | Unit | Tol |
|----------|-------|------|-----|
| Total energy \(E\) (4×1×1) | **−98.9944775364** | eV | ±0.05 eV |

Also required:

- \(|E - E_\Gamma| > 0.01\) eV where \(E_\Gamma\) is the same cell at
  `k_mesh={1,1,1}`
- \(E_\Gamma = -113.5976598145\) eV ± 0.05 eV (control: matches
  `reg_dftb2-periodic-gamma`)

## Public API

```cpp
cpu::semiempirical::DFTB2 dftb;
dftb.k_mesh = {4, 1, 1};
dftb.k_time_reversal = true;
dftb.with_forces = false;
// members only — no *Config, no make_*/build_*
```

## Perf

```text
perf_exempt: true
perf_exempt_reason: no offline DFTB+ binary on host; fair wall deferred
```
