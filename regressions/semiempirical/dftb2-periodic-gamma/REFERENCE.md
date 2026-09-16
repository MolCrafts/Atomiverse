# dftb2-periodic-gamma — Γ-point periodic SCC-DFTB2

```text
oracle: literature
oracle_version: ATV-mio-style-sketch (Hamiltonian twin; not DFTB.org mio-1-1)
family_oracle: dftb+
```

**Note:** No offline DFTB+ binary was available on the build host. Absolute
energy is pinned to the ATV static mio-style SK Hamiltonian at the geometry
below (same hygiene as `h2o_dftb2_mio`). Family numerical oracle for this
method remains **DFTB+** once bit-identical SK tables ship
(`dftb2-dftbplus-energy-pin` / `dftb2-3ob-organic`).

## Method

Self-consistent-charge DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260
(1998), DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260))
at the **Γ-point**: lattice-summed two-center \(H^0\) and \(S\), screened
DFTB+ Elstner \(\gamma\), and exponential repulsive within the static
pair cutoff from `basis_data/dftb_mio_tables.h`. No k-mesh; no Ewald
long-range Coulomb (γ is real-space truncated at \(R_\mathrm{cut}\)).

## Data

In-tree static tables `basis_data/dftb_mio_tables.h` for H/C/N/O only.
**Not** a full DFTB.org mio-1-1 `.skf` dump.

## Geometry

1-D carbon dimer crystal (two C atoms per orthorhombic cell):

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C    | 0.00  | 0.00  | 0.00  |
| C    | 1.40  | 0.00  | 0.00  |

Cell: orthorhombic \(L_x = 2.8\) Å, \(L_y = L_z = 12.0\) Å.

With \(r(\mathrm{C–C})=1.4\) Å and \(L_x=2.8\) Å, each carbon has
nearest-neighbor contacts along ±x from the partner and its periodic
images; lattice-sum H/S differs from a single min-image molecular dimer.

## SCC settings

- `scc_tol = 1e-10`
- `max_scc = 50` (default)
- `with_forces = false` (energy gate)

## Pin

| Quantity | Value | Unit | Tol |
|----------|-------|------|-----|
| Total energy \(E\) | **−94.5030857189** | eV | ±0.05 eV |

Also required: \(|E - E_\mathrm{mol}| > 0.05\) eV where \(E_\mathrm{mol}\) is
the same dimer in a 20 Å cubic box (molecular limit).

## Re-pin (2026-07-29)

Full mio-1-1 C–C SK tables replace the sketch pair scales; Γ-PBC energy
moved from −113.60 eV (sketch) to **−94.503 eV**. Homonuclear Δq ≡ 0 so
Elstner γ is inert here — this is an H⁰/repulsive twin, not an SCC check.
Still **not** a DFTB+ absolute pin (no offline C-chain twin input yet).

## Perf

```text
perf_exempt: true
perf_exempt_reason: no offline DFTB+ binary on host; fair wall deferred
```
