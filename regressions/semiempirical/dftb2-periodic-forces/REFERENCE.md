# dftb2-periodic-forces — lattice-image Γ-PBC nuclear forces

```text
oracle: literature
oracle_version: formula-level (Elstner 1998; Hourahine et al. JCP 152, 124101 (2020); ATV mio-style sketch)
family_oracle: dftb+
```

**Note:** No offline DFTB+ binary on the pin host. Absolute \(E\)/\(F\) vs
DFTB.org mio-1-1 is **not** claimed (sketch SK tables). The pin is the
**module-boundary identity**: analytic \(F=-\nabla E\) of the lattice-summed
SCC-DFTB2 energy equals a re-SCC central finite difference of that energy under
the same Γ-point lattice walk. Family numerical oracle remains **DFTB+**.

## Method

Self-consistent-charge DFTB2 at the **Γ-point** with lattice-summed \(H^0\),
\(S\), screened Elstner \(\gamma\), and exponential repulsive (same as
`dftb2-periodic-gamma`). Nuclear forces are the **closed-form** fixed-\(\Delta q\)
gradient (Hellmann–Feynman + Pulay + \(\partial\gamma\) + \(\partial E_{\mathrm{rep}}\))
walked over **all lattice images within \(r_{\mathrm{cut}}\)** — not min-image
only. Self-images (\(A=A+\mathbf{T}\), \(\mathbf{T}\ne 0\)) contribute zero
nuclear force (geometry-independent \(|\mathbf{T}|\)).

## Data

In-tree static tables `basis_data/dftb_mio_tables.h` (mio-**style** sketch
H/C/N/O). **Not** a full DFTB.org mio-1-1 dump.

## Geometry

Off-equilibrium 1-D carbon dimer crystal (two C per orthorhombic cell):

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C    | 0.00  | 0.00  | 0.00  |
| C    | 1.20  | 0.00  | 0.00  |

Cell: orthorhombic \(L_x = 2.8\) Å, \(L_y = L_z = 12.0\) Å.

With \(r(\mathrm{C–C})=1.20 \ne L_x/2\), left/right lattice images sit at
1.20 Å and 1.60 Å (plus farther shells inside \(r_{\mathrm{cut}}=4.5\) Å), so
net \(F_x \ne 0\). The equilibrium twin \(r=1.4\) of `dftb2-periodic-gamma`
has vanishing force by symmetry and is not used here.

## SCC settings

- `scc_tol = 1e-10`
- `max_scc = 50` (default)
- `with_forces = true`
- `sk_set = MioSketch` (default)
- `d3 = false`

## Pins (hard-coded gates)

| Gate | Criterion |
|------|-----------|
| Energy | finite, \(\|E\| > 10^{-3}\) eV |
| Forces | all components finite; \(\|F\|^2 > 10^{-8}\) |
| Translation | \(\|\sum_i F_i\| \le 10^{-6}\) eV/Å per Cartesian |
| FD identity | analytic \(F_{0,x}\) and \(F_{1,x}\) vs re-SCC central FD (\(h=10^{-4}\) Å) within **1×10⁻⁵ eV/Å** |
| PBC vs molecular | \(\|E_{\mathrm{PBC}} - E_{\mathrm{mol}}\| > 0.05\) eV (20 Å cubic same dimer) |

## Perf

```text
perf_exempt: true
perf_exempt_reason: no offline DFTB+ binary on host; fair wall deferred
```

Offline twin note: `tools/tb_oracle/inputs/dftbplus/dftb2-periodic-forces/`.
