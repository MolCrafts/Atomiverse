# Reference — DFTB2 Γ-PBC lattice-sum D3(BJ)

## Method

SCC-DFTB2 (ATV static mio-style SK sketch) with the opt-in **Grimme DFT-D3
Becke–Johnson** two-body correction, evaluated as a **lattice sum over
orthorhombic cell images** instead of an isolated cluster
(`DFTB2::d3_periodic = true`).

Damping = published **DFTB3-D3(BJ)** set (`s-dftd3 --bj dftb3`), same as the
molecular sibling `dftb2-dispersion-d3`:

| param | value |
| ----- | ----- |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.5883 |
| \(a_1\) | 0.5719 |
| \(a_2\) | 3.6017 Bohr |

CN form: molecular Grimme (simple-dftd3 path), **not** CP2K tanh. ATM off.

## Why this is not a second molecular pin

The molecular sibling pins the isolated-cluster sum. Here the *same* geometry
is evaluated both ways in one run and the gate is on **all three** of:

1. the cluster sum (must still reproduce the molecular pin),
2. the lattice sum (absolute periodic pin), and
3. **their difference — the image term**, which a cluster-only implementation
   reports as exactly `0`.

The image term is what this feature adds; gate 3 is the discriminating one.

## The cutoff is part of the pinned quantity

A lattice sum is only defined up to its real-space truncation: the
\(C_6/R^6\) tail beyond the cutoff is discarded, and D3 engines disagree on
where to stop. ATV's `DFTB2::d3_cutoff` defaults to **94.2 Bohr** (legacy
`dftd3` `rthr` convention, unchanged); **simple-dftd3 truncates the two-body
term at 60 Bohr** and exposes no CLI flag to change it.

Measured on this geometry (cluster term identical in every row, so the spread
is pure image tail):

| `d3_cutoff` [Bohr] | ATV \(E_{\mathrm{disp}}\) [Ha] |
| ------------------ | ------------------------------ |
| 40.0 | −1.6956613842920e−03 |
| 60.0 | −1.6959658556667e−03 |
| 94.2 (default) | −1.6960673778506e−03 |
| 160.0 | −1.6960985763136e−03 |

Against the oracle's −1.6959579071183e−03, the default cutoff looks 3.0e−6 eV
"wrong" and the matched 60 Bohr cutoff agrees to **2.2e−7 eV**. The
regression therefore sets `d3_cutoff = 60 Bohr` so the comparison is
like-for-like. `d3_cutoff` was added as a **public member** for exactly this
reason — before it, the periodic sum could not be pinned against any external
engine. Default behaviour is bit-identical to the previous hard-coded 94.2
Bohr (gated in `test_dftb2_dispersion_d3.cpp`).

The residual 2.2e−7 eV is the **coordination-number** cutoff: ATV drives CN
and the pair sum from one radius, simple-dftd3 uses a separate 40 Bohr CN
cutoff. CN enters only through the interpolated \(C_6(\mathrm{CN})\), so the
effect is third-order small. See Gaps.

## Oracle

```text
oracle: literature
oracle_version: Simple-DFTD3/s-dftd3 1.2.1 (offline)
```

DFTB+ is the family oracle for the electronic DFTB ladder; for this
geometry-only dispersion correction the simple-dftd3 engine is the correct pin
— it is the same D3 model DFTB+ calls. ctest never launches s-dftd3 or DFTB+.

## Hard gates

**Lattice (periodic) — \(E_{\mathrm{disp}} = -1.6959579071183\times10^{-3}\) Ha**
**Cluster (molecular) — \(E_{\mathrm{disp}} = -1.6892886213669\times10^{-3}\) Ha**
**Image term — difference \(= -6.6692857514\times10^{-6}\) Ha**

Conversion: ATV `kHartreeInEv = 27.211386245988`.

1. `|E_disp_cluster − pin_cluster| ≤ 1e-6 eV` (measured 1.5e-12)
2. `|E_disp_lattice − pin_lattice| ≤ 1e-6 eV` (measured 2.2e-7)
3. `|image − pin_image| ≤ 1e-6 eV` (measured 2.2e-7) — **discriminating**
4. `E_disp_lattice < E_disp_cluster` (images must attract)
5. `E_tot(d3 periodic) − E_tot(off) = E_disp_lattice` within 1e-8 eV

## Geometry (Å)

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000 | 0.000 | 0.000 |
| H | 0.757 | 0.000 | 0.586 |
| H | −0.757 | 0.000 | 0.586 |
| O | 0.000 | 0.000 | 2.900 |
| H | 0.757 | 0.000 | 3.486 |
| H | −0.757 | 0.000 | 3.486 |

Cubic cell **12 Å** — small enough that the nearest images sit well inside the
D3 cutoff (the 20 Å cell of the molecular sibling would make the image term
~10× smaller and the pin correspondingly less discriminating).

## Offline reproduce

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/Simple-DFTD3/1.2.1-gfbf-2025b/bin:$PATH
cd tools/tb_oracle/inputs/dftbplus/dftb2-dispersion-d3-periodic
s-dftd3 --bj dftb3 dimer_pbc12.gen   # lattice sum (periodic .gen input)
s-dftd3 --bj dftb3 dimer_mol.xyz     # isolated cluster
```

Oracle inputs: `tools/tb_oracle/inputs/dftbplus/dftb2-dispersion-d3-periodic/`.

## Unit gates (companion, `tests/cpu/semiempirical/test_dftb2_dispersion_d3.cpp`)

The oracle pin fixes the number; these fix the *structure* of the sum:

- `BeyondCutoffBoxEqualsMolecular` — a 120 Å cell (every image past the cutoff)
  collapses onto the cluster sum to 1e-12: no residual image term, no
  double-counted home cell.
- `SupercellEnergyIsExtensive` — \(E(2{\times}2{\times}2) = 8\,E(1{\times}1{\times}1)\)
  to 1e-8 relative. The decisive Bravais-sum identity: a wrong image count, a
  half-counted home cell, or a non-image-aware coordination number all break it.
- `ForcesMatchCentralFDOfEDisp` — periodic analytic \(F_{D3}\) vs central FD of
  the periodic \(E_{\mathrm{disp}}\), < 1e-6 eV/Å.
- `StressTraceMatchesVolumeDerivative` — image-aware virial:
  \(\mathrm{tr}\,\sigma_{D3} = -(3/V)\,\partial E_{\mathrm{disp}}/\partial\ln V\)
  under affine scaling of **cell and positions together**.
- `CutoffTruncatesTheLatticeTail` — monotone, converging tail; default member
  value reproduces the historical 94.2 Bohr result exactly.
- `NonOrthorhombicBoxThrows`, `NonPositiveCutoffThrows`.

## Gaps

1. **Orthorhombic only.** `d3_periodic` reads `box.length(d)` and throws on a
   triclinic cell. General-lattice image enumeration is owed.
2. **One CN cutoff.** ATV drives the coordination number from the same radius
   as the pair sum; simple-dftd3 uses a separate (40 Bohr) CN cutoff. Worth
   2.2e-7 eV here. Splitting the two would close the pin to ~1e-12 like the
   molecular case.
3. **No ATM.** Three-body \(C_9\) is off on both sides (`s9 = 0`), so this
   pins the two-body lattice sum only.
4. **Not gated against a periodic DFTB+ run.** The dispersion model is pinned
   against the engine DFTB+ itself calls, but no end-to-end DFTB+ periodic
   DFTB2+D3 total energy is pinned — no `dftb+` binary exists on this host.

## Perf

`perf_exempt: true` — the D3 lattice sum is microseconds; DFTB2 wall time is
electronics-dominated.
