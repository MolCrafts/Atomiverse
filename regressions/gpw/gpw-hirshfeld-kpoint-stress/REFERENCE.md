# Reference — multi-k GPW cell stress with Hirshfeld-scaled TS/MBD volumes

## Method

**Monkhorst–Pack multi-k GPW** analytic cell stress with **Hirshfeld volume
ratios** \(v_A = V_A/V_A^{\mathrm{free}}\) scaling free-atom **TS** and
**MBD@rsSCS** parameters under orthorhombic diagonal strain.

Closes the residual that multi-k energy Hirshfeld
(`gpw-hirshfeld-kpoints`), multi-k force Hirshfeld
(`gpw-hirshfeld-kpoint-forces`), and Γ stress Hirshfeld (gate inside
`gpw-hirshfeld-forces`) were product-pinned while the multi-k stress path
already folds `scf.volume_ratios` into the fixed-\(P_k\) Lagrangian without a
dedicated dual-gate pin.

Product definition (density-chain \(\partial v/\partial\varepsilon\) out of
scope):

\[
\sigma_{aa}
  = -\Omega^{-1}
    \frac{\partial E_L[\{P_k\},\{W_k\};\varepsilon]}{\partial\varepsilon_{aa}}
\Big|_{v}
\quad\text{with \(v\) fixed at the multi-k SCF density}.
\]

Public API:

```cpp
GpwDft gpw;
gpw.ts = true;          // or gpw.mbd = true
gpw.hirshfeld = true;
gpw.k_mesh[0] = gpw.k_mesh[1] = 1;
gpw.k_mesh[2] = 2;
gpw.build(atoms);
const auto& scf = gpw.run(atoms);
auto sig = gpw_stress(atoms, gpw, scf, /*h=*/2e-3);
```

Scaling of free-atom \(\alpha_0\) / \(C_6\) / \(R_{\mathrm{vdW}}\):
\(\alpha\propto v\), \(C_6\propto v^2\), \(R\propto v^{1/3}\).

## Reference values (hard gates)

### A/B — Geometry-only fixed-\(v\) dual (water dimer)

| Quantity | Source | Value / tol |
| -------- | ------ | ----------- |
| **E_disp isolated MBD@rsSCS + fixed \(v\)** | product kernel = dual (same volumes as `gpw-hirshfeld-forces`) | **−1.1479717148086300×10⁻³ Ha**, tol 1×10⁻⁸ |
| **Geometry stress dual (TS, MBD)** | central FD of fixed-\(v\) \(E_{\mathrm{disp}}\) under orthorhombic strain \(h=2\times10^{-3}\) | \(\sigma_{\mathrm{disp}}\) finite and nonzero; self dual ≤ **1×10⁻¹²** Ha/Bohr³ |
| **Not free-atom** | max\|σ_hirsh − σ_free\| (TS) | must exceed **1×10⁻¹²** Ha/Bohr³ |

Representative fixed volumes (O,H,H,O,H,H):

`0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784`

### C/D — Multi-k product wiring (H₂, k-mesh 1×1×2)

Compressed H₂ (same geometry as `gpw-hirshfeld-kpoint-forces` /
`gpw-mbd-kpoint-forces`): PADE, DZV-GTH-PADE, `k_mesh={1,1,2}`, CUTOFF 100 Ry.

| Quantity | Gate |
| -------- | ---- |
| SCF converged; multi-k payload (`n_kpts ≥ 1`, `C_occ_k` non-empty) | hard |
| `volume_ratios` size 2; \(E_{\mathrm{disp}} < 0\) | hard |
| \(\sigma^{k+\mathrm{ts}+\mathrm{hirsh}}-\sigma^{k,\mathrm{plain}}\approx\sigma_{\mathrm{TS}}(v_{\mathrm{SCF}})\) | max\|Δ\| ≤ **1×10⁻⁸** Ha/Bohr³ |
| same for MBD@rsSCS (isolated when `mbd_k_mesh` default zero) | max\|Δ\| ≤ **1×10⁻⁸** Ha/Bohr³ |
| residual vs free-atom stress nontrivial | max\|Δ_free\| > **1×10⁻¹²** Ha/Bohr³ |
| orthorhombic off-diagonals | exactly 0 |

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — **no** stock TS/MBD. Hard pin is **libMBD /
product dual** + multi-k product wiring. Twin notes:
`tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoint-stress/`.

**oracle:** literature / libMBD (volume-scaled strain dual).  
**perf_exempt:** true — no fair CP2K twin algorithm (same policy as
`gpw-hirshfeld-forces` / `gpw-hirshfeld-kpoint-forces` / `gpw-mbd-kpoint-stress`).

Host binary check (2026-08-01):

```
cp2k.psmp --version → CP2K version 2026.1 (git:5e54ba2)
```

## Geometry

### Water dimer (Å) — fixed-\(v\) dual

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Isolated MBD dual uses these nuclei without lattice images. Periodic TS dual
uses the same coordinates in a **12 × 12 × 12 Å** cube (`PERIODIC XYZ`).

### H₂ (Bohr) — multi-k wiring

| Atom | x | y | z |
| ---- | - | - | - |
| H | 5.5 | 6.0 | 6.0 |
| H | 6.5 | 6.0 | 6.0 |

Cell: **12 × 12 × 12 Bohr**, periodic; electronic **k = 1×1×2**;
basis `DZV-GTH-PADE` + `GTH-PADE-q1`; XC PADE; CUTOFF **100 Ry**.

## Perf gate

**perf_exempt = true** — primary deliverable is fixed-\(v\) stress dual-gate +
multi-k product wiring. CP2K has no matched TS/MBD `STRESS_TENSOR` twin for a
fair wall ratio. Regression wall budget ≤ 60 s (two H₂ multi-k SCFs +
geometry duals).
