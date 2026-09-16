# Reference — zero-SCF linearized XL-BOMD forces (H₂O NVE, GPW)

## Method

**Zero-SCF linearized-functional XL-BOMD** (Niklasson shadow Born–Oppenheimer
potential) on Gaussian-and-plane-waves (GPW) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE)
- Force/energy: `cpu::gpw::GPWEvaluator` with
  **`xl_bomd = true`, `xl_bomd_k = 5`, `xl_bomd_linearized = true`**,
  `compute_forces = true`
- Cold seed (step 0): full SCF → seed XL history with \(D_{\mathrm{SCF}}\)
- Warm steps: auxiliary \(n = P_{\mathrm{aux}}\); one \(H[n]\) construction
  (fixed-density, `max_iter = 0`); one diagonalization \(D = q[n]\); **no
  iterative SCF** (`last_result().iters == 0`)
- Shadow energy
  \(\mathcal{E} = \mathrm{Tr}(H[n]\,D) - \mathrm{Tr}(H[n]\,n) + E_{\mathrm{KS}}[n]\)
  (recovers \(E_{\mathrm{KS}}\) at \(D = n\))
- Forces: Hellmann–Feynman of \(\mathcal{U}(R,n)\) via
  `gpw_forces_at_density(D, C_{\mathrm{occ}}, H[n])`
- XL advance(\(D\)) with Table-I \(K=5\) dissipation

**Literature:**

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) — Table I
  (unit-pinned in `test_cpu_gpw_xl_bomd`)
- A. M. N. Niklasson & M. J. Cawkwell, *Phys. Rev. B* **86**, 174308 (2012);
  Niklasson, *J. Chem. Phys.* **147**, 054103 (2017) — shadow / linearized
  free-energy functional
- VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW

## System

Identical twin of `xl-bomd-water` (only `xl_bomd_linearized` differs):

| Quantity | Value |
| -------- | ----- |
| Species | H₂O (O + 2 H), isolated / MT |
| Geometry | CP2K getting-started |
| Cell | cubic \(L = 10\) Å; nonperiodic |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 200 Ry |
| XL-BOMD | \(K = 5\); linearized forces on |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Steps | 1 cold seed + 4 production |

| Atom | x | y | z (Å) |
| ---- | - | - | - |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

## Offline oracle (CP2K)

**Status:** no `cp2k` binary / Lmod module on the worker host used for this
pin (Arrhenius login-class, 2026-07-26 inventory) — same as prior XL/BOMD
AIMD slugs. Twin sketch kept at
`tools/cp2k_oracle/inputs/xl-bomd-linearized-forces/h2o_xl_linearized_nve.inp`
(Quickstep MD with low `MAX_SCF` / density-matrix propagation notes). When a
binary is available, re-run offline via `tools/cp2k_oracle/` and replace this
section with version + measured drift + fair wall.

Until then the regression asserts the **literature shadow-XL / NVE envelope**
plus **zero warm-step SCF iterations** and trajectory diagnostics. Table-I
coefficients remain unit-pinned (`test_cpu_gpw_xl_bomd`).

## Observable and tolerance

Let \(E(t) = E_{\mathrm{pot}}(t) + E_{\mathrm{kin}}(t)\) [eV], with \(E_0\) after
the cold seed.

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- Warm steps: `last_result().iters == 0` and `converged == true`
- Cold seed: `iters > 0` and converged
- PE finite and bound (\(< -100\) eV after seed)
- KE finite, \(\ge 0\), non-trivial after seed
- Soft O–H: both \(r_{\mathrm{OH}} < 1.8\) Å after production

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary for fair AIMD wall twin (2026-07-26) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

Note: warm steps are cheaper than full-SCF XL (one \(H\) build + one
diagonalization vs iterative DIIS).

## Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | A1–A4 unit; A5 `reg_xl-bomd-linearized-forces` |
| Speed | `perf_exempt=true` |
