# Reference — zero-SCF linearized XL-BOMD forces on Γ-UKS (O₂ triplet NVE, GPW)

## Method

**Zero-SCF linearized-functional XL-BOMD** (Niklasson shadow Born–Oppenheimer
potential) on collinear Γ-UKS Gaussian-and-plane-waves (GPW) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE)
- Force/energy: `cpu::gpw::GPWEvaluator` (`n_alpha = 7`, `n_beta = 5`) wrapped
  by `cpu::electronic::XlBomdForce{gpw, 5, XlBomdLinearized{}}`
- Cold seed (step 0): full UKS SCF → seed **both** spin XL histories with
  \(P^\sigma_{\mathrm{SCF}}\)
- Warm steps: dual auxiliary densities \(n^\sigma = P^\sigma_{\mathrm{aux}}\);
  one dual-KS construction \(H^\sigma[n]\) (fixed-density, `max_iter = 0`);
  one generalized diagonalization **per spin** → \(D^\sigma = q^\sigma[n]\)
  (occupancy 1); **no iterative SCF** (`last_result().iters == 0`)
- Dual-spin shadow energy
  \(\mathcal{E} = \sum_\sigma\left[\mathrm{Tr}(H^\sigma[n]\,D^\sigma)
  - \mathrm{Tr}(H^\sigma[n]\,n^\sigma)\right] + E_{\mathrm{KS}}[n^\alpha,n^\beta]\)
  (recovers \(E_{\mathrm{KS}}\) at \(D^\sigma = n^\sigma\))
- Forces: Hellmann–Feynman of \(\mathcal{U}(R,n^\alpha,n^\beta)\) via the
  dual-spin fixed-density overload
  `gpw_forces_at_density(D^α, C^α_occ, H^α, D^β, C^β_occ, H^β)`
- XL advance(\(D^\sigma\)) per spin with Table-I \(K = 5\) dissipation
- ⟨S²⟩: Löwdin at the shadow densities (same formula as the converged-SCF
  store; Szabo & Ostlund §2.5)

**Literature:**

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) — Table I
  (unit-pinned in `test_cpu_gpw_xl_bomd`; identical coefficients per spin)
- A. M. N. Niklasson & M. J. Cawkwell, *Phys. Rev. B* **86**, 174308 (2012);
  Niklasson, *J. Chem. Phys.* **147**, 054103 (2017) — shadow / linearized
  free-energy functional (spin-diagonal for collinear UKS)
- VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW

## System

**Identical twin of `xl-bomd-uks`** — only the `XlBomdLinearized{}` tag is
added, so the pair is a direct SCF-XL vs zero-SCF linearized-XL A/B on the
same open-shell system (same family as `gpw-uks-o2-gamma` / `gpw-uks-forces`):

| Quantity | Value |
| -------- | ----- |
| Species | O₂ triplet (2 × O), isolated / MT |
| Multiplicity | 3 (`n_alpha = 7`, `n_beta = 5`; 12 valence e⁻) |
| Geometry | \(r_{\mathrm{OO}} = 1.2075\) Å, centered |
| Cell | cubic \(L = 12\) Å; nonperiodic |
| XC | PADE (LDA) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 |
| CUTOFF | 200 Ry |
| SCF | `tol = 1e-5`, `max_iter = 80`, `mix = 0.4`, DIIS diagonalization |
| XL-BOMD | \(K = 5\); linearized forces on; dual α/β propagators |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_z = \pm 0.0008\) Å/fs |
| Steps | 1 cold seed + 4 production (chunked) |

Note (system choice): the queue sketched an OH-radical "water-family" twin,
but the OH doublet cold seed is a pathological DIIS case in this tree (locks
in only at ~196 iterations at every `mix`/`tol` probed, PBE; PADE worse) —
measured 2026-07-31 on the worker host. The O₂ triplet twin converges its
seed in ~23 iterations and pins the same dual-spin linearized-XL physics.

## Offline oracle (CP2K)

**Status:** no `cp2k` binary / CP2K Lmod module on the worker host used for
this pin (Arrhenius login-class, **2026-07-31** inventory — `module avail`
and `PATH` both empty of cp2k) — same as the whole XL/BOMD AIMD family
(`xl-bomd-water`, `xl-bomd-uks`, `xl-bomd-linearized-forces`). Twin sketch
kept at
`tools/cp2k_oracle/inputs/xl-bomd-linearized-uks/o2_uks_xl_linearized_nve.inp`
(Quickstep UKS MD, low `MAX_SCF` / density-matrix propagation notes). When a
binary is available, re-run offline via `tools/cp2k_oracle/` and replace this
section with version + measured drift + fair wall.

Until then the regression asserts the **literature shadow-XL / NVE envelope**
plus **zero warm-step SCF iterations**, the Löwdin ⟨S²⟩ triplet window, and
trajectory diagnostics. Table-I coefficients remain unit-pinned
(`test_cpu_gpw_xl_bomd`); the dual-spin shadow-energy fixed-point identity is
unit-pinned in the same file.

## Observable and tolerance

Let \(E(t) = E_{\mathrm{pot}}(t) + E_{\mathrm{kin}}(t)\) [eV], with \(E_0\)
after the cold seed.

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- Warm steps: `last_result().iters == 0` and `converged == true`
- Cold seed: `iters > 0` and converged
- ⟨S²⟩ in \([1.5, 2.5]\) (triplet window; exact 2) after seed **and** at
  every sampled production step (shadow densities)
- PE finite and bound (\(< -100\) eV after seed)
- KE finite, \(\ge 0\), non-trivial after seed
- Soft O–O window: \(0.9 < r_{\mathrm{OO}} < 1.6\) Å after production
- All coordinates finite

Measured on the worker host (2026-07-31, 8 threads): worst relative drift
**1.8e-07**, warm iters **0**, ⟨S²⟩ 2.0047 (seed) → 2.0047 (final),
\(r_{\mathrm{OO}}\) 1.220 Å, wall 40 s.

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary for fair AIMD wall twin (2026-07-31 inventory) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

Note: warm steps are cheaper than full-SCF XL-UKS (one dual-KS build + two
diagonalizations vs iterative dual-spin DIIS).

## Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | A1–A3 unit (`test_cpu_gpw_xl_bomd`); A4 `reg_xl-bomd-linearized-uks` |
| Speed | `perf_exempt=true` |
