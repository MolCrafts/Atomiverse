# Reference — linear-response TD-DFTB (Casida) on SCC-DFTB2

```text
oracle: literature
oracle_version: Niehaus et al., Phys. Rev. B 63, 085108 (2001) — Casida
  TD-DFTB. No citable H2/mio-1-1 TD-DFTB value exists (see below).
```

## Method

Ground state: SCC-DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998),
DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)),
restricted closed shell, Γ point, `MioSketch` Slater–Koster tables.

Linear response: the tight-binding formulation of time-dependent
density-functional response theory of T. A. Niehaus, S. Suhai, F. Della Sala,
P. Lugli, M. Elstner, G. Seifert, Th. Frauenheim, *Phys. Rev. B* **63**, 085108
(2001), DOI
[10.1103/PhysRevB.63.085108](https://doi.org/10.1103/PhysRevB.63.085108).

Mulliken transition charges [e]

\[
q^{ia}_A = \tfrac12\sum_{\mu\in A}\Big(c_{\mu i}(Sc_a)_\mu + c_{\mu a}(Sc_i)_\mu\Big)
\]

coupling kernel [eV]

\[
K^{S}_{ia,jb} = \sum_{AB} q^{ia}_A\,\gamma_{AB}\,q^{jb}_B ,\qquad
K^{T}_{ia,jb} = \sum_{A} q^{ia}_A\,W_A\,q^{jb}_A
\]

(the singlet kernel uses the **same** \(\gamma_{AB}\) as the second-order SCC
energy; the triplet magnetization kernel is on-site,
\(\gamma^{m}_{AB}=\delta_{AB}W_A\)), and the Hermitian Casida eigenproblem
\(\Omega F=\omega^2F\) [eV²]

\[
\Omega_{ia,jb} = \delta_{ij}\delta_{ab}(\varepsilon_a-\varepsilon_i)^2
+ 2\sqrt{\varepsilon_a-\varepsilon_i}\;K_{ia,jb}\;
\sqrt{\varepsilon_b-\varepsilon_j}
\]

solved densely (LAPACK `dsyevd` through `xt::linalg::eigh`).

- ATV surface: `cpu::semiempirical::DFTB2::td_casida(frame, n_states, channel,
  n_occ_window, n_virt_window, coupling_scale)` → `TdDftbResult`
  (`include/atomiverse/cpu/semiempirical/td_dftb2.h`). Public headers + Frame
  only; no `*Config` bag, no factory, no `*System`.
- **Energies only.** No oscillator strengths, no transition dipoles, no
  excited-state gradients.

## Hamiltonian / tables

`basis_data/dftb_mio_tables.h` + transcribed mio-1-1 H–H. The triplet kernel
uses the published PBE \(W_{\mathrm{ss}}(\mathrm{H})=-0.072\) Ha
\(= -1.959219809711\) eV via `spin_w_ha = basis_data::kDftbSpinConstantsPbeHa`
(HOAO collapse; `shell_resolved_spin` default `false`). The singlet kernel
is \(\gamma_{AB}\) and does not take \(W\).

## Geometry

**H₂** — Z = 1, 1, at (0,0,0) and (r,0,0) Å, cubic box 20 Å (isolated),
r = 0.74 Å (pinned point) and r = 1.20 Å (stretched check).
`scc_tol = 1e-12`, `max_scc = 120`, `sk_set = MioSketch`, `k_mesh = {1,1,1}`,
`charge = 0`, `multiplicity = 1`.

Minimal basis ⇒ H₂ has exactly one occupied (σ_g) and one virtual (σ_u)
orbital, i.e. exactly **one** occupied→virtual channel and a 1×1 Casida matrix.

**H₂O** (multi-pair coverage) — O (0,0,0), H (1.05,0,0), H (−0.26,1.0,0) Å,
cubic box 20 Å. n_ao = 6, n_occ = 4, n_virt = 2 ⇒ **8** pairs.

## Pins (hard-coded gates)

| # | Gate | Tolerance | Measured |
| - | ---- | --------- | -------- |
| 1 | **λ = 0 ⇒ ω = ε_a − ε_i** (H₂, 1 state) | 1e-10 eV | 0.000e+00 |
| 2 | **λ = 0 ⇒ ω = ε_a − ε_i** (H₂O, 4 states) | 1e-10 eV | 0.000e+00 |
| 3 | **Ω symmetric** (H₂O, 8×8, as assembled — never re-symmetrized) | 1e-12 eV² | 3.553e-15 |
| 4 | **Σ_A q^{ia}_A = 0** (H₂O, all 8 pairs) | 1e-10 e | 2.776e-16 |
| 5 | H₂: ω_T < (ε_a−ε_i) < ω_S, all finite and > 0 | strict | 13.676 < 15.441 < 16.392 |
| 6 | H₂: q_H1 = −q_H2 and \|q\| > ½ | 1e-10 e / strict | ±0.6515851161 |
| 7 | H₂: ω_S(1.20 Å) < ω_S(0.74 Å) (σ_g/σ_u splitting closes) | strict | 7.445 < 16.392 |
| 8 | H₂ **snapshot** ω_S, ω_T, gap (drift guard, **not** an oracle) | 1e-6 eV | ≤ 5e-09 |

Gate 1/2 is the decisive structural check: at zero coupling Ω is diagonal in
\((\varepsilon_a-\varepsilon_i)^2\), so every root must be a bare
single-particle gap. It exercises the whole path — SCC, overlap, MO ordering,
window selection, √Δε prefactors, eigensolve — while being independent of the
parameter quality.

Gate 6 is analytic: for a symmetric two-level system \(c_i=(a,a)\),
\(c_a=(b,-b)\) with \(a=1/\sqrt{2+2S_{12}}\), \(b=1/\sqrt{2-2S_{12}}\) give
\(q_1 = ab = 1/(2\sqrt{1-S_{12}^2}) = -q_2 > \tfrac12\) — a closed form the
general Mulliken code path must satisfy without being told \(S_{12}\).
(The measured 0.65159 back-solves to \(S_{12}=0.6412\), the mio-1-1 H–H overlap
at 0.74 Å.)

Measured run (2026-09-04, `OMP_NUM_THREADS=1`):

```text
TD-DFTB H2 r=0.74 A: gap=15.44081455  S1=16.39162450  T1=13.67637744 eV
  q_H1=-0.6515851161  q_H2=0.6515851161 e
  lambda=0: |omega - (eps_a - eps_i)| = 0.000e+00 eV
  S1(1.20 A)=7.44545235 eV (must be below S1(0.74 A))
TD-DFTB H2O: n_pair=8  S1..S4 = 14.090670 15.869295 17.148433 18.385602 eV
  max |Omega_pq - Omega_qp| = 2.487e-14 eV^2
  max |sum_A q^ia_A| = 2.903e-16 e
  H2O lambda=0 max |omega - gap| = 0.000e+00 eV
```

T1 was re-pinned after published \(W_{\mathrm{ss}}(\mathrm{H})\) landed
(`dftb-data-00-honesty`); it supersedes the sketch-\(W_{\mathrm{H}}=-0.50\) eV
snapshot 15.01024822 eV. S1/gap are independent of \(W\) (singlet kernel is
\(\gamma\); H₂ ground state has \(\Delta q\equiv 0\)).

## Why there is no absolute pin

The `regressions/` contract asks for a **cited literature value**. None exists
for this system, and none was invented. What was searched (2026-07-29):

- **Niehaus et al., PRB 63, 085108 (2001)** — closed access; APS,
  ResearchGate, academia.edu (HTTP 403), the Regensburg repository, arXiv and
  Unpaywall/OpenAlex (`is_oa: false`, `any_repository_has_fulltext: false`) all
  failed. **Its table contents could not be verified**, so nothing from it is
  quoted here beyond the formalism, which is standard and reproduced above.
- **DFTB+ test suite** (`test/app/dftb+/timedep`, 35 cases) and **DFTB+
  Recipes** (`docs/linresp/`) — no H₂ case.
- **Niehaus, JCTC 9, 4901 (2013)** (arXiv:1302.3362) Table I — NO, N₂, O₂ only,
  and mio-**0**-1.
- **Rüger, van Lenthe, Heine, Visscher, JCP 144, 184103 (2016)** — Schreiber's
  28-molecule set, N₂, C₆₀. No H₂.
- **Humeniuk & Mitrić** (arXiv:1503.01714) states the reason directly: *"the
  molecules N2, CO, H2CO and HCl were excluded because the minimal basis set of
  occupied valence orbitals used in DFTB is not suitable for describing Rydberg
  states."* In mio-1-1 H carries only a 1s shell, so H₂ is a two-level toy —
  nobody benchmarks it.

Consequently gate 8 pins **Atomiverse's own converged values** and is labelled
in the source as a drift guard, not an oracle. `S1 = 16.39 eV` for H₂ is
**not** a physical excitation energy (experiment: B¹Σ⁺_u ≈ 12.3–12.7 eV
vertical); a minimal-basis σ_u cannot produce one.

## Gaps (open)

1. **No absolute excitation-energy oracle.** No citable H₂/mio-1-1 TD-DFTB
   value exists (see above). Everything gated here is a structural / analytic
   limit plus a drift snapshot. Closing this needs an offline DFTB+ `Casida{}`
   run on a molecule with a published number — N₂ is the candidate (next
   item). DFTB+ 24.1 is installed; it was not run for this H₂ toy.
2. **N₂ would be the citable case, and the tree cannot run it.** The DFTB+
   Recipes ship a fully specified mio-1-1 N₂ `Casida{}` example
   (d = 1.151128 Å, `SpinConstants N = {-0.026}`, `SCCTolerance = 1e-10`) whose
   `EXC.DAT` gives lowest triplet **7.321 eV**, lowest singlet **8.118 eV**, and
   the bright ¹Σ⁺_u at **12.750 eV** (f = 0.80120458)
   (<https://dftbplus-recipes.readthedocs.io/en/latest/linresp/diatomic.html>).
   Atomiverse cannot target it: `basis_data/dftb_mio_tables.h` has only a
   **sketch** N–N Slater–Koster row and a sketch \(W_\mathrm{N}\). Transcribing
   the real mio-1-1 N–N grid (as was already done for H–H) would make that the
   first genuinely citable TD-DFTB pin in the tree.
3. **γ_AB carries a units defect** that scales every singlet coupling — see the
   next section. It is pre-existing (DFTB2 **and** DFTB3 ground state) and is
   *not* introduced by this case; the gates above are all invariant under it
   (they are structural limits or homonuclear/λ=0 identities), but the absolute
   singlet numbers here inherit it.
4. **Energies only.** No oscillator strengths / transition dipoles (so no
   bright-vs-dark ordering can be gated), no excited-state gradients, no NAC.
5. **Restricted, molecular, MioSketch only.** Open-shell (spin-polarized)
   TD-DFTB, the 3ob path, and periodic / k-point linear response all throw.
6. **Dense solver only** (`n_pair ≤ 512`); no Davidson.
7. **No long-range correction and no on-site correction** — charge-transfer
   excitations are the known failure mode of plain TD-DFTB.

## Blocking defect found in the ground state (NOT fixed here)

`src/cpu/semiempirical/dftb2.cpp:118` (and the identical
`internal/dftb_gamma.h`) implement the DFTB+ Elstner γ as

```cpp
double gamma_ab(double r, double ua, double ub) {          // r [Å], U [eV]
    const double tau = 0.5 * (1.0 / ua + 1.0 / ub);        // [1/eV]
    return 1.0 / std::sqrt(r * r + tau * tau);             // Å² + eV⁻²  ✗
}
```

\(R^2\) [Å²] and \(\tau^2\) [eV⁻²] are added, and the Coulomb constant
\(k_e = 14.399645\) eV·Å/e² (already in the tree as `units::coulomb`,
`include/atomiverse/core/units.h:140`) is missing. The dimensionally consistent
form in eV/Å is \(\gamma = k_e/\sqrt{R^2 + (k_e\tau)^2}\). Consequences:

- \(\gamma_{AA}=U_A\) is accidentally **correct** (\(1/\tau\) at \(R=0\)),
- \(\gamma_{AB}(R\to\infty)\to 1/R\) instead of \(k_e/R\),
- the crossover radius is \(\tau = 0.088\) Å instead of \(k_e\tau = 1.26\) Å,
  so γ collapses to the (already 14× too small) tail almost immediately:
  for H–H at 0.74 Å the code gives **1.342 eV** where the consistent form gives
  **9.845 eV** (7.3× low); at 3 Å, 0.333 eV vs 4.425 eV (13× low).

This was invisible to the existing pins because the one *real*-mio-1-1 system in
the tree is H₂, where \(\Delta q = 0\) by symmetry and \(\gamma_{AB}\) never
enters the energy (`dftb2-dftbplus-energy-pin`), while every polar case is
already declared a "sketch". It is **not fixed on this branch**: correcting it
moves every `MioSketch` DFTB2/DFTB3 energy, force, stress and charge for any
system with non-zero Δq and re-pins ~10 regressions owned by other waves
(`h2o_dftb2_mio`, `dftb2-analytic-forces`, `dftb2-analytic-stress`,
`dftb2-periodic-gamma`, `dftb2-periodic-forces`, `dftb2-kpoints`,
`dftb2-dispersion-d3`, `spin-dftb2-o2`, `dftb2-spin-analytic-forces`,
`dftb3-water-mio`). Route: `/mol:fix` as its own change, with the whole
MioSketch pin set re-derived in one pass.

## Perf

`perf_exempt: true` — no DFTB+ `Casida{}` twin was run for this H₂ toy (no
citable number exists). Runtime is ~15 ms single-threaded.
