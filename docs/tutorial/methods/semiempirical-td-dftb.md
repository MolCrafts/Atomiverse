# Linear-response TD-DFTB (Casida)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Case: `ctest -R '^semiempirical/td-dftb$'` — sources and oracle record in [`regressions/semiempirical/td-dftb/`](../../../regressions/semiempirical/td-dftb/). One section of the case per `##` heading below: `atv_regressions --case semiempirical/td-dftb --section <name>`.

Vertical **singlet** and **triplet** excitation energies from the
tight-binding formulation of time-dependent density-functional response
theory (Niehaus et al. 2001), on a converged restricted **SCC-DFTB2** ground
state. The concrete method `cpu::semiempirical::DFTB2` carries a `const` query
method `td_casida(frame, n_states, channel, n_occ_window, n_virt_window,
coupling_scale)` returning the result-only struct `TdDftbResult`
(`cpu/semiempirical/td_dftb2.h`): excitation energies, oscillator strengths
and transition dipoles, single-particle gaps, Mulliken transition charges and
the assembled Casida matrix. Every Casida quantity is private workspace of
`DFTB2`; Frame in, result out — no public `*Config` bag, no `*System`, no
`make_*` / `build_*` factory. This mirrors the tree's house shape for linear
response (`TDA<RHF>::run`, `RKS::tddft`, `TDA<UKS>::run`, `SfTDA<UHF>::run`
beside the concrete ground-state method, result-only `cpu/scf/tdhf.h`).

The case is CPU-only with one section, `td-dftb-h2` (H₂ plus an H₂O
multi-pair check).

## Oracle

```text
oracle: literature
oracle_version: Niehaus et al., Phys. Rev. B 63, 085108 (2001) — Casida
  TD-DFTB. No citable H2/mio-1-1 TD-DFTB value exists.
```

Family oracle is **DFTB+** (never OpenMOPAC / xtb). The pin is the formalism
only: every gate is a structural or analytic limit, plus an ATV drift
snapshot that is labelled as such in the source. `perf_exempt: true` — no
DFTB+ `Casida{}` twin is run for the H₂ toy. Roadmap: TD-DFTB ✓
(closed-shell Γ MioSketch).

## td-dftb-h2 — TD-DFTB singlet / triplet — H₂ (+ H₂O) {#td-dftb-h2}

### Method

Ground state: SCC-DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998)),
restricted closed shell, Γ point, `DftbSkSet::MioSketch` (H onsite / \(U\) and
the H–H SK + repulsive grid are transcribed mio-1-1).

Response: Niehaus, Suhai, Della Sala, Lugli, Elstner, Seifert, Frauenheim,
*Phys. Rev. B* **63**, 085108 (2001). Mulliken transition charges [e]

\[
q^{ia}_A = \tfrac12\sum_{\mu\in A}
\Big(c_{\mu i}\,(Sc_a)_\mu + c_{\mu a}\,(Sc_i)_\mu\Big)
\]

contract the coupling kernel [eV]

\[
K^{S}_{ia,jb} = \sum_{AB} q^{ia}_A\,\gamma_{AB}\,q^{jb}_B
\qquad\text{(singlet)},\qquad
K^{T}_{ia,jb} = \sum_{A} q^{ia}_A\,W_A\,q^{jb}_A
\qquad\text{(triplet)}
\]

— the singlet kernel is the **same** \(\gamma_{AB}\) the second-order SCC
energy uses; the triplet magnetization kernel is on-site,
\(\gamma^{m}_{AB}=\delta_{AB}W_A\), with \(W_A\) [eV] the highest-occupied
atomic-orbital diagonal of `DFTB2::spin_w_ha` (Ha, converted with
`units::kHartreeInEv`). Here `spin_w_ha = basis_data::kDftbSpinConstantsPbeHa`,
the published PBE \(W_{\mathrm{ss}}(\mathrm{H})=-0.072\) Ha
\(= -1.959219809711\) eV; the triplet channel throws when `spin_w_ha` is
empty. The Hermitian Casida eigenproblem [eV²]

\[
\Omega_{ia,jb} = \delta_{ij}\delta_{ab}(\varepsilon_a-\varepsilon_i)^2
+ 2\sqrt{\varepsilon_a-\varepsilon_i}\;K_{ia,jb}\;
\sqrt{\varepsilon_b-\varepsilon_j},
\qquad \Omega F = \omega^2 F
\]

is solved densely (LAPACK `dsyevd`), and \(\omega_n=\sqrt{\lambda_n}\).
Singlet oscillator strengths are length-form from pair dipoles
\(\boldsymbol{\mu}_{ia}=\sum_A q^{ia}_A\mathbf{R}_A\); triplets are
dipole-dark (\(f = 0\)).

`td_casida` is **stateless**: it runs its own converged SCC at the geometry
in `frame`, through the same routine `launch()` uses, so it never depends on
whether or where `launch()` last ran. Cost is one extra SCC solve.

### Systems and settings

- **H₂** — at (0,0,0) and (r,0,0) Å, cubic box 20 Å,
  `boundary = DftbBoundary::cluster`; r = 0.74 Å (pinned) and r = 1.20 Å
  (stretched check). Minimal basis ⇒ one σ_g → σ_u channel, a 1×1 Casida
  matrix.
- **H₂O** — O (0,0,0), H (1.05,0,0), H (−0.26,1.0,0) Å, cubic box 20 Å;
  n_ao = 6, n_occ = 4, n_virt = 2 ⇒ **8** pairs.
- `scc_tol = 1e-12`, `max_scc = 120`, `charge = 0`, `multiplicity = 1`.

### Implementation

```cpp
cpu::semiempirical::DFTB2 dftb{cpu::semiempirical::DftbSkSet::MioSketch};
dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
dftb.scc_tol = 1e-12;
dftb.max_scc = 120;
dftb.spin_w_ha = cpu::semiempirical::basis_data::kDftbSpinConstantsPbeHa;

cpu::Frame h2 = make_h2(0.74);          // box = cpu::Box::cubic(20.0)
const auto s = dftb.td_casida(h2, /*n_states=*/1);
const auto t = dftb.td_casida(h2, 1,
                              cpu::semiempirical::TdDftbChannel::triplet);

s.energy_ev[0];                          // lowest singlet omega [eV]
t.energy_ev[0];                          // lowest triplet omega [eV]
s.ks_gap_ev[0];                          // eps_a - eps_i [eV]
s.oscillator_strength[0];                // f_S1 (t.oscillator_strength[0] == 0)
s.transition_charge[0];                  // q^{ia}_{H1} [e]
```

Optional arguments narrow the active window and scale the kernel:

```cpp
// HOMO -> LUMO only, coupling switched off (non-interacting limit).
const auto bare = dftb.td_casida(h2o, 1,
                                 cpu::semiempirical::TdDftbChannel::singlet,
                                 /*n_occ_window=*/1, /*n_virt_window=*/1,
                                 /*coupling_scale=*/0.0);
// bare.energy_ev[0] == bare.ks_gap_ev[0] to solver precision.
```

### Validation

| Gate | Tolerance | Measured |
| ---- | --------- | -------- |
| λ = 0 ⇒ ω = ε_a − ε_i (H₂, H₂O) | 1e-10 eV | 0.000e+00 |
| Ω symmetric as assembled (H₂O, 8×8) | 1e-12 eV² | 3.553e-15 |
| Σ_A q^{ia}_A = 0 (all 8 H₂O pairs) | 1e-10 e | 2.776e-16 |
| H₂: ω_T < ε_a−ε_i < ω_S | strict | 13.676 < 15.441 < 16.392 |
| H₂: q_H1 = −q_H2, \|q\| > ½ | analytic | ±0.6515851161 |
| H₂: ω_S red-shifts on stretching | strict | 7.445 (1.20 Å) < 16.392 |
| H₂: f_S1 ≥ 0, \|f_T1\| ≤ 1e-12 | strict | — |
| H₂ snapshot gap, ω_S, ω_T (drift guard, **not** an oracle) | 1e-6 eV | 15.44081455, 16.39162450, 13.67637744 |

The λ = 0 gate is the decisive structural check: at zero coupling Ω is
diagonal in \((\varepsilon_a-\varepsilon_i)^2\), so every root is a bare
single-particle gap; it exercises SCC, overlap, MO ordering, window
selection, √Δε prefactors and the eigensolve independently of parameter
quality. The transition-charge gate is analytic: for a symmetric two-level
system \(q_1 = 1/(2\sqrt{1-S_{12}^2}) = -q_2 > \tfrac12\); the measured
0.65159 back-solves to \(S_{12}=0.6412\), the mio-1-1 H–H overlap at 0.74 Å.
H₂O S1..S4 = 14.090670, 15.869295, 17.148433, 18.385602 eV (not gated).
Runtime is ~15 ms single-threaded.

**There is no absolute excitation-energy oracle.** No published
TD-DFTB/mio-1-1 value for H₂ exists — DFTB+'s own test suite and Recipes have
no H₂ linear-response case, and Humeniuk & Mitrić state the reason directly
(a minimal valence basis cannot describe H₂'s Rydberg-like excited states).
`S1 = 16.39 eV` is therefore **not** a physical excitation energy
(experiment: B¹Σ⁺_u ≈ 12.3–12.7 eV vertical); it is pinned only as a drift
guard. The citable candidate is the DFTB+ Recipes mio-1-1 N₂ `Casida{}`
example (lowest triplet 7.321 eV, lowest singlet 8.118 eV, bright ¹Σ⁺_u
12.750 eV), which needs a transcribed mio-1-1 N–N grid.

### Limitations

- Restricted closed shell, molecular Γ point, `MioSketch` only — ThreeOb,
  multi-k and open shell all throw.
- Dense solver, ≤ 512 occupied→virtual pairs; no Davidson.
- No excited-state gradients or non-adiabatic couplings.
- No long-range correction, no on-site correction — charge-transfer states are
  the known failure mode of plain TD-DFTB.

## References

- T. A. Niehaus, S. Suhai, F. Della Sala, P. Lugli, M. Elstner, G. Seifert,
  Th. Frauenheim, "Tight-binding approach to time-dependent density-functional
  response theory", *Phys. Rev. B* **63**, 085108 (2001),
  DOI [10.1103/PhysRevB.63.085108](https://doi.org/10.1103/PhysRevB.63.085108).
- M. Elstner, D. Porezag, G. Jungnickel, J. Elsner, M. Haugk, Th. Frauenheim,
  S. Suhai, G. Seifert, *Phys. Rev. B* **58**, 7260 (1998),
  DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260).
- M. E. Casida, in *Recent Advances in Density Functional Methods*, Part I
  (World Scientific, 1995).
- R. Rüger, E. van Lenthe, T. Heine, L. Visscher, *J. Chem. Phys.* **144**,
  184103 (2016), DOI [10.1063/1.4948647](https://doi.org/10.1063/1.4948647).
- A. Humeniuk, R. Mitrić, arXiv:1503.01714.
- DFTB+ Recipes, linear response of diatomics:
  <https://dftbplus-recipes.readthedocs.io/en/latest/linresp/diatomic.html>
