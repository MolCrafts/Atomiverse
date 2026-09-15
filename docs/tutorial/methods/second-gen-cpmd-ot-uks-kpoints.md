# Multi-k UKS second-generation CP with OT corrector — H₂ NVE (GPW)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `second-gen-cpmd-ot-uks-kpoints` |
| Regression | `regressions/dynamics/second-gen-cpmd-ot-uks-kpoints/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::SecondGenCp{cpu::gpw::GPWEvaluator{…, n_alpha, n_beta, k_mesh, use_ot=true}, aspc_k, corrector_max_iter}` bound with `set_evaluator_ref` |
| Dual gate | Correctness (corrector A/B identity + NVE drift + budget + dual-spin multi-k payload); speed **`perf_exempt`** |
| Wave | CP2K capability loop — worker wave 3 (shift) (2026-07-31) |
| Oracle | **CP2K cannot run this method** (OT is Γ-only in CP2K; `&OT` + `&KPOINTS` refused, UKS included) — literature envelope + internal A/B identity; unexecutable sketch at `tools/cp2k_oracle/inputs/second-gen-cpmd-ot-uks-kpoints/` |

## Summary

**Second-generation Car–Parrinello MD** on **multi-k collinear UKS** GPW with
the **Orbital Transformation (OT)** minimizer as the partial-SCF corrector.
Per sampled \(k\) and spin \(\sigma\), a Kolafa **ASPC** predictor
(`ASPCPredictorCplx`) extrapolates the complex Bloch spin density
\(P_k^\sigma\); each predicted \(P_k^\sigma\) is **purified to natural
orbitals** (eigenvectors of \(X_k^\dagger P_k^\sigma X_k\), largest
occupations, Löwdin-orthonormalized against \(S_k\)) which **seed the
per-(k, σ) OT manifold** — the complex dual-spin mirror of the Γ-UKS
`guess_P`/`guess_P_beta` OT warm start (`second-gen-cpmd-uks-ot`) and of the
multi-k RKS `guess_P_k` warm start (`second-gen-cpmd-ot-kpoints`). A
**partial OT** corrector (`second_gen_cp_max_iter` ≪ full BOMD) refines the
state; analytic multi-k UKS forces are taken at the post-corrector density.

This closes the last SGCP corrector×sampling cell for collinear UKS: Γ-DIIS
(`second-gen-cpmd-uks`), Γ-OT (`second-gen-cpmd-uks-ot`), multi-k
diagonalization (inside `GPWEvaluator::launch`), and multi-k OT.

## Usage

```cpp
cpu::gpw::GPWEvaluator gpw({H_kind}, /*cutoff_ry=*/100.0, /*max_iter=*/60,
                           /*tol=*/1e-5, /*mix=*/0.3,
                           cpu::gpw::XcKind::PADE, /*nonperiodic=*/false,
                           /*compute_forces=*/true);
gpw.n_alpha = 1;               // collinear UKS path
gpw.n_beta = 1;
gpw.k_mesh[2] = 2;             // 1×1×2 Monkhorst–Pack
gpw.k_time_reversal = false;   // two sampled k
gpw.use_ot = true;             // dual-spin per-k OT partial-SCF corrector
cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                  /*corrector_max_iter=*/5};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.5});
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, n_steps);       // Frame in / Frame out; energy in eV on meta
```

## Pin

No CP2K number can exist for this method — CP2K's OT implementation is
Γ-point-only and its input validation refuses `&OT` together with
`&KPOINTS` (UKS does not change the refusal). The decisive gates
(regression, ≤ 60 s):

1. **Corrector A/B identity** — full-budget multi-k UKS OT vs multi-k UKS
   diagonalization at the initial H₂ geometry (1×1×2 TR-off mesh, 100 Ry,
   DZV-GTH-PADE / GTH-PADE-q1, \(n_\alpha=n_\beta=1\)): measured
   \(|\Delta E| \approx 3.6\times10^{-11}\) Ha (gate ≤ 1e-4 Ha, the
   `ot-uks-kpoints` A2 pin level); ⟨S²⟩ OT ≡ diag on the multi-k UKS integer
   seed formula \(s_z(s_z+1)+N_\beta\) (= 1 for this filling).
2. **NVE drift envelope** (Kühne PRL **98**, 066401 (2007); Kolafa JCC
   **25**, 335 (2004)): worst relative drift ≤ 0.10, production OT iters ≤
   budget 5, \(n_k = 2\) with dual-spin \(P_k / P_k^\beta\) payloads, soft
   H–H window.

## Honest scope

- Collinear multi-k UKS product pin on singlet fillings
  (\(n_\alpha = n_\beta = 1\)) — the dual-spin plumbing gates (β payloads,
  dual ASPC, per-(k, σ) OT) are exercised; open-shell k-UKS physics rides
  the same code path via `ot-uks-kpoints`.
- No metallic smearing / +U under the SGCP corrector (integer-occupation
  warm start only; evaluator exposes no smearing).
- `perf_exempt`: no oracle wall twin can exist (see above).
