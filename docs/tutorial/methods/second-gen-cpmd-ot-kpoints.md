# Multi-k second-generation CP with OT corrector — H₂ NVE (GPW)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `second-gen-cpmd-ot-kpoints` |
| Regression | `regressions/dynamics/second-gen-cpmd-ot-kpoints/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::SecondGenCp{cpu::gpw::GPWEvaluator{…, k_mesh, use_ot=true}, aspc_k, corrector_max_iter}` bound with `set_evaluator_ref` |
| Roadmap / queue | [cp2k-method-queue.md](../../developer/cp2k-method-queue.md) row `second-gen-cpmd-ot-kpoints` |
| Dual gate | Correctness (corrector A/B identity + NVE drift + budget + multi-k payload); speed **`perf_exempt`** |
| Wave | CP2K capability loop — worker wave 2 (shift) (2026-07-31) |
| Oracle | **CP2K cannot run this method** (OT is Γ-only in CP2K; `&OT` + `&KPOINTS` refused) — literature envelope + internal A/B identity; unexecutable sketch at `tools/cp2k_oracle/inputs/second-gen-cpmd-ot-kpoints/` |

## Summary

**Second-generation Car–Parrinello MD** on **multi-k closed-shell RKS** GPW
with the **Orbital Transformation (OT)** minimizer as the partial-SCF
corrector. Per sampled \(k\), a Kolafa **ASPC** predictor
(`ASPCPredictorCplx`) extrapolates the complex Bloch density \(P_k\); the
predicted \(P_k\) is **purified to natural orbitals** (eigenvectors of
\(X_k^\dagger P_k X_k\), largest occupations, Löwdin-orthonormalized against
\(S_k\)) which **seed the per-k OT manifold** — the complex mirror of the Γ
`guess_P` OT warm start (`second-gen-cpmd-ot`). A **partial OT** corrector
(`second_gen_cp_max_iter` ≪ full BOMD) refines the state; analytic multi-k
forces are taken at the post-corrector density.

This closes the last SGCP corrector×sampling cell for RKS: Γ-DIIS
(`second-gen-cpmd-water`), Γ-OT (`second-gen-cpmd-ot`), multi-k
diagonalization (`second-gen-cpmd-kpoints`), and now multi-k OT. Multi-k
**UKS** + OT stays refused with a precise throw (dual-spin k-OT warm start
is not productized).

## Usage

```cpp
cpu::gpw::GPWEvaluator gpw({H_kind}, /*cutoff_ry=*/100.0, /*max_iter=*/60,
                           /*tol=*/1e-5, /*mix=*/0.3,
                           cpu::gpw::XcKind::PADE, /*nonperiodic=*/false,
                           /*compute_forces=*/true);
gpw.k_mesh[2] = 2;             // 1×1×2 Monkhorst–Pack
gpw.k_time_reversal = false;   // two sampled k
gpw.use_ot = true;             // OT partial-SCF corrector
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
`&KPOINTS`. The decisive gates (regression, ≤ 60 s):

1. **Corrector A/B identity** — full-budget multi-k OT vs multi-k
   diagonalization at the initial H₂ geometry (1×1×2 TR-off mesh, 100 Ry,
   DZV-GTH-PADE / GTH-PADE-q1): measured \(|\Delta E| = 3.0\times10^{-12}\)
   Ha (gate ≤ 1e-4 Ha, the `ot-kpoints` pin level).
2. **NVE drift envelope** (Kühne PRL **98**, 066401 (2007); Kolafa JCC
   **25**, 335 (2004)): measured worst relative drift \(5.1\times10^{-4}\)
   (gate ≤ 0.10), production OT iters ≤ budget 5, \(n_k = 2\) with full
   \(P_k\) payload, soft H–H window.

## Honest scope

- Closed-shell multi-k RKS only; multi-k UKS + OT refused.
- No metallic smearing / +U pinned under the SGCP corrector.
- `perf_exempt`: no oracle wall twin can exist (see above).
