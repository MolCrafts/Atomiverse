# Acceptance — second-gen-cpmd-ot-kpoints

- [x] type: unit — `test_cpu_gpw_second_gen_cpmd` green: multi-k **RKS** SGCP +
      `use_ot` build **accepted**; multi-k **UKS** SGCP + `use_ot` build
      **rejected** (`std::invalid_argument`); all pre-existing guards
      unchanged.
- [x] type: scientific — `reg_second-gen-cpmd-ot-kpoints` green (≤ 60 s,
      8 threads):
      - corrector A/B: full-budget multi-k OT vs multi-k diagonalization
        single point at the initial geometry, |ΔE| ≤ 1e-4 Ha (the
        `ot-kpoints` pin level);
      - `n_kpts ≥ 2` and `P_k.size() == n_kpts` after seed;
      - production OT corrector iters ≤ `second_gen_cp_max_iter` once ASPC
        history is warm (partial SCF, not full BOMD);
      - NVE drift envelope `max |E − E0| / max(1, |E0|) ≤ 0.10` (family
        tolerance, Kühne PRL 2007 / Kolafa JCC 2004);
      - PE bound, KE ≥ 0 and non-trivial, coordinates finite, soft H–H
        window 0.4–1.5 Å.
- [x] type: performance — `perf_exempt`: CP2K's OT is Γ-only (`&OT` +
      `&KPOINTS` is refused by CP2K itself), so no oracle wall twin can exist
      for this method; family policy per `second-gen-cpmd-kpoints` /
      `second-gen-cpmd-ot`.
- [x] REFERENCE.md cites method literature + the CP2K OT/k-point
      incompatibility + the internal A/B identity as the decisive gate.
- [x] Scoped build/test only; builds wiped before exit.
