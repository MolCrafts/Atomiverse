---
spec: gpw-ts-kpoint-stress
created: 2026-08-01
criteria:
  - id: ac-001
    summary: Multi-k fixed-P_k stress Lagrangian rebuilds free-atom TS at strained cells
    type: code
    pass_when: |
      energy_fixed_P_k (gpw_stress.cpp) calls ts_pbe when cfg.ts is set, with
      the same rcut/s_R/d as the Γ path and energy path.
    status: met
  - id: ac-002
    summary: E_disp multi-k product matches Γ libMBD pure-RS pin
    type: runtime
    pass_when: |
      reg_gpw-ts-kpoint-stress gate A: |E_disp − pin| ≤ 1e-8 Ha
      (pin −6.0996307733860822e-4 Ha).
    status: met
  - id: ac-003
    summary: Strained-cell E_disp and σ(ts)−σ(plain) match libMBD strain pins
    type: runtime
    pass_when: |
      Gates B/D in reg_gpw-ts-kpoint-stress: max |E_disp(strain)−libMBD| ≤ 1e-8 Ha;
      max |σ shift − libMBD FD term| ≤ 2e-9 Ha/Bohr³.
    status: met
  - id: ac-004
    summary: Analytic multi-k stress matches multi-SCF strain-FD
    type: runtime
    pass_when: |
      Gate C: max |σ_an − σ_fd| ≤ 2e-5 Ha/Bohr³ on diagonals; off-diagonals 0;
      product wall ≤ 60 s.
    status: met
  - id: ac-005
    summary: REFERENCE + offline pin provenance
    type: docs
    pass_when: |
      REFERENCE.md cites libMBD version/git, pure-RS settings, pin lines, and
      states CP2K has no stock TS (oracle:literature/libMBD, not fake cp2k).
    status: met
acceptance_met: true
measured_2026_08_01: |
  B max|E_disp strain|=9.98e-18 Ha; A |dE_disp|=6.19e-12 Ha;
  C max|an-fd|=1.36e-09 Ha/Bohr3; D max|wire|=4.01e-12 Ha/Bohr3;
  wall_product=10.03 s; wall_fd_twin=48.42 s; SCF 11 iters; nk=1 (1×1×2 TR).
