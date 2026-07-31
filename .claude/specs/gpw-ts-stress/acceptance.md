spec: gpw-ts-stress
created: 2026-07-31
criteria:
  - id: ac-001
    summary: libMBD strained-cell TS pins exist with full provenance
    type: docs
    pass_when: |
      tools/cp2k_oracle/inputs/gpw-ts-stress/ holds the pin driver
      (pin_ts_strain.f90) + note with libMBD git rev, build line, and the
      printed unstrained + 6 strained E_disp values; the unstrained pure-RS
      number reproduces the shipped gpw-ts-ice periodic pin.
    status: pending
  - id: ac-002
    summary: Strained E_disp parity is 1e-12 Ha class
    type: runtime
    pass_when: |
      reg_gpw-ts-stress gate B: ATV ts_pbe at each of the 6 strained cells
      matches the libMBD pin within 1e-8 Ha (measured ~1e-15 recorded in
      REFERENCE.md).
    status: pending
  - id: ac-003
    summary: Analytic vs strain-FD dual gate passes
    type: runtime
    pass_when: |
      reg_gpw-ts-stress gate C: analytic fixed-P diagonal stress vs
      gpw_stress_fd_impl (both ts=true) within 2e-5 Ha/Bohr3, and gate D:
      sigma(ts)-sigma(plain) matches the libMBD strain-FD dispersion term
      within 2e-9 Ha/Bohr3 per axis.
    status: pending
  - id: ac-004
    summary: Regression registered and green under the scoped build
    type: runtime
    pass_when: |
      regressions/gpw/gpw-ts-stress/ registered via atv_add_regression; the
      binary passes (all gates + wall <= 60 s) in the scoped build; never
      launches libMBD or CP2K at ctest time.
    status: pending
  - id: ac-005
    summary: Existing TS pin provenance upgraded, no product-code drift
    type: docs
    pass_when: |
      gpw-ts-ice / gpw-ts-forces REFERENCE periodic-pin provenance rows cite
      the genuine libMBD pure-RS run; no changes to src/ or include/ (pin-only
      slug) unless a defect is found and named.
    status: pending
