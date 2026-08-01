---
spec: dfpt-gga-spin-smearing
created: 2026-08-01
criteria:
  - id: ac-001
    summary: Joint spin x GGA x smear DFPT C00 matches offline free-energy FD pin
    type: runtime
    pass_when: |
      reg_dfpt-gga-spin-smearing runs DfptPhonon::compute with spin_polarized=true,
      XCType::PBE, and Smearing::fermi_dirac on the diamond-Si twin;
      |C[(0x),(0x)] - FD pin| <= 0.30 eV/A^2 where the pin is the offline
      central second difference of the spin-polarized PBE free energy
      (h = 0.03 A, tol 1e-9), hard-coded in REFERENCE.md.
    status: verified
  - id: ac-002
    summary: Physics window gates (acoustic, optical triplet, ASR, symmetry)
    type: runtime
    pass_when: |
      Acoustic |omega| <= 30 cm^-1 with no ASR imposition; optical triplet in
      [450, 650] cm^-1 with spread <= 10; DFPT row-sum <= 0.05 eV/A^2;
      pre-symmetrization asymmetry <= 1e-5 eV/A^2.
    status: verified
  - id: ac-003
    summary: Regression is an out-of-library consumer, <= 60 s
    type: code
    pass_when: |
      regressions/response/dfpt-gga-spin-smearing/reg_dfpt-gga-spin-smearing.cpp
      uses public headers + link atomiverse only, registered via
      atv_add_regression; wall <= 60 s on the login node.
    status: verified
  - id: ac-004
    summary: REFERENCE.md + oracle note document a re-runnable pin
    type: docs
    pass_when: |
      REFERENCE.md cites method, twin settings, measured pin values with date
      and build type; tools/cp2k_oracle/inputs/dfpt-gga-spin-smearing.note
      records the offline recipe and the CP2K 2026.1 run (binary, input,
      output line, wall) that anchors the physics window and the fair ratio.
    status: verified
  - id: ac-005
    summary: Fair wall within_10x from a real CP2K run
    type: runtime
    pass_when: |
      ratio_atv_over_oracle = T_atv / T_cp2k <= 10.0 from the matched
      VIBRATIONAL_ANALYSIS twin (serial 1 MPI x OMP=1); perf_exempt false.
    status: verified
  - id: ac-006
    summary: Queue + roadmap close the row
    type: docs
    pass_when: |
      docs/developer/cp2k-method-queue.md moves dfpt-gga-spin-smearing to done;
      docs/roadmap/dft.md marks joint spin x GGA x smear DFPT shipped.
    status: verified
out_of_scope:
  - Finite-q three-way combo; hybrid XC; M != 0 cells; CUDA
  - Live CP2K/QE at ctest time
---

# Acceptance — `dfpt-gga-spin-smearing`

**Done** means the joint collinear-spin × pure-GGA (PBE) × metallic-smear
Γ DFPT phonon path of `cpu::pw::DfptPhonon` is gated by an offline dual pin
(same-Hamiltonian free-energy FD + CP2K physics window) in one ≤ 60 s
regression with fair wall ratio ≤ 10, queue row closed.
