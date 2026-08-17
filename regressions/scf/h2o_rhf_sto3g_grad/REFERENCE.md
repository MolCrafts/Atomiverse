# Reference — H2O RHF/STO-3G analytic gradient (device s/p force path)

**One case, multi-device:** `reg_h2o_rhf_sto3g_grad --device=cpu|cuda|both`
(default `both` when built with CUDA). Same PySCF pin on every device — not a
`*_cuda` sibling. Public RHF + Frame only.

**Why this case exists:** the device analytic s/p force path
(`launch_rhf_forces_sp`, `cuda::scf::RHF::forces`) is a delivered feature that
previously had only unit coverage (`tests/cuda/scf/test_rhf.cu`). This is its
e2e regression. The CUDA leg is **fail-closed**: it asserts
`last_scf_on_device()` *and* `last_force_on_device()`, so a silent host
fallback (grading `cpu::scf::RHF::forces` under a "cuda" banner) fails the
case instead of passing it. The out-of-coverage counterpart
(`scf/h2o_rhf_def2svp_grad` CUDA leg) asserts that `forces()` **throws**.

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).

| Field | Value |
|---|---|
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| Basis | STO-3G — H2O carries s/p shells only, so Cartesian ≡ spherical |
| Convergence | `conv_tol = 1e-12`, RHF |
| PySCF version | 2.14.0 |
| Command | `~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis sto-3g --grad --conv-tol 1e-12` |
| **E_RHF** | **−74.963023138463 Ha** (same run as the `mp2-grad-h2o-sto3g` E_HF) |

## Pinned analytic gradient dE/dR [Ha/Bohr]

`mf.nuc_grad_method().kernel()`. Molecule lies in the yz-plane, so all x
components are numerically zero (~1e-16).

| Atom | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| O      | 0 | 0 | −6.142775911808e-02 |
| H (+y) | 0 | −2.364133873480e-02 | +3.071387955904e-02 |
| H (−y) | 0 | +2.364133873480e-02 | +3.071387955904e-02 |

Σ over atoms ≈ (0, 0, 0) to ~1e-15 Ha/Bohr.

## Gates and their rationale

| Leg | Energy [Ha] | Element-wise vs pin [Ha/Bohr] | ‖Σ_A g_A‖ [Ha/Bohr] | Extra |
|---|---|---|---|---|
| cpu | 1e-8 | 1e-6 | 1e-8 | — |
| cuda | 1e-6 | 5e-6 | 1e-7 | `last_scf_on_device()` ∧ `last_force_on_device()`; CPU parity ≤ 2e-6 Ha/Bohr |

Both engines deliver **analytic** gradients, so the CPU leg reproduces PySCF
to well below 1e-6 Ha/Bohr; a wrong/missing Pulay or operator-derivative term
shows up at ≥ 1e-3 Ha/Bohr. The CUDA leg computes the same closed-form
quantities with device kernels whose accumulation order is not bit-identical
to the host, hence the looser 5e-6/1e-7 windows. On top of the shared external
pin the CUDA leg grades **same-settings CPU parity** at 2e-6 Ha/Bohr — the
Ha/Bohr equivalent of the 1e-4 eV/Å CUDA↔CPU force tolerance stated in
`tests/cuda/scf/test_rhf.cu` — per the sanctioned CUDA-ES rule that
same-settings CPU results are the correctness oracle.

Literature: Pulay, *Mol. Phys.* **17**, 197 (1969) (energy-weighted density /
Pulay forces); Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure
Theory* §10 (molecular gradients); Hehre, Stewart, Pople, *J. Chem. Phys.*
**51**, 2657 (1969), doi:10.1063/1.1672392 (STO-3G).

## Measurement log

| Item | Value |
| ---- | ----- |
| GH200 job id | **1039997** (node `n188`, NVIDIA GH200 120GB) — the committed tree |
| Toolchain | `GPU/buildenv-gcccuda/2026.03-cu13.0`, nvcc **13.0.88** / GNU 14.3.0; cubins sm_75 + runtime PTX JIT to ISA 90 |
| GH200 log path | `.claude/reg-cuda-grad-honesty-1039997.log` |
| cpu leg | PASS — max element-wise err **3.870e-11** Ha/Bohr (gate 1e-6); ‖Σ_A g_A‖ 1.133e-14 (gate 1e-8) |
| cuda leg | PASS — `path=device` (both residency flags asserted true); max element-wise err **3.867e-11** Ha/Bohr (gate 5e-6); ‖Σ_A g_A‖ 4.430e-14 (gate 1e-7) |
| CUDA↔CPU parity | **3.200e-14** Ha/Bohr (gate 2e-6) |
| Job verdict | `=== ALL CUDA LEGS PASS ===` (this case + `h2o_rhf_def2svp_grad`, `--device=both`) |
