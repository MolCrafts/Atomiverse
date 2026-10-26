# Restricted open-shell Hartree–Fock (ROHF)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Restricted open-shell Hartree–Fock describes a high-spin open-shell state
with one set of spatial MOs split into closed, open and virtual spaces
(\(n_\alpha \ge n_\beta\)). The spin Focks \(F_\alpha,F_\beta\) from the
unrestricted Pople–Nesbet build are coupled into a single effective Fock
\(F_{\mathrm{eff}}\) (PySCF canonical / Guest–Saunders projectors) before one
restricted diagonalization; the energy is the coupling-free Pople–Nesbet
functional on the true spin Focks. The public method is
`cpu::scf::ROHF{basis, n_alpha, n_beta}` (members `tol`, `max_iter`) with
`run(frame)` and analytic `forces(frame)`; the nuclear Hessian is the
companion `cpu::scf::Hessian<cpu::scf::ROHF>{rohf}.run(frame)`
(`cpu/scf/response.h`), not a member of the method.

The case also carries the sections `o2_rohf_sto3g` (triplet O₂ ROHF/STO-3G
energy) and `o2_rohf_grad` (its analytic nuclear gradient), whose oracle
records live in the case `REFERENCE.md`; the page below documents the Hessian
section. All sections run on CPU.

Case: `ctest -R '^scf/rohf$'` — sources and oracle record in [`regressions/scf/rohf/`](../../../regressions/scf/rohf/). One section of the case per `##` heading below: `atv_regressions --case scf/rohf --section <name>`.

## Common setup and oracle

The case needs `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU build
environment `buildenv-gcc/2026.03-mpich`, see the repository build notes).
Pins are offline PySCF **2.14.0** runs; nothing in ctest imports PySCF.
Roadmap row: `docs/roadmap/hartree-fock.md`.

## References

- C. C. J. Roothaan, *Rev. Mod. Phys.* **32**, 179 (1960) — ROHF.
- Guest & Saunders, *Mol. Phys.* **28**, 819 (1974) — coupling convention.
- J. A. Pople et al., *Int. J. Quantum Chem. Symp.* **13**, 225 (1979) — analytic HF second derivatives.
- Y. Yamaguchi et al., *A New Dimension to Quantum Chemistry* (Oxford, 1994).

## rohf-hessian-o2 — ROHF nuclear Hessian, triplet O₂ / STO-3G {#rohf-hessian-o2}

### Summary

Computes the Cartesian nuclear Hessian
\(\partial^2 E_{\mathrm{ROHF}}/\partial R_i\partial R_j\) by central finite
difference of **analytic** ROHF nuclear forces (not a full three-space ROHF
CPHF). The system is gas-phase triplet O₂ (\(2S=2\), \(n_\alpha=9\),
\(n_\beta=7\)) at HF/STO-3G. `Hessian<ROHF>::run` returns a dense
\(3N\times 3N\) matrix in eV/Å² (`cpu::vib::HessianResult`). The accuracy pin
is a PySCF 2.14.0 FD of analytic ROHF gradients (PySCF has no
`hessian.ROHF`); performance is **perf_exempt** (no fair analytic twin).

### Method

**Forces.** Analytic nuclear gradient from the ROHF Lagrangian
\[
W = O_\alpha F_\alpha O_\alpha + O_\beta F_\beta O_\beta
\]
(`ROHF::forces`). Units on the public path: eV/Å.

**Hessian.** Central FD of analytic forces, step \(h = 10^{-4}\) Bohr
(displaced in Å via CODATA conversion), then symmetrized:
\[
g = -F,\qquad
H_{ij} \approx \frac{g_i(+h\,e_j) - g_i(-h\,e_j)}{2h},
\]
with \(H\) reported in eV/Å². A fresh ROHF SCF runs per displacement (no
DIIS/MO state bleed) — the same algorithm class as the offline oracle.

**Not included.**

- Analytic three-space ROHF CPHF / `hessian.ROHF`-class second derivatives.
- Mass-weighting, frequencies, IR intensities (those live on `vib::Hessian*` /
  RHF analytic Hessian paths).
- RI density fitting, SOSCF, level shift (throw; no silent fallback).

### Implementation

```cpp
// O₂: Z=8, r = 1.2075 Å along z; 20 Å cubic box (gas phase)
ATV::cpu::scf::ROHF rohf{ATV::cpu::scf::BasisSet::STO3G, /*n_alpha=*/9, /*n_beta=*/7};
rohf.tol = 1e-12;
rohf.max_iter = 200;

const ATV::cpu::vib::HessianResult res =
    ATV::cpu::scf::Hessian<ATV::cpu::scf::ROHF>{rohf}.run(frame);
// res.hessian: [6,6] dense, eV/Å²; res.n_atoms == 2
// rohf.energy() [Ha] at the restored reference geometry
```

| Path | Role |
| ---- | ---- |
| `rohf.h`, `response.h` | Public `ROHF`, `Hessian<Method>` |
| `src/cpu/scf/rohf.cpp` | FD-of-forces implementation |
| `hessian_result.h` | `HessianResult` contract |

### Validation (vs PySCF)

Oracle class: offline central FD of the PySCF **2.14.0** analytic ROHF nuclear
gradient (`mf.nuc_grad_method().kernel()`), \(h=10^{-4}\) Bohr, symmetrized;
units via Atomiverse CODATA factors. PySCF `mf.Hessian()` / `hessian.ROHF`
raises `NotImplementedError` and is not used as a twin.

| Item | Value |
| ---- | ----- |
| Molecule | O₂, charge 0, spin \(2S=2\) |
| Geometry [Å] | O (0, 0, 0); O (0, 0, 1.2075) |
| Box | 20 Å cubic (does not enter molecular SCF) |
| Basis | STO-3G |
| Angular | PySCF pin: `cart=True` (s/p only; cart ≡ spherical for this basis) |
| SCF | ROHF, `conv_tol = 1e-12` |
| \(E_{\mathrm{ROHF}}\) | −147.632166990682 Ha (checked to 1e-8 Ha) |
| Tol | **1e-5 eV/Å²** on nine gated elements; off-axis \(\lvert H_{ij}\rvert < 10^{-3}\) |

**Pinned Cartesian Hessian elements [eV/Å²]** (dense index \(c = 3\cdot\mathrm{atom}+\mathrm{axis}\)):

| Element (row, col) | atoms/axes | Pin [eV/Å²] | Tol |
| ------------------ | ---------- | ----------: | --- |
| \(H_{xx}(0,0)\) | O1-x / O1-x | −1.1102153966 | 1e-5 |
| \(H_{yy}(1,1)\) | O1-y / O1-y | −1.1102153968 | 1e-5 |
| \(H_{zz}(2,2)\) | O1-z / O1-z | +135.5737365643 | 1e-5 |
| \(H_{xx}(0,3)\) | O1-x / O2-x | +1.1102153967 | 1e-5 |
| \(H_{yy}(1,4)\) | O1-y / O2-y | +1.1102153968 | 1e-5 |
| \(H_{zz}(2,5)\) | O1-z / O2-z | −135.5737365648 | 1e-5 |
| \(H_{xx}(3,3)\) | O2-x / O2-x | −1.1102153969 | 1e-5 |
| \(H_{yy}(4,4)\) | O2-y / O2-y | −1.1102153968 | 1e-5 |
| \(H_{zz}(5,5)\) | O2-z / O2-z | +135.5737365643 | 1e-5 |

The section also asserts Hessian symmetry (\(\lvert H-H^\top\rvert \le 10^{-8}\))
and the ROHF energy within 1e-8 Ha of the pin after restore.

### Performance

**`perf_exempt=true`.** PySCF 2.14.0 has no analytic ROHF Hessian; the offline
oracle is multi-SCF FD of analytic gradients — the same work class as
`Hessian<ROHF>`, not a one-shot CPHF wall-time twin.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| analytic ROHF Hessian | n/a | n/a | **n/a** | n/a | `perf_exempt` — no `hessian.ROHF` |

Confirm the PySCF gap:

```bash
python -c "
from pyscf import gto, scf
m=gto.M(atom='O 0 0 0; O 0 0 1.2075',basis='sto-3g',spin=2,cart=True)
mf=scf.ROHF(m); mf.kernel(); mf.Hessian()
"
# → NotImplementedError
```

A same-class FD-vs-FD wall comparison (`python <pyscf fair-bench> --method rohf -- <section binary>`)
is informational only: both sides count many ROHF SCFs + gradients, so ratios
are not comparable to analytic-RHF Hessian vs PySCF CPHF, and the \(r\le 10\)
gate does not apply while exempt.

### Limitations

- **Algorithm:** FD of analytic forces, not analytic ROHF CPHF;
  \(O(N_{\mathrm{atoms}})\) SCFs dominate cost.
- **PySCF parity:** no `hessian.ROHF` oracle for an element-wise analytic twin
  or a fair speed gate.
- **Scope gaps:** no RI Hessian; no frequencies/IR from this path; open-shell
  SOSCF / level shift throw on ROHF.
- **Open:** a full three-space ROHF CPHF Hessian is the natural follow-up (see
  `docs/roadmap/hartree-fock.md`); UHF analytic Hessian and RHF FD/analytic
  Hessian are separate methods.
