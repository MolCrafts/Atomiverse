# ROHF nuclear Hessian — triplet O₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `rohf-hessian-o2` |
| Regression | `regressions/scf/rohf-hessian-o2/` |
| Public API | `cpu::scf::ROHF{n_α,n_β}.hessian(frame)` |
| Roadmap | `docs/roadmap/hartree-fock.md` (ROHF Hessian row) |

## Summary

Computes the Cartesian nuclear Hessian \(\partial^2 E_{\mathrm{ROHF}}/\partial R_i\partial R_j\) for restricted open-shell Hartree–Fock by central finite difference of **analytic** ROHF nuclear forces (not a full three-space ROHF CPHF). Target system is gas-phase triplet O₂ (\(2S=2\), \(n_\alpha=9\), \(n_\beta=7\)) at HF/STO-3G. Public surface: construct `cpu::scf::ROHF`, set members, call `hessian(frame)` → dense \(3N\times 3N\) matrix in eV/Å² (`cpu::vib::HessianResult`). Offline accuracy pin is PySCF 2.14.0 FD of analytic ROHF gradients (PySCF has no `hessian.ROHF`). End-to-end gate: `regressions/scf/rohf-hessian-o2` (`reg_rohf-hessian-o2`); performance is **perf_exempt** (no fair analytic twin).

## Method

**Energy.** ROHF uses one set of spatial MOs with closed/open/virtual three-space occupation (\(n_\alpha \ge n_\beta\)). Spin Focks \(F_\alpha,F_\beta\) from the unrestricted Pople–Nesbet build are coupled into a single effective Fock \(F_{\mathrm{eff}}\) (PySCF canonical / Guest–Saunders projectors) before one restricted diagonalization; energy is the coupling-free Pople–Nesbet functional on the true spin Focks.

**Forces.** Analytic nuclear gradient from the ROHF Lagrangian
\[
W = O_\alpha F_\alpha O_\alpha + O_\beta F_\beta O_\beta
\]
(`ROHF::forces` / force path). Units on the public path: eV/Å.

**Hessian (shipped).** Central FD of analytic forces, step \(h = 10^{-4}\) Bohr (displaced in Å via CODATA conversion), then symmetrized:
\[
g = -F,\qquad
H_{ij} \approx \frac{g_i(+h\,e_j) - g_i(-h\,e_j)}{2h},
\]
with \(H\) reported in eV/Å². Fresh ROHF SCF per displacement (no DIIS/MO state bleed). Same algorithm class as the offline oracle.

**Not included.**

- Analytic three-space ROHF CPHF / `hessian.ROHF`-class second derivatives.
- Mass-weighting, frequencies, IR intensities (those links live on `vib::Hessian*` / RHF analytic Hessian paths).
- RI density fitting, SOSCF, level shift (throw; no silent fallback).
- Live PySCF inside ctest (pin is offline only).

## Implementation

Public C++ (Frame + concrete method; no `*Config`, no factories):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

ATV::cpu::Frame frame(2);
// O₂: Z=8, r = 1.2075 Å along z; 20 Å cubic box (gas phase)
// ... fill type / mass / x,y,z / box as in the regression ...

ATV::cpu::scf::ROHF rohf(/*n_alpha=*/9, /*n_beta=*/7);
rohf.basis = ATV::cpu::scf::BasisSet::STO3G;
rohf.tol = 1e-12;
rohf.max_iter = 200;

const ATV::cpu::vib::HessianResult res = rohf.hessian(frame);
// res.hessian: [6,6] dense, eV/Å²; res.n_atoms == 2
// rohf.energy() [Ha] after restore of reference geometry
```

Build and run the regression binary:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_rohf-hessian-o2 -j$(nproc)
  ./build/regressions/reg_rohf-hessian-o2
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/rohf.h` | Public `ROHF` / `hessian` |
| `src/cpu/scf/rohf.cpp` | FD-of-forces implementation |
| `include/atomiverse/cpu/vibration/hessian_result.h` | `HessianResult` contract |
| `regressions/scf/rohf-hessian-o2/reg_rohf-hessian-o2.cpp` | Product gate |
| `regressions/scf/rohf-hessian-o2/REFERENCE.md` | Offline pins |

## Validation (vs PySCF)

Source: [`regressions/scf/rohf-hessian-o2/REFERENCE.md`](../../../regressions/scf/rohf-hessian-o2/REFERENCE.md)
Oracle class: offline central FD of PySCF **2.14.0** analytic ROHF nuclear gradient (`mf.nuc_grad_method().kernel()`), \(h=10^{-4}\) Bohr, symmetrized; units via Atomiverse CODATA factors. PySCF `mf.Hessian()` / `hessian.ROHF` → `NotImplementedError` (not used as twin). Never live-import PySCF in ctest.

**Setup (from REFERENCE):**

| Item | Value |
| ---- | ----- |
| Molecule | O₂, charge 0, spin \(2S=2\) |
| Geometry [Å] | O (0, 0, 0); O (0, 0, 1.2075) |
| Box | 20 Å cubic (does not enter molecular SCF) |
| Basis | STO-3G |
| Angular | PySCF pin: `cart=True` (s/p only; cart ≡ spherical for this basis) |
| SCF | ROHF, `conv_tol = 1e-12` |
| \(E_{\mathrm{ROHF}}\) (ref, not product gate) | −147.632166990682 Ha |
| Product tol | **1e-5 eV/Å²** on nine gated elements; off-axis \(\lvert H_{ij}\rvert < 10^{-3}\) |

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

Regression also asserts Hessian symmetry (\(\lvert H-H^\top\rvert \le 10^{-8}\)) and ROHF energy within 1e-8 Ha of the pin after restore.

## Performance

**`perf_exempt=true`.** No external wall-time twin for this case. PySCF 2.14.0 has no analytic ROHF Hessian twin; the offline oracle is multi-SCF FD of analytic gradients—the same work class as Atomiverse `ROHF::hessian`, not a one-shot CPHF wall-time twin.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| analytic ROHF Hessian | n/a | n/a | **n/a** | n/a | `perf_exempt` — no `hessian.ROHF` |

Confirm PySCF gap:

```bash
~/work/.x86_64/bin/python -c "
from pyscf import gto, scf
m=gto.M(atom='O 0 0 0; O 0 0 1.2075',basis='sto-3g',spin=2,cart=True)
mf=scf.ROHF(m); mf.kernel(); mf.Hessian()
"
# → NotImplementedError
```

If a same-class FD-vs-FD wall comparison is needed later (not the fair-bench gate):

```bash
# Not a fair analytic twin; informational only.
python tools/pyscf_oracle/fair_bench.py --method rohf -- \
  ./build/regressions/reg_rohf-hessian-o2
```

**Caveats:** twin would count many ROHF SCFs + gradients on both sides; ratios are not comparable to analytic-RHF Hessian vs PySCF CPHF. Gate \(r\le 10\) does not apply while exempt.

## Limitations

- **Algorithm:** FD of analytic forces, not analytic ROHF CPHF; \(O(N_{\mathrm{atoms}})\) SCFs dominate cost.
- **PySCF parity:** no `hessian.ROHF` oracle for element-wise analytic twin or fair speed gate.
- **Scope gaps:** no RI Hessian; no frequencies/IR from this path; open-shell SOSCF / level shift still throw on ROHF.
- **Roadmap:** full three-space ROHF CPHF Hessian is the natural follow-up (see `docs/roadmap/hartree-fock.md`); UHF analytic Hessian and RHF FD/analytic Hessian are separate methods.
- **Unit tests:** product accuracy lives only under `regressions/`; do not place e2e Hessian solves in `tests/`.

## References

- Offline pin: `regressions/scf/rohf-hessian-o2/REFERENCE.md`
- Roadmap: `docs/roadmap/hartree-fock.md`
- Oracle tooling: `tools/setup_pyscf_env.sh`, `tools/pyscf_oracle/pin_scf.py`
- C. C. J. Roothaan, *Rev. Mod. Phys.* **32**, 179 (1960) — ROHF.
- Guest & Saunders, *Mol. Phys.* **28**, 819 (1974) — coupling convention.
- J. A. Pople et al., *Int. J. Quantum Chem. Symp.* **13**, 225 (1979) — analytic HF second derivatives.
- Y. Yamaguchi et al., *A New Dimension to Quantum Chemistry* (Oxford, 1994).
