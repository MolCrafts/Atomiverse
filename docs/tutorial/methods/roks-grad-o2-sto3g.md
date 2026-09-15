# ROKS analytic gradient — triplet O₂ / LSDA / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `roks-grad-o2-sto3g` |
| Regression | `regressions/scf/roks-grad-o2-sto3g/` |
| Public API | `cpu::scf::ROKS::forces(Frame&)` → `GradientResult` |
| Energy sibling | `regressions/scf/roks-o2-sto3g` |
| ROHF grad sibling | `regressions/scf/o2_rohf_grad` |

## Summary

Restricted open-shell Kohn–Sham (ROKS) analytic nuclear forces on high-spin O₂
(triplet, \(n_\alpha=9\), \(n_\beta=7\)) with LSDA (`XCType::LSDA` ≡ PySCF
`LDA,VWN`) and Cartesian STO-3G. Public path is the concrete method
`cpu::scf::ROKS` on a `cpu::Frame`; parameters are constructor args / members —
no `*Config` bag. Forces return as `GradientResult` ([eV/Å]); the regression
converts to Ha/Bohr for the pin. End-to-end gate:
`regressions/scf/roks-grad-o2-sto3g` (`reg_roks-grad-o2-sto3g`). Offline oracle:
PySCF 2.14.0 `dft.ROKS.nuc_grad_method` pinned in `REFERENCE.md` (never live in
ctest). Fair wall \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 1.117\).

## Method

**Spin treatment.** One spatial MO set with closed / open / virtual three-space
and split occupation \(n_\alpha \ge n_\beta\). High-spin \(\langle S^2\rangle =
S(S+1)\) exactly (here triplet \(S=1\), \(\langle S^2\rangle=2\)). SCF couples
the unrestricted KS Fock pair into one Roothaan effective Fock
(Guest–Saunders / PySCF `get_roothaan_fock`); total energy is the coupling-free
UKS functional on that constrained RO density.

**Gradient composition.**

- Electronic gradient reuses the **unrestricted KS** path: Coulomb \(J\), hybrid
  \(K\) (if any), and XC grid response on the split-occupation \(\alpha/\beta\)
  densities from the shared spatial orbitals (PySCF `grad/uks.py` / `grad_elec`
  shape).
- ROKS-specific piece is the **energy-weighted density** (same Lagrangian form
  as ROHF):

\[
W = O_\alpha F_\alpha O_\alpha + O_\beta F_\beta O_\beta,
\]

  with **KS spin Focks** \(F_\sigma\), not the Roothaan effective Fock and not
  \(\sum_i \varepsilon_i C_i C_i^\top\). Matches PySCF `make_rdm1e` on the KS
  Fock pair. A naive effective-Fock eigenvalue \(W\) fails this pin by
  \(\gtrsim 10^{-1}\) Ha/Bohr (same pathology as the ROHF sibling).

**Approximations / scope of this case.**

- Functional: LSDA (VWN-class as in PySCF `LDA,VWN`); no hybrid, no GGA on this
  regression.
- Basis: STO-3G **Cartesian** (`cart=True`).
- Analytic nuclear gradient only; no Hessian, no ROMP2 forces on this slug.
- XC grid discretization: PySCF analytic vs FD of the same PySCF energy differ
  by \(\sim 1\times 10^{-5}\) Ha/Bohr here; ATV analytic tracks the **FD**
  surface more closely (see Validation).

## Implementation

Headers: `include/atomiverse/cpu/scf/roks.h`,
`include/atomiverse/cpu/scf/methods.h`,
`include/atomiverse/cpu/scf/gradient_result.h`,
`include/atomiverse/cpu/frame/frame.h`. SCF kernel / W path live under
`include/atomiverse/cpu/scf/` (+ mirrored `src/` where implemented).

```cpp
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"

cpu::Frame frame(2);
// O₂: Z=8, r = 1.2075 Å along z; cubic box 20 Å
frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 8;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_z)[1] = 1.2075;
frame.box = cpu::Box::cubic(20.0);

cpu::scf::ROKS roks(/*n_alpha=*/9, /*n_beta=*/7, cpu::scf::XCType::LSDA);
roks.basis = cpu::scf::BasisSet::STO3G;
roks.tol = 1e-12;
roks.max_iter = 200;
const cpu::scf::GradientResult res = roks.forces(frame);
// res.forces(A, d) in eV/Å; convert for Ha/Bohr pins:
// units::convert(..., units::Force::eVPerAngstrom, units::Force::HartreePerBohr)
```

Build / run the regression (CPU login toolchain; see `AGENTS.md`):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_roks-grad-o2-sto3g -j$(nproc)
  ./build/regressions/reg_roks-grad-o2-sto3g
'
```

Consumer source: `regressions/scf/roks-grad-o2-sto3g/reg_roks-grad-o2-sto3g.cpp`.

## Validation (vs PySCF)

Source: [`regressions/scf/roks-grad-o2-sto3g/REFERENCE.md`](../../../regressions/scf/roks-grad-o2-sto3g/REFERENCE.md)
(offline pin; PySCF never in ctest).

| Field | Value |
| ----- | ----- |
| Geometry (Å) | O(0, 0, 0); O(0, 0, 1.2075) |
| Basis | STO-3G, Cartesian |
| Method | ROKS(9,7) / LSDA (`XCType::LSDA` ≡ `LDA,VWN`) |
| Spin | triplet, \(\langle S^2\rangle = 2\) exactly |
| `conv_tol` | \(1\times 10^{-12}\) |
| PySCF | **2.14.0** |
| Oracle | `dft.ROKS` + `nuc_grad_method().kernel()` |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{ROKS}}\) (energy sibling) | −147.193715448518 Ha | \(1\times 10^{-6}\) Ha | `roks-o2-sto3g` |
| \(\partial E/\partial z\) [atom 0] | **+0.16292653404646806** Ha/Bohr | element-wise **\(2\times 10^{-5}\)** Ha/Bohr | reg gate |
| \(\partial E/\partial z\) [atom 1] | **−0.16292653404646806** Ha/Bohr | same | equal–opposite |
| \(\partial E/\partial x\), \(\partial E/\partial y\) | ~0 | same | diatomic on \(z\) |
| Translational invariance \(\lvert\sum_A g_A\rvert_{\max}\) | — | \(1\times 10^{-8}\) Ha/Bohr | reg gate |

Nominal element-wise tol is \(1\times 10^{-5}\) Ha/Bohr; product gate is
**\(2\times 10^{-5}\)** because PySCF analytic and central FD of the PySCF
energy (\(h=10^{-4}\) Bohr, atom 0 \(z\)) differ by **1.004e-5** Ha/Bohr on
this system (LDA grid-response). Atomiverse analytic matches that FD value to
\(\sim 7\times 10^{-9}\) Ha/Bohr and sits \(\sim 1.005\times 10^{-5}\) from the
PySCF analytic pin. Wrong \(W\) still fails by \(\ge 10^{-1}\).

Oracle re-pin:

```bash
# tools/setup_pyscf_env.sh  # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py roks \
    --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN --conv-tol 1e-12 --grad
```

## Performance

Source: `REFERENCE.md` fair-bench block (2026-07-23, login node,
`OMP_NUM_THREADS=4`, `OMP_PROC_BIND=close`, median of \(n=3\)). Handoff ratio
**1.117** (the case `REFERENCE.md`). Not yet a row in the
rollup (that table still lists energy-only
`roks-o2-sto3g` with ROHF twin).

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `roks_grad` (SCF + analytic grad) | **0.1767** | **0.1974** | **1.117** | ✓ | matched method |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\).

```bash
export PATH="$HOME/work/.x86_64/bin:$PATH"
export LD_LIBRARY_PATH="$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
python tools/pyscf_oracle/fair_bench.py --method roks_grad --geom o2 \
    --basis sto-3g --spin 2 --xc LDA,VWN --repeat 3 -- \
    ./build/regressions/reg_roks-grad-o2-sto3g
```

**Twin caveats.** Reg binary includes full element-wise + translational checks;
PySCF side is SCF + `nuc_grad_method` only. Walls are short (\(\sim 0.2\) s) —
ratio noise from OS jitter can be material at this scale. Energy sibling
`roks-o2-sto3g` used a **ROHF** twin (\(r\sim 1.31\)); do not conflate that with
this matched `roks_grad` pair.

## Limitations

- Single textbook case: O₂ triplet, STO-3G cart, LSDA only; no hybrid ROKS
  gradient regression, no GGA/meta-GGA force pin on this slug.
- Grid-response residual vs PySCF analytic forces the \(2\times 10^{-5}\) gate
  rather than a hard \(1\times 10^{-5}\) match to analytic PySCF.
- `ROKS::mp2()`, `fci()`, SOSCF, level shift, density fitting throw / unsupported
  (header contract).
- CUDA ROKS / ROKS forces: not a product path.
- Hessian / vibrational analysis on ROKS density: not this slug (ROHF Hessian
  is a separate queue item).
- Perf rollup table may lag REFERENCE until a bulk refresh of
  ``.

## References

- Offline pin: `regressions/scf/roks-grad-o2-sto3g/REFERENCE.md`
- Energy sibling pin: `regressions/scf/roks-o2-sto3g/REFERENCE.md`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`;
  `tools/pyscf_oracle/pin_scf.py`, `tools/setup_pyscf_env.sh`
- Energy handbook: [`roks-o2-sto3g.md`](roks-o2-sto3g.md)
- Roothaan, *Rev. Mod. Phys.* **32**, 179 (1960)
- Guest & Saunders, *Mol. Phys.* **28**, 819 (1974)
- Johnson, Gill, Pople, *J. Chem. Phys.* **98**, 5612 (1993) (KS XC grid gradient)
- PySCF `pyscf/grad/rohf.py` (`make_rdm1e`), `pyscf/grad/uks.py` (`grad_elec`);
  `dft.ROKS`
