# GPW Γ-RKS BLYP — isolated H₂O / DZVP-MOLOPT-GTH (Martyna–Tuckerman)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft` + `run(frame)`; `xc = XcKind::BLYP`; `Kind` + `GaussianBasis` / `GthPotential` lookup ctors |
| Roadmap | handoff ratio **perf_exempt** (dual-gate vs PBE); sibling PBE pin [gpw-water-pbe-mt](gpw-water-pbe-mt.md); narrative [ch09-gpw](../ch09-gpw.md) |

## Summary

Γ-point closed-shell **Gaussian-and-plane-waves (GPW)** Kohn–Sham DFT total
energy for an **isolated** water molecule with the **BLYP** GGA functional
(Becke exchange + Lee–Yang–Parr correlation). Setup matches the PBE foothold
`gpw-water-pbe-mt`: contracted MOLOPT Gaussians, uniform grid + FFT for Hartree
and XC, GTH pseudopotentials, density cutoff **400 Ry**, Martyna–Tuckerman
cluster electrostatics (`nonperiodic = true`). Public path is concrete
`cpu::gpw::GpwDft` (kinds ctor arg + members; no `*Config` bag) run as
`dft.run(frame)` with `xc = XcKind::BLYP`. The recorded check (no standing
regression) asserted SCF convergence, a **dual-gate** against identical
geometry PBE (\(|E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha), and a soft
absolute band on first-green Atomiverse \(E_{\mathrm{BLYP}}\) until an offline
CP2K twin replaces it. Fair wall vs CP2K is **`perf_exempt`** (no CP2K binary
on the pin host).

## Method

**GPW dual representation** (Quickstep family; Lippert, Hutter, Parrinello
1997/1999). Density matrix \(P\) lives in a small contracted Gaussian basis
\(\{\chi_\mu\}\). Each SCF cycle:

1. Collocate \(\rho(\mathbf{r}) = \sum_{\mu\nu} P_{\mu\nu}\,\chi_\mu(\mathbf{r})\chi_\nu(\mathbf{r})\)
   onto a uniform real-space grid set by the plane-wave cutoff \(E_{\mathrm{cut}}\)
   (here 400 Ry).
2. FFT → reciprocal space; solve Poisson for the Hartree potential; evaluate
   \(v_{\mathrm{xc}}[\rho]\) on the grid — \(\mathcal{O}(N\log N)\).
3. Integrate grid potentials back against the Gaussians to form
   \(V_{\mu\nu}^{\mathrm{Hxc}}\); add kinetic + GTH local/nonlocal PP in the
   AO basis; diagonalize; rebuild \(P\).

**BLYP** (Becke, *Phys. Rev. A* **38**, 3098 (1988) + Lee–Yang–Parr, *Phys.
Rev. B* **37**, 785 (1988)). Pure GGA exchange–correlation:

\[
E_{\mathrm{xc}}^{\mathrm{BLYP}}[\rho]
=
E_{\mathrm{x}}^{\mathrm{B88}}[\rho]
+
E_{\mathrm{c}}^{\mathrm{LYP}}[\rho].
\]

libxc / CP2K names: `GGA_X_B88` + `GGA_C_LYP` / `XC_FUNCTIONAL BLYP`.
Atomiverse routes BLYP through the shared pointwise engine already used by
all-electron RKS and by GPW PBE/PBEsol:

\[
\texttt{evaluate\_xc}(\texttt{XCType::BLYP},\,\rho,\,\sigma,\,\ldots),
\qquad
\texttt{xc\_grid\_blyp}
=
\texttt{xc\_grid\_gga}(\ldots,\,\texttt{XCType::BLYP}).
\]

On the GPW grid, closed-shell GGA assembly builds spectral \(\nabla\rho\) and
the divergence of the \(v_\sigma\) flux exactly as for `XcKind::PBE`, with
`XcKind::BLYP` selecting the B88+LYP kernels. This is the **GPW** BLYP product
(distinct from molecular all-electron `blyp-water-sto3g` vs PySCF).

**Core treatment.** Norm-conserving **GTH** analytic pseudopotentials remove
core electrons. Valence counts on this system: O **q6**, H **q1** → 8 valence
electrons → \(n_{\mathrm{occ}} = 4\) closed-shell pairs. PP rows are the same
**GTH-PBE** tables as the PBE twin (CP2K `GTH_POTENTIALS` / in-tree
`GthPotential(Element, name)`); XC is independent of the PP name.

**Isolated electrostatics.** Martyna–Tuckerman Poisson (`nonperiodic = true`;
CP2K `PERIODIC NONE` / `PSOLVER MT`) so the finite cubic cell does not couple
the molecule to periodic images. Cell \(L = 10\) Å matches
`gpw-water-pbe-mt`.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Spin | Closed-shell RKS (Γ only) |
| XC | **BLYP** GGA (`XcKind::BLYP`); PBE twin for dual-gate |
| Basis | DZVP-MOLOPT-GTH (O, H) |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| Cutoff | 400 Ry |
| SCF | direct \(P\)-mixing \(\alpha=0.3\), `max_iter=200`, `tol=1e-7` |
| BC | MT isolated (`nonperiodic=true`) |
| Valence | 8 e⁻ → \(n_{\mathrm{occ}} = 4\) |

**Key exclusions (non-goals of this slug)**

- Absolute total-energy digit match to a live CP2K process at pin time (no
  binary on host; do **not** invent CP2K digits from memory). Soft absolute
  band is first-green ATV until offline re-pin.
- Analytic forces / stress / UKS / k-points / D3 / BOMD as *this* product’s gate
  (other queue slugs: `gpw-forces-water`, `gpw-uks-o2-gamma`,
  `gpw-pbe-d3bj-ice`, …).
- Live CP2K (or any third-party) inside ctest.
- MPI multi-rank; OT minimizer (`use_ot`); public Config bags / factories.
- Molecular all-electron BLYP/STO-3G (`blyp-water-sto3g` is the PySCF twin).

## Implementation

Public surface only (`cpu::Frame` + `GpwDft` ctor args / members; lookup
constructors for basis/PP tables — not factories for a parallel world model).
BLYP single-point + PBE dual-gate:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// CP2K getting-started h2o.inp geometry [Å], 10 Å cubic cell.
cpu::Frame frame(3);
const int z[3] = {8, 1, 1};
const double x[3] = {5.0000, 5.7586, 4.2414};
const double y[3] = {5.0000, 5.0000, 5.0000};
const double zc[3] = {5.0000, 5.5043, 5.5043};
for (int i = 0; i < 3; ++i) {
    frame.span(kAtom_type)[i] = z[i];
    frame.span(kAtom_x)[i] = x[i];
    frame.span(kAtom_y)[i] = y[i];
    frame.span(kAtom_z)[i] = zc[i];
}
frame.box = cpu::Box::cubic(10.0);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
const cpu::gpw::Kind H{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

cpu::gpw::GpwDft dft{{O, H}};
dft.cutoff_ry = 400.0;  dft.max_iter = 200;  dft.tol = 1e-7;  dft.mix = 0.3;
dft.xc = cpu::gpw::XcKind::BLYP;  dft.nonperiodic = true;
dft.build(frame);
const cpu::gpw::GpwResult& r = dft.run(frame);

const double e_ha = r.e_total;   // [Hartree]
const bool ok = r.converged;
// frame.meta.get(kEnergy) is also set [eV]
```

Dual-gate (product regression): second `GpwDft` with
`xc = XcKind::PBE` on the same geometry; require
\(|E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha.

Build configuration (CPU login; modules as in `AGENTS.md`):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_GPW=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
'
```

| Path | Role |
| ---- | ---- |
| `gpw_solver.h` | Public `Kind`, `GpwDft`, `XcKind::{PADE,PBE,PBEsol,BLYP}` |
| `gaussian_basis.h` | `GaussianBasis(Element, name)` lookup |
| `gth_potential.h` | `GthPotential(Element, name)` lookup |
| `xc_grid_gga.h` | `xc_grid_blyp` / shared GGA grid path |
| `src/cpu/gpw/` | GPW SCF / collocate / integrate (private) |
| CP2K oracle | Offline CP2K twin (never ctest) |

## Validation (vs CP2K)

Source: values recorded when this page was written (no standing regression).
Oracle: offline **CP2K Quickstep** twin (same geometry, DZVP-MOLOPT-GTH +
GTH-PBE, CUTOFF 400, PERIODIC NONE / PSOLVER MT, `XC_FUNCTIONAL BLYP`).
External-engine absolute digits were **not** available at pin time (no CP2K
binary on host); the product therefore uses a **dual-gate** vs PBE plus a soft
absolute first-green ATV band. **Never** launch CP2K from ctest. Twin input:
CP2K oracle. Sibling absolute CP2K pin for
the PBE twin: [gpw-water-pbe-mt](gpw-water-pbe-mt.md)
(\(E_{\mathrm{CP2K}}^{\mathrm{PBE}} = -17.212337\) Ha).

**Setup:**

| Item | Value |
| ---- | ----- |
| Molecule | H₂O, charge 0, singlet (closed-shell RKS) |
| Geometry [Å] | O (5.0000, 5.0000, 5.0000); H (5.7586, 5.0000, 5.5043); H (4.2414, 5.0000, 5.5043) |
| Cell | cubic \(L = 10\) Å, non-periodic (MT) |
| Basis | DZVP-MOLOPT-GTH |
| Pseudopotentials | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| XC | BLYP (product); PBE (dual-gate twin) |
| Density cutoff | 400 Ry |
| Valence | 8 e⁻ → \(n_{\mathrm{occ}} = 4\) |
| SCF | mix 0.3, max_iter 200, tol \(10^{-7}\) |
| Provenance | CP2K getting-started geometry; BLYP activation dual-gate (2026-07-25) |

**Accuracy gate (product):**

| Quantity | Atomiverse (verified 2026-07-25) | Oracle / pin | Tol | Notes |
| -------- | -------------------------------: | -----------: | --- | ----- |
| \(E_{\mathrm{BLYP}}\) | **−17.1982170663 Ha** | first-green ATV soft pin | **\(1\times 10^{-3}\) Ha** | Absolute soft band until offline CP2K |
| \(E_{\mathrm{PBE}}\) (twin) | −17.2127580657 Ha | CP2K −17.212337 Ha (`gpw-water-pbe-mt`) | 1 mHa on PBE slug | Residual ~0.42 mHa on PBE stack |
| \(\lvert E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}\rvert\) | \(1.454\times 10^{-2}\) Ha | dual-gate floor \(10^{-4}\) Ha | **hard** \(> 10^{-4}\) | BLYP activation |
| SCF (BLYP) | converged, 12 iters | — | `converged == true` | Product gate |
| Frame energy | finite `frame.meta.get(kEnergy)` [eV] | — | finite | Product gate |

Tolerance rationale (from REFERENCE): without a CP2K binary, absolute Ha digits
must not be invented. Dual-gate proves the BLYP kernel is active on the GPW
GGA path (energy shift ≫ floor). Soft absolute band freezes setup integrity
after first green; when CP2K is available offline, re-run the twin and replace
the absolute pin (cite CP2K version), tighten tol toward ~1 mHa like
`gpw-water-pbe-mt`.

Offline re-pin (human only; never ctest):

```bash
# when a CP2K binary is available
cp2k.psmp -i <CP2K oracle input for this case> \
          -o /tmp/gpw-blyp-water.out
# record TOTAL ENERGY [Ha] into REFERENCE.md; tighten kEnergyRef / kTol
```

## Performance

Source: pin notes
(2026-07-25).

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the pin host (`cp2k`, `cp2k.psmp`, `cp2k.sopt` not found; `module avail cp2k` empty). Fair wall twin deferred. |
| \(T_{\mathrm{ATV}}\) BLYP (OMP=4, Release) | ~34.4 s |
| \(T_{\mathrm{ATV}}\) PBE twin (same host) | ~19.1 s |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |

. Gate when a twin exists:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on the same host/core count
(single-rank).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| H₂O BLYP MT (400 Ry) | — | ~34.4 | **exempt** | n/a | No CP2K binary at pin; remeasure when available |

## Limitations

- **Energy-only** Γ-RKS + MT isolated BC with BLYP. Forces:
  `gpw-forces-water` (PBE-era path; BLYP force parity is follow-on);
  UKS+GGA: `gpw-uks-pbe`; D3(BJ)/D4/TS: other queue slugs.
- **Absolute CP2K pin deferred** — dual-gate + soft ATV band only until offline
  CP2K is run on CP2K oracle.
- **No k-points** on this pin (Γ only); bulk GGA k-mesh is other slugs.
- **CPU only** — `cuda-gpw` remains blocked on the CP2K loop.
- **`perf_exempt`**: no fair \(r\) until an offline CP2K binary is timed on the
  same host; do not invent a speed ratio.
- Basis/PP tables are static in-tree headers (`cpu/gpw/basis_data/`); unsupported
  element/name **throws** (no silent fallback).
- Further pure-GGA `XcKind` expansion (revPBE, RPBE, …) is residual queue work
  — one functional per wave; BLYP is the first non-PBE pure GGA on GPW.

## References

- Offline pin: values recorded when this page was written (no standing regression)
- Oracle twin: CP2K oracle
- Oracle notes: CP2K oracle
- Sibling PBE pin / tutorial: case `gpw-water-pbe-mt` pin,
  [gpw-water-pbe-mt](gpw-water-pbe-mt.md)
- Molecular all-electron BLYP (PySCF): [blyp-water-sto3g](blyp-water-sto3g.md)
- Narrative chapter: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Public API: `gpw_evaluator.h`,
  `gpw_solver.h` (`XcKind::BLYP`)
- A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988) — B88 exchange.
- C. Lee, W. Yang, R. G. Parr, *Phys. Rev. B* **37**, 785 (1988) — LYP
  correlation.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997);
  *Theor. Chem. Acc.* **103**, 124 (1999) — GPW / Quickstep.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — GTH.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT
  Poisson.
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — CP2K
  Quickstep; MOLOPT / DZVP-MOLOPT-GTH table family.
