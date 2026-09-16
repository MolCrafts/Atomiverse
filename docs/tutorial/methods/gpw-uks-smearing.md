# Collinear UKS + metallic Fermi–Dirac smearing — BCC Na / GPW multi-k

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-uks-smearing` |
| Regression | `regressions/gpw/gpw-uks-smearing/` |
| Public API | `cpu::gpw::GpwDft` members `n_alpha`/`n_beta`, `smearing`, `degauss` [Ry], `extra_bands`, `mp_order`, `k_mesh`; `GpwResult::{e_total,e_fermi,n_elec_occ,e_entropy}` |
| Roadmap | [dft.md](../../roadmap/dft.md) — Smearing for metals (GPW multi-k UKS ✓) |
## Summary

Metals need **fractional band occupations** and a shared chemical potential
\(\mu\) that conserves the valence electron count. On **collinear
spin-polarized GPW**, Atomiverse productizes that path on the concrete
method `cpu::gpw::GpwDft`: set `n_alpha`/`n_beta` (UKS seed),
`smearing = Smearing::fermi_dirac` (or Methfessel–Paxton), `degauss` [Ry],
`extra_bands`, and a multi-k mesh — **no** public `*Config` bag and **no**
`make_*` factory. Occupations are spin-orbital \(f_{\sigma kb}\in[0,1]\)
with **one shared** \(\mu\) over \(\alpha\) and \(\beta\); magnetization is
**free under smear** (`n_alpha`/`n_beta` seed only). The SCF total is the
**Mermin free energy** \(F = E_{\mathrm{KS}} - TS\).

Textbook pin: **BCC Na** conventional cubic (\(a=4.225\) Å, 2 atoms), PADE
LSD, GTH-PBE-q9 + DZVP-MOLOPT-SR-GTH, Monkhorst–Pack \(2\times2\times2\),
Fermi–Dirac `degauss = 0.02` Ry, ferro seed \(n_\alpha=10\), \(n_\beta=8\).
Regression: `reg_gpw-uks-smearing`. Closes the product gap after multi-k
RKS smear (`regressions/gpw/gpw-metallic-smearing`) and fixed-occ multi-k
UKS (`gpw-kpoints-uks`). Offline CP2K 2026.1 documents `LSD` +
`&SMEAR METHOD FERMI_DIRAC`; absolute free energy is an **ATV-frozen** pin
(grid / multigrid differ from CP2K by ~0.6 Ha class). Fair wall
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 0.017\) (≤10×).

Related: PW closed-shell metal smear [pw-smearing-al](pw-smearing-al.md);
Γ UKS O₂ foothold [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md); narrative
[Ch. 9 GPW](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in a contracted Gaussian AO basis;
density collocation on a uniform real-space grid; Hartree via FFT Poisson;
XC on the grid; potential integrated back into the AO basis. Core electrons
removed by analytic **GTH** pseudopotentials. Grid resolution set by the
plane-wave density cutoff in Rydberg. Citation: Lippert, Hutter,
Parrinello, *Mol. Phys.* **92**, 477 (1997).

**Collinear UKS / LSD.** Separate \(\rho_\alpha\), \(\rho_\beta\); Hartree
from \(\rho=\rho_\alpha+\rho_\beta\); spin-resolved **PADE** LSD
(`XcKind::PADE`); separate \(H_\alpha\), \(H_\beta\). Integer members
`n_alpha` / `n_beta` select the UKS path and seed occupations; under
metallic smear they are **not** locked fillings — \(\mu\) and fractional
\(f_{\sigma kb}\) set the magnetization freely. Citation: Goedecker,
Teter, Hutter, *Phys. Rev. B* **54**, 1703 (1996).

**Occupations (UKS spin-orbital convention).** Reuse
`cpu::pw::internal` with `spin_degen = 1` so each spin-orbital occupation
is \(f\in[0,1]\). With reduced energy \(x = (\varepsilon-\mu)/\sigma\) and
\(\sigma =\) `degauss` converted to Hartree inside the SCF:

| Kind | \(f(x)\) | Public enum |
| ---- | -------- | ----------- |
| Fixed integer | step fill of `n_alpha`/`n_beta` | `Smearing::none` (default) |
| Fermi–Dirac | \(1/(1+e^{x})\) | `Smearing::fermi_dirac` |
| Methfessel–Paxton order \(N\) | MP series of order `mp_order` | `Smearing::methfessel_paxton` |

**Electron count.** One shared \(\mu\) (Fermi search) so

\[
N_{\mathrm{occ}}
  = \sum_{\sigma,k,b} w_k\, f_{\sigma kb}(\varepsilon_{\sigma kb},\mu,\sigma)
  = Z_{\mathrm{val}}\cdot N_{\mathrm{atoms}}
\]

(for BCC Na with GTH-PBE-q9: \(N_e = 2\times 9 = 18\)). Extra empty states
above \(\max(n_\alpha,n_\beta)\) come from `extra_bands` (CP2K
`ADDED_MOS` parity; GPW still diagonalizes the full AO spectrum).

**Free energy.** With entropy weight \(s\) for the chosen smear (FD:
\(s = -[f\ln f+(1-f)\ln(1-f)]\) per spin-orbital),

\[
F = E_{\mathrm{KS}} - TS,\qquad
E_{\mathrm{entropy}} = -TS = -\sigma\sum_{\sigma,k,b} w_k\, s_{\sigma kb}.
\]

`GpwResult.e_total` [Ha] is this free energy when smearing is on;
`e_entropy` reports the \(-TS\) piece; `e_fermi` is \(\mu\) [Ha];
`n_elec_occ` is \(N_{\mathrm{occ}}\). Citation: Mermin, *Phys. Rev.*
**137**, A1441 (1965); Methfessel & Paxton, *Phys. Rev. B* **40**, 3616
(1989).

**Identity (closed-shell).** On multi-k H₂, UKS \(n_\alpha=n_\beta=1\) + FD
matches RKS \(n_{\mathrm{occ}}=1\) + FD to machine-level free energy
(\(\lvert\Delta F\rvert \lesssim 10^{-15}\) Ha on the same binary; regression
gate \(\lvert\Delta F\rvert \le 5\times 10^{-6}\) Ha). BCC Na is
**non-magnetic**: a ferro seed collapses under smear to the same free
energy as RKS FD (`gpw-metallic-smearing`) to \(\sim 4\times 10^{-13}\) Ha.

**Excluded (this slug).**

- Absolute free-energy match to CP2K (grid / `EPS_DEFAULT` / REL_CUTOFF /
  multigrid differ).
- OT + smearing (throws at `build()`).
- Forces / stress under UKS smear as acceptance pins.
- Cold / Marzari–Vanderbilt / Gaussian public smear on GPW (PW has
  Gaussian; e2e gate here is Fermi–Dirac).
- CUDA GPW smearing; molecular all-electron UKS (PySCF loop).

## Implementation

Public surface only: construct `cpu::gpw::GpwDft`, set members, `build` /
`run` on `std::vector<GpwAtom>` (Bohr positions, lookup-table basis/PP).
Headers: `include/atomiverse/cpu/gpw/gpw_solver.h` (API + `Smearing`);
occupations shared with PW internals (not a user type).

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using cpu::gpw::GpwDft;
using cpu::gpw::Smearing;
using cpu::gpw::XcKind;

// BCC conventional cubic: 2 Na at (0,0,0) and (½,½,½); a in Bohr
constexpr double a_ang = 4.225;
constexpr double a = a_ang / 0.52917720859;
const cpu::gpw::GaussianBasis bNa(molrs::Element::Na, "DZVP-MOLOPT-SR-GTH");
const cpu::gpw::GthPotential pNa(molrs::Element::Na, "GTH-PBE-q9");
std::vector<cpu::gpw::GpwAtom> atoms;
const double frac[2][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
for (int i = 0; i < 2; ++i) {
    const double pos[3] = {frac[i][0] * a, frac[i][1] * a, frac[i][2] * a};
    atoms.emplace_back(pos, bNa, pNa, 11);
}

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = a;
m.cutoff_ry = 150.0;
m.n_alpha = 10;  // ferro seed; M free under smear
m.n_beta = 8;
m.max_iter = 80;
m.tol = 1e-6;
m.eps_default = 1e-7;
m.xc = XcKind::PADE;
m.diis_depth = 6;
m.mix = 0.4;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
m.k_time_reversal = true;
m.smearing = Smearing::fermi_dirac;
m.degauss = 0.02;   // Ry
m.extra_bands = 4;  // CP2K ADDED_MOS parity

m.build(atoms);
const cpu::gpw::GpwResult& r = m.run(atoms);
// r.e_total [Ha] free energy F; r.e_fermi [Ha]; r.n_elec_occ ≈ 18;
// r.e_entropy [Ha] (−TS); r.converged, r.iters, r.n_kpts
```

`GpwResult` fields used by the gate: `e_total`, `e_fermi`, `n_elec_occ`,
`e_entropy`, `converged`, `iters`, `n_kpts`. Closed-shell identity is
exercised first in the same binary (H₂ multi-k UKS+FD ≡ RKS+FD).

```bash
# configure (login shell + Lmod — see AGENTS.md)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-uks-smearing
  ./build/regressions/reg_gpw-uks-smearing
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Solver API | `include/atomiverse/cpu/gpw/gpw_solver.h` (`GpwDft`, `Smearing`, `GpwResult`) |
| Solver impl | `src/cpu/gpw/gpw_solver.cpp` |
| Basis / PP lookup | `gaussian_basis.h`, `gth_potential.h` (in-tree tables) |
| Regression | `regressions/gpw/gpw-uks-smearing/reg_gpw-uks-smearing.cpp` |
| CP2K twin input | `tools/cp2k_oracle/inputs/gpw-uks-smearing/na_bcc_uks_fermi_dirac.inp` |
| RKS twin wall class | `tools/cp2k_oracle/inputs/gpw-metallic-smearing/na_bcc_fermi_dirac.inp` |

## Validation (CP2K)

Source:
[`regressions/gpw/gpw-uks-smearing/REFERENCE.md`](../../../regressions/gpw/gpw-uks-smearing/REFERENCE.md)
(offline pins only; **never** live CP2K in ctest).

### ATV system (free energy + \(N_{\mathrm{occ}}\) gate)

| Setting | Value |
| ------- | ----- |
| Geometry | BCC Na conventional cubic, \(a=4.225\) Å, 2 atoms |
| Basis | `DZVP-MOLOPT-SR-GTH` (Na) — in-tree table |
| PP | `GTH-PBE-q9` (9 valence e⁻ / atom → \(N_e=18\)) |
| XC | PADE (LSD) |
| Density cutoff | 150 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\), time-reversal fold |
| Spin seed | \(n_\alpha=10\), \(n_\beta=8\) (ferro; free under smear) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| Extra bands | 4 |
| SCF | DIIS 6, `tol = 1e-6` Ha, `EPS_DEFAULT = 1e-7`, `mix = 0.4` |
| Pin date | 2026-07-26 (Arrhenius x86_64) |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| Free energy \(F\) | **−92.126330608004 Ha** | **\(10^{-4}\) Ha** | frozen ATV offline pin (primary \(E\) gate) |
| \(N_{\mathrm{occ}}=\sum_{\sigma k b} w_k f\) | **18.0** | \(10^{-5}\) | valence count |
| \(E_F\) | **0.055825 Ha** (finite) | \(\lvert E_F\rvert < 10^3\) Ha | among bands; not NaN |
| Entropy term \(-\sigma S\) | **−2.26×10⁻⁴ Ha** | finite | diagnostic |
| RKS Na FD identity | \(\lvert F_{\mathrm{UKS}}-F_{\mathrm{RKS}}\rvert\sim 4\times 10^{-13}\) Ha | — | non-magnetic collapse |
| H₂ closed-shell identity | \(\lvert F_{\mathrm{UKS}}-F_{\mathrm{RKS}}\rvert\) | \(\le 5\times 10^{-6}\) Ha | multi-k \(1\times1\times2\); both \(N_e=2\) |

Recorded wall ≈ **1.56 s** single rank (`OMP_NUM_THREADS=4`), 16 SCF iters.

### Offline CP2K twin (algorithm / SMEAR documentation)

**Not an absolute-energy twin** (ATV single-grid / `EPS_DEFAULT` vs CP2K
multigrid `NGRIDS 4` / REL_CUTOFF differ by ~0.6 Ha class). Documents that
CP2K `LSD` + `&SMEAR METHOD FERMI_DIRAC` + `ADDED_MOS` is the product
counterpart of ATV multi-k UKS metallic occupations.

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (local install under `…/work/cp2k/install`) |
| Input | `tools/cp2k_oracle/inputs/gpw-uks-smearing/na_bcc_uks_fermi_dirac.inp` |
| Method | Quickstep GPW **LSD** + FERMI_DIRAC, PADE, GTH-PBE-q9 + DZVP-MOLOPT-SR-GTH |
| Cell | BCC conventional \(a=4.225\) Å, 2 Na |
| SMEAR | FERMI_DIRAC, \(T_e = 3157.75\) K (\(\approx 0.02\) Ry \(k_B T\)) |
| k-points | Monkhorst–Pack \(2\times2\times2\) |
| CUTOFF / REL_CUTOFF | 150 / 40 Ry, NGRIDS 4 |
| MULTIPLICITY | 3 (ferro seed; collapses under smear for BCC Na) |

RKS algorithm twin (same system, closed shell; fair wall class):
`tools/cp2k_oracle/inputs/gpw-metallic-smearing/na_bcc_fermi_dirac.inp`
(**≈ 89.5 s**, OMP=4). Oracle path: `tools/cp2k_oracle/README.md` (offline
only).

### Dual gate

1. **Correctness:** R1–R5 (converge, \(N_e\), finite \(E_F\), free-energy
   pin, closed-shell UKS≡RKS identity).
2. **Perf:** \(r \le 10\) vs CP2K Na FD twin wall class.

## Performance

Source: same REFERENCE (single-rank wall, 2026-07-26). Gate:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) unless exempted. Handoff:
the case `REFERENCE.md`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| Na FD twin class (`na_bcc_fermi_dirac.inp`) vs `reg_gpw-uks-smearing` (Na pin) | **89.51** | **≈ 1.56** | **≈ 0.017** | ✓ | UKS CP2K twin is same system with `LSD`; wall class matches RKS FD twin already timed offline; ATV own multigrid |

\[
r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 0.017 \ll 10
\]

`within_10x = true`. Fairness note: same system/basis/PP class; do **not**
treat \(r\) as bit-identical Hamiltonian throughput. Queue records ratio
**~0.017**.

## Limitations

- **Absolute free energy is ATV-frozen, not CP2K-matched.** Grid /
  `EPS_DEFAULT` / REL_CUTOFF / multigrid differ (~0.6 Ha class as RKS twin);
  CP2K documents SMEAR + LSD settings / algorithm only.
- **BCC Na is non-magnetic under smear.** The ferro seed exercises the
  UKS+smear code path; the physical solution collapses to RKS FD energy.
  For a permanently magnetic metallic pin, see PW bulk Fe
  (`pw-uks-fe-smearing` / roadmap).
- **OT + smearing throws** at `build()` — diagonalization SCF only on this
  path.
- **No force/stress acceptance pin** under UKS smear on this slug.
- **e2e gate is Fermi–Dirac**; Methfessel–Paxton is API-present
  (`mp_order`) and shared with RKS metallic smear — not the Na free-energy
  pin here.
- **CUDA GPW smearing** out of scope.
- **Public Frame energy** for outer drivers remains eV when using
  evaluators; this regression gates internal `GpwResult` free energy in Ha
  (cited pin units).
- **Speed.** \(r\approx 0.017\) on the documented dual; no multi-system
  standing CP2K perf suite in ctest.

## References

- Offline pin: [`regressions/gpw/gpw-uks-smearing/REFERENCE.md`](../../../regressions/gpw/gpw-uks-smearing/REFERENCE.md)
- Regression: [`reg_gpw-uks-smearing.cpp`](../../../regressions/gpw/gpw-uks-smearing/reg_gpw-uks-smearing.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-uks-smearing/`](../../../tools/cp2k_oracle/inputs/gpw-uks-smearing/)
- RKS twin / wall class: [`tools/cp2k_oracle/inputs/gpw-metallic-smearing/`](../../../tools/cp2k_oracle/inputs/gpw-metallic-smearing/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Handoff ratio **~0.017**: the case `REFERENCE.md`
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md) (Smearing for metals — GPW multi-k UKS ✓)
- Public header: `include/atomiverse/cpu/gpw/gpw_solver.h`
- Related tutorials: [pw-smearing-al](pw-smearing-al.md), [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md)
- Literature: Mermin, *Phys. Rev.* **137**, A1441 (1965); Methfessel &
  Paxton, *Phys. Rev. B* **40**, 3616 (1989); Lippert, Hutter, Parrinello,
  *Mol. Phys.* **92**, 477 (1997); Goedecker, Teter, Hutter, *Phys. Rev. B*
  **54**, 1703 (1996)
- Narrative GPW background: [ch09-gpw](../ch09-gpw.md)
