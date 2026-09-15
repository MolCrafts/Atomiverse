# Metallic Fermi–Dirac smearing — BCC Na / GPW multi-k

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-metallic-smearing` |
| Regression | `regressions/gpw/gpw-metallic-smearing/` |
| Public API | `cpu::gpw::GpwDft` members `smearing`, `degauss` [Ry], `mp_order`, `extra_bands`, `k_mesh`; `GpwResult::{e_total,e_fermi,n_elec_occ,e_entropy}` |
| Roadmap | [dft.md](../../roadmap/dft.md) — Smearing for metals (CPU GPW ✓ multi-k RKS) |
## Summary

Metals need **fractional band occupations** and a chemical potential \(\mu\)
that conserves the valence electron count under multi-\(k\) sampling.
Atomiverse productizes this on the concrete GPW method `cpu::gpw::GpwDft` as
public members (`Smearing::fermi_dirac` / `methfessel_paxton`, width
`degauss` [Ry], `extra_bands`, Monkhorst–Pack `k_mesh`) — **no** public
`*Config` bag and **no** `make_*` factory. Occupations reuse the same
`cpu::pw::internal` fill (Fermi bisection + FD / MP kernels) as plane-wave
metallic smearing. Density and band energy use \(f_b(\varepsilon,\mu)\);
the SCF total is the **Mermin free energy** \(F = E_{\mathrm{KS}} - TS\)
reported as `GpwResult.e_total` [Ha]. Textbook pin: **BCC Na** conventional
cell (2 atoms), GTH + MOLOPT-SR, PADE, Fermi–Dirac `degauss = 0.02` Ry,
\(2\times2\times2\) MP mesh. Regression binary: `reg_gpw-metallic-smearing`.
Offline CP2K 2026.1 documents the Quickstep counterpart
(`&SMEAR METHOD FERMI_DIRAC` + `ADDED_MOS`); absolute \(E\) is an **ATV free-energy
freeze**, not a bit-identical twin (grid / multigrid / `EPS_DEFAULT` differ
by ~0.6 Ha). Fair wall \(r \approx 0.011\) (≤10×).

Related: PW metallic twin [pw-smearing-al](pw-smearing-al.md); narrative GPW
[ch09-gpw](../ch09-gpw.md).

## Method

**Level.** Spin-unpolarized **GPW** Kohn–Sham SCF with metallic electronic
smearing on a Monkhorst–Pack \(k\) mesh. Wavefunction in a contracted
Gaussian AO basis; density collocation on a uniform grid; Hartree via FFT
Poisson; XC on the grid; GTH pseudopotentials. Citation: Lippert, Hutter,
Parrinello, *Mol. Phys.* **92**, 477 (1997).

**Occupations (closed-shell band form).** Spinless \(f(x)\in[0,1]\) is
doubled to band occupation \(f_b\in[0,2]\). With reduced energy
\(x = (\varepsilon-\mu)/\sigma\) and \(\sigma =\) `degauss` converted to
Hartree inside the SCF:

| Kind | Spinless \(f(x)\) | Public enum |
| ---- | ----------------- | ----------- |
| Fermi–Dirac | \(1/(1+e^{x})\) | `Smearing::fermi_dirac` |
| Methfessel–Paxton order \(N\) | MP series of order `mp_order` | `Smearing::methfessel_paxton` |

**Electron count.** Shared \(\mu\) over all \(k\) (Fermi bisection) so

\[
n_{\mathrm{elec}}
  = \sum_{k,b} w_k\, f_b(\varepsilon_{kb},\mu,\sigma)
  = Z_{\mathrm{val}}\cdot N_{\mathrm{atoms}}
\]

(for this pin: \(Z_{\mathrm{val}}=9\) per Na → \(N_e = 18\)). `extra_bands`
mirrors CP2K `ADDED_MOS` / PW `extra_bands` (API parity; GPW already
diagonalizes the full AO spectrum).

**Free energy.** With entropy weight \(s\) for the chosen smear (FD:
\(s = -2[f\ln f+(1-f)\ln(1-f)]\) in band form),

\[
F = E_{\mathrm{KS}} - TS,\qquad
E_{\mathrm{entropy}} = -TS = -\sigma\sum_{k,b} w_k\, s_{kb}.
\]

`GpwResult.e_total` [Ha] is this free energy when smearing is on;
`e_entropy` reports the \(-TS\) piece; `e_fermi` is \(\mu\) [Ha];
`n_elec_occ` is \(\sum w f\).

**Default path.** `Smearing::none` keeps fixed integer fillings
(`n_occ` double-occupied) — prior insulator/molecule GPW pins unchanged.

**Product constraints (build-time).** Metallic smear is the **multi-k
closed-shell RKS** path only. `build()` throws if:

- UKS is configured (`n_alpha` / `n_beta` set), or
- OT is on (`use_ot`), or
- `degauss <= 0` when `smearing != none`, or
- Methfessel–Paxton with `mp_order < 1`.

**Excluded (this slug).** GPW UKS + smear; OT + smear; cold /
Marzari–Vanderbilt; Gaussian smear enum on GPW (PW has it);
forces/stress under smear as acceptance pins; CUDA GPW; Frame /
`GPWEvaluator` wrapper path for this regression (uses `GpwDft` +
`GpwAtom` directly).

## Implementation

Public surface only: construct `cpu::gpw::GpwDft`, set members, `build` /
`run` on a `std::vector<GpwAtom>` (positions [Bohr]). Headers:
`include/atomiverse/cpu/gpw/gpw_solver.h` (API + `Smearing` + `GpwResult`);
occupation kernels live in `cpu/planewave/internal/smearing.h` (not a user
type). Lookup constructors only: `GaussianBasis(Element, name)`,
`GthPotential(Element, name)`.

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

#include <vector>

using namespace ATV;
using cpu::gpw::GpwDft;
using cpu::gpw::Smearing;
using cpu::gpw::XcKind;

// BCC Na conventional cubic, a = 4.225 Å → Bohr (REFERENCE.md)
constexpr double kA2B = 1.0 / 0.52917720859;
const double a = 4.225 * kA2B;

const cpu::gpw::GaussianBasis bNa(molrs::Element::Na, "DZVP-MOLOPT-SR-GTH");
const cpu::gpw::GthPotential pNa(molrs::Element::Na, "GTH-PBE-q9");
const double frac[2][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
std::vector<cpu::gpw::GpwAtom> atoms;
for (int i = 0; i < 2; ++i) {
    const double pos[3] = {frac[i][0] * a, frac[i][1] * a, frac[i][2] * a};
    atoms.emplace_back(pos, bNa, pNa, 11);
}

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = a;
m.cutoff_ry = 150.0;
m.n_occ = 9;              // Ne/2 seed; fractional fill under smear
m.max_iter = 80;
m.tol = 1e-6;             // Ha
m.eps_default = 1e-7;
m.xc = XcKind::PADE;
m.diis_depth = 6;
m.mix = 0.4;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
m.k_time_reversal = true;
m.smearing = Smearing::fermi_dirac;
m.degauss = 0.02;         // Ry (≈ CP2K ELECTRONIC_TEMPERATURE 3157.75 K)
m.extra_bands = 4;        // CP2K ADDED_MOS parity

m.build(atoms);
const cpu::gpw::GpwResult& r = m.run(atoms);
// r.e_total [Ha] free energy F; r.e_fermi [Ha]; r.n_elec_occ ≈ 18;
// r.e_entropy [Ha] (−TS); r.converged, r.iters, r.n_kpts > 1
```

`GpwResult` fields used by the gate: `e_total`, `e_fermi`, `n_elec_occ`,
`e_entropy`, `converged`, `iters`, multi-\(k\) payload (`n_kpts` / `k_frac`).

```bash
# configure (login shell + Lmod — see AGENTS.md)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse test_cpu_gpw_smearing \
        reg_gpw-metallic-smearing
  ctest --test-dir build -R "^(test_cpu_gpw_smearing)$" --output-on-failure
  ./build/regressions/reg_gpw-metallic-smearing
'
```

Source consumer:
`regressions/gpw/gpw-metallic-smearing/reg_gpw-metallic-smearing.cpp`.
Unit tests (API + occupation closed forms, not e2e SCF):
`tests/cpu/gpw/test_cpu_gpw_smearing.cpp` → `test_cpu_gpw_smearing`.

| Piece | Path |
| ----- | ---- |
| Method / result / enum | `include/atomiverse/cpu/gpw/gpw_solver.h` |
| SCF + Mermin free energy | `src/cpu/gpw/gpw_solver.cpp` |
| Shared occupations | `include|src/.../planewave/internal/smearing.*` |
| Unit tests | `tests/cpu/gpw/test_cpu_gpw_smearing.cpp` |
| Regression | `regressions/gpw/gpw-metallic-smearing/` |
| CP2K twin input | `tools/cp2k_oracle/inputs/gpw-metallic-smearing/na_bcc_fermi_dirac.inp` |

## Validation (CP2K)

Source:
[`regressions/gpw/gpw-metallic-smearing/REFERENCE.md`](../../../regressions/gpw/gpw-metallic-smearing/REFERENCE.md)
(offline pins only; **never** live CP2K in ctest).

### ATV system (free energy + \(N_{\mathrm{occ}}\) gate)

| Setting | Value |
| ------- | ----- |
| Geometry | BCC Na conventional cubic, \(a=4.225\) Å, 2 atoms at (0,0,0) and \((\tfrac12,\tfrac12,\tfrac12)\) |
| Basis | `DZVP-MOLOPT-SR-GTH` (in-tree table) |
| PP | `GTH-PBE-q9` (\(Z_{\mathrm{val}}=9\) → \(N_e=18\)) |
| XC | PADE (LDA) |
| Density cutoff | 150 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\), time-reversal fold |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| Extra bands | 4 (CP2K `ADDED_MOS` parity) |
| SCF | DIIS 6, `tol = 1e-6` Ha, `eps_default = 1e-7`, `mix = 0.4` |
| Pin date | 2026-07-26 (Arrhenius x86_64) |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| Free energy \(F\) | **−92.126330608378 Ha** | **\(10^{-4}\) Ha** | frozen ATV offline pin (primary \(E\) gate) |
| \(N_{\mathrm{occ}}=\sum w f\) | **18.0** | \(10^{-5}\) | valence count |
| \(E_F\) | **0.055826 Ha** (finite) | \(\lvert E_F\rvert < 10^3\) Ha | among bands; not NaN |
| Entropy term \(-\sigma S\) | **−2.26×10⁻⁴ Ha** | finite | diagnostic |
| SCF | converged (~7 iters) | `r.converged` | multi-\(k\): \(n_{\mathrm{kpts}}>1\) |

Occupation formulas (unit tests): Fermi–Dirac fill recovers \(N_e\) within
\(10^{-8}\) on a synthetic spectrum; public members carry no `*Config` bag
(Methfessel–Paxton PRB **40**, 3616 (1989); Mermin *Phys. Rev.* **137**,
A1441).

### Offline CP2K twin (algorithm / SMEAR documentation)

**Not an absolute-energy twin** (ATV single-grid / DIIS vs CP2K multigrid
`NGRIDS 4` / `REL_CUTOFF 40` / Broyden). Documents that CP2K
`&SMEAR METHOD FERMI_DIRAC` + `ADDED_MOS` is the product counterpart of ATV
metallic occupations on the **same** system class (GTH + MOLOPT-SR, PADE,
BCC Na, \(2^3\) MP).

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, source 85b8a9b49c) |
| Input | `tools/cp2k_oracle/inputs/gpw-metallic-smearing/na_bcc_fermi_dirac.inp` |
| Method | Quickstep GPW, PADE, `GTH-PBE-q9` + `DZVP-MOLOPT-SR-GTH` |
| Cell | BCC conventional \(a=4.225\) Å, 2 Na |
| SMEAR | FERMI_DIRAC, \(T_e = 3157.75\) K (\(\approx 0.02\) Ry \(k_B T\)) |
| k-points | Monkhorst–Pack \(2\times2\times2\) |
| CUTOFF / REL_CUTOFF | 150 / 40 Ry, NGRIDS 4 |
| Example \(E_{\mathrm{tot}}\) | **−92.76593904 Ha** (8 SCF steps; **not** the ATV pin) |
| Fermi energy (CP2K) | **0.054489 Ha** |
| Al algorithm twin (PW era) | `tools/cp2k_oracle/inputs/al_fermi_dirac.inp` |

Offline re-run (operator only; not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/gpw-metallic-smearing/na_bcc_fermi_dirac.inp
mpirun -np 1 cp2k.psmp -i na_bcc_fermi_dirac.inp -o na_bcc_fermi_dirac.out
```

Oracle path: `tools/cp2k_oracle/README.md` (offline only).

## Performance

Source: same REFERENCE (single-rank wall, 2026-07-26, `OMP_NUM_THREADS=4`).
Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) unless exempted.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| `na_bcc_fermi_dirac.inp` vs `reg_gpw-metallic-smearing` | **89.51** | **0.999** | **~0.011** | ✓ | same system/basis/PP class; ATV in-tree tables + own grid; wall indicative for dual gate |

\[
r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 0.011 \ll 10
\]

`within_10x = true`. Fairness note: do **not** treat \(r\) as a
same-multigrid throughput claim — CP2K multigrid + Broyden vs ATV DIIS /
single density pathway. Handoff records ratio **~0.011**
(the case `REFERENCE.md`).

## Limitations

- **Spin-unpolarized multi-k RKS only.** UKS + metallic smear is not a
  product path on GPW (`build` rejects). Bulk Fe UKS+smear is the **PW**
  pin `pw-uks-fe-smearing`.
- **Absolute free energy is ATV-frozen, not CP2K-matched.** ~0.6 Ha gap from
  grid / `EPS_DEFAULT` / multigrid / mixer differences; electron count +
  SMEAR algorithm class are the product twin.
- **No cold / Marzari–Vanderbilt** public enum on GPW; e2e gate is
  Fermi–Dirac (MP accepted at `build` / unit level).
- **OT + smear incompatible** (OT needs integer occupations).
- **Forces / stress under smear** may inherit the occupation path; not an
  acceptance pin for this slug.
- **Regression uses `GpwDft` + `GpwAtom`**, not `GPWEvaluator` /
  `cpu::Frame` single-point — Frame `kEnergy` [eV] is the evaluator
  contract when that path is used; this pin reports Ha free energy on
  `GpwResult`.
- **CUDA GPW** out of scope.
- **Speed.** \(r\approx 0.011\) on the documented dual; no multi-system
  standing CP2K perf suite in ctest.

## References

- Offline pin: [`regressions/gpw/gpw-metallic-smearing/REFERENCE.md`](../../../regressions/gpw/gpw-metallic-smearing/REFERENCE.md)
- Regression: [`reg_gpw-metallic-smearing.cpp`](../../../regressions/gpw/gpw-metallic-smearing/reg_gpw-metallic-smearing.cpp)
- CP2K twin input: [`tools/cp2k_oracle/inputs/gpw-metallic-smearing/`](../../../tools/cp2k_oracle/inputs/gpw-metallic-smearing/)
- Oracle notes: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Handoff ratio **~0.011**: the case `REFERENCE.md`
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md) (Smearing for metals ✓ CPU GPW)
- Public header: `include/atomiverse/cpu/gpw/gpw_solver.h`
- Shared occupations: `include|src/.../planewave/internal/smearing.*`
- Unit tests: `tests/cpu/gpw/test_cpu_gpw_smearing.cpp`
- PW metallic twin: [pw-smearing-al](pw-smearing-al.md)
- Literature: Mermin, *Phys. Rev.* **137**, A1441 (1965); Methfessel &
  Paxton, *Phys. Rev. B* **40**, 3616 (1989); Lippert, Hutter, Parrinello,
  *Mol. Phys.* **92**, 477 (1997)
- Narrative GPW background: [ch09-gpw](../ch09-gpw.md)
