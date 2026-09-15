# Metallic Fermi–Dirac smearing — FCC Al / plane-wave

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pw-smearing-al` |
| Regression | `regressions/pw/pw-smearing-al/` |
| Public API | `cpu::pw::PW` members `smearing`, `degauss` [Ry], `mp_order`, `extra_bands`; `PWResult::{e_fermi,n_elec_occ,e_entropy}` |
| Roadmap | [dft.md](../../roadmap/dft.md) — Smearing for metals (CPU PW ✓) |
## Summary

Plane-wave DFT for **metals** needs **fractional band occupations** and a
Fermi level \(\mu\) that conserves the valence electron count. Atomiverse
exposes this on the concrete method `cpu::pw::PW` as public members
(`Smearing::fermi_dirac` / `methfessel_paxton` / `gaussian`, width
`degauss` [Ry], `extra_bands`) — **no** public `*Config` bag and **no**
`make_*` factory. Density and band energy use \(f_b(\varepsilon,\mu)\);
the SCF total is the **Mermin free energy** \(F = E_{\mathrm{KS}} - TS\).
Textbook pin: **FCC Al** primitive cell, NC UPF, Fermi–Dirac
`degauss = 0.02` Ry. Regression binary: `reg_pw-smearing-al`. Offline
CP2K 2026.1 documents the product counterpart (`&SMEAR METHOD FERMI_DIRAC`);
absolute \(E\) is **not** a GTH-GPW ↔ NC-UPF twin — the energy gate freezes
ATV free energy on identical ATV settings.

## Method

**Level.** Spin-unpolarized plane-wave Kohn–Sham SCF with metallic electronic
smearing. Pseudopotential path is **norm-conserving UPF** (QE-style);
reciprocal-space Poisson / FFT XC / Davidson as in
[ch08-plane-wave](../ch08-plane-wave.md).

**Occupations (closed-shell convention).** Spinless \(f(x)\in[0,1]\) is
doubled to band occupation \(f_b\in[0,2]\). With reduced energy
\(x = (\varepsilon-\mu)/\sigma\) and \(\sigma =\) `degauss` (same unit as
eigenvalues inside the SCF):

| Kind | Spinless \(f(x)\) | Public enum |
| ---- | ----------------- | ----------- |
| Fermi–Dirac | \(1/(1+e^{x})\) | `Smearing::fermi_dirac` |
| Gaussian | \(\tfrac12\mathrm{erfc}(x)\) | `Smearing::gaussian` |
| Methfessel–Paxton order \(N\) | MP series of order `mp_order` | `Smearing::methfessel_paxton` |

**Electron count.** \(\mu\) is chosen (Fermi bisection) so

\[
n_{\mathrm{elec}}
  = \sum_{k,b} w_k\, f_b(\varepsilon_{kb},\mu,\sigma)
  = Z_{\mathrm{val}}\cdot N_{\mathrm{atoms}}
\]

(for FCC Al, \(N_e = 3\)). Extra empty bands above \(\lceil N_e/2\rceil\)
come from `extra_bands` (Davidson needs them near \(E_F\)).

**Free energy.** With entropy weight \(s\) for the chosen smear (FD:
\(s = -2[f\ln f+(1-f)\ln(1-f)]\) in band form),

\[
F = E_{\mathrm{KS}} - TS,\qquad
E_{\mathrm{entropy}} = -TS = -\sigma\sum_{k,b} w_k\, s_{kb}.
\]

`PWResult.energy` [Ry] is this free energy when smearing is on;
`e_entropy` reports the \(-TS\) piece; `e_fermi` is \(\mu\) [Ry].

**Default path.** `Smearing::none` keeps fixed integer occupations
(\(n_{\mathrm{bands}}=\max(N_e/2,1)\)) — prior Si/H PW pins unchanged.

**Excluded (this slug).** Spin-polarized metallic UKS (`pw-uks-kmesh-fe`,
`pw-spin-polarized-o2`); cold / Marzari–Vanderbilt smearing; GPW `&SMEAR`
productization; CUDA PW smearing; forces under smear as a pin.

## Implementation

Public surface only: construct `cpu::pw::PW`, set members, `build` / `run`
on a `cpu::Frame`. Headers:
`include/atomiverse/cpu/planewave/pw_solver.h` (API);
occupations live in `internal/smearing.h` (not a user type).

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

// FCC Al primitive, a = 4.05 Å, one atom at origin (REFERENCE.md)
cpu::Frame f(1);
f.span(kAtom_type)[0] = 13;
f.span(kAtom_mass)[0] = 26.9815385;
f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0.0;
const double h = 4.05 / 2.0;
Mat3 cell;
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
f.box.set_cell(cell);

PW pw;
pw.upf_path = "tests/data/pw/Al_pz_nc.upf"; // NC, LDA/PZ, Z_val = 3
pw.ecut_ry = 20.0;
pw.k_mesh = 2;
pw.tol = 1e-5;
pw.max_iter = 60;
pw.mix_beta = 0.4;
pw.pulay_depth = 5;
pw.smearing = Smearing::fermi_dirac;
pw.degauss = 0.02;   // Ry
pw.extra_bands = 4;  // → 6 bands for N_e = 3

pw.build(f);
const PWResult& r = pw.run(f);
// r.energy [Ry] free energy; r.e_fermi [Ry]; r.n_elec_occ ≈ 3;
// r.e_entropy [Ry] (−TS); r.converged, r.iterations
```

`PWResult` fields used by the gate: `energy`, `e_fermi`, `n_elec_occ`,
`e_entropy`, `converged`, `iterations`. A pseudopotential is not optional:
leaving both `upf_path` and `gth_name` empty throws at `build()` — the
bare-Coulomb "toy" fallback that once ran here has been deleted.

```bash
# configure (login shell + Lmod — see AGENTS.md)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_pw-smearing-al -j$(nproc)
  ./build/regressions/reg_pw-smearing-al
'
```

Source consumer:
`regressions/pw/pw-smearing-al/reg_pw-smearing-al.cpp`.
Unit tests (occupation closed forms, not e2e):
`tests/cpu/planewave/test_pw_smearing.cpp` → `test_cpu_pw_smearing`.

## Validation (CP2K)

Source:
[`regressions/pw/pw-smearing-al/REFERENCE.md`](../../../regressions/pw/pw-smearing-al/REFERENCE.md)
(offline pins only; **never** live CP2K in ctest).

### ATV system (energy + \(N_{\mathrm{occ}}\) gate)

| Setting | Value |
| ------- | ----- |
| Geometry | FCC Al primitive, \(a=4.05\) Å, 1 atom at origin |
| PP | `tests/data/pw/Al_pz_nc.upf` (NC, LDA/PZ, \(Z_{\mathrm{val}}=3\)) |
| \(E_{\mathrm{cut}}\) | 20 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| Extra bands | 4 (\(\Rightarrow\) 6 bands for \(N_e=3\)) |
| SCF | `tol=1e-5`, `max_iter=60`, `mix_beta=0.4` |
| Pin date | 2026-07-23 (Arrhenius x86_64) |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| Free energy \(F\) | **−7.54535571 Ry** | **0.05 Ry** | frozen ATV offline pin (primary \(E\) gate) |
| \(N_{\mathrm{occ}}=\sum w f\) | **3.0** | \(10^{-5}\) | valence count |
| \(E_F\) | finite, \(\lvert E_F\rvert < 10^3\) Ry | — | among bands; not NaN |

Occupation formulas (unit tests): Fermi–Dirac / Gaussian / MP-1 \(f(x)\) vs
closed form to \(10^{-12}\); Fermi search recovers \(N_e\) within \(10^{-8}\)
on a synthetic spectrum (Methfessel–Paxton PRB **40**, 3616 (1989); Mermin
*Phys. Rev.* **137**, A1441).

### Offline CP2K twin (algorithm / SMEAR documentation)

**Not an absolute-energy twin** (GTH-GPW ≠ NC-UPF PW). Documents that CP2K
`&SMEAR METHOD FERMI_DIRAC` + `ADDED_MOS` is the product counterpart of ATV
metallic occupations.

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Input | `tools/cp2k_oracle/inputs/al_fermi_dirac.inp` |
| Method | Quickstep GPW, PADE, `GTH-PADE-q3` + `DZVP-MOLOPT-SR-GTH` |
| Cell | same primitive FCC \(a=4.05\) Å |
| SMEAR | FERMI_DIRAC, \(T_e = 3157.75\) K (\(\approx 0.02\) Ry \(k_B T\)) |
| k-points | Monkhorst–Pack \(2\times2\times2\) |
| Example \(E_{\mathrm{tot}}\) | ≈ −2.096 Ha (grid-dependent; **not** compared to ATV Ry pin) |

Oracle path: `tools/cp2k_oracle/README.md` (offline only).

## Performance

Source: same REFERENCE (single-rank wall, 2026-07-23). Gate:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) unless exempted.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| `al_fermi_dirac.inp` vs `reg_pw-smearing-al` | 211 | **1.70** | **0.008** | ✓ | different basis/PP/grid; wall indicative for dual gate only |

\[
r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 0.008 \ll 10
\]

`within_10x = true`. Fairness note: ATV NC-UPF PW at 20 Ry / \(2^3\) k vs
CP2K GPW GTH + MOLOPT at CUTOFF 200 Ry; do **not** treat \(r\) as a
same-Hamiltonian throughput claim. Handoff records ratio **0.008**
(the case `REFERENCE.md`).

## Limitations

- **Spin-unpolarized only on this pin.** Collinear spin is a separate
  product (`pw-spin-polarized-o2`); metallic k-point UKS is queue-pending
  (`pw-uks-kmesh-fe`).
- **Absolute energy is ATV-frozen, not CP2K-matched.** GTH-GPW and NC-UPF PW
  are different Hamiltonians; CP2K documents SMEAR settings / algorithm
  only.
- **No cold / Marzari–Vanderbilt** public enum; Gaussian and MP are
  unit-tested, e2e gate is Fermi–Dirac.
- **GPW SMEAR** not productized here (PW-only).
- **CUDA PW smearing** out of scope.
- **Forces under smear** may inherit the occupation path; not an acceptance
  pin for this slug.
- **No pseudopotential-free path.** Empty `upf_path` *and* empty `gth_name`
  throws at `build()`; the old bare-Coulomb "toy" fallback is gone.
- **Speed.** \(r\approx 0.008\) on the documented dual; no multi-system
  standing CP2K perf suite in ctest.

## References

- Offline pin: `regressions/pw/pw-smearing-al/REFERENCE.md`
- Regression: `regressions/pw/pw-smearing-al/reg_pw-smearing-al.cpp`
- CP2K twin input: `tools/cp2k_oracle/inputs/al_fermi_dirac.inp`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Handoff ratio **0.008**: the case `REFERENCE.md`
- Roadmap: `docs/roadmap/dft.md` (Smearing for metals ✓ CPU)
- Public header: `include/atomiverse/cpu/planewave/pw_solver.h`
- Internal occupations: `include|src/.../planewave/internal/smearing.*`
- Unit tests: `tests/cpu/planewave/test_pw_smearing.cpp`
- Spec history (closed wave 1): git `72ebb870` (`.claude/specs/pw-smearing-al/`)
- Literature: Mermin, *Phys. Rev.* **137**, A1441 (1965); Methfessel &
  Paxton, *Phys. Rev. B* **40**, 3616 (1989)
- Narrative PW background: [ch08-plane-wave](../ch08-plane-wave.md)
