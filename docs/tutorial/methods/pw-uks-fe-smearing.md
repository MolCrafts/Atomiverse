# PW collinear UKS + metallic smearing — BCC Fe / plane-wave

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pw-uks-fe-smearing` |
| Regression | `regressions/pw/pw-uks-fe-smearing/` |
| Public API | `cpu::pw::PW` (`spin_polarized`, `multiplicity`, `smearing`, `degauss` [Ry], `extra_bands`, `k_mesh`) → `PWResult::{energy, energy_ha, magnetization, e_fermi, n_elec_occ, e_entropy}` |
| Track | CP2K capability (harness 2026-07-25) |
| Roadmap | [dft.md](../../roadmap/dft.md) § spin + metallic smearing (CPU PW ✓); queue `pw-uks-fe-smearing` (done) |

## Summary

**Collinear spin-polarized plane-wave Kohn–Sham with metallic Fermi–Dirac
smearing** — the product step that joins restricted metal occupations
(`pw-smearing-al`) and fixed-occupation UKS (`pw-spin-polarized-o2` /
`pw-uks-kmesh-fe`). Concrete method `cpu::pw::PW` with
`spin_polarized = true` **and** `smearing = Smearing::fermi_dirac` on the
same `run`. Shared chemical potential \(\mu\) over spin channels;
occupations \(f_{\sigma kb}\in[0,1]\); magnetization
\(M = N_\uparrow - N_\downarrow\) is **self-consistent** (multiplicity only
seeds the initial \(\rho_\uparrow/\rho_\downarrow\) split). Total energy is
the Mermin free energy \(F = E_{\mathrm{KS}} - TS\). Textbook pin: **BCC Fe**
primitive cell, SG15 ONCV PBE Fe NC-UPF, `degauss = 0.02` Ry, MP
\(2\times2\times2\). Regression binary: `reg_pw-uks-fe-smearing`. **No live
CP2K/QE in ctest** — offline pins only. Absolute \(E\) is **not** a
GTH-GPW ↔ NC-UPF twin (same class as `pw-smearing-al` / `pw-uks-kmesh-fe`).

## Method

**Level.** Collinear spin-DFT on the plane-wave grid (LSD channel as in
`pw-spin-polarized-o2`) with metallic occupations (same smear library as
`pw-smearing-al`):

- Separate \(\alpha/\beta\) KS orbitals and densities \(\rho_\uparrow\),
  \(\rho_\downarrow\) at every \(\mathbf{k}\) of the MP mesh.
- Hartree from total density \(\rho = \rho_\uparrow + \rho_\downarrow\).
- Spin-dependent XC \(V_{\mathrm{xc}}^\sigma(\rho_\uparrow,\rho_\downarrow)\)
  via the existing PW XC path (`XCType::PBE` on this pin).
- Dual Davidson diagonalizations per **k** and spin; k-weights from
  `generate_mp_mesh`.
- One shared \(\mu\) for both spin channels (Fermi bisection on the joint
  spectrum).

**Occupations (spinor convention).** With metallic smearing each KS
spin-orbital carries \(f_{\sigma kb}\in[0,1]\) (not the closed-shell
\([0,2]\) band form of restricted smear). Reduced energy
\(x = (\varepsilon_{\sigma kb}-\mu)/\sigma\) with \(\sigma =\) `degauss`
[Ry]:

| Kind | Spinor \(f(x)\) | Public enum |
| ---- | --------------- | ----------- |
| Fermi–Dirac | \(1/(1+e^{x})\) | `Smearing::fermi_dirac` |
| Gaussian | \(\tfrac12\mathrm{erfc}(x)\) | `Smearing::gaussian` |
| Methfessel–Paxton order \(N\) | MP series of order `mp_order` | `Smearing::methfessel_paxton` |

e2e gate is **Fermi–Dirac** only.

**Electron count and magnetization.** \(\mu\) is chosen so

\[
N_{\mathrm{occ}}
  = \sum_{\sigma,k,b} w_k\, f_{\sigma kb}(\varepsilon_{\sigma kb},\mu,\sigma)
  = Z_{\mathrm{val}}\cdot N_{\mathrm{atoms}}
\]

(for BCC Fe, \(Z_{\mathrm{val}}=16\), \(N_{\mathrm{atoms}}=1\) → \(N_e=16\)).

\[
M = \sum_{k,b} w_k\bigl(f_{\uparrow kb}-f_{\downarrow kb}\bigr).
\]

With `Smearing::none`, fixed integer fillings enforce
\(M = m-1\) from multiplicity \(m\). With metallic smear, **multiplicity
only seeds** the initial density split; \(M\) is free under SCF (ferro
expected for BCC Fe with a ferro seed).

**Free energy.** Entropy weight \(s\) for FD spinors gives

\[
F = E_{\mathrm{KS}} - TS,\qquad
E_{\mathrm{entropy}} = -TS = -\sigma\sum_{\sigma kb} w_k\, s_{\sigma kb}.
\]

`PWResult.energy` [Ry] is this free energy when smearing is on;
`e_entropy` reports the \(-TS\) piece; `e_fermi` is \(\mu\) [Ry].

**Basis / PP / sampling.** Plane-wave cutoff \(E_{\mathrm{cut}} = 12\) Ry
(algorithm pin for ≤60 s wall — **not** production ONCV completeness);
Monkhorst–Pack \(2\times2\times2\) (`k_mesh = 2`); BCC primitive from
cubic \(a = 2.87\) Å; UPF `tests/data/pw/Fe_ONCV_PBE-1.2.upf` (Hamann
ONCVPSP / SG15, PBE, \(Z_{\mathrm{val}}=16\)).

**Observables**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(F\) | Mermin free energy (`PWResult::energy`) | Ry |
| \(E_{\mathrm{Ha}}\) | \(F/2\) (`PWResult::energy_ha`) | Ha |
| \(M\) | \(N_\uparrow - N_\downarrow\) (`PWResult::magnetization`) | e⁻ |
| \(N_{\mathrm{occ}}\) | \(\sum w f\) (`PWResult::n_elec_occ`) | e⁻ |
| \(E_F\) | chemical potential (`PWResult::e_fermi`) | Ry |
| \(E_{\mathrm{ent}}\) | \(-TS\) (`PWResult::e_entropy`) | Ry |
| — | SCF convergence flag / iteration count | — |

**Excluded (this slug).** Noncollinear / SOC; spin-polarized analytic
forces or stress under smear; GPW UKS + SMEAR productization; CUDA PW;
hybrid / meta-GGA XC; cold / Marzari–Vanderbilt smear; production Fe
magnetization (needs higher ecut / denser k-mesh). Restricted smear
without spin remains `pw-smearing-al`; fixed-occupation multi-k UKS
remains `pw-uks-kmesh-fe`.

## Implementation

Public surface only: construct `cpu::pw::PW`, set members (no `*Config`
bag, no `make_*` / `build_*` factory), `build(frame)` → `run(frame)`.
This slug gates the **joint** path `spin_polarized && smearing != none`
that previously threw. Header:
`include/atomiverse/cpu/planewave/pw_solver.h`. Occupations:
`internal/smearing.h` (not a user type).

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

// BCC Fe primitive, a = 2.87 Å, one atom at origin (REFERENCE.md)
cpu::Frame f(1);
f.span(kAtom_type)[0] = 26;       // Z(Fe)
f.span(kAtom_mass)[0] = 55.845;
f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0.0;
const double h = 2.87 / 2.0;
Mat3 cell;
// Primitive BCC vectors: (-h,h,h), (h,-h,h), (h,h,-h)
cell(0, 0) = -h; cell(0, 1) =  h; cell(0, 2) =  h;
cell(1, 0) =  h; cell(1, 1) = -h; cell(1, 2) =  h;
cell(2, 0) =  h; cell(2, 1) =  h; cell(2, 2) = -h;
f.box.set_cell(cell);

PW pw;
pw.upf_path = "tests/data/pw/Fe_ONCV_PBE-1.2.upf";
pw.ecut_ry = 12.0;                 // algorithm pin (not production ecut)
pw.k_mesh = 2;                     // 2×2×2 MP
pw.tol = 2e-3;                     // density L2
pw.max_iter = 30;
pw.mix_beta = 0.3;
pw.pulay_depth = 5;
pw.xc = cpu::scf::XCType::PBE;
pw.spin_polarized = true;
pw.multiplicity = 5;               // seed M≈4; free under smear
pw.smearing = Smearing::fermi_dirac;
pw.degauss = 0.02;                 // Ry
pw.extra_bands = 4;                // per spin

pw.build(f);
const PWResult& r = pw.run(f);
// r.energy [Ry] free energy; r.magnetization; r.e_fermi [Ry];
// r.n_elec_occ ≈ 16; r.e_entropy [Ry]; r.converged, r.iterations
```

`PWEvaluator` can own a moved `PW` for Driver binding; this regression
calls `PW` directly. Forces/stress APIs are **not** part of the Fe-smear
acceptance. Public energy on Frame remains eV when routed through
`PWEvaluator` / `single_point`; `PWResult` reports Ry/Ha internally.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_PW=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_pw-uks-fe-smearing -j$(nproc)
  ./build/regressions/reg_pw-uks-fe-smearing
'
```

**File pointers**

| Role | Path |
| ---- | ---- |
| Public API | `include/atomiverse/cpu/planewave/pw_solver.h` |
| SCF / spin × smear branch | `src/cpu/planewave/pw_solver.cpp` |
| Internal occupations | `include|src/.../planewave/internal/smearing.*` |
| Unit (smear closed forms) | `tests/cpu/planewave/test_pw_smearing.cpp` |
| Regression | `regressions/pw/pw-uks-fe-smearing/reg_pw-uks-fe-smearing.cpp` |
| Offline pin | `regressions/pw/pw-uks-fe-smearing/REFERENCE.md` |
| Oracle note | `tools/cp2k_oracle/inputs/pw-uks-fe-smearing.note` |

## Validation (CP2K)

Source:
[`regressions/pw/pw-uks-fe-smearing/REFERENCE.md`](../../../regressions/pw/pw-uks-fe-smearing/REFERENCE.md)
(offline only; **no live CP2K or QE in ctest**). Closed acceptance
(harness wave 2026-07-25): collinear UKS + FD smear, \(N_{\mathrm{occ}}\),
free-energy pin, ferro \(M\), finite \(E_F\), `perf_exempt`.

| Setting | Value |
| ------- | ----- |
| System | BCC Fe primitive, 1 atom at origin |
| Geometry | cubic lattice constant \(a = 2.87\) Å → primitive cell vectors above |
| PP | SG15 ONCV PBE Fe, `tests/data/pw/Fe_ONCV_PBE-1.2.upf` (\(Z_{\mathrm{val}}=16\)) |
| XC | PBE (`XCType::PBE`) |
| \(E_{\mathrm{cut}}\) | 12 Ry |
| **k**-mesh | \(2\times 2\times 2\) (`k_mesh = 2`) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| Extra bands | 4 (per spin) |
| Multiplicity seed | 5 (initial \(M\approx 4\); free under smear) |
| SCF | density L2 tol \(2\times10^{-3}\), `max_iter=30`, `mix_beta=0.3`, `pulay_depth=5` |
| Pin date | 2026-07-25 (Arrhenius x86_64, OMP=8, single rank, ~11 SCF iters) |
| Twin note | CP2K Quickstep is **GPW/GTH**, not NC-UPF PW; absolute \(F\) is a frozen ATV method pin |

### Electron count

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(N_{\mathrm{occ}}\) | **16** | \(Z_{\mathrm{val}}=16\) (SG15 Fe) | \(1\times10^{-4}\) | \(\sum_{\sigma kb} w f\) |

### Magnetization (ferro seed, self-consistent under smear)

BCC Fe is ferromagnetic; experimental moment is ~2.2 \(\mu_B\). At the
modest pin ecut, the gate is **ferro sign + stability under smear**, not
chemical magnetization accuracy. Recorded SCF value
**\(M \approx 0.991\)** (undershoots experiment — expected at 12 Ry /
\(2^3\) k).

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(M\) [e⁻] | \(\approx 0.991\) (gate: \(> 0.1\)) | ferro seed / literature ferro BCC Fe | \(M > 0.1\) | `PWResult::magnetization` |

### Free energy (offline ATV freeze)

| Quantity | Atomiverse / pin | Twin | Tol | Notes |
| -------- | ---------------: | ---- | --- | ----- |
| \(F\) [Ry] | **−205.8365363098** | offline method pin (same PP/ecut/cell/geom/k/smear) | **0.05 Ry** | Mermin free energy; ~11 SCF iters |
| \(E_F\) [Ry] | finite, \(\approx 2.16\) | — | \(\lvert E_F\rvert < 10^3\) | among bands; not NaN |
| SCF | must converge | — | — | regression fails if `!converged` |

Absolute PW free energies shift with cutoff, k-mesh, and PP family; the
regression gate is **stability under the frozen method string**, not a
literature chemical-accuracy claim.

### Offline CP2K twin (algorithm / SMEAR documentation)

**Not an absolute-energy twin** (GTH-GPW ≠ NC-UPF PW). Documents that
CP2K Quickstep UKS + `&SMEAR METHOD FERMI_DIRAC` is the product
counterpart of ATV collinear metallic occupations.

| Item | Value |
| ---- | ----- |
| Note | `tools/cp2k_oracle/inputs/pw-uks-fe-smearing.note` |
| Method class | Quickstep UKS / LSD + `FERMI_DIRAC` + Fe GTH (if re-run offline) |
| SMEAR width | \(T_e \approx 3158\) K matching `degauss ≈ 0.02` Ry |
| Host CP2K | not loadable as fair NC-UPF PW twin at pin time |
| Example energy | **not** compared to ATV Ry pin |

Oracle path: `tools/cp2k_oracle/README.md` (offline only).

## Performance

Source: `REFERENCE.md` § Perf; queue/handoff mark this slug **perf_exempt**.

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | no fair single-rank NC-UPF PW twin wall vs CP2K GTH-GPW; same class as `pw-uks-kmesh-fe` |
| \(T_{\mathrm{ATV}}\) (this host, OMP=8, single rank) | **~20 s** |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a** |

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| BCC Fe PW UKS+FD (NC-UPF) | n/a | ~20 | **n/a** | exempt | No fair CP2K/QE wall twin at pin host |

Dual gate: (1) converge + \(N_{\mathrm{occ}}\) + free-energy pin +
\(M>0.1\) + finite \(E_F\); (2) perf exempt with reason above. Queue
row: ratio **perf_exempt** (2026-07-25 harness). Wall budget: e2e ≤ 60 s
single-rank CPU.

## Limitations

- **Algorithm ecut (12 Ry), not production Fe completeness.** Absolute
  \(F\) and \(M\) are not chemically converged; recorded \(M\approx 1\)
  undershoots experimental BCC Fe (~2.2 \(\mu_B\)). Higher ecut / denser
  k-mesh is a re-pin exercise, not this slug’s gate.
- **Absolute energy is ATV-frozen, not CP2K-matched.** GTH-GPW and NC-UPF
  PW are different Hamiltonians; CP2K documents UKS + SMEAR settings /
  algorithm only.
- **Perf exempt** — no fair NC-UPF twin wall on the pin host.
- **No spin-polarized forces/stress under smear** in acceptance.
- **No cold / Marzari–Vanderbilt** public enum; Gaussian and MP are
  unit-tested; e2e gate is Fermi–Dirac.
- **GPW UKS + SMEAR** not productized here (PW-only path).
- **CUDA PW** out of scope.
- **Out of scope forever for this page:** noncollinear spin, SOC, hybrid
  PW, restricted-only smear (`pw-smearing-al`), fixed-occupation multi-k
  UKS without smear (`pw-uks-kmesh-fe`).

## References

- Offline pin: [`regressions/pw/pw-uks-fe-smearing/REFERENCE.md`](../../../regressions/pw/pw-uks-fe-smearing/REFERENCE.md)
- Regression consumer: `regressions/pw/pw-uks-fe-smearing/reg_pw-uks-fe-smearing.cpp`
- Oracle note: `tools/cp2k_oracle/inputs/pw-uks-fe-smearing.note`
- Roadmap: `docs/roadmap/dft.md` (bulk Fe UKS+smear ✓ CPU)
- Public header: `include/atomiverse/cpu/planewave/pw_solver.h`
- Internal occupations: `include|src/.../planewave/internal/smearing.*`
- Unit tests: `tests/cpu/planewave/test_pw_smearing.cpp`
- Spec history (closed 2026-07-25): git (`.claude/specs/pw-uks-fe-smearing/` deleted at done)
- Literature: Mermin, *Phys. Rev.* **137**, A1441 (1965); Methfessel &
  Paxton, *Phys. Rev. B* **40**, 3616 (1989); Hamann ONCVPSP / SG15 NC-UPF
- Related restricted smear: [`pw-smearing-al`](pw-smearing-al.md)
- Related Γ UKS PW: [`pw-spin-polarized-o2`](pw-spin-polarized-o2.md)
- Related multi-k fixed-occ UKS: [`pw-uks-kmesh-fe`](pw-uks-kmesh-fe.md)
- Narrative PW background: [ch08-plane-wave.md](../ch08-plane-wave.md)
