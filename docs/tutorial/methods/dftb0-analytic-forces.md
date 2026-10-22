# Non-SCC DFTB0 analytic nuclear forces — H₂, H₂O, CO (mio-1-1)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dftb0-analytic-forces` |
| Public API | `cpu::semiempirical::DFTB0` (`with_forces = true`, default); `cpu::Evaluate` |
| Roadmap | `docs/roadmap/semiempirical.md` §2 DFTB0 ✓ (analytic F; no SCC γ) |
| Oracle | **DFTB+ 24.1** only (`SCC = No`, mio-1-1). Never OpenMOPAC, never xtb |

```text
oracle: dftb+
oracle_version: 24.1
```

## Summary

Closed-shell **non-SCC DFTB / DFTB0** (Elstner band energy plus pair
repulsive) on H/C/N/O with static in-tree **mio-1-1** Slater–Koster tables.
Forces are the closed-form nuclear gradient of that energy at the non-SCC
density — Hellmann–Feynman on \(\partial H^0\), Pulay energy-weighted density
on \(\partial S\), and the pair-repulsive derivative — written onto
`frame.span(kAtom_f*)` in eV/Å. There is no Hubbard \(\gamma\), no
\(\Delta q\), and no SCC shift on the overlap. This is **not** a \(6N\)
finite difference. Public surface: concrete method
`cpu::semiempirical::DFTB0` (member `with_forces`; no public `*Config`, no
`make_*` / `build_*`). Product gate: `reg_dftb0-analytic-forces`. Oracle
family: **DFTB+** 24.1, offline transcripts only — never live in ctest.
Fair wall \(r = T_{\mathrm{ATV}}/T_{\mathrm{DFTB+}} = 0.45\) (cold ATV
launch vs warm DFTB+), within 10×. Not `perf_exempt`.

## Method

**Non-SCC energy** (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998), the
zeroth-order band + repulsive piece, before the second-order charge
fluctuation):

\[
E = \sum_i 2\varepsilon_i + E_{\mathrm{rep}}.
\]

Parameter path: static SK tables → \(H^0\), \(S\) → generalized eigenproblem
→ occupied density and energy-weighted density → band + repulsive. In ATV
this is `ParameterSet → SK → H/S → eigen → P → repulsive → E/F` with **no**
SCC(Hubbard, \(\gamma\)) stage. Tables are transcribed DFTB.org mio-1-1 for
every ordered H/C/N/O pair (not an exponential sketch). Unsupported \(Z\)
throws.

**Ionic forces (closed form)** at that density. The Hamiltonian is pure
\(H^0\), so

\[
\frac{\partial E}{\partial R}
  = \operatorname{Tr}\!\bigl(P\,\partial H^0\bigr)
  - \operatorname{Tr}\!\bigl(W\,\partial S\bigr)
  + \partial E_{\mathrm{rep}},
\qquad
\mathbf{F} = -\nabla E,
\]

with closed-shell occupation already inside
\(P = 2CC^{\mathsf T}\) and
\(W = 2C\,\mathrm{diag}(\varepsilon)\,C^{\mathsf T}\). Pairwise, only the
A→B SK block is built; the B→A block is its transpose at \(-R\), and the
two electronic contributions are equal (factor 2). The repulsive term is
the radial derivative of the mio-1-1 pair spline times the bond direction.
Net \(\sum_A\mathbf{F}_A = 0\) by antisymmetry of each pair. Force-module
boundary: Hourahine et al., *J. Chem. Phys.* **152**, 124101 (2020).

The SK **radial** derivative that feeds \(\partial H^0\) and \(\partial S\)
is still a finite difference of the hopping interpolant
(\(h = 10^{-5}\) Å). This page does not replace that interpolant derivative
with a closed radial form.

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| Method | Non-SCC DFTB0, closed shell, integer occupation from valence count |
| SK | transcribed mio-1-1, all ordered H/C/N/O pairs |
| DFTB+ | 24.1, `SCC = No`, `PrintForces = Yes`, mio-1-1 |
| Cell | Cubic **30 Å** isolation for the four oracle geometries |
| Units | \(E\) [eV], \(\mathbf{R}\) [Å], \(\mathbf{F}\) [eV/Å] |
| Ha → eV | \(27.211386245988\) (`kHartreeInEv`) |
| Ha/Bohr → eV/Å | \(51.42206747467393\) (`kHartreePerBohrInEvPerAngstrom`) |
| Forces | `with_forces = true` (the default) → closed-form path |

**Geometries [Å]** (the same coordinates as the offline DFTB+ inputs):

| Case | Atoms \((Z, x, y, z)\) |
| ---- | ---------------------- |
| H₂, \(R = 0.74\) Å | H (0, 0, 0), H (0.74, 0, 0) |
| H₂, \(R = 0.90\) Å | H (0, 0, 0), H (0.90, 0, 0) |
| Off-eq H₂O | O (0, 0, 0), H (1.05, 0, 0), H (−0.26, 1.00, 0) |
| Tilted CO | C (0, 0, 0), O (1.20, 0.15, 0) |

Masses used only as Frame columns (not in the Hamiltonian): H 1.008, C 12.011,
O 15.999. A fifth geometry, H atoms at 0.20 Å and 7.60 Å inside an **8 Å**
cube, is an ATV minimum-image identity only. DFTB+ `C` geometry does not
wrap, so that case gates \(F = -\nabla E\) and \(\sum F = 0\), not an
absolute oracle number.

**Key exclusions:**

- Hubbard \(\gamma\), \(\Delta q\), SCC shift on \(S\) — that is DFTB2, a
  different Hamiltonian. Heteronuclear water and CO **must** be pinned with
  `SCC = No`. An SCC transcript is not a DFTB0 number. Homonuclear H₂ is the
  one case where \(\Delta q = 0\) by symmetry; the pins below are still from
  a fresh non-SCC run, not from that reduction.
- Open shell, non-integer occupation, elements outside H/C/N/O — throw.
- Cell stress, k-mesh, lattice images beyond the single minimum image the
  energy already uses.
- OpenMOPAC as a DFTB reference, or xtb / GFN as a DFTB0 or DFTB3
  successor — family lock forbids both.
- Live DFTB+ from ctest — forbidden. Offline pins only.

## Implementation

Public surface only (`cpu::Frame`, `cpu::semiempirical::DFTB0` members; no
public `*Config`, no `make_*` / `build_*`). Energy and forces **accumulate**
(`+=`), so the caller zeros `kEnergy` and `kAtom_f*` before `run`. The
snippet is off-equilibrium water; H₂ and CO are the same call with the
coordinates in the table above.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb0.h"

using namespace ATV;

cpu::Frame frame(3);
frame.span(kAtom_type)[0] = 8;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_type)[1] = 1;
frame.span(kAtom_x)[1] = 1.05;
frame.span(kAtom_y)[1] = 0.0;
frame.span(kAtom_z)[1] = 0.0;
frame.span(kAtom_type)[2] = 1;
frame.span(kAtom_x)[2] = -0.26;
frame.span(kAtom_y)[2] = 1.0;
frame.span(kAtom_z)[2] = 0.0;
for (int i = 0; i < 3; ++i) {
    frame.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
    frame.span(kAtom_tag)[i] = i + 1;
    frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
        frame.span(kAtom_fz)[i] = 0;
}
frame.box = cpu::Box::cubic(30.0);
frame.meta.get(kEnergy) = 0;

cpu::semiempirical::DFTB0 dftb0;
dftb0.with_forces = true;  // default; closed-form, not a re-solve FD

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add(std::move(dftb0));
eval.build();
eval.run(frame);

// Energy: frame.meta.get(kEnergy) [eV]
// Forces: kAtom_fx / fy / fz [eV/Å]
```

SE-only build and regression (never bare `cmake --build build`):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_EVALUATE=ON \
        -DATV_BUILD_MD=OFF -DATV_BUILD_MC=OFF -DATV_BUILD_MINIMIZER=OFF \
        -DATV_BUILD_VIB=OFF -DATV_BUILD_PATH=OFF -DATV_BUILD_POTENTIALS=OFF \
        -DATV_BUILD_NEIGHBOR=OFF -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF \
        -DATV_BUILD_GPW=OFF -DATV_BUILD_KSPACE=OFF -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_dftb0-analytic-forces
  cmake --build build --target run_reg_dftb0-analytic-forces
'
```

| Path | Role |
| ---- | ---- |
| `dftb0.h` | Public `DFTB0` (`with_forces`, `last_energy`) |
| `dftb_mio_sk_tables.h` | Static mio-1-1 SK for H/C/N/O |
| `src/cpu/semiempirical/dftb0.cpp` | Non-SCC solve + analytic \(F\) |
| `reg_dftb0-analytic-forces.cpp` | Product gate |
| case `dftb0-analytic-forces` pin | Offline DFTB+ 24.1 transcripts |

## Validation (DFTB+)

Source: case `dftb0-analytic-forces` pin. Oracle: **`dftb+`** 24.1,
`SCC = No`, mio-1-1, `PrintForces = Yes`. Nothing is executed at ctest
time. Hard-coded gates in the regression binary.

DFTB+ prints total energy in Hartree and forces in Hartree/Bohr
(`Maximal derivative component` / force block, atomic units). The
regression multiplies by the two constants above and compares on the
public eV / eV/Å surface. The eV figures below are that product, not a
fit to an ATV residual.

**H₂, \(R = 0.74\) Å.** Transcript total energy **−0.6749509324 Ha**
(public pin **−18.3663505186 eV**). The sibling published figure
**−18.36635052 eV** must not move (\(\lvert\Delta\rvert \le 10^{-6}\) eV).
\(F_{1x} = -0.001976354621\) Ha/Bohr, \(F_{2x} = +0.001976354621\) Ha/Bohr
(all other components 0).

**H₂, \(R = 0.90\) Å.** Total energy **−0.6630166170 Ha**.
\(F_{1x} = +0.065975227977\) Ha/Bohr, \(F_{2x} = -0.065975227977\) Ha/Bohr.

**Off-equilibrium water.** Total energy **−4.0964033746 Ha**.

| Atom | \(F_x\) [Ha/Bohr] | \(F_y\) [Ha/Bohr] | \(F_z\) |
| ---- | ----------------: | ----------------: | ------: |
| O | 0.039206443065 | 0.036588999897 | 0 |
| H1 | −0.048418613962 | −0.000229747051 | 0 |
| H2 | 0.009212170898 | −0.036359252846 | 0 |

**Tilted CO** (ppσ / ppπ and a transverse component). Total energy
**−5.0270725023 Ha**.

| Atom | \(F_x\) [Ha/Bohr] | \(F_y\) [Ha/Bohr] | \(F_z\) |
| ---- | ----------------: | ----------------: | ------: |
| C | 0.183901804021 | 0.022987724205 | 0 |
| O | −0.183901804021 | −0.022987724205 | 0 |

**Hard-coded gates (public units):**

| Quantity | Atomiverse / pin | DFTB+ 24.1 | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| H₂ 0.74 \(E\) [eV] | −18.3663505186 | −0.6749509324 Ha | \(1\times10^{-6}\) eV | also −18.36635052 published |
| H₂ 0.74 \(F_{1x}\) [eV/Å] | −0.1016282407 | −0.001976354621 Ha/Bohr | \(1\times10^{-5}\) eV/Å | \(F_{2x}=-F_{1x}\) |
| H₂ 0.90 \(E\) [eV] | −18.0416012527 | −0.6630166170 Ha | \(1\times10^{-6}\) eV | stretched ss + repulsive |
| H₂ 0.90 \(F_{1x}\) [eV/Å] | +3.3925826247 | +0.065975227977 Ha/Bohr | \(1\times10^{-5}\) eV/Å | bond pulls inward |
| H₂O \(E\) [eV] | −111.4688144456 | −4.0964033746 Ha | \(1\times10^{-6}\) eV | heteronuclear; `SCC=No` required |
| H₂O \(F_{\mathrm{O}}\) [eV/Å] | (2.0160763607, 1.8814820215, 0) | transcript row above | \(1\times10^{-5}\) eV/Å | each component |
| H₂O \(F_{\mathrm{H1}}\) [eV/Å] | (−2.4897852342, −0.0118140684, 0) | transcript row above | \(1\times10^{-5}\) eV/Å | each component |
| H₂O \(F_{\mathrm{H2}}\) [eV/Å] | (0.4737088735, −1.8696679532, 0) | transcript row above | \(1\times10^{-5}\) eV/Å | each component |
| CO \(E\) [eV] | −136.7936115467 | −5.0270725023 Ha | \(1\times10^{-6}\) eV | |
| CO \(F_{\mathrm{C}}\) [eV/Å] | (9.4566109751, 1.1820763052, 0) | transcript row above | \(1\times10^{-5}\) eV/Å | \(F_{\mathrm{O}}=-F_{\mathrm{C}}\) |
| Analytic \(F\) vs central FD of the same \(E\) | identity | — | \(1\times10^{-5}\) eV/Å | \(h = 10^{-4}\) Å; all five geometries |
| \(\lvert\sum_i F_i\rvert\) [eV/Å] | 0 | 0 | \(1\times10^{-8}\) eV/Å | each geometry |
| Wrapped H₂ \(\lvert F_{1x}\rvert\) | \(> 1\) eV/Å | — | hard | min-image only; no DFTB+ twin |

**Tolerance rationale.** The transcript prints energy to \(10^{-10}\) Ha
(\(\approx 3\times10^{-9}\) eV) and forces to \(\sim10^{-12}\) Ha/Bohr
(\(\approx 5\times10^{-11}\) eV/Å). The \(10^{-6}\) eV / \(10^{-5}\) eV/Å
gates sit above that print precision and above the central-difference
floor of the SK interpolant derivative. They are not fitted to an ATV
residual. A factor-of-two or a sign error in the Pulay contraction is
eV/Å, not \(10^{-5}\).

**Observed (2026-10-09, Release, 1 thread; not a retune of the gates).**
Max \(\lvert\Delta E\rvert\) vs the Ha transcript was \(8.3\times10^{-7}\) eV
(CO). Max \(\lvert\Delta F\rvert\) stayed under the \(1\times10^{-5}\) eV/Å
gate (H₂ 0.74 Å, \(\sim 2\times10^{-6}\) eV/Å). FD-identity max residual
\(1.0\times10^{-6}\) eV/Å (wrapped H₂). H₂ 0.74 Å energy
−18.3663505230 eV, \(\lvert\Delta\rvert = 3\times10^{-9}\) eV from the
published −18.36635052 eV pin.

## Performance

Fair wall, same thread count. DFTB+ 24.1 single point,
`OMP_NUM_THREADS=1`, process wall, five repeats, 2026-10-09. Cold start
of the DFTB+ binary was 0.92 s and is **not** the comparison. ATV is
`Evaluate::run` in-process chrono inside `reg_dftb0-analytic-forces`.

```bash
cmake --build build --target run_reg_dftb0-analytic-forces
```

| Twin | \(T_{\mathrm{DFTB+}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| H₂ \(R=0.90\) Å, cold ATV vs warm DFTB+ | 0.03 | 0.014 | **0.45** | ✓ | slower reading; first launch includes runtime init |
| H₂ later repeats | 0.03 | \(< 0.0001\) | \(\sim 0.003\) | ✓ | warm ATV |
| Off-eq H₂O | 0.04 | 0.0010 | 0.025 | ✓ | warm both |
| Tilted CO | — | 0.0003 | — | ✓ | no separate DFTB+ median quoted; same binary class |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{DFTB+}} \le 10\). The binding
ratio is **0.45** (0.014 / 0.03). Not `perf_exempt`.

## Limitations

- **Closed shell, integer occupation, H/C/N/O only.** Any other \(Z\), or
  an odd valence-electron count, throws. No spin, no \(+U\), no SOC.
- **No SCC.** Do not compare these forces to a DFTB+ `SCC = Yes` run on
  water or CO. H₂ happens to agree because \(\Delta q = 0\); that is not
  a licence to drop `SCC = No`.
- **No cell stress, no k-mesh, no lattice sum** beyond the one minimum
  image the energy already uses. The wrapped-H₂ case is an ATV identity,
  not a DFTB+ pin.
- **SK radial derivative is still a finite difference of the interpolant.**
  The nuclear gradient is closed-form given that derivative; the radial
  piece is not analytic.
- **Family lock.** DFTB rows pin to DFTB+ only. OpenMOPAC does not pin
  DFTB0. xtb does not pin DFTB0, and GFN-xTB is not a DFTB3 successor.
- **SE-only cmake** for this case. Do not enable MD, MINIMIZER, HF, PW,
  GPW, KSPACE, or CUDA to run this gate.

## References

- Offline pin: case `dftb0-analytic-forces` (`oracle: dftb+`, `oracle_version: 24.1`)
- M. Elstner et al., *Phys. Rev. B* **58**, 7260 (1998). DOI 10.1103/PhysRevB.58.7260
- B. Hourahine et al., *J. Chem. Phys.* **152**, 124101 (2020). DOI 10.1063/1.5143190
- Roadmap: `docs/roadmap/semiempirical.md` §2 DFTB0
- TB oracle inputs: `tools/tb_oracle/inputs/dftbplus/dftb0-analytic-forces/` (offline only)
