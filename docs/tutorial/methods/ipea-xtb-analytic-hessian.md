# IPEA-xTB FD Hessian — Td CH₄ vs xtb `--hess`

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ipea-xtb-analytic-hessian` |
| Regression | `regressions/semiempirical/ipea-xtb-analytic-hessian/` |
| Public API | `cpu::semiempirical::IPEA` (`with_forces = true`) on `cpu::vib::Hessian` at `vib::Phase::force` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md); [Ch. 14 Vibrations](../ch14-vibrations.md); [ipea-xtb-forces](ipea-xtb-forces.md) |
| Queue | TB method loop — wave 12 done (`c7db3d8f` / `f5e7c78a`) |
| Roadmap | `docs/roadmap/semiempirical.md` §3 IPEA-xTB analytic Hessian ✓; VIP/VEA is [`ipea-xtb-vip`](ipea-xtb-vip.md) |
| Oracle | **xtb 6.7.1** (compiled 2026-06-09) only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

Nuclear Hessian of production **IPEA-xTB** (Ásgeirsson, Bauer, Grimme,
*Chem. Sci.* **8**, 4879 (2017)) on tetrahedral methane, assembled as a
**\(6N\) central finite difference of the already-shipped analytic
gradient** (\(h=10^{-3}\) Å) through `cpu::vib::Hessian` with
`cpu::semiempirical::IPEA` (`with_forces = true`) on `vib::Phase::force`.
This is the **xtb IPEA `--hess` module boundary**. xtb 6.7.1 prints an
**analytical** Hessian (log: “analytical hessian”); ATV stays FD of
analytic \(G\) (same shape as
[`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md) /
[`gfn0-xtb-analytic-hessian`](gfn0-xtb-analytic-hessian.md)), **not** a
closed-form \(\partial^2 E/\partial R_i\partial R_j\) kernel and **not**
`libxtb`. Polar water is **not** this pin (0.5 eV kernel hole vs xtb).
After mass-weighting (IUPAC-2013 from \(Z\)) and TR projection the nine
vibrational wavenumbers are written on `HessianResult::frequencies`
[cm⁻¹]; the Cartesian matrix is `HessianResult::hessian` [eV/Å²]. Public
surface: concrete method plus the vibration Driver preset; no public
`*Config`, no `make_*` / `build_*`. Product gate:
`reg_ipea-xtb-analytic-hessian`. Energy + analytic \(G\) already ship
(`reg_ipea-xtb-small`, `reg_ipea-xtb-forces`). Fair wall
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.061\le 10\).

## Method

**IPEA-xTB energy** (V. Ásgeirsson, C. Bauer, S. Grimme, *Chem. Sci.* **8**,
4879 (2017), DOI
[10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)) — GFN1-shaped
extended tight-binding **reparametrized for IPs and EAs**, already shipped
by `reg_ipea-xtb-small`:

\[
E = \mathrm{Tr}(P H_0)
  + \tfrac12 q\,\gamma\,q
  + \tfrac13 \Gamma q^3
  + E_{\mathrm{rep}}
  + E_{\mathrm{D3}}.
\]

| Term | Role |
| ---- | ---- |
| \(E_{H^0}=\mathrm{Tr}(P H_0)\) | Occupied density on a CN-shifted EHT Hamiltonian; extra \((n+1)s\) on C/N/O; \(S\) from Stewart STO-\(n\)G |
| \(E_\gamma=\tfrac12 q\,\gamma\,q\) | Shell-resolved Klopman–Ohno SCC (**\(g_{\exp}=1\)**, harmonic-mean hardness) |
| \(E_\Gamma=\tfrac13\sum_A\Gamma_A q_A^3\) | On-site third-order (atomic, not shell-resolved); frozen-\(q\) \(\partial E_\Gamma/\partial R=0\) |
| \(E_{\mathrm{rep}}\) | Classical pair \(Z^{\mathrm{eff}}_A Z^{\mathrm{eff}}_B\,e^{-\sqrt{\alpha_A\alpha_B}\,r^{1.5}}/r\) |
| \(E_{\mathrm{D3}}\) | Molecular D3(BJ); GFN1 set \(s_6=1\), \(s_8=2.4\), \(a_1=0.63\), \(a_2=5.0\) (xtb \(s_9=0\)) |

Hydrogen is double-ζ (1s + orthogonalized 2s polarization). C/N/O carry an
extra 3s (Gram–Schmidt against valence s). CH₄: **13 AO**, 11 shells, 8
electrons. IPEA is **not** GFN1, **not** GFN2, **not** DFTB3 — different
Hamiltonian, tables, and oracle (**xtb** only). Parameters as members of
`IPEA`; tables in `basis_data/ipea_tables.h` (tblite `ipea1.f90` / xtb
`param_ipea-xtb.txt`). Unsupported \(Z\) throws.

**Analytic nuclear forces** already ship (`ipea-xtb-forces`):
\(\mathbf{F}=-\nabla E\) at frozen SCC density — Hellmann–Feynman + Pulay
on \(\partial S\), CN/shell-polynomial chains, \(\partial\gamma/\partial R\),
pair repulsion, and D3(BJ). Written to `frame.span(kAtom_f*)` in eV/Å.

**IPEA vs GFN1 on this geometry (same CH₄, different Hamiltonian):**

| Knob | IPEA-xTB | GFN1-xTB (`gfn1-xtb-organic`) |
| ---- | -------- | ----------------------------- |
| Extra AO | \((n+1)s\) on C/N/O | H 2s polarization only |
| CH₄ AO / shells | 13 / 11 | 12 / 10 |
| \(k_s, k_p, k_{sp}, k_{\mathrm{diff}}\) | 2.15, 2.30, 2.25, 2.25 | 1.85, 2.25, 2.08, 2.85 |
| \(g_{\exp}\) | 1 | 2 |
| Pair \(k_{\mathrm{HH}}, k_{\mathrm{NH}}\) | none | 0.96, 1.04 |
| D3(BJ) | same GFN1 set | same |
| xtb \(E_{\mathrm{tot}}\) | **−4.670476737294 Eh** | −4.274232224875 Eh |

**Hessian (this page).** Central FD of those analytic forces, then
symmetrized. With \(g=-F\) and \(h=10^{-3}\) Å
(`cpu::vib::kHessianStepAngstrom`):

\[
H_{ij}
  \approx
  \frac{g_i(+h\,e_j)-g_i(-h\,e_j)}{2h},
  \qquad
  H\leftarrow\tfrac12(H+H^\top).
\]

For \(N=5\) this is a fixed **\(6N=30\)** force budget plus one finalize
pass (restore \(R_0\), mass-weight, TR-project, diagonalize). The driver
is `cpu::vib::Hessian` (`Driver<vib::Phase>`); stages
`setup → displace → force → accumulate → diagonalize → thermochem →
end_of_step`. Masses for \(\omega\) come from IUPAC-2013 `mass_amu(Z)`
(`kAtom_type`), **not** `kAtom_mass`. xtb 6.7.1 `--hess` on IPEA is an
**analytical** Hessian (different algorithm class from ATV’s FD of \(G\));
the product gate is still \(\omega\) and one Cartesian \(H\) element vs
the xtb printout, with tols that absorb the path mismatch.

**Not included.** Closed-form \(\partial^2 E/\partial R_i\partial R_j\)
(CPHF / IPEA second-derivative kernel). IR intensities from an APT (FD
driver does not solve CPHF). xtb `--vip` / `--vea` is
[`ipea-xtb-vip`](ipea-xtb-vip.md). PBC / spin / solvent. GFN-FF
(classical). Polar-molecule water Hessian (0.5 eV energy hole vs xtb).

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| System | Isolated CH₄, **tetrahedral** (same xyz as `reg_ipea-xtb-small`; **not** the off-eq stretch of `reg_ipea-xtb-forces`) |
| Cell | Cubic 20 Å (isolated; min-image present, images not engaged) |
| Method | IPEA-xTB, closed shell, `charge = 0` |
| Tables | Static H/C/N/O from tblite `ipea1.f90` / xtb `param_ipea-xtb.txt` |
| Forces | `with_forces = true` → closed-form \(G\) at each displacement |
| Hessian | `cpu::vib::Hessian`, \(h=10^{-3}\) Å central FD of \(G\) |
| Accuracy | xtb `--acc 0.01` (SCF \(10^{-8}\) Eh) |
| xtb Hessian path | **analytical** (log: “analytical hessian”) |
| Units | \(E\) [eV] on `kEnergy`; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å]; \(H\) [eV/Å²]; \(\omega\) [cm⁻¹] |
| Conversion | \(a_0=0.52917721092\) Å (`units::kBohrInAngstrom`); \(1\,\mathrm{Ha}=27.211386245988\) eV |
| Thermo ctor | \(T=298.15\) K, \(P=101325\) Pa, multiplicity 1, \(\sigma=1\); empty recorder paths |

**Geometry [Å]** (`tools/tb_oracle/inputs/xtb/ch4_ipea_hess/mol.xyz`):
same tetrahedral CH₄ as `reg_ipea-xtb-small` (\(s=1.09/\sqrt{3}\)).

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| C | 6 | 0 | 0 | 0 |
| H | 1 | \(s\) | \(s\) | \(s\) |
| H | 1 | \(s\) | \(-s\) | \(-s\) |
| H | 1 | \(-s\) | \(s\) | \(-s\) |
| H | 1 | \(-s\) | \(-s\) | \(s\) |

xtb reports Td, gradient norm \(7.87\times10^{-4}\) Eh/\(a_0\),
\(n_{\mathrm{imag}}=0\). All nine vibrational eigenvalues are real.

**Key exclusions (this page / acceptance):**

- Closed-form IPEA Hessian kernel — not shipped; this page is FD of analytic
  \(G\). xtb 6.7.1 `--hess` is analytical; the pin is still \(\omega\) /
  \(H_{C_x,C_x}\).
- IPEA **energy** (this geometry) — sibling [`ipea-xtb-small`](ipea-xtb-small.md).
- IPEA **analytic \(G\)** (off-eq stretch) — sibling [`ipea-xtb-forces`](ipea-xtb-forces.md).
- **GFN0 / GFN1 / GFN2 Hessian** — siblings
  [`gfn0-xtb-analytic-hessian`](gfn0-xtb-analytic-hessian.md) /
  [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md).
- xtb **`--vip` / `--vea`** ΔSCC IP/EA — sibling [`ipea-xtb-vip`](ipea-xtb-vip.md)
  (empirical shift; not this closed-shell Hessian).
- Polar **water** vs xtb — polar Δq still carries a ~0.5 eV kernel hole
  (named in `reg_ipea-xtb-small` Gaps). Not this pin.
- PBC, spin, sTDA, ALPB/GBSA, CPCM-X — other queue rows. GFN-FF is
  classical (scope-out of this loop).
- Halogen-bond correction (xtb term; identically 0 on CH₄; not implemented).
- Beyond-CHNO tables — unsupported \(Z\) throws; deepen via
  `gfn2-beyond-chno`.
- Live xtb / DFTB+ / OpenMOPAC from ctest — **forbidden**; offline pins only.
- Using DFTB+ or OpenMOPAC as an IPEA Hessian reference — family lock forbids
  it.
- Treating IPEA / GFN-xTB as a DFTB3 successor — **forbidden**.

## Implementation

Public surface only (`cpu::Frame`, `cpu::semiempirical::IPEA` members,
`cpu::vib::Hessian` ctor args; no public `*Config`, no `make_*` /
`build_*`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/ipea.h"
#include "atomiverse/cpu/vibration/hessian.h"

using namespace ATV;

cpu::Frame frame(5);
const double s = 1.09 / std::sqrt(3.0);
const double x[5] = {0, s, s, -s, -s};
const double y[5] = {0, s, -s, s, -s};
const double z[5] = {0, s, -s, -s, s};
const int Z[5] = {6, 1, 1, 1, 1};
for (int i = 0; i < 5; ++i) {
    frame.span(kAtom_type)[i] = Z[i];
    frame.span(kAtom_x)[i] = x[i];
    frame.span(kAtom_y)[i] = y[i];
    frame.span(kAtom_z)[i] = z[i];
    frame.span(kAtom_mass)[i] = (i == 0) ? 12.011 : 1.008;
    frame.span(kAtom_tag)[i] = i + 1;
    frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
        frame.span(kAtom_fz)[i] = 0;
}
frame.box = cpu::Box::cubic(20.0);
frame.meta.get(kEnergy) = 0;

cpu::semiempirical::IPEA ipea;
ipea.charge = 0;
ipea.with_forces = true;

// Empty recorder paths: do not write frequencies.tsv / mode_*.xyz.
cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
hess.on(vib::Phase::force).add(std::move(ipea));
hess.build();
const cpu::vib::HessianResult res = hess.run(frame);
// res.hessian: [15,15] dense, eV/Å²; res.frequencies: 9 modes, cm⁻¹
// Frame geometry is restored to R0; leftover kEnergy is the last displacement.
```

The regression also re-evaluates energy after restore via `cpu::Evaluate`
(fresh method; leftover `kEnergy` from the last displacement is not the
pin).

SE-only build **plus the VIB exception** for this slug (never bare
`cmake --build build`; never HF/MD/CUDA):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_EVALUATE=ON \
        -DATV_BUILD_VIB=ON -DATV_BUILD_CLI=OFF \
        -DATV_BUILD_MD=OFF -DATV_BUILD_MC=OFF -DATV_BUILD_MINIMIZER=OFF \
        -DATV_BUILD_PATH=OFF -DATV_BUILD_POTENTIALS=OFF \
        -DATV_BUILD_NEIGHBOR=OFF -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF \
        -DATV_BUILD_GPW=OFF -DATV_BUILD_KSPACE=OFF -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse \
        test_cpu_xtb_ipea_hessian reg_ipea-xtb-analytic-hessian
  ctest --test-dir build -R "^test_cpu_xtb_ipea_hessian$" --output-on-failure
  cmake --build build --target run_reg_ipea-xtb-analytic-hessian
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/ipea.h` | Public `IPEA` (`charge`, `scc_tol`, `with_forces`) |
| `include/atomiverse/cpu/semiempirical/basis_data/ipea_tables.h` | Static H/C/N/O IPEA rows + global scales |
| `src/cpu/semiempirical/ipea.cpp` | Stewart overlap / extra 3s / EHT+SCC / analytic \(\nabla E\) |
| `include/atomiverse/cpu/vibration/hessian.h` | `cpu::vib::Hessian` FD preset |
| `include/atomiverse/cpu/vibration/hessian_result.h` | `HessianResult` (\(H\), \(\omega\), \(n_{\mathrm{imag}}\)) |
| `include/atomiverse/cpu/vibration/hessian_workspace.h` | \(h=10^{-3}\) Å (`kHessianStepAngstrom`) |
| `regressions/semiempirical/ipea-xtb-analytic-hessian/reg_ipea-xtb-analytic-hessian.cpp` | Product gate |
| `regressions/semiempirical/ipea-xtb-analytic-hessian/REFERENCE.md` | Offline pins + oracle line |
| `tools/tb_oracle/inputs/xtb/ch4_ipea_hess/` | xtb 6.7.1 twin (`mol.xyz`, `xtb.out.txt`, `hessian.txt`, `vibspectrum.txt`) |
| `tests/cpu/semiempirical/test_xtb_ipea_hessian.cpp` | Unit: `Hessian::build` accepts IPEA force kind (no `run`) |

## Validation (xtb)

Source:
[`regressions/semiempirical/ipea-xtb-analytic-hessian/REFERENCE.md`](../../../regressions/semiempirical/ipea-xtb-analytic-hessian/REFERENCE.md).
Oracle: **`xtb`** IPEA (`param_ipea-xtb.txt` as GFN1) `--hess` (Ásgeirsson,
Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017); Bannwarth et al., *WIREs
Comput. Mol. Sci.* **11**, e01493 (2021)).
`oracle_version`: **6.7.1** (compiled 2026-06-09).
Hard-coded gates in the regression binary — **never** live xtb from ctest.

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine (identity) | ATV IPEA-xTB closed shell, H/C/N/O tables + `cpu::vib::Hessian` |
| Oracle family | xtb (not OpenMOPAC, not DFTB+) |
| Offline twin | `tools/tb_oracle/inputs/xtb/ch4_ipea_hess/` |
| Command | `XTBPATH=… xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess` (`OMP_NUM_THREADS=1`; `param_gfn1-xtb.txt` ← `param_ipea-xtb.txt`; banner `I P E A - x T B`) |
| Atomiverse path | `IPEA` on `Hessian` at `vib::Phase::force`, `with_forces=true` |
| FD protocol | Central of analytic \(G\), \(h=10^{-3}\) Å (xtb: **analytical** Hessian) |
| Conversion | \(H[\mathrm{eV}/\mathrm{Å}^2]=H[\mathrm{Eh}/a_0^2]\times 27.211386245988/(0.52917721092)^2\) |

**xtb projected \(\omega\) [cm⁻¹]** (`vibspectrum.txt` modes 7–15; six TR
zeros projected out):

| Mode | Assignment | Degeneracy | xtb 6.7.1 |
| ---- | ---------- | ---------: | --------: |
| \(\omega_{\mathrm{T2}}\) | bend | 3 | **1541.68** |
| \(\omega_{\mathrm{E}}\) | bend | 2 | **1705.89** |
| \(\omega_{\mathrm{A1}}\) | stretch | 1 | **3548.78** |
| \(\omega_{\mathrm{T2}}\) | stretch | 3 | **3576.13** |

xtb also printed TOTAL ENERGY **−4.670476737294 Eh**, Cartesian
\(H_{C_x,C_x}=0.6813937129\) Eh/\(a_0^2\) (`hessian.txt`), Td, gradient
norm \(7.87\times10^{-4}\) Eh/\(a_0\), 13 AO / 11 shells / 8 electrons,
HOMO–LUMO 7.751 eV. IR intensities in the twin (T2 bend 9.62 / E 0 / A1 0
/ T2 stretch 53.36 km mol⁻¹) are **not** a product gate (ATV FD Hessian
has no APT).

**xtb energy decomposition (documentation; Eh):** same geometry as
`reg_ipea-xtb-small`.

| Term | Eh |
| ---- | --: |
| SCC (electronic) | −4.766828456418 |
| → electrostatic | 0.001132006171 |
| repulsion | 0.096806468732 |
| dispersion | −0.000454749608 |
| halogen | 0 |
| **Total** | **−4.670476737294** |

**Hard-coded gates (regression):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E\) | `kEnergy` (fresh Evaluate after restore) | **−4.670476737294 Eh** | \(5\times10^{-2}\) eV | same kernel-residual floor as `reg_ipea-xtb-small` |
| \(\omega\) T2 bend (×3) | `res.frequencies(0..2)` | **1541.68 cm⁻¹** | 10 cm⁻¹ | projected, ascending |
| \(\omega\) E bend (×2) | `res.frequencies(3..4)` | **1705.89 cm⁻¹** | 10 cm⁻¹ | |
| \(\omega\) A1 stretch | `res.frequencies(5)` | **3548.78 cm⁻¹** | 10 cm⁻¹ | |
| \(\omega\) T2 stretch (×3) | `res.frequencies(6..8)` | **3576.13 cm⁻¹** | 10 cm⁻¹ | |
| \(H_{C_x,C_x}\) | `res.hessian(0,0)` | **66.21350 eV/Å²** | 0.2 eV/Å² | from \(0.6813937129\) Eh/\(a_0^2\) |
| Hessian shape | identity | \(15\times 15\) | hard | finite, \(\lvert H-H^\top\rvert\le 10^{-8}\) eV/Å² |
| Rotor | identity | nonlinear | hard | `res.is_linear == false`; 9 modes |
| \(n_{\mathrm{imag}}\) | identity | 0 | hard | all nine \(\omega>100\) cm⁻¹ |
| Geometry restore | identity | \(R_0\) | \(10^{-12}\) Å | `Hessian::run` restores displacements |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=8.90\) meV,
\(\Delta\omega = 0.36/0.64/1.30/0.26\) cm⁻¹ (T2-bend / E / A1 / T2-stretch;
Td degeneracies unsplit),
\(\lvert H_{C_x,C_x}-H_{\mathrm{pin}}\rvert=1.60\times10^{-2}\) eV/Å².

**Tolerance rationale.** Energy floor \(5\times10^{-2}\) eV is the
`reg_ipea-xtb-small` kernel-residual floor (D3 C6 interpolation vs xtb’s
internal D3, Stewart primitive screening vs xtb `intcut`, Anderson vs
Broyden) on a ~127 eV total — the same 8.90 meV CH₄ hole. Frequency /
Cartesian-\(H\) tols absorb FD-of-\(G\) vs xtb **analytical** Hessian plus
IUPAC-2013 vs xtb isotope masses. The 10 cm⁻¹ / 0.2 eV/Å² gates sit well
above the measured residual (max \(\Delta\omega=1.30\) cm⁻¹; \(H\) residual
0.016 eV/Å²) and still catch a sign, unit, or mass-weighting error on the
stiff T2 stretches (\(\sim 3576\) cm⁻¹). Polar-molecule residual vs water
(~0.5 eV) is **not** this pin.

Discriminates IPEA from GFN1: eq CH4 GFN1 is −4.274232224875 Eh vs IPEA
−4.670476737294 Eh.

Offline re-pin (human only; never ctest):

```bash
# when xtb 6.7.1 is available; see tools/tb_oracle/inputs/xtb/ch4_ipea_hess/
# XTBPATH with param_ipea as GFN1; --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess
# Td CH4 r_CH=1.09 Å; record TOTAL ENERGY [Eh], vibspectrum.txt ω, hessian.txt H_Cxx
# → H[eV/Å²] via 27.211386245988 / (0.52917721092)²
```

## Performance

Source: `REFERENCE.md`. Fair wall asserted in
`reg_ipea-xtb-analytic-hessian` (\(r\le 10\)). Same *product* (projected
\(\omega\) of Td CH₄); **not** the same Hessian algorithm class (ATV:
analytic \(G\), numerical \(H\), 30 force evaluations + finalize; xtb
6.7.1: analytical Hessian).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| CH₄ IPEA `--hess` | **0.170** (1 thread, printed) | **\(1.04\times10^{-2}\)** | **≈ 0.061** | ✓ | `/usr/bin/time` process wall 0.43 s; regression pins printed wall 0.17 s; ATV tables are static; xtb Hessian is analytical |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) when a twin exists.
`perf_exempt`: **false**. Ratio is a ctest assertion (`ratio > 10` fails).

## Limitations

- **FD of analytic \(G\), not closed-form \(H\).** Cost is
  \(O(N_{\mathrm{atoms}})\) IPEA force evaluations. An IPEA
  second-derivative kernel is not on this page. xtb 6.7.1 `--hess` is
  analytical; \(\omega\) tols absorb that path difference.
- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; not a
  silent skip.
- **Closed-shell even-electron only.** `multiplicity != 1` throws
  (`spin-gfn-xtb`).
- **Kernel hole vs xtb.** Energy residual 8.90 meV on this Td CH₄ is the
  same D3/intcut/mixer mismatch as `reg_ipea-xtb-small`; it does not move
  \(\omega\) past 10 cm⁻¹. Polar-molecule residual vs water (~0.5 eV) is
  an energy-path hole, not a Hessian defect, and is **not** this pin.
- **No PBC / spin / solvent / IR** on this page —
  `--vip`/`--vea` is [`ipea-xtb-vip`](ipea-xtb-vip.md). GFN-FF is classical
  (never pending on this loop).
- **GFN0 / GFN1 / GFN2 Hessian.** Siblings
  [`gfn0-xtb-analytic-hessian`](gfn0-xtb-analytic-hessian.md) /
  [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md). This page
  closes IPEA FD \(H\) only.
- **VIB exception.** This slug (and `se-analytic-hessian` /
  `gfn0-xtb-analytic-hessian` / `gfn-xtb-analytic-hessian`) may set
  `-DATV_BUILD_VIB=ON`. Do not enable MD/MINIMIZER/HF/PW/GPW/CUDA for the
  TB product gate; do not build `reg_am1_water_bomd` /
  `reg_am1_water_minimize` under this configure.
- Family lock: **xtb only** for IPEA / GFN rows — never DFTB+ or OpenMOPAC as
  an IPEA oracle; never treat GFN-xTB as a DFTB3 successor.
- Foothold ≠ closed: IPEA energy + \(G\) + FD \(H\) + VIP/VEA is not the GFN
  product closed (ALPB, CPCM-X, PBC, spin, sTDA, beyond-CHNO, extra-Z IPEA,
  CAMM-vs-\(R\) remain).

## References

- Offline pin: [`regressions/semiempirical/ipea-xtb-analytic-hessian/REFERENCE.md`](../../../regressions/semiempirical/ipea-xtb-analytic-hessian/REFERENCE.md)
- Twin: `tools/tb_oracle/inputs/xtb/ch4_ipea_hess/` (`mol.xyz`,
  `xtb.out.txt`, `hessian.txt`, `vibspectrum.txt`)
- Energy sibling: [`ipea-xtb-small.md`](ipea-xtb-small.md) /
  `regressions/semiempirical/ipea-xtb-small/REFERENCE.md`
- Force sibling: [`ipea-xtb-forces.md`](ipea-xtb-forces.md) /
  `regressions/semiempirical/ipea-xtb-forces/REFERENCE.md`
- VIP/VEA sibling: [`ipea-xtb-vip.md`](ipea-xtb-vip.md) /
  `regressions/semiempirical/ipea-xtb-vip/REFERENCE.md`
- GFN Hessian siblings: [`gfn0-xtb-analytic-hessian.md`](gfn0-xtb-analytic-hessian.md),
  [`gfn-xtb-analytic-hessian.md`](gfn-xtb-analytic-hessian.md)
- SE Hessian foothold (DFTB2/H₂, structural): `regressions/semiempirical/se-analytic-hessian/REFERENCE.md`
- Oracle README: `tools/tb_oracle/README.md`
- Roadmap: `docs/roadmap/semiempirical.md`
- Spec history (closed, deleted at done): git `wf/ipea-xtb-analytic-hessian`
- V. Ásgeirsson, C. Bauer, S. Grimme, *Chem. Sci.* **8**, 4879 (2017),
  DOI [10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
