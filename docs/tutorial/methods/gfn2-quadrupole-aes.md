# GFN2-xTB quadrupole AES — H₂O vs xtb 6.7.1

*(Technical-report page. Narrative house style TBD.)*

| Field | Value |
| ----- | ----- |
| Slug | `gfn2-quadrupole-aes` |
| Public API | `cpu::Evaluate` + `cpu::semiempirical::GFN2` (`with_forces = false`) |
| Roadmap / queue | `docs/roadmap/semiempirical.md` §3 GFN2-xTB; queue **done** |
| Oracle | **xtb 6.7.1** (`edcfbbe`) only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

GFN2-xTB (Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019)) is a third-generation extended tight-binding Hamiltonian whose
anisotropic electrostatics (AES) go beyond the isotropic shell-resolved SCC
of GFN1: cumulative atomic dipoles **and** traceless quadrupoles interact through a
damped multipole kernel, with on-site anisotropic XC (AXC) kernels
\(d_{\mathrm{kernel}}\) and \(q_{\mathrm{kernel}}\). ATV evaluates
charge–dipole, dipole–dipole and **charge–quadrupole** AES plus dipole and
**quadrupole** AXC on the cumulative atomic multipoles of every atom (H
included); this page pins that energy on the public Evaluate surface (eV on
`frame.meta[kEnergy]`). The measured residual against archived xtb is
**11 µeV**: AES + AXC is \(-1.94391\times10^{-4}\) Eh vs xtb anisotropic
ES + XC \(-1.94401\times10^{-4}\) Eh, and the remainder is the
self-consistent D4 term. Parameters are static in-tree tables — there is no
`libxtb` wrap and no runtime parameter file. Analytic forces ship on
`gfn2-xtb-forces`; AES and D4 enter the SCC Fock as
\(\partial E/\partial P\) ([`gfn2-aes-fock`](gfn2-aes-fock.md)).

## Method

### GFN2 total energy (closed shell, molecular)

\[
E = \operatorname{Tr}(P H^{0})
  + E_{\mathrm{IES}}
  + E^{(3)}
  + E_{\mathrm{rep}}
  + E_{\mathrm{D4}}
  + E_{\mathrm{AES}}.
\]

\(H^{0}\) is the STO-\(n\)G valence Hamiltonian with self-energies shifted
by the double-exponential CN' (\(k_a=10\), \(k_b=20\), 2 Bohr shift,
\(4/3\)-scaled Pyykkö radii) and the distance polynomial
\(\Pi(R)=\prod\bigl(1+k\,(R/(R_A+R_B))^{1/2}\bigr)\) (\(\Pi\) radius
O = 0.64 Å; tblite `get_hamiltonian` scales \(k_{s}=1.85\), \(k_{p}=2.23\),
\(w_{\mathrm{exp}}=0.5\)). Isotropic electrostatics are shell-resolved
DFTB2-style \(\gamma_{AB}\) with exponent \(g_{\mathrm{exp}}=2\) plus the
third order \(E^{(3)}=\tfrac13\sum_l\Gamma_l q_l^3\) in the shell charge.
Repulsion is the GFN2 screened \(Z_{\mathrm{eff}}\) exponential; dispersion is
self-consistent Grimme D4 (\(C_6\) scaled by the SCC Mulliken charges)
with GFN2 damping (\(s_{6}=1\), \(s_{8}=2.7\), \(a_{1}=0.52\), \(a_{2}=5\),
\(s_{9}=5\)). Occupation is restricted integer filling (8 valence electrons
on this H₂O).

### Anisotropic electrostatics

AES enters the SCC Fock ([`gfn2-aes-fock`](gfn2-aes-fock.md)); its
energy is evaluated on the cumulative atomic multipoles of the density on
**every** atom (H included), with the operator centred on the bra atom
\(A\) of \(\kappa\):

\[
\mu_{A,\alpha}
  = -\sum_{\kappa\in A}\sum_{\nu} P_{\kappa\nu}
    \langle\kappa|r_\alpha-R_{A,\alpha}|\nu\rangle,
\]

\[
\Theta_{A,\alpha\beta}
  = -\sum_{\kappa\in A}\sum_{\nu} P_{\kappa\nu}
    \bigl\langle\kappa\bigr|
      \tfrac32(r-R_A)_\alpha(r-R_A)_\beta
      -\tfrac12\delta_{\alpha\beta}\lvert r-R_A\rvert^2
    \bigl|\nu\bigr\rangle.
\]

The \(R_A\)-centred integrals come from origin-0 \(S\), \(D^\alpha\),
\(Q_{\alpha\beta}\): \(D^\alpha-R_{A,\alpha}S\) and the Cartesian origin
shift \(q^{\mathrm{shift}}_{\alpha\beta}
 = Q_{\alpha\beta}-R_{\alpha}D_{\beta}-R_{\beta}D_{\alpha}+R_{\alpha}R_{\beta}S\),
mapped to the **traceless** \(3/2\) form
\(\tfrac32 q^{\mathrm{shift}}_{\alpha\beta}
 -\tfrac12\delta_{\alpha\beta}\operatorname{Tr}q^{\mathrm{shift}}\)
(tblite packed \((xx,xy,yy,xz,yz,zz)\)).

Pair damping uses a CN-dependent AES radius (tblite `damped_multipole` 0d):

\[
R_{A}
  = R_{\mathrm{mp},A}
  + \frac{R_{\max}-R_{\mathrm{mp},A}}
        {1+\exp\bigl(-k_{\mathrm{aes}}(\mathrm{CN}_{A}
          -\mathrm{CN}_{\mathrm{mp},A}-\mathrm{shift})\bigr)},
\]

with \(k_{\mathrm{aes}}=4\), \(\mathrm{shift}=1.2\), \(R_{\max}=5\) Bohr, and
the same double-exponential CN' as \(H^0\). For pair
\(AB\) with \(r=\lvert\mathbf{r}\rvert\), \(\mathbf{r}=\mathbf{R}_{A}-\mathbf{R}_{B}\),
\(R_{\mathrm{avg}}=(R_{A}+R_{B})/2\),

\[
f_{n}(r)
  = \bigl(1+6(R_{\mathrm{avg}}/r)^{n}\bigr)^{-1},
\qquad n=3\text{ (odd), }n=5\text{ (even)},
\qquad g_n=f_n/r^n.
\]

The energy assembled in `solve_gfn2` (tblite `get_energy` / `amat_sq` /
`amat_sd`) is, for every pair \(A\neq B\),

\[
\begin{aligned}
E_{q\mu}
  &= g_{3}\,(q_{A}\boldsymbol{\mu}_{B}-q_{B}\boldsymbol{\mu}_{A})\!\cdot\mathbf{r},\\
E_{\mu\mu}
  &= g_{5}\bigl(
        (\boldsymbol{\mu}_{A}\!\cdot\boldsymbol{\mu}_{B})\,r^{2}
      - 3(\boldsymbol{\mu}_{A}\!\cdot\mathbf{r})(\boldsymbol{\mu}_{B}\!\cdot\mathbf{r})
      \bigr),\\
E_{q\Theta}
  &= g_{5}\bigl(
        q_{A}\,\mathbf{r}\!\cdot\Theta_{B}\!\cdot\mathbf{r}
      + q_{B}\,\mathbf{r}\!\cdot\Theta_{A}\!\cdot\mathbf{r}
      \bigr).
\end{aligned}
\]

Here \(q_{A}=-\sum_{l\in A}\Delta n_{l}\) is the Mulliken charge (shell
electron excess \(\Delta n_l=\mathrm{pop}_l-n^0_l\)). \(\Theta\) is already
the traceless \(3/2\) form, so no extra factor enters; charge–quadrupole is
even in \(\mathbf{r}\).

On-site AXC (off-diagonal packed quadrupole scaled by 2, tblite):

\[
E_{\mathrm{AXC}}
  = \sum_{A}
    \Bigl(
      d_{A}\lvert\boldsymbol{\mu}_{A}\rvert^{2}
    + q_{A}\bigl(
        \Theta_{xx}^{2}+2\Theta_{xy}^{2}+\Theta_{yy}^{2}
        +2\Theta_{xz}^{2}+2\Theta_{yz}^{2}+\Theta_{zz}^{2}
      \bigr)
    \Bigr).
\]

This page’s focus is the \(E_{q\Theta}\) + quadrupole-\(q_{\mathrm{kernel}}\)
terms; charge–dipole, dipole–dipole and dipole AXC are the same kernel as
`gfn2-xtb-water`.

### SK / tables (binding for this pin)

| Piece | Source in ATV | Oracle twin |
| ----- | ------------- | ----------- |
| H/C/N/O element rows (ζ, ε, Hubbard, \(d/q_{\mathrm{kernel}}\), AES radii) | `basis_data/gfn2_xtb_tables.h` | xtb `param_gfn2-xtb.txt` / tblite `gfn2.f90` |
| Global AES damp (\(d_{\mathrm{mp3}}=3\), \(d_{\mathrm{mp5}}=4\)) | same header | tblite 0d molecular kernel |
| D4 damping | self-consistent `scf::dftd4_fixed_charges` at the SCC charges + GFN2 \(s_{6,8,9},a_{1,2}\) | xtb GFN2 D4 |
| Other elements | **unsupported** (throw) | xtb is all-element |

O `qkernel` is tabulated and non-zero (\(-3.10828\times10^{-3}\) Eh); the
unit test `test_gfn2_params` asserts that.

### Geometry and units

| Setting | Value |
| ------- | ----- |
| System | H₂O, closed shell, charge 0 |
| Geometry (Å) | O \((0,0,0)\); H \((0.96,0,0)\); H \((-0.24,0.93,0)\) |
| Box | cubic 20 Å (isolation only; molecular min-image) |
| Basis (xtb transcript) | 6 AO, 4 shells, 8 electrons; HOMO–LUMO 14.519 eV |
| Mulliken (xtb) | \(q(\mathrm{O})=-0.563\), \(q(\mathrm{H})=0.282/0.281\) |
| Public energy unit | **eV** on `frame.meta.get(kEnergy)` |
| Internal AES / Hamiltonian | Hartree / Bohr (tblite convention) |
| Ha → eV (public write) | \(27.211386245988\) (CODATA-2018, `units::kHartreeInEv`) |
| Ha → eV (table ε) | \(27.21138505\) (xtb/tblite `kGfn2EvToEh`) |

Same geometry as `gfn2-xtb-water`. The pin tolerance here is
\(5\times10^{-4}\) eV, tighter than that page’s \(2\times10^{-3}\) eV.

### Explicit exclusions

- **Not** analytic nuclear forces (this energy pin; sibling
  `gfn2-xtb-forces` ships closed-form \(F=-\nabla E\)).
- **Not** the derivation of the AES / D4 Fock potential (the production SCC
  uses it; documented on [`gfn2-aes-fock`](gfn2-aes-fock.md)).
- **Not** dipole–quadrupole or quadrupole–quadrupole pair ranks (not GFN2
  terms).
- **Not** ALPB/GBSA, CPCM-X, spin, PBC, IPEA, sTDA, Hessian.
- **Not** elements beyond H/C/N/O (`gfn2-beyond-chno`).
- **Not** GFN-FF (classical; out of this electronic loop).
- **Not** OpenMOPAC or DFTB+ as reference — family lock is **xtb only**.
  GFN2 is **not** a DFTB3 successor.
- **Not** live xtb in ctest: hard-coded pin only.

## Implementation

Public surface only: construct `GFN2`, set members, bind into `Evaluate`.
No `*Config` bag, no `make_*` / `build_*` factory.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/gfn2.h"

using namespace ATV;

cpu::Frame frame(3);
frame.span(kAtom_type)[0] = 8;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_type)[1] = 1;
frame.span(kAtom_x)[1] = 0.96;
frame.span(kAtom_y)[1] = 0.0;
frame.span(kAtom_z)[1] = 0.0;
frame.span(kAtom_type)[2] = 1;
frame.span(kAtom_x)[2] = -0.24;
frame.span(kAtom_y)[2] = 0.93;
frame.span(kAtom_z)[2] = 0.0;
for (int i = 0; i < 3; ++i) {
    frame.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::cubic(20.0);

cpu::Evaluate eval;
cpu::semiempirical::GFN2 gfn2;
gfn2.with_forces = false;
gfn2.scc_tol = 1e-8;
eval.on(evaluate::Phase::evaluate).add_ref(gfn2);
eval.build();
eval.run(frame);

const double E_eV = frame.meta.get(kEnergy);
const double pin_eV = -5.070383095219 * units::kHartreeInEv;
// gate: |E_eV − pin_eV| ≤ 1e−4 eV
```

### SE-only build and run

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
  cmake --build build -j"$(nproc)" --target atomiverse \
        test_cpu_gfn2_params reg_gfn2-quadrupole-aes
  ctest --test-dir build -R "^(test_cpu_gfn2_params)$" --output-on-failure
  cmake --build build --target run_reg_gfn2-quadrupole-aes
'
```

Relevant paths:

| Role | Path |
| ---- | ---- |
| Method | `gfn2.h` |
| Impl (SCC with AES + self-consistent D4 in the Fock; AES in `aes_eval`) | `src/cpu/semiempirical/gfn2.cpp` |
| D4 at given charges (`dftd4_fixed_charges`) | `src/cpu/scf/dftd4.cpp` |
| Static H/C/N/O + AES globals | `…/basis_data/gfn2_xtb_tables.h` |
| Unit tests | `tests/cpu/semiempirical/test_gfn2_params.cpp` |
| Offline twin | TB oracle (OpenMOPAC / DFTB+ / xtb) |

## Validation (vs xtb)

Source: case `gfn2-quadrupole-aes` pin
(offline pin; **never** live xtb in ctest).

```text
oracle: xtb
oracle_version: 6.7.1 (edcfbbe)
```

### Oracle decomposition (xtb 6.7.1, `--gfn 2 --chrg 0 --uhf 0`)

| Quantity | Value | Unit |
| -------- | ----: | ---- |
| Total energy | **−5.070383095219** | Eh |
| Total energy (public pin) | −5.070383095219 × `units::kHartreeInEv` | eV |
| SCC | −5.104517627658 | Eh |
| isotropic ES | +0.031400392671 | Eh |
| anisotropic ES | +0.000610908598 | Eh |
| anisotropic XC | −0.000805309300 | Eh |
| dispersion | −0.000141270483 | Eh |
| repulsion | +0.034134532439 | Eh |

xtb inputs (reproducible offline only):

- TB oracle (OpenMOPAC / DFTB+ / xtb) — off-eq H₂O (Å)
- Command: `xtb mol.xyz --gfn 2 --chrg 0 --uhf 0`
- Archived stdout: TB oracle (OpenMOPAC / DFTB+ / xtb)

### Accuracy gate (ctest)

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E_{\mathrm{tot}}\) | `kEnergy` | **−5.070383095219 Eh** | **1×10⁻⁴ eV** | hard-coded in `reg_*.cpp` |
| Host residual | \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert\) | — | — | **11 µeV** (D4 term) |
| SCC residual | `last_scc_residual()` | — | \(<\) `scc_tol` \(=10^{-8}\) | must converge, not exhaust `max_scc` |
| Energy finite / non-zero | `kEnergy` | — | \(\lvert E\rvert>1\) eV | sanity gate |

Tolerance is 0.1 meV — tighter than `gfn2-xtb-water`
(\(2\times10^{-3}\) eV) because this pin isolates the quadrupole AES.
AES + AXC on the cumulative multipoles of every atom is
\(-1.94391\times10^{-4}\) Eh vs xtb anisotropic ES + XC
\(-1.94401\times10^{-4}\) Eh; the 11 µeV remainder is the self-consistent
D4 term (ATV \(-1.40847\times10^{-4}\) vs xtb \(-1.41270\times10^{-4}\) Eh).

## Performance

Informative only — **not** asserted in ctest.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN2 Evaluate (forces off) | 0.155 (xtb 6.7.1, 1 thread; SCC 0.115 s) | — | **perf_exempt** | — | no xtb binary on the SE build host for a fair wall |

`perf_exempt: true` in REFERENCE. Archived xtb wall is 0.155 s. Revisit
when xtb is on PATH. Gate convention (TB loop): \(r\le 10\) unless
`perf_exempt`.

## Limitations

| Item | Status |
| ---- | ------ |
| Charge–quadrupole AES + \(q_{\mathrm{kernel}}\) AXC on H₂O | **closed** (residual 11 µeV) |
| GFN2 energy H/C/N/O | closed (`gfn2-xtb-water`) |
| AES + self-consistent D4 as SCC Fock potential | shipped ([`gfn2-aes-fock`](gfn2-aes-fock.md)) |
| Dipole–quadrupole / quadrupole–quadrupole pair AES | not assembled |
| Analytic nuclear forces | sibling `gfn2-xtb-forces` |
| Elements beyond H/C/N/O | throw; sibling `gfn2-beyond-chno` |
| ALPB / CPCM-X / spin / PBC / IPEA / sTDA / Hessian | open or blocked |
| Self-consistent D4 term (ATV \(-1.40847\times10^{-4}\) vs xtb \(-1.41270\times10^{-4}\) Eh) | residual contributor (11 µeV) |
| Live oracle in CI | **forbidden** — offline pins only |

Foothold ≠ closed: GFN2 product energy on CHNO is ✓,
`gfn2-xtb-forces` ships closed-form \(F=-\nabla E\), and AES + D4 are in
the SCC Fock ([`gfn2-aes-fock`](gfn2-aes-fock.md)); solvent, PBC and
all-element tables remain ○.

## References

- Offline pin: case `gfn2-quadrupole-aes` pin
- Water foothold: case `gfn2-xtb-water` pin
- Oracle twin: TB oracle (OpenMOPAC / DFTB+ / xtb)
- Family rules: TB oracle (OpenMOPAC / DFTB+ / xtb)
- Roadmap: [`docs/roadmap/semiempirical.md`](../../roadmap/semiempirical.md) §3
- C. Bannwarth, S. Ehlert, S. Grimme, *J. Chem. Theory Comput.* **15**, 1652
  (2019), DOI [10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
- xtb 6.7.1 (`edcfbbe`); tblite `damped_multipole` molecular 0d
