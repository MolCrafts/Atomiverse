# GFN1/GFN2-xTB FD Hessian — off-eq H₂O vs xtb `--hess`

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gfn-xtb-analytic-hessian` |
| Regression | `regressions/semiempirical/gfn-xtb-analytic-hessian/` |
| Public API | `cpu::semiempirical::GFN1` / `GFN2` (`with_forces = true`) on `cpu::vib::Hessian` at `vib::Phase::force` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md); [Ch. 14 Vibrations](../ch14-vibrations.md) |
| Queue | TB method loop — wave 10 (`gfn-xtb-analytic-hessian`) |
| Roadmap | `docs/roadmap/semiempirical.md` §3 GFN-xTB analytic Hessian |
| Oracle | **xtb 6.7.1** only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

Nuclear Hessian of production **GFN1-xTB** (primary) and **GFN2-xTB**
(family close) on off-equilibrium water, assembled as a **\(6N\)
central finite difference of the already-shipped analytic gradient**
(\(h=10^{-3}\) Å) through `cpu::vib::Hessian`. This is the
**xtb `--gfn 1 --hess` / `--gfn 2 --hess` module boundary**. xtb 6.7.1
prints an **analytical** Hessian; ATV stays FD of analytic \(G\) (same
shape as `gfn0-xtb-analytic-hessian` / `se-analytic-hessian`), **not** a
closed-form \(\partial^2 E/\partial R_i\partial R_j\) kernel and **not**
`libxtb`. GFN0 FD Hessian already ships. GFN2 AES remains post-SCC
(angular \(F\) hole — `gfn2-aes-fock`). Public surface: concrete
methods plus the vibration Driver preset; no public `*Config`, no
`make_*` / `build_*`. Product gate: `reg_gfn-xtb-analytic-hessian`.
Fair wall \(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.11\le 10\) (GFN1).

## Method

**GFN1-xTB energy** (Grimme, Bannwarth, Shushkov, JCTC 2017) —
SCC EHT + Klopman–Ohno \(\gamma\) + on-site \(\Gamma q^3\) + pair
repulsion + D3(BJ). Analytic \(G\) already ships (`gfn1-xtb-forces`).

**GFN2-xTB energy** (Bannwarth, Ehlert, Grimme, JCTC 2019) — SCC +
anisotropic electrostatics + D4. Analytic \(G\) already ships
(`gfn2-xtb-forces`); AES is still a post-SCC energy.

**Hessian (this page).** Central FD of those analytic forces, then
symmetrized, through `cpu::vib::Hessian` (`Driver<vib::Phase>`). Masses
for \(\omega\) come from IUPAC-2013 `mass_amu(Z)`.

**Not included.** Closed-form second-derivative kernel. IR intensities
from an APT. PBC / spin / solvent / IPEA. GFN-FF (classical).

**Twin settings:**

| Knob | Value |
| ---- | ----- |
| System | Isolated H₂O, 0.96 Å (same xyz as `reg_gfn1-xtb-forces`) |
| Cell | Cubic 20 Å (isolated) |
| GFN1 | `--gfn 1 --acc 0.01 --hess` → ω **1474.96 / 3657.78 / 3761.56** cm⁻¹ |
| GFN2 | `--gfn 2 --acc 0.01 --hess` → ω **1565.10 / 3634.95 / 3643.62** cm⁻¹ |
| Hessian | `cpu::vib::Hessian`, \(h=10^{-3}\) Å central FD of \(G\) |
| Units | \(E\) [eV]; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å]; \(H\) [eV/Å²]; \(\omega\) [cm⁻¹] |

**Key exclusions:** live xtb in ctest; DFTB+ / OpenMOPAC as a GFN Hessian
reference; treating GFN-xTB as a DFTB3 successor.

## Reproduce

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
        test_cpu_xtb_gfn_hessian reg_gfn-xtb-analytic-hessian
  ctest --test-dir build -R "^test_cpu_xtb_gfn_hessian$" --output-on-failure
  cmake --build build --target run_reg_gfn-xtb-analytic-hessian
'
```

Pins and provenance:
[`regressions/semiempirical/gfn-xtb-analytic-hessian/REFERENCE.md`](../../../regressions/semiempirical/gfn-xtb-analytic-hessian/REFERENCE.md).

**VIB exception.** This slug (and `se-analytic-hessian` /
`gfn0-xtb-analytic-hessian`) may set `-DATV_BUILD_VIB=ON`. Do not enable
HF/MD/CUDA/MINIMIZER.
