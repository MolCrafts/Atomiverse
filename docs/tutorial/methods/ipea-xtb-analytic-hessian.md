# IPEA-xTB FD Hessian — Td CH4 vs xtb `--hess`

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ipea-xtb-analytic-hessian` |
| Regression | `regressions/semiempirical/ipea-xtb-analytic-hessian/` |
| Public API | `cpu::semiempirical::IPEA` (`with_forces = true`) on `cpu::vib::Hessian` at `vib::Phase::force` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md); [Ch. 14 Vibrations](../ch14-vibrations.md); [ipea-xtb-forces](ipea-xtb-forces.md) |
| Queue | TB method loop — wave 12 |
| Roadmap | `docs/roadmap/semiempirical.md` §3 IPEA-xTB analytic Hessian |
| Oracle | **xtb 6.7.1** only (family-locked; never OpenMOPAC / DFTB+) |

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

**IPEA-xTB energy** already shipped by `reg_ipea-xtb-small`:

\[
E = \mathrm{Tr}(P H_0)
  + \tfrac12 q\,\gamma\,q
  + \tfrac13 \Gamma q^3
  + E_{\mathrm{rep}}
  + E_{\mathrm{D3}}.
\]

Analytic \(\mathbf{F}=-\nabla E\) already ships (`ipea-xtb-forces`):
Hellmann–Feynman + Pulay, CN/shell-poly chains, \(\partial\gamma/\partial R\),
pair repulsion, D3(BJ). Extra \((n+1)s\) on C/N/O. Tables in
`basis_data/ipea_tables.h`. Unsupported \(Z\) throws.

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
pass. Masses for \(\omega\) come from IUPAC-2013 `mass_amu(Z)`.

## Public API

```cpp
cpu::semiempirical::IPEA ipea;
ipea.with_forces = true;
cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
hess.on(vib::Phase::force).add(std::move(ipea));
hess.build();
const cpu::vib::HessianResult res = hess.run(frame);
// res.hessian: [15,15] dense, eV/Å²; res.frequencies: 9 modes, cm⁻¹
```

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
| `include/atomiverse/cpu/vibration/hessian.h` | `cpu::vib::Hessian` FD preset |
| `regressions/semiempirical/ipea-xtb-analytic-hessian/` | Product gate |
| `tools/tb_oracle/inputs/xtb/ch4_ipea_hess/` | xtb 6.7.1 IPEA twin |
| `tests/cpu/semiempirical/test_xtb_ipea_hessian.cpp` | Unit: `Hessian::build` accepts IPEA (no `run`) |

## Validation (xtb)

Source:
[`regressions/semiempirical/ipea-xtb-analytic-hessian/REFERENCE.md`](../../../regressions/semiempirical/ipea-xtb-analytic-hessian/REFERENCE.md).
Oracle: **`xtb`** IPEA `--hess`. `oracle_version`: **6.7.1**. Never live
xtb from ctest.

| Quantity | xtb 6.7.1 |
| -------- | --------- |
| \(E\) | **−4.670476737294 Eh** |
| \(\omega\) | **1541.68×3, 1705.89×2, 3548.78, 3576.13×3** cm⁻¹ |
| \(H_{C_x,C_x}\) | **0.6813937129 Eh/\(a_0^2\)** |

`--vip`/`--vea` is **not** this pin (`ipea-xtb-vip`).
