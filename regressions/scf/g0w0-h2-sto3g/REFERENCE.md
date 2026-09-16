# H₂ RHF/STO-3G spectral G0W0 quasiparticle gaps — reference values

## Method

Closed-shell **G0W0** on an **RHF** reference: frequency-exact (spectral)
correlation self-energy with **direct RPA (dRPA)** screening of \(W\).
Observable: HOMO / LUMO quasiparticle energies and the **HOMO–LUMO QP gap
[eV]** (also IP \(= -\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\),
EA \(= \varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\)).

This is the Green's-function foothold for roadmap **G0W0** (excited-states /
many-body table). It is **not** TDDFT, **not** TDA/CIS, and **not** the
analytic-continuation (AC) `gw.GW` / `GWAC` path (AC numbers are recorded
only as a near twin; the regression pins the spectral / exact-frequency
solver).

Public API:

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.run(frame);
cpu::scf::G0W0Result gw = rhf.g0w0();
// gw.gap_ev, gw.ip_ev, gw.ea_ev, gw.qp_energy_ha
```

## Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

## Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells)

## Convergence / solver

- Ground-state SCF: `conv_tol = 1e-12`
- Screening: full dRPA spectrum (`n_states = n_occ · n_vir = 1` on H₂/STO-3G)
- Self-energy: spectral sum, broadening \(\eta = 10^{-8}\) Ha
- QP equation: non-linear Newton, tolerance \(10^{-8}\) Ha (not linearized)

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py g0w0 \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

(`g0w0` builds `dft.RKS(mol, xc='hf')` then `gw.gw_exact.GWExact` + `tdscf.dRPA`
— spectral twin of Atomiverse `RHF::g0w0()`. Default `gw.GW` / AC is also
reported under `gap_ac_ev` for comparison.)

### Verbatim twin

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, dft, tdscf
from pyscf.gw import gw_exact
import pyscf
HA2EV = 27.211386245988
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
            unit="Angstrom", cart=True, verbose=0)
mf = dft.RKS(mol); mf.xc = "hf"; mf.conv_tol = 1e-12
e = mf.kernel()
td = tdscf.dRPA(mf); td.nstates = 1; td.kernel()
ge = gw_exact.GWExact(mf, tdmf=td); ge.verbose = 0; ge.kernel()
print("pyscf", pyscf.__version__)
print("E_RHF", e)
print("QP", ge.mo_energy)
print("gap_eV", (ge.mo_energy[1]-ge.mo_energy[0])*HA2EV)
print("IP_eV", -ge.mo_energy[0]*HA2EV)
print("EA_eV", ge.mo_energy[1]*HA2EV)
EOF
```

### Pinned output (2026-07-23)

| Quantity | Value |
|----------|-------|
| PySCF | 2.14.0 |
| \(E_{\mathrm{RHF}}\) | **−1.116759307396** Ha |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{HF}}\) | −0.578553859832 Ha |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{HF}}\) | +0.671143491914 Ha |
| HF gap | 34.005997328943 eV |
| dRPA \(\Omega_1\) | 1.570852128278 Ha = 42.745063997913 eV |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{G0W0}}\) | **−0.596957673490** Ha |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{G0W0}}\) | **+0.689547305573** Ha |
| **QP gap** | **35.007583892680 eV** |
| IP \(-\varepsilon_H^{\mathrm{QP}}\) | **16.244045825851 eV** |
| EA \(\varepsilon_L^{\mathrm{QP}}\) | **18.763538066829 eV** |
| AC twin gap (`gw.GW`) | 35.007177297967 eV (info only; ~0.4 meV off spectral) |

### Regression gates

- Primary: QP gap within **1e-4 eV** of 35.007583892680 eV
- IP / EA within 1e-4 eV
- QP orbital energies within 1e-5 Ha
- \(E_{\mathrm{RHF}}\) within 1e-6 Ha
- Sanity: G0W0 gap > HF gap on this system

## Performance

Fair bench vs PySCF spectral twin (same geom/basis/method class):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method g0w0 --geom h2 --basis sto-3g --repeat 5 \
    -- ./build/regressions/reg_g0w0-h2-sto3g
'
```

Recorded 2026-07-23 (median of 5; `OMP_NUM_THREADS=1`):

| Field | Value |
|-------|-------|
| \(T_{\mathrm{pyscf}}\) median | 0.1148 s |
| \(T_{\mathrm{atv}}\) median | 0.0125 s |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **0.109** |
| within_10x | **true** (≪ 10.0) |
| max_ratio | 10.0 |
| notes | Regression binary includes RHF + spectral G0W0; PySCF twin is RKS(hf)+dRPA+GWExact |

## Citations

- Hybertsen & Louie, Phys. Rev. B **34**, 5390 (1986), DOI 10.1103/PhysRevB.34.5390.
- van Setten et al., J. Chem. Theory Comput. **11**, 5665 (2015).
- PySCF GW: Zhu & Chan, J. Chem. Theory Comput. **17**, 727 (2021) (AC path);
  Berkelbach et al. spectral / exact-frequency implementation in `pyscf.gw.gw_exact`.
