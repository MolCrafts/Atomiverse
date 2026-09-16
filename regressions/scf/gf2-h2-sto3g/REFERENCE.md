# H₂ RHF/STO-3G one-shot GF2 (G0F2) quasiparticle gaps — reference values

## Method

Closed-shell **one-shot second-order Green's function** (G0F2 / GF2) on an
**RHF** reference: the 2h1p + 2p1h second-order self-energy is built in the
auxiliary (Lehmann) representation from the HF Green's function, then the
Dyson equation is solved **once** with the HF Fock matrix
\(F = \mathrm{diag}(\varepsilon_{\mathrm{HF}})\). Observable: HOMO / LUMO
quasiparticle energies and the **HOMO–LUMO QP gap [eV]** (also
IP \(= -\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\),
EA \(= \varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\)), plus the Galitskii–Migdal
correlation energy (equals canonical MP2 on a pure HF reference).

This is the Green's-function foothold for roadmap **GF2** (excited-states /
many-body table). It is **not** G0W0 (screened \(W\); bare \(v\) here),
**not** ADC(2), and **not** self-consistent AGF2 (outer density/Fock loop —
separate roadmap row). The regression pins the **one-shot** path of PySCF
`agf2.RAGF2` (SE from `init_gf()` → `se.get_greens_function(F_HF)`).

Public API:

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.run(frame);
cpu::scf::GF2Result gf = rhf.gf2();
// gf.gap_ev, gf.ip_ev, gf.ea_ev, gf.qp_energy_ha, gf.e_corr
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
- Self-energy: PySCF AGF2 construction (`os_factor = ss_factor = 1`,
  `weight_tol = 1e-11` to drop uncoupled poles)
- Dyson: single Hermitian diagonalization of the extended Fock matrix
  \([F, V; V^\dagger, \varepsilon_{\mathrm{aux}}]\)
- Chemical potential: binary search for \(N = 2\) electrons (RHF occupancy 2)

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py gf2 \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

### Verbatim twin

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, scf, agf2
import pyscf, numpy as np
HA2EV = 27.211386245988
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
            unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e = mf.kernel()
g1 = agf2.RAGF2(mf); g1.verbose = 0
eri = g1.ao2mo()
g0 = g1.init_gf()
se0 = g1.build_se(eri, g0)
fock = np.diag(mf.mo_energy)
from pyscf.agf2.chempot import binsearch_chempot
w, v = se0.eig(fock)
mu, _ = binsearch_chempot((w, v), g1.nmo, g1.nocc * 2)
g_one = agf2.GreensFunction(w, v[:g1.nmo], chempot=mu)
ip = g1.get_ip(g_one, nroots=1)[0][0]
ea = g1.get_ea(g_one, nroots=1)[0][0]
e_init = g1.energy_mp2(mf.mo_energy, se0)
print("pyscf", pyscf.__version__)
print("E_RHF", e)
print("IP_Ha", ip, "EA_Ha", ea)
print("gap_eV", (ip + ea) * HA2EV)
print("e_init", e_init)
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
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{GF2}}\) | **−0.591623589738** Ha |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{GF2}}\) | **+0.684213221821** Ha |
| **QP gap** | **34.717288266166 eV** |
| IP \(-\varepsilon_H^{\mathrm{QP}}\) | **16.098898012595 eV** |
| EA \(\varepsilon_L^{\mathrm{QP}}\) | **18.618390253572 eV** |
| \(E_{\mathrm{corr}}\) (GM / MP2) | **−0.013138073590** Ha |

SC-AGF2 companion (info only; not the regression pin): full `RAGF2.kernel()`
gap ≈ 34.451051 eV, e_corr ≈ −0.012920 Ha.

### Regression gates

- Primary: QP gap within **1e-4 eV** of 34.717288266166 eV
- IP / EA within 1e-4 eV
- QP orbital energies within 1e-5 Ha
- \(E_{\mathrm{RHF}}\) and \(E_{\mathrm{corr}}\) within 1e-6 Ha
- Sanity: gap > 0 and finite

## Performance

Fair bench vs PySCF one-shot twin (same geom/basis/method class):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method gf2 --geom h2 --basis sto-3g --repeat 5 \
    -- ./build/regressions/reg_gf2-h2-sto3g
'
```

Recorded 2026-07-23 (median of 5; `OMP_NUM_THREADS=1`):

| Field | Value |
|-------|-------|
| \(T_{\mathrm{pyscf}}\) median | 0.2449 s |
| \(T_{\mathrm{atv}}\) median | 0.0174 s |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **0.071** |
| within_10x | **true** (≪ 10.0) |
| max_ratio | 10.0 |
| notes | Regression binary includes RHF + one-shot GF2; PySCF twin is RHF+RAGF2 SE+Dyson |

## Citations

- Backhouse, Holzmann & Booth, J. Chem. Theory Comput. **17**, 3414 (2021)
  (PySCF auxiliary GF2).
- Galitskii & Migdal, Sov. Phys. JETP **7**, 96 (1958).
- Roadmap distinguishes **GF2** (one-shot / G0F2) from **self-consistent GF2**.
