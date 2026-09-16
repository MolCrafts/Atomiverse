# Reference — H₂O / STO-3G restricted B2PLYP double-hybrid total energy

## Method

Restricted Kohn–Sham double hybrid **B2PLYP** (Grimme, *J. Chem. Phys.*
**124**, 034108 (2006), doi:10.1063/1.2148954) on a fixed water geometry in
the STO-3G Gaussian basis:

\[
E_{\mathrm{xc}}
  = (1-a_x)\,E_x^{\mathrm{B88}}
  + a_x\,E_x^{\mathrm{HF}}
  + (1-a_c)\,E_c^{\mathrm{LYP}}
  + a_c\,E_c^{\mathrm{MP2}},
\qquad a_x = 0.53,\quad a_c = 0.27.
\]

Concrete public API (no Config bag, no factory):

```cpp
cpu::scf::RKS rks(cpu::scf::XCType::B2PLYP);
rks.basis = cpu::scf::BasisSet::STO3G;
const double a_c = cpu::scf::hybrid_pt2_fraction(cpu::scf::XCType::B2PLYP);
rks.mp2_os_scale = a_c;
rks.mp2_ss_scale = a_c;
auto dh = rks.mp2(frame);   // dh.e_tot [Ha] = E_KS + a_c * E_MP2
```

- Hybrid KS semi-local part: `0.47 * B88 + 0.73 * LYP` via `evaluate_xc`.
- Exact exchange fraction: `hybrid_exchange_fraction(B2PLYP) = 0.53`.
- PT2 scale: `hybrid_pt2_fraction(B2PLYP) = 0.27`, applied through
  `mp2_os_scale` / `mp2_ss_scale` so both spin channels share a_c.
- MP2 is the closed-shell canonical RMP2 kernel on the converged KS orbitals
  and orbital energies (same composition as offline PySCF `mp.MP2(mf)`).

## Reference values

| Quantity | Value [Hartree] |
|---|---|
| E_KS (hybrid only) | −75.198832360592 |
| E_MP2 (unscaled) | −0.044726379786 |
| a_c · E_MP2 | −0.012076122542 |
| **E_B2PLYP total** | **−75.210908483134** |

at the pin geometry below (cart=True, conv_tol=1e-12).

Source: offline PySCF **2.14.0** pin. Hosted libxc in this environment does
not register a named `B2PLYP` composite, so the hybrid-KS part is composed as
`0.53*HF + 0.47*GGA_X_B88 + 0.73*GGA_C_LYP` and PT2 is
`a_c * mp.MP2(mf).e_corr` offline. External-engine oracle — not a golden
snapshot of Atomiverse output.

## Geometry

PySCF oracle preset `water` from `tools/pyscf_oracle/pin_scf.py` [Å]:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.000000 | 0.117300 |
| H | 0.000000 | 0.757200 | −0.469200 |
| H | 0.000000 | −0.757200 | −0.469200 |

20 Å cubic box (gas-phase cell only; does not enter the isolated-molecule SCF).

## Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True`; STO-3G has no d shells)

## Convergence

- Oracle SCF: `conv_tol = 1e-12`
- Atomiverse: `tol = 1e-10`, default MatchPySCF grid

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py b2plyp \
  --geom water --basis sto-3g --conv-tol 1e-12
```

### Pinned JSON excerpt (2026-07-23)

```json
{
  "a_c": 0.27,
  "a_x": 0.53,
  "basis": "sto-3g",
  "cart": true,
  "conv_tol": 1e-12,
  "converged": true,
  "e_corr_ha": -0.012076122542341092,
  "e_corr_mp2_ha": -0.044726379786448486,
  "e_corr_pt2_ha": -0.012076122542341092,
  "e_ks_ha": -75.19883236059151,
  "e_tot_ha": -75.21090848313385,
  "geom": "water",
  "method": "b2plyp",
  "pyscf": "2.14.0",
  "xc": "0.53*HF + 0.47*GGA_X_B88 + 0.73*GGA_C_LYP"
}
```

## Tolerance and its rationale

**1 × 10⁻⁶ Ha** on e_ks, e_mp2 (unscaled), a_c·E_MP2, and e_tot. Job gate for
this double-hybrid total-energy case: with the MatchPySCF grid and the same
geometry/basis, Atomiverse is expected to track the composed PySCF B2PLYP
energy to micro-Hartree.

## What is checked

`cpu::scf::RKS{XCType::B2PLYP}` with `mp2_os_scale = mp2_ss_scale = 0.27` and
`mp2(frame)` converges the hybrid KS reference, reports finite energies, and
matches the pinned e_ks / unscaled MP2 / PT2 / e_tot within 1e-6 Ha.
Reconstructs `e_corr == a_c * (e_corr_os + e_corr_ss)` and
`e_tot == e_ks + e_corr`.

## Performance

Fair wall-time vs offline PySCF (same geom/basis/method class; median of ≥3
runs). Protocol: `tools/pyscf_oracle/fair_bench.py`.

```bash
# OMP_NUM_THREADS=1 for a fair single-thread A/B on this tiny system
export OMP_NUM_THREADS=1 OMP_PROC_BIND=close
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method b2plyp --geom water --basis sto-3g --repeat 3 --max-ratio 10 \
  -- ./build/regressions/reg_b2plyp-h2o
```

| T_pyscf median [s] | T_atv median [s] | ratio T_atv/T_pyscf | within_10x |
|---|---|---|---|
| 0.291 | 0.509 | **1.75** | ✓ |

Worktree bench (2026-07-23): `ratio_atv_over_pyscf = 1.7475`, `within_10x = true`.
No optimize pass required (ratio ≤ 10).
