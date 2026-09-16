# H2 EA-ADC(2)/STO-3G — electron-attachment energy pin

Closed-shell **strict EA-ADC(2)** electron attachment (1p / 2p–1h manifold) for
the lowest root of H₂ / STO-3G. Product API: `RHF::ea_adc2(n_states)` after
`run(frame)`.

Literature / theory:

- J. Schirmer, *Beyond the random-phase approximation: A new approximation
  scheme for the polarization propagator*, Phys. Rev. A **26**, 2395 (1982).
  DOI [10.1103/PhysRevA.26.2395](https://doi.org/10.1103/PhysRevA.26.2395)
- Intermediate-state representation of the electron-attachment propagator
  through second order (strict ADC(2), not ADC(2)-x): second-order singles
  intermediate \(M_{ab}\) on the virtual (1p) space, first-order
  singles–doubles coupling via \((ia|bc)\), zeroth-order 2p–1h doubles
  diagonal.
- Implementation twin: PySCF `adc.ADC` with `method='adc(2)'`,
  `method_type='ea'` (Loos et al., J. Chem. Phys. **158**, 044112 (2023)).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂, charge 0, singlet |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — `pin_scf` `h2` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (no \(d\) shells; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| ADC | `pyscf.adc.ADC(mf)`, `method='adc(2)'`, `method_type='ea'`, `nroots≥1`, `conv_tol=1e-12` |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py ea-adc2 \
    --geom h2 --basis sto-3g --nstates 2 --conv-tol 1e-12
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396 Ha | 1e-6 Ha (reference) |
| \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) | −0.013138073590 Ha | 1e-6 Ha (diagnostic) |
| EA-ADC(2) EA0 \(\omega\) | 0.684281565504 Ha = **18.620249979933 eV** | **1e-4 eV** (product gate) |
| EA-ADC(2) EA1 \(\omega\) | 1.920840843660 Ha ≈ 52.269 eV | (diagnostic) |

CODATA \(1\,\mathrm{Ha}=27.211386245988\,\mathrm{eV}\)
(`units::kHartreeInEv`) — both sides of the eV gate use the same factor.

On this minimal basis the lowest attachment eigenvalue is endothermic
(~18.62 eV): the LUMO is unbound relative to the N-electron ground state.
The pin is the PySCF strict-EA-ADC(2) root above, not an experimental EA.

## Gates (product regression)

- EA-ADC(2) EA0: \(|\omega_{\mathrm{ours}}-\mathrm{pin}| < 10^{-4}\,\mathrm{eV}\).
- \(E_{\mathrm{RHF}}\) and MP2 \(E_{\mathrm{corr}}\): \(10^{-6}\) Ha.

## Performance

Fair wall-time protocol (`tools/pyscf_oracle/fair_bench.py`): same geom/basis,
median of ≥3 runs, ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\), target ≤ 10.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=8
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method ea-adc2 --geom h2 --basis sto-3g --nstates 1 --repeat 3 \
    ./build/regressions/reg_ea-adc2-h2-sto3g
'
```

Recorded 2026-07-23 (login node, `OMP_NUM_THREADS=8`, Release build):

| Field | Value |
|---|---|
| PySCF median wall | 0.735 s |
| ATV median wall | 0.157 s |
| `ratio_atv_over_pyscf` | **0.213** |
| `within_10x` | **true** |

```text
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method ea-adc2 --geom h2 --basis sto-3g --nstates 1 --repeat 3 \
  ./build/regressions/reg_ea-adc2-h2-sto3g
# → ratio_atv_over_pyscf=0.2135, within_10x=true
```
