# H2O IP-ADC(2)/STO-3G — lowest ionization potential pin

Closed-shell **strict ADC(2)** ionization potential (IP manifold, 1h / 2h1p
ISR) for the lowest root of water / STO-3G. Product API:
`RHF::ip_adc2(n_states)` after `run(frame)`.

Literature / theory:

- J. Schirmer, *Beyond the random-phase approximation: A new approximation
  scheme for the polarization propagator*, Phys. Rev. A **26**, 2395 (1982).
  DOI [10.1103/PhysRevA.26.2395](https://doi.org/10.1103/PhysRevA.26.2395)
- J. Schirmer, L. S. Cederbaum, O. Walter, *New approach to the one-particle
  Green's function for finite Fermi systems*, Phys. Rev. A **28**, 1237 (1983).
- Strict ADC(2) (not ADC(2)-x): second-order occupied–occupied intermediate
  \(M_{ij}\), first-order 1h–2h1p coupling, zeroth-order 2h1p diagonal.
- Implementation twin: PySCF `adc.ADC` with `method='adc(2)'`,
  `method_type='ip'` (Loos et al., J. Chem. Phys. **158**, 044112 (2023)).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂O (water), charge 0, singlet |
| Geometry (Å) | O(0,0,0.1173); H(0,±0.7572,−0.4692) — `pin_scf` `water` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (no \(d\) shells; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| ADC | `pyscf.adc.ADC(mf)`, `method='adc(2)'`, `method_type='ip'`, `nroots≥1`, `conv_tol=1e-12` |
| AO / MO | 7 AO; 5 occupied, 2 virtual spatial MOs |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py ip-adc2 \
    --geom water --basis sto-3g --nstates 5 --conv-tol 1e-12
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −74.963023138463 Ha | 1e-6 Ha (reference) |
| \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) | −0.035545651647 Ha | 1e-6 Ha (diagnostic) |
| IP-ADC(2) IP1 | 0.304180883162 Ha = **8.277183500364 eV** | **1e-4 eV** (product gate) |
| IP-ADC(2) IP2 | 0.395907981066 Ha ≈ 10.773 eV | (diagnostic) |
| IP-ADC(2) IP3 | 0.604352918337 Ha ≈ 16.445 eV | (diagnostic) |
| IP-ADC(2) IP4 | 1.191953547066 Ha ≈ 32.435 eV | (diagnostic) |
| IP-ADC(2) IP5 | 1.403753901455 Ha ≈ 38.198 eV | (diagnostic) |

CODATA \(1\,\mathrm{Ha}=27.211386245988\,\mathrm{eV}\)
(`units::kHartreeInEv`) — both sides of the eV gate use the same factor.

On this minimal basis the lowest IP is ~8.28 eV (valence). The pin is the
PySCF strict IP-ADC(2) root above, not an experimental vertical IP.

## Gates (product regression)

- IP1: \(|\mathrm{IP}_{\mathrm{ours}}-\mathrm{pin}| < 10^{-4}\,\mathrm{eV}\).
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
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method ip-adc2 --geom water --basis sto-3g --nstates 1 --repeat 3 \
    -- -- ./build/regressions/reg_ip-adc2-h2o-sto3g
'
```

Recorded 2026-07-23 (login node, `OMP_NUM_THREADS=8`, Release build):

| Field | Value |
|---|---|
| PySCF median wall | 2.766 s |
| ATV median wall | 0.528 s |
| `ratio_atv_over_pyscf` | **0.191** |
| `within_10x` | **true** |

```text
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method ip-adc2 --geom water --basis sto-3g --nstates 1 --repeat 3 \
  ./build/regressions/reg_ip-adc2-h2o-sto3g
# → ratio_atv_over_pyscf=0.1907, within_10x=true
```
