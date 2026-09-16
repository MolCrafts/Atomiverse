# H2CO ADC(2)/STO-3G — S1 excitation energy pin

Closed-shell **strict ADC(2)** electronic excitation (singlet EE manifold) for
the lowest root of formaldehyde / STO-3G. Product API: `RHF::adc2(n_states)`
after `run(frame)`.

Literature / theory:

- J. Schirmer, *Beyond the random-phase approximation: A new approximation
  scheme for the polarization propagator*, Phys. Rev. A **26**, 2395 (1982).
  DOI [10.1103/PhysRevA.26.2395](https://doi.org/10.1103/PhysRevA.26.2395)
- A. B. Trofimov, J. Schirmer, *An efficient polarization propagator approach
  to valence electron excitation spectra*, J. Phys. B **28**, 2299 (1995).
  DOI [10.1088/0953-4075/28/12/003](https://doi.org/10.1088/0953-4075/28/12/003)
- Strict ADC(2) (not ADC(2)-x): second-order singles intermediate \(M_{ss}\),
  first-order singles–doubles coupling, zeroth-order doubles diagonal.
- Implementation twin: PySCF `adc.ADC` with `method='adc(2)'`,
  `method_type='ee'` (Loos et al., J. Chem. Phys. **158**, 044112 (2023)).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂CO (formaldehyde), charge 0, singlet |
| Geometry (Å) | C(0,0,0); O(0,0,1.21); H(0,±0.943,−0.587) — `pin_scf` `h2co` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (no \(d\) shells; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| ADC | `pyscf.adc.ADC(mf)`, `method='adc(2)'`, `method_type='ee'`, `nroots≥1`, `conv_tol=1e-12` |
| AO / MO | 12 AO; 8 occupied, 4 virtual spatial MOs |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py adc2 \
    --geom h2co --basis sto-3g --nstates 3 --conv-tol 1e-12
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −112.353990369522 Ha | 1e-6 Ha (reference) |
| \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) | −0.117214030417 Ha | 1e-6 Ha (diagnostic) |
| ADC(2) S1 \(\omega\) | 0.166669832218 Ha = **4.535317180049 eV** | **1e-4 eV** (product gate) |
| ADC(2) S2 \(\omega\) | 0.371358612315 Ha ≈ 10.105 eV | (diagnostic) |
| ADC(2) S3 \(\omega\) | 0.508095678166 Ha ≈ 13.826 eV | (diagnostic) |

CODATA \(1\,\mathrm{Ha}=27.211386245988\,\mathrm{eV}\)
(`units::kHartreeInEv`) — both sides of the eV gate use the same factor.

On this minimal basis the S1 (\(n\to\pi^*\)) is ~4.54 eV; CIS(D) on the same
geometry/basis is 4.545 eV (`reg_cisd-h2co-sto3g`). The pin is the PySCF
strict-ADC(2) root above, not an experimental band origin.

## Gates (product regression)

- ADC(2) S1: \(|\omega_{\mathrm{ours}}-\mathrm{pin}| < 10^{-4}\,\mathrm{eV}\).
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
    --method adc2 --geom h2co --basis sto-3g --nstates 1 --repeat 3 \
    -- -- ./build/regressions/reg_adc2-h2co-sto3g
'
```

Recorded 2026-07-23 (login node, `OMP_NUM_THREADS=8`, Release build):

| Field | Value |
|---|---|
| PySCF median wall | 0.247 s |
| ATV median wall | 0.066 s |
| `ratio_atv_over_pyscf` | **0.267** |
| `within_10x` | **true** |

```text
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method adc2 --geom h2co --basis sto-3g --nstates 1 --repeat 3 \
  ./build/regressions/reg_adc2-h2co-sto3g
# → ratio_atv_over_pyscf=0.2669, within_10x=true
```
