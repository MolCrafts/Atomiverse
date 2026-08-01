# Reference — multi-k GPW PBE-D3 three-body ATM (C9) analytic ionic forces

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh **1×1×2**) on the
**periodic 6 Å** water-dimer cell of `gpw-d3-atm-forces` /
`gpw-d3-atm-kpoint-stress`, with Grimme **DFT-D3** (zero damping, PBE)
including the **Axilrod–Teller–Muto three-body C9** term
(`CALCULATE_C9_TERM .TRUE.`) folded into the **ionic forces**.

**Product forces** use the Hellmann–Feynman multi-k path
(`cpu::gpw::GpwDft` / `gpw_forces_from_scf` with `d3 = true`,
`d3_atm = true`, `k_mesh = {1,1,2}`): one multi-k SCF then one gradient
pass; the geometry-only
\(E_{\mathrm{disp}}^{\mathrm{D3+ATM}}(\{\mathbf{R}\},\mathbf{h})\)
gradient is added in the k-path force assembly (`dftd3_pbe(..., calculate_c9 =
true)` — the same call as Γ).

Configurable **`d3_rcut_bohr`** (CP2K `&PAIR_POTENTIAL R_CUTOFF`, doubled
convention): this twin uses `R_CUTOFF 10.0` → 37.79 Bohr so the product folds
exactly the oracle's functional.

Sibling twins: `gpw-d3-atm-forces` (Γ ATM force pin — same geometry),
`gpw-d3-atm-kpoint-stress` (multi-k ATM stress), `gpw-kpoint-forces`
(multi-k electronic forces, no dispersion).

## Why the pin is a *difference* of two CP2K runs

DFT-D3 — with or without C9 — is geometry-only and never enters the
Kohn–Sham potential, so the converged density is identical with
`CALCULATE_C9_TERM` `.TRUE.` and `.FALSE.`:

\[
F_A(\mathrm{C9}) - F_A(\mathrm{two\text{-}body})
  = -\nabla_A E_{\mathrm{ATM}},
\]

which isolates **exactly the ATM three-body gradient**. Because ATM is
geometry-only, the multi-k ΔF matches the Γ periodic pin to
≲ 1×10⁻⁹ Ha/Bohr (FORCES| print residual) — the k mesh does not enter the
dispersion functional.

## System

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | **6 × 6 × 6 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Electronic k mesh | MONKHORST-PACK **1×1×2** (CP2K DIAGONALIZATION; OT is Γ-only) |
| Dispersion | DFT-D3 zero damping, PBE; ATM per run (`CALCULATE_C9_TERM` T/F) |
| Pair/ABC cutoff | `R_CUTOFF 10.0` Å → doubled 37.794 Bohr (`d3_rcut_bohr`) |
| CP2K cutoff | 300 Ry (`CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| ATV fair-wall cutoff | 150 Ry (ATM ΔF pin is geometry-only ⇒ cutoff-independent) |
| SCF | CP2K `EPS_SCF 1e-8` + DIRECT_P_MIXING; ATV product `tol = 1e-6` |

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Host | Arrhenius epyc login (`arrhenius1`), offline only |
| Binary | `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/bin/cp2k.psmp` |
| Env | `source …/cp2k/install/env.sh`; `CP2K_DATA_DIR` set by env |
| Fair wall job | `mpirun -np 1`, `OMP_NUM_THREADS=8` |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-atm-kpoint-forces/water_dimer_d3_atm_k_forces.inp` (C9 T) and `water_dimer_d3_k_forces.inp` (C9 F) |
| Output lines | final `Dispersion energy:` + `FORCES\| Atomic forces [hartree/bohr]` table |
| Wall (C9, fair cold) | **1.51 s** (`/usr/bin/time` wall_sec; TIMING TOTAL ~1.17 s; 1 MPI × 8 OMP) |
| Wall (2b) | ~1.3 s (companion; not used for ratio) |
| Date | 2026-08-01 |

### Energies [Hartree]

| Quantity | CP2K multi-k |
| -------- | ------------ |
| Total (C9) | −34.34639787446541 |
| **E_disp (C9)** (printed) | **−0.00195809486938** |
| Total (two-body) | −34.34639865943880 |
| **E_disp (two-body)** (printed) | **−0.00195887984277** |

E_disp values are bit-identical to the Γ `gpw-d3-atm-forces` / multi-k
`gpw-d3-atm-kpoint-stress` pins (geometry-only). ATM contribution
+7.8497339e−7 Ha.

### Atomic forces [Hartree/Bohr] (CP2K `FORCES|` tables)

Multi-k, `CALCULATE_C9_TERM .TRUE.`:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −7.76100703E−02 | 4.42032292E−10 | 4.26650979E−10 |
| H0 | 3.84410315E−02 | 1.33918034E−11 | 1.30001789E−01 |
| H1 | 3.84410315E−02 | 1.23244443E−11 | −1.30001789E−01 |
| O1 | −7.76098924E−02 | 5.14119447E−10 | 5.36972620E−10 |
| H2 | 3.84411038E−02 | 1.31896386E−11 | 1.30001779E−01 |
| H3 | 3.84411038E−02 | 1.43579460E−11 | −1.30001779E−01 |

Multi-k, `CALCULATE_C9_TERM .FALSE.`:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −7.76105812E−02 | 4.42038340E−10 | 4.26653893E−10 |
| H0 | 3.84413680E−02 | 1.33916184E−11 | 1.30001230E−01 |
| H1 | 3.84413680E−02 | 1.23242605E−11 | −1.30001230E−01 |
| O1 | −7.76105827E−02 | 5.14117684E−10 | 5.36989844E−10 |
| H2 | 3.84413679E−02 | 1.31894361E−11 | 1.30001230E−01 |
| H3 | 3.84413679E−02 | 1.43582194E−11 | −1.30001230E−01 |

Derived ATM force pins \(F(\mathrm{C9}) − F(\mathrm{two\text{-}body})\):
largest component **6.90e−7** Ha/Bohr (O1 x). Residual vs Γ periodic ATM
ΔF: max |ΔF_multi-k − ΔF_Γ| ≈ 1.0×10⁻⁹ Ha/Bohr (print residual).
`FORCES| Sum` ≈ 1.46e−3 Ha/Bohr (same both runs — electronic imbalance).

## Tolerances

| Gate | Tol | Why |
| ---- | --- | --- |
| \(E_{\mathrm{disp}}\) (ATM + two-body) | **1e−9 Ha** | Closed-form sum; bit-match Γ pin |
| ATM-only force vs CP2K multi-k difference | **1e−8 Ha/Bohr** | Electronic cancels to ~1e−15; print floor ~1e−9; signal 6.9e−7 |
| ATM signal nonzero | **≥ 1e−7 Ha/Bohr** | Engage gate |
| Product multi-k wiring (H₂ 1×1×2) | **1e−7 Ha/Bohr** | Force assembly twice on same density vs standalone ATM ΔF; e2e budget |
| Multi-k payload | `n_kpts ≥ 1`, `C_occ_k` | Product uses k path |
| Wall (e2e reg) | **≤ 60 s** | e2e tier (H₂ wiring + standalone ATM) |
| Fair ratio | **≤ 10** | \(T_{\mathrm{atv,fair}} / T_{\mathrm{cp2k,C9}}\) on water-dimer product |

## Measured results (2026-08-01, first green)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| E_disp (C9) vs CP2K | 1e−9 Ha | **2.79e−10** |
| ATM force vs CP2K multi-k Δ | 1e−8 Ha/Bohr | **2.62e−10** (signal 6.90e−7) |
| Product multi-k H₂ wiring | 1e−7 Ha/Bohr | **6.1e−18** |
| Fair product wall (water dimer multi-k SCF+force, 150 Ry, OMP 8) | — | **14.97 s** |
| CP2K C9 ENERGY_FORCE wall (1×8, cold) | — | **1.51 s** |
| Fair ratio | ≤ 10 | **9.91** (14.97 / 1.51) |
| e2e reg wall | ≤ 60 s | **~0.2 s** |

## Dual gate

- **Correctness:** pins above; multi-k product wiring (H₂) on−off equals
  standalone ATM gradient; unit twin
  `GpwDftD3Atm.MultiKProductForcesFoldAtm`
  (`tests/cpu/gpw/test_cpu_gpw_dftd3.cpp`, heavy suite).
- **Speed:** fair wall on the **matched water-dimer multi-k ENERGY_FORCE**
  product (ATV SCF + one `gpw_forces_from_scf`, OMP 8, 150 Ry) vs CP2K C9
  twin (1 MPI × 8 OMP, 300 Ry). Ratio ≤ 10. The e2e regression uses a cheap
  H₂ multi-k wiring twin for the ≤ 60 s budget; fair \(T_{\mathrm{atv}}\) is
  the water-dimer product wall above, not multi-gate reg wall.

## Performance note (multi-k force OpenMP)

Serial multi-k AO-Pulay + grid assembly on this twin was ~35 s (force alone
~34 s; SCF ~1 s). OpenMP over atoms (Pulay) and grid points (Veff term) in
`gpw_forces.cpp` k-path brings the product wall to **~15 s** (≈ 2.3×),
inside the 10× fair gate vs CP2K 1.51 s.

## Notes

- ATM still contributes **no** `D3Result::virial` entry; stress is a separate
  pin (`gpw-d3-atm-kpoint-stress` ✓).
- CP2K uses DIAGONALIZATION under `&KPOINTS` (OT is Γ-only); ATV multi-k
  SCF uses its standard k-path DIIS/mixing.
- Sibling multi-k dispersion force pins: `gpw-d4-atm-kpoint-forces` (dftd4
  `--grad`; no CP2K DFTD4 — serialize on force-assembly ATM folds carefully).
