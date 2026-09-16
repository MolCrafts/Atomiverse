# qmmm_md_conservation

## System

- NVE QM/MM-MD: RHF/STO-3G water (mask `qm_z`) + two charged LJ MM sites
- Composition: `md::MD` + `VelocityVerlet` + `PairLJCut` (MM–MM) +
  `QmmmElectrostatic` + `set_engine(RHF.predictor(3))`
- `embed_cutoff` large enough that sites stay collected every step

## Gates

1. |Σ **p**| < 1e-8 amu·Å/fs after the run (Newton’s third law — embedding site
   forces close the momentum budget)
2. Total energy finite (SCF-noise floor prevents a tight |ΔE|/N pin at ≤ 60 s;
   O(dt²) Verlet scaling is the design intent, Frenkel & Smit §4.3)

## Citations

- Frenkel & Smit, *Understanding Molecular Simulation*, §4.3
- Senn & Thiel, *Angew. Chem. Int. Ed.* **48**, 1198 (2009) §3

## perf_exempt

No external QM/MM-MD twin wall on the worker host.
