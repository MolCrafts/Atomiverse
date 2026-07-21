# debug_amorphous — Atomiverse vs LAMMPS amorphous-PE comparison

Reproduces the
[MSU amorphous polyethylene tension protocol][msu] (Stages 1 + 2 from
`amorphous.in`) on the Atomiverse CPU + MPI backend so the trajectories
can be put next to LAMMPS line-by-line.

[msu]: https://icme.hpc.msstate.edu/mediawiki/index.php/Deformation_of_Amorphous_Polyethylene.html

## Files

| File | Role |
|---|---|
| `PE_nc10_cl1000.dat` | LAMMPS data file: 10 chains × 1000 united atoms, 80.06 Å³ box |
| `amorphous.in` | LAMMPS input script (DREIDING-like force field, NVE/limit + Langevin Stage 1, NPT Stage 2) |
| `log.PE_nc10_cl1000.txt` | LAMMPS reference log (Stages 1 + 2 already run) |
| `lammps_loader.h` | C++ loader: LAMMPS data → `cpu::Frame` |
| `gather_frame.h` | Native-schema inter-stage all-gather; preserves runtime columns, topology, configured precision, and full-width tags |
| `run_atomiverse.cpp` | Main MPI driver |
| `run.sh` | Build-and-launch helper |
| `compare.py` | Side-by-side T / P / Lx comparison + plot |
| `CMakeLists.txt` | Subdirectory build (gated on `-DATV_BUILD_DEBUG_AMORPHOUS=ON`) |

## Force-field mapping

Both engines use V = K(r − r₀)² (no ½ factor), V = K(θ − θ₀)², V = Σ Cₙ
cosⁿ(φ).  All `kcal/mol` ↔ `eV` and `degree` ↔ `rad` conversions live
inside `run_atomiverse.cpp` so the LAMMPS `.in` numbers can be pasted in
verbatim.

| | LAMMPS (real units) | Atomiverse (eV/Å/fs/amu/K) |
|---|---|---|
| Bond | K=350, r₀=1.53 | K=15.18 eV/Å², r₀=1.53 |
| Angle | K=60, θ₀=109.5° | K=2.602 eV/rad², θ₀=1.911 rad |
| Dihedral | A=1.73,−4.49,0.776,6.99,0 (kcal/mol) | C=0.0750,−0.1947,0.0337,0.3032,0 (eV) |
| LJ | ε=0.112, σ=4.01, cut=10.5 | ε=4.857e-3 eV, σ=4.01 Å, cut=10.5 Å |

## Stage protocol

| Stage | LAMMPS | Atomiverse mapping |
|---|---|---|
| 1 | `fix nve/limit 0.05` + `fix langevin T=500 τ=10 fs`, dt=1 fs, 10 000 steps | `cpu::md::NVELimit{1, 0.05}` + `cpu::md::FixLangevin{500, 10, 1, 904297}` |
| 2 | `fix npt T=500 τ_T=50 iso P=0 τ_P=1000 drag 2` (mtchain=mpchain=3) + `fix momentum 1 linear 1 1 1`, dt=0.5 fs, 50 000 steps | `cpu::md::NoseHooverNPT{.T=500, .tau_T=50, .P=0, .tau_P=1000, .drag=2, .mtchain=3, .mpchain=3, .dt=0.5}` + `cpu::md::ZeroMomentum{1}` |

**Algorithm match rule.** Both stages use the exact same thermostat /
barostat family on both sides — Stage 1 Langevin vs Langevin, Stage 2
MTK-iso NHC vs MTK-iso NHC. No Langevin-vs-NHC or Berendsen-vs-NHC
mixing. `NoseHooverNPT` supports the LAMMPS `drag` flag and chain
lengths (`mtchain`, `mpchain`) so every tunable in `fix npt iso` maps
1:1.

## Build

```bash
cmake -B build -DATV_USE_MPI=ON -DATV_BUILD_DEBUG_AMORPHOUS=ON
cmake --build build --target run_atomiverse_amorphous -j
```

## Run

```bash
# Smoke test (1 000 + 500 steps, 4 ranks)
tools/debug_amorphous/run.sh --np 4 --steps-stage1 1000 --steps-stage2 500

# Full LAMMPS-equivalent protocol on 8 ranks (~5 minutes)
tools/debug_amorphous/run.sh --np 8

# Stage 1 only
tools/debug_amorphous/run.sh --np 4 --stage1-only --steps-stage1 10000

# Single-rank reference
tools/debug_amorphous/run.sh --np 1
```

The runner writes thermo every 100 steps to `log.atomiverse.txt` in a
LAMMPS-comparable format:

```
#       Step           Temp     Press[atm]       PotE[eV]       KinE[eV]     EtotE[eV]    Dens[g/cc]      Vol[A^3]        Lx        Ly        Lz
         0      10658.80  1.9296393e+08       15632920      13776.189       15646697     0.453705      513125.94    80.0586    80.0586    80.0586
       100        836.45      58125.806      6589.9213      1081.0836      7671.0048     0.453705      513125.94    80.0586    80.0586    80.0586
       ...
```

Columns mirror LAMMPS `thermo_style custom step temp press pe ke etotal
density vol lx ly lz`. The only unit difference is energy: LAMMPS
logs kcal/mol ('real' units); we log eV (`kKcalPerMolToEv = 1 /
23.060547830619026` is the conversion `compare.py` applies on read).

## Compare

```bash
python3 tools/debug_amorphous/compare.py
```

Output: `compare.png` with one row per observable, one column per
stage (T, P, PE, KE, E_total, ρ, v_rms), plus a text summary table
on stdout. `v_rms = sqrt(2·KE / (M_total·units::mass_v2_to_energy))` is derived identically
on both sides from the KE column, so it visualises the same data as
the KE panel but in Å/fs.

## Known caveats

- **Stage 1 step-0 offset**: LAMMPS reports step 0 BEFORE any integration
  (so T == the `velocity create` target, P == the raw initial virial);
  Atomiverse reports step 0 AFTER the first NVE/limit step, which clamps
  the displacement-per-atom to ≤ 0.05 Å and absorbs most of the initial
  LJ blow-up. The two engines agree closely from step ≳ 200 onward.
- **Pressure sign convention**: `frame.stress[0..2]` stores the
  compression-positive virial directly (Atomiverse matches LAMMPS).
  The runner's `ThermoPrinter` prints that sum as a positive pressure
  (compression > 0) so the output is line-comparable to LAMMPS's
  `thermo_style custom ... press`. The `Barostat` and `NoseHooverNPT`
  components apply their own sign-flipped formula internally — they
  see the same virial but subtract instead of add in the pressure
  expression; both outcomes (physical expansion under compression,
  physical compression under tension) match LAMMPS.
- **Dynamics divergence**: LAMMPS Langevin (RanMars stream, seed 904297)
  and Atomiverse's splitmix-per-(seed,step,tag) Langevin use independent
  random streams, so individual Stage 1 trajectories differ pointwise;
  ⟨T⟩ and ⟨P⟩ match within statistical noise. Stage 2's NHC NPT is
  deterministic on both sides, so Stage 2 should match pointwise once
  the Stage-1 RNG-induced divergence has been absorbed.
- **Inter-stage handoff**: each driver creates its own
  `DomainDecomposition`, so we explicitly `MPI_Allgatherv` the per-rank
  owners back to every rank between stages (`gather_frame.h`).
