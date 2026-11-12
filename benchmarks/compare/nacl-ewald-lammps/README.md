# Small dense Coulomb MD: Atomiverse vs LAMMPS

512-ion molten NaCl (Petravic & Delhommelle, J. Chem. Phys. 118, 7477
(2003)): Tosi–Fumi Born + real-space Coulomb + direct Ewald. The reciprocal
sum dominates a step this small, and every parallel region is short, so this
is the case where per-pair cost and thread overhead show.

| Arm | Atomiverse | LAMMPS |
|---|---|---|
| Model | `PairBorn` + `CoulLong` + `KSpaceEwald`, `GearPC` + Gauss isokinetic | `born/coul/long` + `ewald`, `fix nvt` |
| Binary | `bench_nacl_ewald` (`benchmarks/perf`) | `in.nacl` |

No results are kept here. A performance claim is the pair of tables `run.py`
prints on one named node, with the compiler, the LAMMPS version and the
thread binding.

## Cases

| | `liquid` | `fluid` |
|---|---|---|
| T, L, dt | 1500 K, 25.4904 Å, 1 fs | 3200 K, 49.8957 Å, 0.5 fs |

- `paper`: κ = 1.8π/L, Born and real-space cutoff L/2, reciprocal sphere
  |k| ≤ 6·2π/L (accuracy e^{−100/9}), skin 2 Å. The k-vector count does not
  depend on L.
- `accuracy`: cutoff 8 Å, α = 3/8 Å⁻¹, accuracy 1e-6, skin 0.5 Å. The
  sphere has a fixed radius in Å⁻¹, so its k-vector count grows with the
  volume (≈ 7.5× from liquid to fluid).

LAMMPS cuts a cubic cell's sphere at an integer radius in units of 2π/L
(`kmax/ewald`), so the written script rounds the radius to the nearest
integer and both counts are printed. Atomiverse keeps the boundary shell of
`paper` out (its accuracy is rounded up), LAMMPS keeps it in (462 against
447); with `accuracy` LAMMPS's sphere is the slightly smaller one (2787
against 3015 in the liquid, 22236 against 22776 in the fluid).

LAMMPS has no Gear predictor–corrector or Gauss thermostat; `fix nvt`
stands in. Integration is a small share of either engine's step.

## Run

```sh
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_BENCHMARKS=ON -DATV_BUILD_MD=ON ...
cmake --build "$BUILD" --target bench_nacl_ewald
python3 run.py --atv atv="$BUILD/bin/bench_nacl_ewald" --lmp "$LMP" --out "$OUT" \
    --threads 1 4 8 16 64 --steps 1000 --warm 200
python3 run.py --summarize --out "$OUT"
```

- Repeat `--atv LABEL=PATH` to time several builds (before / after a change)
  in the same allocation.
- `run.py` sets `OMP_NUM_THREADS` per point; LAMMPS runs threaded points
  with `-sf omp -pk omp N` (needs the OPENMP package).
- `--thermostat config` times Atomiverse's configurational thermostat
  instead of the Gauss one; it starts in `liquid` only (`--states liquid`),
  since the expanded `fluid` lattice has a negative ∇²U.
- Run the whole matrix inside one batch job, one engine after the other.
  Record `OMP_PROC_BIND` / `OMP_PLACES`: on a shared node, unbound OpenMP
  threads can stall each barrier for a scheduler time slice.
