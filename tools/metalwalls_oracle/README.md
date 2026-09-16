# Offline oracle for the MetalWalls capability loop

Mirrors `tools/pyscf_oracle/` and `tools/cp2k_oracle/`: produce **hard-coded
pins** for regressions under `metalwalls-method-loop`.

**ctest never launches LAMMPS or MetalWalls.**

## Binding oracle priority

| Condition | Primary offline oracle |
| --- | --- |
| Feature/algorithm **also exists in LAMMPS** | **LAMMPS** |
| Feature is **MetalWalls-only** (no LAMMPS twin) | **MetalWalls** |
| Neither binary available for the agent | **Literature** value with full citation; `perf_exempt` for speed |

## Both binaries are installed — literature fallback is no longer a default

**Do not pin MW-only physics from literature formulas.** Both oracle binaries
exist on this machine; the literature row above is now reachable only for
quantities neither code computes.

| Oracle | Binary | Version |
| --- | --- | --- |
| LAMMPS | `lmp` (on `PATH`) | see `lmp -h` |
| MetalWalls | `/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw` | git `c8ad175d` (release branch, 2026-06-18) |

Rebuild MetalWalls with `bash -l tools/metalwalls_oracle/build_metalwalls.sh`
(GNU + MPICH + OpenBLAS; the script ends with an upstream self-check that must
print `PASS` before any pin is taken from that build). Run it as:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cd <case-dir> && mpirun -n 1 /nobackup/.../work/opt/metalwalls/mw
'
```

MW writes `energies.out`, `forces.out`, `charges.out`, `energies_breakdown.out`
in **atomic units** (Hartree, Bohr, \(e\)); ATV pins are eV / Å — convert
explicitly in `REFERENCE.md` and show both.

### Upstream reference cases → ATV slugs

MetalWalls ships its own regression inputs with `*.ref` outputs. **Prefer
adapting one of these over hand-writing an input**, and cite which one:

| MW case | Covers ATV slug(s) |
| --- | --- |
| `tests/maze/electrodes/*`, `tests/maze/dipoles` | `maze-charge-dynamics`, `maze-symplectic-charge-dynamics`, `maze-molten-salt-al` |
| `tests/thomas_fermi/`, `example/thomas-fermi-gold-capacitor` | `thomas-fermi-electrode`, `voronoi-volume-tf` |
| `tests/pim/`, `example/EMIM_TFSI_bulk_polarizability` | `pim-induced-dipole-bulk`, `extended-lagrangian-dipoles` |
| `example/NaCl_PIM_NPT` | `pim-npt-polarizable-il` |
| `tests/dip_plus_elec/` | `dipole-electrode-joint` |
| `tests/piston/`, `example/python-piston-mgo` | `piston-electrode-pz` |
| `tests/external_field/`, `example/electric-field` | `finite-field-single-electrode`, `displacement-dfield-md`, `constant-d-polarization-feedback` |
| `tests/supercap/` | `const-pot-cg-large-electrode`, `electrode-charge-predictor`, `per-electrode-eta` |
| `tests/charge_neutrality/`, `tests/non_neutral/` | `global-electroneutrality-bo` |
| `example/EMIM_BF4_graphene`, `example/EMIM_TFSI_graphene_polarizability` | `const-pot-il-edl-box` |
| `example/nanoporous-graphene`, `example/blue-energy-cdc` | `nanoporous-carbon-const-pot` (pending) |
| `example/python-thermo_integration_charge` | `thermo-integration-capacitance` (pending) |
| `tests/four_site_model/`, `example/tip4p-water` | `four-site-tip4p-foothold`, `const-pot-tip4p-edl` |
| `tests/steele/`, `example/steele-argon` | `steele-wall-argon` |
| `tests/tosi_fumi/` | `fumi-tosi-nacl` |
| `tests/nist/` | SPC/E Ewald cross-checks |

Examples that **must** use LAMMPS when the binary is available offline:

- `pair_style lj/cut`, `lj/cut/coul/long`, `coul/long`
- `kspace_style pppm` / `ewald` / PME
- `bond_style harmonic`, angles, dihedrals, `special_bonds`
- `fix nve`, `fix nvt` (Nose–Hoover), `fix npt`
- `fix setforce 0` / freeze
- `fix efield`
- `pair_style lj/cut/coul/long` soft-core variants when matching ATV soft-core
- LAMMPS **ELECTRODE** / constant-potential packages **when installed** —
  prefer over MetalWalls for that twin

Examples that use **MetalWalls**:

- Gaussian electrode charge + BO / CG / MaZe as in MW
- Thomas–Fermi electrode metallicity
- PIM + electrode joint polarization
- Piston electrodes, MW 2D electrochemical demos without LAMMPS twin

## Contract

1. Human or agent runs **LAMMPS or MetalWalls offline** (login / batch) with a
   fixed input.
2. Record energy / forces / charges / temperature / capacitance into
   `REFERENCE.md` under the regression (cite code version, units, pair/kspace
   settings, timestep, ensemble).
3. Every `REFERENCE.md` must state:

   ```text
   oracle: lammps | metalwalls | literature
   oracle_version: <string>
   ```

4. Regression binary asserts Atomiverse vs those numbers within stated tol.
5. Optional fair wall: same geometry / comparable settings; write
   \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) into REFERENCE (not ctest).

## Suggested layout (grow as needed)

```text
tools/metalwalls_oracle/
  README.md           ← this file
  inputs/lammps/      ← minimal in.lammps twins (optional)
  inputs/metalwalls/  ← minimal MW runtime/data twins (optional)
  pin_notes.md        ← how pins were taken
  fair_bench.py       ← optional wall twin (later)
```

## MD-only build reminder

Workers in `metalwalls-method-loop` **must not** enable HF/PW/GPW/CUDA. See
`docs/developer/metalwalls-method-loop.md` Build section.

## Relation to capability map

Delivery slices **E1–E12**:
`.claude/notes/metalwalls-capability-map.md`.

Loop product pages: `docs/tutorial/methods/<slug>.md`.
