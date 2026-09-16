# Pin notes — metalwalls-method-loop

## four-site-tip4p-foothold (2026-07-23)

- **Oracle:** LAMMPS `lj/cut/tip4p/cut` (units metal)
- **Binary:** `/home/jicli594/.local/x86_64/bin/lmp` — 30 Mar 2026 Development
- **Inputs:** `tools/metalwalls_oracle/inputs/lammps/four-site-tip4p-foothold/`
- **Command:** `lmp -in in.tip4p_dimer_metal -log log.metal`
- **Pin:** PE = 0.52924228894955327451 eV; forces in REFERENCE.md
- **Conversion:** metal units match ATV eV; ε_OO = 0.1550 kcal/mol × `kKcalPerMolInEv`

## born-repulsion-foothold (2026-07-24)

- **Oracle:** LAMMPS `pair_style born` (units metal, shift yes)
- **Binary:** `/home/jicli594/.local/x86_64/bin/lmp` — 30 Mar 2026 Development
- **Inputs:** `tools/metalwalls_oracle/inputs/lammps/in.born_dimer`
- **Command:** `lmp -in in.born_dimer`
- **Pin:** PE = 509.3671885909988 eV; F1x = -201.72730961087780 eV/Å
- **Formula:** A exp((σ−r)/ρ) − C/r⁶ + D/r⁸ (ρ softness length)
## fumi-tosi-nacl (2026-07-24)

- **Oracle:** LAMMPS `hybrid/overlay born + coul/cut` (units metal; `pair_modify pair born shift yes`)
- **Inputs:** `tools/metalwalls_oracle/inputs/lammps/in.fumi_tosi_nacl4`
- **Command:** `lmp -in in.fumi_tosi_nacl4`
- **Pin:** PE = 54.7707332302457 eV; forces in REFERENCE.md
- **ATV:** multi-type `cpu::PairBorn` + `cpu::PairCoulCut` on 4-ion NaCl square
## thomas-fermi-electrode (re-pinned 2026-07-29 against the real binary)

- **Oracle:** MetalWalls git `c8ad175d` binary, `mpirun -n 1 mw`
- **Why not LAMMPS:** ELECTRODE package is perfect conductor only (no \(l_{\mathrm{TF}}\))
- **Inputs:** `tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/mw_twin/`
  (`num_pbc 2`, `electrode_charges matrix_inversion`, `charge_neutrality true`;
  substitute `@LTF@` = `0.0` or `3.779452249251540` bohr)
- **Pin:** empty Au capacitor toy, \(\Delta\Psi=1\) V, \(\eta=1/0.56\) Å⁻¹, \(d=4/a_0^3\)
  — `total_charges.out`: \(Q_{\mathrm{top}}(l_{\mathrm{TF}}=0)=4.354124119406114\times 10^{-3}\,e\),
  \(Q_{\mathrm{top}}(l_{\mathrm{TF}}=2\,\mathrm{Å})=3.707555065364343\times 10^{-3}\,e\)
- **Superseded pin:** the wave-2 numbers \(4.620945546104\times10^{-3}\) /
  \(3.934601758679\times10^{-3}\) came from `pin_gold_cap.py` (deleted), which
  re-implemented `ConstPotBO`'s **own** truncated \(xy\)-image sum at
  `xy_image_range = 40`. That is a self-golden, and it is **+6.13 % high**: the
  truncated sum converges to the exact 2D-periodic elastance only as \(O(1/R)\).
  The TF model itself is correct; `xy_image_range` is now 1280 in the regression.
- **perf:** `ratio_atv_over_mw = 4.4` (0.885 s vs 0.20 s per elastance solve, 1 rank)

> **OPEN — same defect, different slug:** `regressions/md/voronoi-volume-tf`
> (`reg_voronoi-volume-tf.cpp:24,26,34`) reuses the identical toy at
> `kImageRange = 40` and the identical superseded numbers. It needs the same
> re-pin against the MetalWalls binary (per-site `voronoi_volume` twin).

## surface-tension-coupling (2026-07-24)
- oracle: lammps (`lmp` develop 30 Mar 2026)
- input: `inputs/lammps/in.surface_tension_coupling`
- pin: after 200 steps, Lx=Ly=16.527384, Lz=16.249165 (L0=15.3)
- Plat fixed = Pn - 2*gamma/Lz0 with gamma=0.001 eV/Å²
## dipole-electrode-joint (2026-07-24)
- **Oracle:** MetalWalls multipole + Gaussian electrode BO form (JCP 2022 / JOSS 2020)
- **Why not LAMMPS:** no SCF PIM ⊕ const-pot joint twin (ELECTRODE non-polarizable; Drude ≠ PIM)
- **Binary:** no `mw` on agent host → offline algebraic pin
- **Script:** `tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/pin_joint.py`
- **Command:** `python3 .../pin_joint.py`
- **Pin (joint α>0):** Q_top = +7.3285008958883610e-02 e; μ_cat,z = 5.8951916148990660e-02 e·Å; μ_an,z = 8.7432701391015569e-02 e·Å
- **Pin (α=0):** Q_top = +7.5339627935767609e-02 e (matches ConstPotBO)
- **perf_exempt:** no MetalWalls twin binary

## global-electroneutrality-bo (re-pinned 2026-07-29 against the real binary)

- **Oracle:** MetalWalls git `c8ad175d`, `mpirun -n 1 mw`
- **Inputs:** `inputs/metalwalls/global-electroneutrality-bo/`
- **Pin:** Q_E1 = -0.5189485789245821 e, Q_E2 = -0.4810514210754179 e, sum = -1.0 e
- **Primary:** global electroneutrality constraint; secondary per-electrode within 2e-5 at xy_image_range=1280
- **Commits:** a423ed1b, 961403b1

## voronoi-volume-tf (re-pinned 2026-07-29 against the real binary)

- **Oracle:** MetalWalls git `c8ad175d`, `mpirun -n 1 mw`
- **Inputs:** `inputs/metalwalls/voronoi-volume-tf/mw_twin/` (`@LTF@`/`@VORONOI@`)
- **Pin:** Q_top(l_tf=0)=4.354124119406114e-3; (l_tf=2,v=bulk)=3.707555065364343e-3; (l_tf=2,v=half)=3.321360199637453e-3
- **Superseded:** pin_voronoi_tf.py self-golden (~6.13% high at image_range=40)
- **Commit:** da688dac

## batch-2 remainder (2026-07-29) — d59e4350

- **const-pot-md-nvt-short:** MW pbc=3 |Q|=9.711e-2 e; ATV PME soft 5% pin
- **const-pot-il-edl-box empty:** MW |Q|=0.2076 e at im=80 (~0.5%); IL Ewald |Q|=4.82 e not matched by image-sum
- **extended-lagrangian-dipoles:** stationary thole_a=0 E_pol=-0.2440 eV / F=0.3493 eV/Å (same as pim bulk)
- **tip4p / maze / electrode-forces:** twins archived; absolute |Q| model gap documented
- **constant-d / displacement-dfield:** literature intentional
