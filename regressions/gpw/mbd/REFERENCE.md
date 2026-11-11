# gpw/mbd

One case; every section below is run by `atv_regressions --case gpw/mbd` and keeps its own oracle record.


## Reference — water-dimer GPW PBE-MBD@rsSCS dispersion energy

### Method

**Many-body dispersion MBD@rsSCS** as an additive correction on top of Γ-point
**GPW** (PBE), free-atom volume ratios \(V/V_{\mathrm{free}}=1\). Ice-relevant
H-bonded water dimer (same geometry as `gpw-ts-ice` / `gpw-pbe-d3bj-ice`).

Public API: `cpu::gpw::GpwDft::mbd = true` (and `GPWEvaluator::mbd`).
Dispersion energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`.

Product path evaluates **isolated** free-atom MBD@rsSCS (geometry-only; basis
independent). Periodic Ewald + k-mesh MBD is a follow-on.

Formula (Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014); libMBD
Hermann et al., *J. Chem. Phys.* **159**, 174802 (2023)):

1. Range-separated self-consistent screening of dynamical polarizabilities
   (Fermi × Gaussian dipole tensor, \(\beta\) damping).
2. Long-range MBD Hamiltonian diagonalization → zero-point energy difference.

PBE: \(\beta = 0.83\), Fermi steepness \(a = 6\). Free-atom \(\alpha_0\), \(C_6\),
\(R_{\mathrm{vdW}}\) from libMBD `ts_vdw_params` (in-tree `basis_data/ts_tables.h`):

| Z | \(\alpha_0\) [Bohr³] | \(C_6\) [Ha·Bohr⁶] | \(R_{\mathrm{vdW}}\) [Bohr] |
| - | -------------------- | ------------------ | -------------------------- |
| H | 4.5 | 6.5 | 3.1 |
| O | 5.4 | 15.6 | 3.19 |

### Reference value (hard gate)

**E_disp (PBE-MBD@rsSCS free-atom, isolated water dimer) =
−1.4928804171407961×10⁻³ Ha**

Cross-checks (offline — never in ctest):

| Source | Geometry | E_disp [Ha] |
| ------ | -------- | ----------- |
| **libMBD** Fortran `cmbd_mbd_scs_energy` (git `88069c6`) | isolated, ATV Å→Bohr | **−1.4928804171407961×10⁻³** |
| pure-Python `pymbd.mbd_energy` | isolated, ATV Å→Bohr | −1.4928804171425725×10⁻³ |
| pure-Python (libmbd Å→Bohr 0.529177249) | isolated | −1.4928806007885598×10⁻³ |

Difference pure-Python vs Fortran \(\sim 2\times10^{-15}\) Ha (same ATV conversion).

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD is via external libMBD when linked. Hard
pin is **libMBD**, same policy as `gpw-ts-ice`. Twin notes under
`tools/cp2k_oracle/inputs/gpw-mbd-ice/`.

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (SCF only): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
MBD pin is isolated free-atom on the same nuclear coordinates.

Basis / PP (GPW regression; E_disp is basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests use **1e-12 Ha** on the
standalone `mbd_rsscs_pbe` kernel.

### Perf gate

MBD evaluation is \(O(N^3)\) eigensolve of a \(3N\times 3N\) matrix on a 6-atom
system (microseconds). Full GPW SCF wall is dominated by the electronic
structure. **perf_exempt = true**: primary deliverable is E_disp pin parity
(same policy as `gpw-ts-ice` / `gpw-pbe-d3bj-ice` / `gpw-d4-water`).


## Reference — water-dimer GPW PBE periodic Ewald + k-mesh MBD@rsSCS

### Method

**Periodic many-body dispersion MBD@rsSCS** (Ewald dipole tensors + Brillouin-zone
k-mesh average) as an additive correction on top of Γ-point **GPW** (PBE), free-atom
volume ratios \(V/V_{\mathrm{free}}=1\). Ice-relevant H-bonded water dimer
geometry (same nuclei as `gpw-mbd-ice` / `gpw-ts-ice`) in a **periodic** cubic cell.

Public API:

```cpp
GpwDft gpw{kinds, /*cutoff_ry=*/200.0, XcKind::PBE};
gpw.mbd = true;
gpw.mbd_k_mesh[0] = gpw.mbd_k_mesh[1] = gpw.mbd_k_mesh[2] = 2; // periodic
```

Default `mbd_k_mesh={0,0,0}` keeps free-atom **isolated** MBD (`reg_gpw-mbd-ice`).
Dispersion energy on `GpwResult::e_disp` [Hartree], folded into `e_total`.

Formula (Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014); libMBD
Hermann et al., *J. Chem. Phys.* **159**, 174802 (2023)):

1. Range-separated self-consistent screening of dynamical polarizabilities
   over lattice images (Fermi × Gaussian dipole tensor, no Ewald on this step).
2. Long-range MBD Hamiltonian with **Ewald** dipole tensor, eigenvalues averaged
   over a shifted k-mesh (libMBD `get_kpts`, shift \(1/2\)).

PBE: \(\beta = 0.83\), Fermi steepness \(a = 6\). Free-atom \(\alpha_0\), \(C_6\),
\(R_{\mathrm{vdW}}\) from libMBD `ts_vdw_params` (in-tree `basis_data/ts_tables.h`).

### Reference value (hard gate)

**E_disp (PBE-MBD@rsSCS free-atom volumes, periodic 12 Å cube, k = 2×2×2) =
−1.4965720704989138×10⁻³ Ha**

Cross-checks (offline — never in ctest):

| Source | Geometry | E_disp [Ha] |
| ------ | -------- | ----------- |
| **pure-Python pymbd** `mbd_energy` (libmbd tree, ATV Å→Bohr) | periodic, k=2×2×2 | **−1.4965720704989138×10⁻³** |
| same, k=1×1×1 | periodic | −1.4474832766229184×10⁻³ |
| same, k=3×3×3 | periodic | −1.5008181948612531×10⁻³ |
| isolated free-atom (no lattice) | `reg_gpw-mbd-ice` / libMBD Fortran | −1.4928804171407961×10⁻³ |

Difference periodic (2×2×2) vs isolated \(\sim 3.7\times10^{-6}\) Ha on this box
(weak image coupling at 12 Å).

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD is via external libMBD when linked. Hard
pin is **pure-Python pymbd** (same algorithm as libMBD Fortran), same policy as
`gpw-mbd-ice`. Twin notes under `tools/cp2k_oracle/inputs/gpw-mbd-periodic/`.

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell: **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
MBD uses the same cell with **k-mesh 2×2×2** (libMBD shift-½).

Basis / PP (GPW regression; E_disp is basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests use **1e-10 Ha** on the
standalone `mbd_rsscs_pbe` periodic kernel.

### Perf gate

MBD evaluation is \(O(N_k \cdot (3N)^3)\) on a 6-atom system (milliseconds).
Full GPW SCF wall is dominated by the electronic structure. **perf_exempt =
true**: primary deliverable is E_disp pin parity (same policy as
`gpw-mbd-ice` / `gpw-ts-ice` / `gpw-pbe-d3bj-ice`).


## Reference — water-dimer GPW PBE + Hirshfeld-scaled MBD@rsSCS

### Method

**Hirshfeld volume ratios** \(V/V_{\mathrm{free}}\) from the final Γ-point
**GPW PBE** valence density, used to density-scale free-atom **MBD@rsSCS**
parameters (Tkatchenko & Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009);
Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014)).

Public API:

```cpp
GpwDft gpw{kinds, /*cutoff_ry=*/200.0, XcKind::PBE};
gpw.mbd = true;
gpw.hirshfeld = true;   // post-SCF volumes → mbd_rsscs_pbe(…, vol_ratio)
```

Dispersion energy on `GpwResult::e_disp` [Hartree]; ratios on
`GpwResult::volume_ratios`. Free-atom path (`hirshfeld=false`) remains
geometry-only (`reg_gpw-mbd-ice`).

Free-atom valence densities (in-tree exponential model,
`basis_data/hirshfeld_tables.h`):

\[
\rho_Z^{\mathrm{free}}(r)=A\exp(-\alpha_Z r),\quad
A=N_{\mathrm{val}}\alpha_Z^3/(8\pi),\quad
V_Z^{\mathrm{free}}=60\,N_{\mathrm{val}}/\alpha_Z^3
\]

H: \(N=1\), \(\alpha=2.0\); O: \(N=6\), \(\alpha=2.2\) (Bohr⁻¹).

Scaling of free-atom α₀ / C₆ / R_vdW (libMBD tables): \(\alpha\propto v\),
\(C_6\propto v^2\), \(R\propto v^{1/3}\). MBD@rsSCS is **isolated** on the
nuclear coordinates (periodic Ewald MBD follow-on).

### Reference value (hard gate)

**E_disp (PBE-MBD@rsSCS, Hirshfeld volumes, water dimer, CUTOFF 200 Ry) =
−1.1479852469005181×10⁻³ Ha**

Cross-checks (offline — never in ctest):

| Source | Note | E_disp [Ha] |
| ------ | ---- | ----------- |
| **ATV product** (`GpwDft` mbd+hirshfeld, CUTOFF 200, tol 1e-6) | dual gate | **−1.1479852469005181×10⁻³** |
| Free-atom MBD (volume_ratio = 1) | `reg_gpw-mbd-ice` / libMBD Fortran | −1.4928804171407961×10⁻³ |
| Dual pure-Python Hirshfeld (same α tables) | free-atom limit on isolated O | \(v\approx 1\) (unit test) |

Hirshfeld scaling reduces |E_disp| vs free-atom (~23% less attractive on this
geometry), as expected for compressed molecular volumes (\(v_A\in[0.58,0.74]\)).

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no TS/MBD. Offline twin notes under
`tools/cp2k_oracle/inputs/gpw-mbd-hirshfeld/`. Free-atom MBD already pinned vs
libMBD Fortran (`gpw-mbd-ice`); this slug pins the **volume-ratio layer**.

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (SCF): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
MBD pin is isolated on the same nuclear coordinates with Hirshfeld \(v_A\).

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).
CUTOFF **200 Ry**, max_iter 16, tol 1e-6, mix 0.4, PBE.

Representative volume ratios at the pin (order O,H,H,O,H,H):

`0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784`

### Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests: free-atom limit \(v\approx 1\)
to 3e-2; volume-scaled MBD dual to 1e-15; product path physical bounds.

### Perf gate

**perf_exempt = true** — primary deliverable is E_disp + volume-ratio parity.
Hirshfeld grid integral is \(O(N_{\mathrm{grid}} N_{\mathrm{atoms}})\); SCF dominates
wall (same policy as `gpw-mbd-ice` / `gpw-ts-ice`).


## Reference — water-dimer GPW PBE-MBD@rsSCS ionic forces

### Method

**Many-body dispersion MBD@rsSCS** ionic forces on top of Γ-point **GPW**
(PBE), free-atom volume ratios \(V/V_{\mathrm{free}}=1\). Same water dimer as
`gpw-mbd-ice` / `gpw-ts-ice`.

Product path:

1. `mbd_rsscs_pbe` evaluates \(E_{\mathrm{disp}}\) matching libMBD
   (`cmbd_mbd_scs_energy` / pure-Python).
2. Forces \(F_A = -\partial E_{\mathrm{disp}}/\partial R_A\) [Ha/Bohr] from the
   same geometry-only energy (central FD product kernel, \(h=10^{-4}\) Bohr).
3. `assemble_gpw_forces` with `GpwDft::mbd = true` **adds** these forces to the
   electronic Hellmann–Feynman / Pulay terms (no longer energy-only).

Public API: `cpu::gpw::GpwDft::mbd = true` (and `GPWEvaluator::mbd`) with
`compute_forces=true` / `gpw_forces_*`.

PBE: \(\beta = 0.83\), Fermi \(a = 6\). Free-atom \(\alpha_0\), \(C_6\),
\(R_{\mathrm{vdW}}\) from `basis_data/ts_tables.h` (libMBD `ts_vdw_params`).

### Reference values (hard gates)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp** | libMBD Fortran `cmbd_mbd_scs_energy` (git `88069c6`, ATV Å→Bohr) | **−1.4928804171407961×10⁻³ Ha** |
| **Force dual-gate** | product \(F\) vs independent central FD \(h=10^{-5}\) Bohr of \(E_{\mathrm{disp}}\) | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| Translation | \(\|\sum_A F_A\|\) | ≤ **1×10⁻⁸** Ha/Bohr |

#### Offline force sample (isolated free-atom, product kernel)

Printed by the regression (geometry-only; not a CP2K absolute pin):

| Atom | \(F_x\) | \(F_y\) | \(F_z\) [Ha/Bohr] |
| ---- | ------- | ------- | ----------------- |
| O0 | (runtime) | ~0 | ~0 |
| … | … | … | … |

Cross-check: pure-Python `pymbd` / libMBD analytic gradients agree with ATV
product forces to \(O(h^2)\) of the FD step (offline — never in ctest).

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD is via external libMBD when linked. Hard
pin is **libMBD energy** + **force dual-gate**, same policy as `gpw-mbd-ice` /
`gpw-ts-ice`. Twin notes under `tools/cp2k_oracle/inputs/gpw-mbd-forces/`.

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (SCF only): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
MBD pin is **isolated free-atom** on the same nuclear coordinates
(`mbd_k_mesh={0,0,0}`).

Basis / PP (GPW short SCF; E_disp and F_MBD are basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Tolerance

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| | **1×10⁻⁸** Ha |
| max\|F_product − F_fd\| | **1×10⁻⁶** Ha/Bohr |
| \|Σ F\| | **1×10⁻⁸** Ha/Bohr |

### Perf gate

MBD energy is \(O(N^3)\) on a \(3N\times 3N\) matrix; force FD is \(6N\) energy
evals (still microseconds on 6 atoms). Full GPW SCF wall dominates if present.
**perf_exempt = true**: primary deliverable is E_disp + force dual-gate parity
(same policy as `gpw-mbd-ice` / `gpw-ts-ice`).


## Reference — Γ GPW analytic cell stress with periodic MBD@rsSCS

### Method

Γ-point **GPW** (PBE) on a periodic water dimer with **MBD@rsSCS** many-body
dispersion (free-atom polarizabilities, PBE \(\beta = 0.83\), Fermi steepness
\(a = 6\), MBD lattice sum on a 2×2×2 mesh).

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path:

1. One SCF at the unstrained cell → \(P\), energy-weighted \(W\).
2. Central FD of the non-self-consistent Lagrangian
   \(E_L[P;W;\varepsilon]\) under orthorhombic diagonal strain at fixed
   fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{MBD}}(\{\mathbf R\},\mathbf h)\) **rebuilt at
   every strained cell** (same `mbd_rsscs_pbe` call as the energy path).
3. \(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; \(E\) in Hartree, \(\Omega\) in Bohr³).

Rebuilding at each strain is the whole point: holding \(E_{\mathrm{disp}}\)
fixed would differentiate a different functional than the SCF converged — the
failure mode that shipped once as a 0.53 Ha/Bohr ZnO force error.

Public path: `cpu::gpw::GPWEvaluator` with `mbd = true`,
`mbd_k_mesh = {2,2,2}`, `compute_stress = true`, `stress_use_fd = false`
→ Frame `kStress` [eV/Å³]; compared here via `last_stress_ha_bohr3()`.

### Oracles

| Gate | Oracle |
| ---- | ------ |
| \(E_{\mathrm{disp}}\) | offline **pure-Python pymbd** `mbd_energy` — the same pin `gpw-mbd-periodic` uses, so the energy being differentiated is externally validated |
| \(\sigma\) | **multi-SCF strain-FD** (`gpw_stress_fd`): full re-SCF at each strained cell, MBD folded through the energy path |
| MBD actually folded | \(\sigma(\text{mbd}) \ne \sigma(\text{plain})\) |

The stress itself has no external twin: CP2K ships no MBD@rsSCS stress class,
and pymbd exposes energies rather than a GPW cell derivative. The strain-FD
oracle is therefore the correct correctness statement for the *new* capability
("the analytic stress is the strain derivative of the functional we converged"),
with the functional pinned externally. Same structure as `gpw-d4-stress`.

### System

Same geometry as `gpw-mbd-periodic` / `gpw-d4-stress`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Dispersion | MBD@rsSCS, free-atom volumes, \(\beta = 0.83\), \(a = 6\), MBD k-mesh 2×2×2 |
| Cutoff | 200 Ry |
| Strain step | \(h = 2\times10^{-3}\) |

Placement is verbatim: a periodic cell wraps by minimum image. (An isolated
MT run would have to be centred — see `gpw-d3-forces/REFERENCE.md`.)

### Pin

**\(E_{\mathrm{disp}}\) = −1.4965720704989138 × 10⁻³ Ha**

Source: offline pure-Python **pymbd** `mbd_energy` (libmbd tree), periodic,
MBD k-mesh 2×2×2, Å→Bohr as in Atomiverse. Identical value and provenance as
`regressions/gpw/mbd`; twin notes under
`tools/cp2k_oracle/inputs/gpw-mbd-periodic/`. Never launched at ctest time.

### Measured results (2026-07-26)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs pymbd | 1e−8 Ha | **1.85e−10** |
| \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **7.39e−07** |
| \(\lvert\sigma\rvert\) nonzero | > 1e−10 | **5.82e−05** |
| MBD shift \(\lvert\sigma(\mathrm{mbd})-\sigma(\mathrm{plain})\rvert\) | > 1e−10 | **1.72e−07** |
| Off-diagonals (orthorhombic) | exactly 0 | **0** |
| `kStress` unit conversion | 1e−9 | exact |
| Product wall | ≤ 60 s | **23.3 s** |

Per-axis: \(\sigma_{xx} = -2.185\times10^{-5}\) (FD \(-2.259\times10^{-5}\)),
\(\sigma_{yy} = 2.007\times10^{-5}\), \(\sigma_{zz} = 5.823\times10^{-5}\)
Ha/Bohr³.

### Tolerance justification

- **\(E_{\mathrm{disp}}\) 1e−8 Ha:** deterministic rsSCS solve against an
  independent implementation of the same algorithm; measured 1.9e−10.
- **Analytic vs FD 2e−5 Ha/Bohr³:** the two differ by the fixed-\(P\)
  (frozen density-matrix) approximation plus SCF residual at `tol = 1e-6`, not
  by the dispersion treatment. Same tier as `gpw-d4-stress`; measured 7.4e−7,
  i.e. 27× inside the gate. The \(xx\) axis carries the largest residual
  because the dimer axis is \(x\), so its strain moves the O–O separation most.
- **MBD shift floor 1e−10:** the MBD stress contribution is ~1.7e−7 Ha/Bohr³
  here; the floor only has to exclude "silently not folded", which is exactly
  what the multi-k path used to do.

### Dual gate

- **Correctness:** the gates above, plus unit gates in
  `tests/cpu/gpw/test_cpu_gpw_stress.cpp` — `AnalyticMatchesFdWithMbd`
  (6.1e−9 Ha/Bohr³ on cheap MT H₂), `MbdChangesGammaDiagonal`, and
  `MbdMultiKStressThrows`.
- **Speed:** no MBD-stress oracle binary exists to time against, so this case is
  **perf_exempt**. For scale: the product analytic pass is 23.3 s versus 23.2 s
  for its own multi-SCF FD twin on this small cell — the fixed-\(P\) path's
  advantage grows with atom count, since the FD twin needs a full re-SCF per
  strain sample.

### Scope

**Not covered here:** multi-k MBD stress — shipped separately as
`regressions/gpw/mbd` (the k-summed fixed-\(P_k\)
Lagrangian folds the same geometry-only \(E_{\mathrm{disp}}\), gated by
strained-cell pymbd pins + the multi-SCF strain-FD dual gate).

### Notes

- Sibling MBD products: `gpw-mbd-ice` (isolated energy), `gpw-mbd-periodic`
  (periodic energy), `gpw-mbd-forces`, `gpw-mbd-kpoint-forces`,
  `gpw-mbd-hirshfeld`.
- Sibling dispersion stress: `gpw-d4-stress`.


## Reference — multi-k GPW analytic forces with MBD@rsSCS

### Method

**Monkhorst–Pack multi-k GPW** ionic forces with additive **MBD@rsSCS**
(geometry-only free-atom volumes). Closes the residual that Γ
`assemble_gpw_forces` already folded MBD (`gpw-mbd-forces`) while
`assemble_gpw_forces_kpoints` counted MBD for mutual exclusion but **skipped**
adding \(F_{\mathrm{MBD}}\).

Product path:

1. Multi-k SCF (`GpwDft::k_mesh`) supplies electronic \(P_k\), \(W_k\).
2. Analytic k-summed Hellmann–Feynman / Pulay assembly (the `GpwDft`
   multi-k force path).
3. Geometry-only \(F_{\mathrm{MBD}} = -\nabla E_{\mathrm{MBD}}\) from
   `mbd_rsscs_pbe` is **added** on the multi-k path (identical switch as Γ:
   isolated when `mbd_k_mesh={0,0,0}`; Ewald + MBD k-mesh when non-zero).

Public API: `GpwDft::mbd = true` + multi-k mesh + `GpwDft::forces(frame)` /
`GPWEvaluator(std::move(dft), /*compute_forces=*/true)`. No new symbols.

PBE MBD: \(\beta = 0.83\), Fermi \(a = 6\). Free-atom \(\alpha_0\), \(C_6\),
\(R_{\mathrm{vdW}}\) from `basis_data/ts_tables.h` (libMBD `ts_vdw_params`).

### Reference values (hard gates)

#### A — Geometry-only MBD (water dimer, isolated free-atom)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp** | libMBD Fortran `cmbd_mbd_scs_energy` (git `88069c6`, ATV Å→Bohr) | **−1.4928804171407961×10⁻³ Ha** |
| **Force dual-gate** | product \(F\) vs independent central FD \(h=10^{-5}\) Bohr | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| Translation | \(\|\sum_A F_A\|\) | ≤ **1×10⁻⁸** Ha/Bohr |

Same nuclei / pin policy as `gpw-mbd-forces` / `gpw-mbd-ice`.

#### B — Multi-k product wiring (H₂, k-mesh 1×1×2)

Compressed H₂: PADE, DZV-GTH-PADE,
`k_mesh={1,1,2}`, free-atom MBD (`mbd_k_mesh` default zero).

| Quantity | Gate |
| -------- | ---- |
| SCF converged; multi-k payload (`n_kpts ≥ 1`, `C_occ_k` non-empty; Γ has `n_kpts==0`) | hard |
| \(F^{k+\mathrm{MBD}}-F^{k,\mathrm{plain}}\approx F_{\mathrm{MBD}}^{\mathrm{geom}}\) | max\|Δ\| ≤ **1×10⁻⁸** Ha/Bohr |
| \(\|F_{\mathrm{MBD},0x}^{\mathrm{geom}}\|\) nontrivial | > **1×10⁻⁸** Ha/Bohr |

Geometry-only MBD does not change the SCF density, so the force difference
must match the standalone `mbd_rsscs_pbe` forces bit-for-bit within tol.

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD is via external libMBD when linked. Hard
pin is **libMBD energy** + **force dual-gate** + **multi-k wiring**, same
policy as sibling MBD slugs. Twin notes under
`tools/cp2k_oracle/inputs/gpw-mbd-kpoint-forces/`.

### Geometry

#### Water dimer (Å) — MBD pin

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

#### H₂ (Bohr) — multi-k wiring

| Atom | x | y | z |
| ---- | - | - | - |
| H | 5.5 | 6.0 | 6.0 |
| H | 6.5 | 6.0 | 6.0 |

Cell (H₂ SCF): **12 × 12 × 12 Bohr**, periodic; electronic **k = 1×1×2**.

### Tolerance

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| (water dimer) | **1×10⁻⁸** Ha |
| max\|F_product − F_fd\| | **1×10⁻⁶** Ha/Bohr |
| \|Σ F_MBD\| | **1×10⁻⁸** Ha/Bohr |
| max\|F_k+MBD − F_k − F_geom\| | **1×10⁻⁸** Ha/Bohr |

### Perf gate

MBD is \(O(N^3)\) geometry-only; multi-k H₂ force assembly is a single
gradient pass after one SCF. Full CP2K QS wall is not a fair twin (no stock
MBD pair type). **`perf_exempt`** — same policy as `gpw-mbd-forces`.


## Reference — multi-k GPW analytic cell stress with periodic MBD@rsSCS

### Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with **MBD@rsSCS** many-body dispersion (free-atom
polarizabilities, PBE \(\beta = 0.83\), Fermi steepness \(a = 6\), MBD
lattice sum on its own 2×2×2 mesh). Closes the residual that the Γ fixed-\(P\)
stress Lagrangian already rebuilt \(E_{\mathrm{disp}}^{\mathrm{MBD}}\) at
every strained cell (`gpw-mbd-stress`) while the **k-summed** Lagrangian
(`energy_fixed_P_k`) refused MBD with a throw.

Product cell stress is the Hellmann–Feynman fixed-\(P_k\) path:

1. One multi-k SCF at the unstrained cell → \(\{P_k, W_k\}\).
2. Central FD of the non-self-consistent k-summed Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{MBD}}(\{\mathbf R\},\mathbf h)\) **rebuilt at
   every strained cell** (same `mbd_rsscs_pbe` call as the Γ path and the
   energy path).
3. \(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; \(E\) Hartree, \(\Omega\) Bohr³).

Public path: `cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `mbd = true`,
`mbd_k_mesh = {2,2,2}`, `compute_stress = true`, `stress_use_fd = false` →
Frame `kStress` [eV/Å³]; compared here via `last_stress_ha_bohr3()`.

### Oracles

| Gate | Oracle |
| ---- | ------ |
| A. \(E_{\mathrm{disp}}\) unstrained | offline **pure-Python pymbd** `mbd_energy_species` — the same pin `gpw-mbd-periodic` / `gpw-mbd-stress` use; MBD is independent of the electronic k mesh, so the multi-k run must reproduce the Γ pin |
| B. \(E_{\mathrm{disp}}\) **under strain** (±h per axis) | fresh offline pymbd pins at each strained geometry (below) — the strain derivative being folded is externally validated point by point |
| C. \(\sigma\) | **multi-SCF strain-FD** (`gpw_stress_fd`): full re-SCF at each strained cell on the same 1×1×2 mesh, MBD folded through the energy path |
| D. MBD actually folded | \(\sigma(\mathrm{mbd}) - \sigma(\mathrm{plain})\) equals the pymbd strain-FD term \(-\Delta E_{\mathrm{disp}}/(2h\Omega)\) per axis |

CP2K 2026.1 stock `PAIR_POTENTIAL TYPE` has no MBD (only DFTD2/D3/D3BJ/D4),
so there is no CP2K stress twin; the strain-FD dual gate plus the externally
pinned dispersion term is the correctness statement for the new fold — same
structure as `gpw-mbd-stress`, strengthened by gate B/D external pins on the
strained cells.

### System

Same geometry as `gpw-mbd-periodic` / `gpw-mbd-stress`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Electronic k mesh | MONKHORST-PACK **1×1×2**, time-reversal folded |
| Dispersion | MBD@rsSCS, free-atom volumes, \(\beta = 0.83\), \(a = 6\), MBD k-mesh 2×2×2 |
| Cutoff | 120 Ry (E_disp is geometry-only — cutoff does not enter the pins) |
| Strain step | \(h = 2\times10^{-3}\) |

### Pins

Source: offline **pure-Python pymbd** `mbd_energy_species` (libmbd git
`88069c6`, `src/pymbd/pymbd.py`), free-atom volume ratios 1, β = 0.83, MBD
k-grid 2×2×2, Å→Bohr = 1/0.52917720859. Reproduce:
`tools/cp2k_oracle/inputs/gpw-mbd-kpoint-stress/pin_mbd_strain.py`
(never launched at ctest time). Output lines:

```
E_disp unstrained            = -1.4965720704989138e-03 Ha
E_disp x +h              = -1.4925684743305467e-03 Ha
E_disp x -h              = -1.5005781658326001e-03 Ha
sigma_MBD_xx (FD, pymbd)    = -1.7171785426e-07 Ha/Bohr3
E_disp y +h              = -1.4965139646472281e-03 Ha
E_disp y -h              = -1.4966306893944292e-03 Ha
sigma_MBD_yy (FD, pymbd)    = -2.5024338482e-09 Ha/Bohr3
E_disp z +h              = -1.4957889518587564e-03 Ha
E_disp z -h              = -1.4973531298823062e-03 Ha
sigma_MBD_zz (FD, pymbd)    = -3.3534037337e-08 Ha/Bohr3
```

The unstrained value is bit-identical to the `gpw-mbd-periodic` pin
(provenance cross-checked at pin time, 2026-07-31).

### Tolerances

| Check | Tol |
| ----- | --- |
| A/B \(\lvert E_{\mathrm{disp}} - \mathrm{pymbd}\rvert\) (unstrained + 6 strained) | 1×10⁻⁸ Ha |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2×10⁻⁵ Ha/Bohr³ (Γ MBD reg tier) |
| C \(\lvert\sigma\rvert\) nonzero | > 1×10⁻¹⁰ |
| D \(\lvert(\sigma_{\mathrm{mbd}}-\sigma_{\mathrm{plain}}) - \sigma^{\mathrm{MBD}}_{\mathrm{pymbd}}\rvert\) | 2×10⁻⁹ Ha/Bohr³ |
| E off-diagonals (orthorhombic) | exactly 0 |
| E `kStress` unit conversion | 1×10⁻⁹ |
| Product wall | ≤ 60 s |

### Perf gate

CP2K 2026.1 stock has no MBD pair potential, so no fair oracle wall exists
for an MBD stress run: **`perf_exempt`** — same policy as `gpw-mbd-stress` /
`gpw-mbd-kpoint-forces`.

### Measured results (2026-07-31, first green)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| B \(E_{\mathrm{disp}}\) under strain vs pymbd (6 cells) | 1e−8 Ha | **1.78e−15** |
| A \(E_{\mathrm{disp}}\) multi-k vs Γ pin | 1e−8 Ha | **1.07e−11** |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **1.36e−09** |
| D wiring vs pymbd FD term | 2e−9 Ha/Bohr³ | **4.00e−12** |
| E off-diagonals | exactly 0 | **0** |
| E `kStress` conversion | 1e−9 | exact |
| Product wall | ≤ 60 s | **13.6 s** (FD twin 22.0 s; SCF 11 iters, nk = 1) |

σ_diag (mbd, analytic, 120 Ry): xx −3.622e−05, yy −8.570e−05,
zz −3.117e−06 Ha/Bohr³; MBD shifts: xx −1.7171e−07, yy −2.500e−09,
zz −3.353e−08 Ha/Bohr³ — matching the pymbd FD terms per axis.
