# Reference — periodic water-dimer GPW PBE + vdW-DF (DRSLL) nonlocal correlation

## Method

**Nonlocal vdW-DF correlation** (Dion–Rydberg–Schröder–Langreth–Lundqvist /
**DRSLL** / libvdwxc **DF1**) as a post-SCF additive on top of Γ-point **GPW
PBE**. Ice-relevant H-bonded water dimer in a periodic 12 Å cell (same geometry
as `gpw-ts-ice` / `gpw-pbe-d3bj-ice` / `gpw-mbd-ice`).

Public API: `cpu::gpw::GpwDft::vdw_df = true` (and `GPWEvaluator::vdw_df`).
Nonlocal energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`.

Formula (Dion et al., *Phys. Rev. Lett.* **92**, 246401 (2004); Román-Pérez &
Soler, *Phys. Rev. Lett.* **103**, 096102 (2009)):

\[
E_c^{\mathrm{nl}}
  = \tfrac12 \iint \rho(\mathbf r)\,\phi(\mathbf r,\mathbf r')\,\rho(\mathbf r')
    \,\mathrm d\mathbf r\,\mathrm d\mathbf r'
\]

with \(q_0(\rho,|\nabla\rho|^2)\) (DF1 \(Z_{ab}=-0.8491\), \(q_{\mathrm{cut}}=5\))
and the community RPS kernel tables (libvdwxc 0.5.0 / QE). Evaluation is
**post-SCF** on the final total density (non-self-consistent potential);
spectral \(\lvert\nabla\rho\rvert^2\) from the GPW FFT gradient.

CP2K twin surface: `&VDW_POTENTIAL` / `POTENTIAL_TYPE NON_LOCAL` /
`&NON_LOCAL TYPE DRSLL` (additive on the chosen XC).

## Reference value (hard gate)

**E_disp (PBE density + DRSLL E_c^{nl}, periodic water dimer, 12 Å cell,
CUTOFF 200 Ry, DZVP-MOLOPT-SR-GTH + GTH-PBE) =
+1.5194390480303990×10⁻¹ Ha**

Cross-checks (offline — never in ctest):

| Source | Geometry / density | E [Ha] |
| ------ | ------------------ | ------ |
| **libvdwxc 0.5.0** `VDWXC_DF1` (synthetic 16³ Gaussian blob, L=8 Bohr) | fixed ρ,σ | **+1.2367739353586624×10⁻¹** |
| Atomiverse `vdw_df_drsll` (same synthetic) | unit test | matches libvdwxc to **&lt; 1e−6 Ha** |
| Atomiverse GPW PBE + `vdw_df` (this regression) | post-SCF ρ | **+1.5194390480303990×10⁻¹** |

Note: pure \(E_c^{\mathrm{nl}}\) on a molecular density can be positive; the
dispersion *binding* is recovered in interaction differences. Pairwise D3/TS
energies are a different functional form and are **not** comparable in
magnitude.

### Why not a live CP2K QS wall?

**CP2K** is not available on the pin host for a fair full-QS twin wall this
wave. The kernel is pinned to **libvdwxc** (the same engine CP2K links when
`libvdwxc` is enabled). Twin input notes under
`tools/cp2k_oracle/inputs/gpw-vdw-df-ice/`.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell: **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

## Tolerance

**5 × 10⁻⁵ Ha** on E_disp (regression). Unit tests use **1e−6 Ha** on the
standalone `vdw_df_drsll` kernel vs libvdwxc.

## Perf gate

Kernel RPS is \(O(N_g \log N_g \times n_\alpha)\) with \(n_\alpha=20\); dominated
by the electronic SCF on the 6-atom cell. **perf_exempt = true**: primary
deliverable is E_disp pin parity (same policy as `gpw-ts-ice` / `gpw-mbd-ice`).
No fair CP2K QS wall on the pin host.
