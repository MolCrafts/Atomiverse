# semiempirical/dftb0

One case; every section below is run by `atv_regressions --case semiempirical/dftb0` and keeps its own oracle record.


## Reference — H₂ non-SCC DFTB0 energy vs DFTB+ (mio-1-1)

```text
oracle: dftb+
oracle_version: 24.1
```

### Method

Non-SCC DFTB / **DFTB0** (band structure of H⁰ + pair repulsive; no SCC):

1. Slater–Koster H⁰ / S from the transcribed mio-1-1 H–H grid
   (`basis_data/dftb_mio_hh_sk.h`, read through
   `internal/mio_hh_table.h` — the same grid, interpolation and Bohr/Hartree
   convention the SCC path uses)
2. Generalized eigen \(H_0 C = S C \varepsilon\)
3. Closed-shell occupation \(n_\mathrm{occ} = n_e/2\)
4. \(E = \sum_i 2\varepsilon_i + E_\mathrm{rep}\), with \(E_\mathrm{rep}\) the
   mio-1-1 H–H repulsive spline

Reference method: DFTB0 / non-SCC DFTB — Elstner et al., *Phys. Rev. B* **58**,
7260 (1998), DOI 10.1103/PhysRevB.58.7260 (the zeroth-order band-structure +
repulsive term of that paper, before the second-order charge-fluctuation
expansion). **Not** OpenMOPAC; **not** xtb.

### Why an SCC oracle run pins a non-SCC method

The offline oracle transcript is a **DFTB2 (SCC)** DFTB+ run. It is a valid
DFTB0 oracle *for this system only*, and by construction rather than by
assumption:

- H₂ is homonuclear, so the two atomic sites are related by symmetry and the
  Mulliken charge fluctuations vanish exactly: \(\Delta q_A = \Delta q_B = 0\).
- The DFTB2 second-order energy is
  \(E_\mathrm{2nd} = \tfrac12 \sum_{AB} \gamma_{AB}\,\Delta q_A \Delta q_B\),
  which is then identically **zero**, and the SCC shift
  \(\sum_B \gamma_{AB}\Delta q_B\) added to H⁰ is likewise zero, so the
  converged Hamiltonian *is* H⁰ and the converged eigenvalues *are* the
  non-SCC eigenvalues.
- DFTB2 therefore reduces exactly to \(E = \sum_i 2\varepsilon_i +
  E_\mathrm{rep}\) — the DFTB0 expression — and the DFTB+ total is a DFTB0
  total.

The oracle transcript states this independently: its `detailed.out`
decomposition is band \(-0.6807883427\) H + repulsive \(+0.0058374104\) H +
SCC \(0\) H (Δq = 0), summing to the pinned total. ATV's own DFTB2 path
reproduces it with converged \(|\Delta q| = 4\times10^{-16}\,e\).

This licence is **specific to homonuclear diatomics**. A heteronuclear DFTB0
pin needs its own non-SCC DFTB+ run.

### Offline oracle run

Carried by the sibling regression
`regressions/semiempirical/dftb2/` (same oracle binary,
same parameter set, same geometry); inputs at
`tools/tb_oracle/inputs/dftbplus/h2_mio_scc/`. Nothing is executed at ctest
time — the numbers below are transcribed constants.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | −0.6749509324 | Ha |
| Total energy | **−18.36635052** | eV |
| Band energy \(\sum_i 2\varepsilon_i\) | −0.6807883427 | Ha |
| Repulsive | +0.0058374104 | Ha |
| SCC (second-order) | 0 (Δq = 0) | Ha |
| Conversion | 1 Ha = 27.211386245988 eV | — |

### Data

In-tree static tables, nothing parsed at build/test/run time:
`basis_data/dftb_mio_hh_sk.h` (H–H, transcribed DFTB.org mio-1-1 `H-H.skf`)
and the H onsite \(E_s\) / Hubbard U rows in `basis_data/dftb_mio_tables.h`
(also mio-1-1). Unsupported Z throws at lookup — no fallback.

### Geometry

Gas-phase H₂, bond along \(x\), \(R = 0.74\) Å (and \(R = 3.0\) Å for
non-flatness). Cubic box 30 Å (isolation only). Charge 0, singlet.

| Atom | x    | y | z |
|------|------|---|---|
| H    | 0.00 | 0 | 0 |
| H    | 0.74 | 0 | 0 |

### Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E(R=0.74\,\text{Å})\) | **−18.36635052** | eV | \(1\times10^{-3}\) eV |
| \(|E(0.74)-E(3.0)|\) | \(> 10^{-6}\) | eV | pipeline not flat |
| Force skeleton at \(R=0.90\) Å | \(\|F\|^2 > 0\) | eV²/Å² | finite, non-zero |

**Tolerance rationale.** \(1\times10^{-3}\) eV is the sibling DFTB2 pin's
tolerance for the *same* oracle number; gating the same transcript twice at
two different precisions would claim a precision the transcript does not
carry (it is printed to \(10^{-10}\) Ha ≈ \(2.7\times10^{-9}\) eV). It also
sits ~230× below the parameter-fidelity defect this gate exists to catch —
reading mio-*style* sketch H–H parameters instead of the transcribed mio-1-1
grid moved this energy by 0.229 eV — while leaving room for host BLAS /
eigensolver variation. Observed residual on the CPU reference host:
\(|E_\mathrm{ATV} - E_\mathrm{pin}| \approx 3\times10^{-9}\) eV, bit-identical
to the DFTB2 path's own energy at this geometry (both −18.3663505230 eV),
which is the symmetry argument above holding numerically.

### Perf

Offline DFTB+ binary is not available on the build host; fair wall
\(T_\mathrm{atv}/T_\mathrm{oracle}\) not measured here → **perf_exempt**
(the sibling pin records ~0.06 s ATV vs ~0.15 s DFTB+ cold for the same
molecule).


## Reference — DFTB0 analytic nuclear forces vs DFTB+ non-SCC

```text
oracle: dftb+
oracle_version: 24.1
```

### Method

Non-SCC DFTB / **DFTB0**. Energy is the band energy of \(H^0\) plus the
pair repulsive,

\[
E = \sum_i 2\varepsilon_i + E_\mathrm{rep}.
\]

Forces are the closed-form gradient of that energy at the non-SCC density
(Hellmann–Feynman + Pulay + repulsive). There is no Hubbard \(\gamma\), no
\(\Delta q\), and no SCC shift on the overlap. The pair walk is one
minimum image, the same geometry the energy uses. Not a \(6N\) finite
difference.

Reference: Elstner et al., *Phys. Rev. B* **58**, 7260 (1998),
DOI 10.1103/PhysRevB.58.7260, the zeroth-order band + repulsive piece,
before the second-order charge fluctuation. Force-module boundary:
Hourahine et al., *J. Chem. Phys.* **152**, 124101 (2020),
DOI 10.1063/1.5143190. **Not** OpenMOPAC. **Not** xtb.

### Offline oracle run

DFTB+ **24.1**, mio-1-1, `SCC = No`, `PrintForces = Yes`. Nothing is
executed at ctest time. Inputs and archived `detailed.out` transcripts:

| Case | Path |
| ---- | ---- |
| H₂, \(R = 0.74\) Å | `tools/tb_oracle/inputs/dftbplus/dftb0-analytic-forces/h2_074/` |
| H₂, \(R = 0.90\) Å | `tools/tb_oracle/inputs/dftbplus/dftb0-analytic-forces/h2_090/` |
| off-eq H₂O | `tools/tb_oracle/inputs/dftbplus/dftb0-analytic-forces/h2o/` |
| tilted CO | `tools/tb_oracle/inputs/dftbplus/dftb0-analytic-forces/co/` |

H₂ at 0.74 Å is also the homonuclear reduction of the SCC transcript in
`h2_mio_scc/` (Δq = 0). The numbers below are from a fresh **non-SCC**
run, not from that reduction. Water and CO are heteronuclear: an SCC run
is a different Hamiltonian and is not a DFTB0 pin.

Conversion, from `core/units.h`: 1 Ha = 27.211386245988 eV,
1 Ha/Bohr = 51.42206747467393 eV/Å. DFTB+ prints forces in atomic units
(`Maximal derivative component: … au`).

The force/charge section expresses the archived GEN coordinates in the same
atomic units before evaluating ATV: multiply each archived Å coordinate by
`units::kBohrInAngstrom / 0.529177249`. DFTB+ 24.1 uses the latter Bohr→Å
constant in its [versioned constants module](https://github.com/dftbplus/dftbplus/blob/24.1/src/dftbp/common/constants.F90#L23-L27).
This is an input-unit conversion; the archived energies, forces and charges
are unchanged. The separate historical H₂ energy section retains its original
nominal Å coordinates. Finite differences use displacements in ATV Å.

Without this conversion, tilted CO compares slightly different bond lengths:
the charge differs by 1.74e-7 e and the energy by 8.30e-7 eV. At identical
atomic-unit coordinates these differences are 2.4e-9 e and 7.2e-10 eV. No
tolerance is widened. The charge gate is 1e-7 e; the transcript rounds to eight
decimals (at most 5e-9 e rounding error).

The same `Atomic gross charges (e)` output blocks supply these charge pins:

| System | Atomic order | Charges [e] |
| --- | --- | --- |
| Water | O, H1, H2 | −0.78403315, 0.39237282, 0.39166033 |
| CO | C, O | 0.17210262, −0.17210262 |

Supplemental checks cover const input, energy-only force preservation,
overwrite versus nested accumulation, atom-count/species rebuilds, invalid
input and retry, and rotation/translation covariance. These checks supplement
the external energy, charge and full-force pins above; they are not new oracles.

#### H₂, \(R = 0.74\) Å (cubic isolation 30 Å)

| Quantity | Transcript | Unit |
| -------- | ---------- | ---- |
| Total energy | −0.6749509324 | Ha |
| \(F_{1x}\) | −0.001976354621 | Ha/Bohr |
| \(F_{2x}\) | +0.001976354621 | Ha/Bohr |

The published eV figure **−18.36635052 eV** (`reg_dftb0-h2`) must not move.

#### H₂, \(R = 0.90\) Å

| Quantity | Transcript | Unit |
| -------- | ---------- | ---- |
| Total energy | −0.6630166170 | Ha |
| \(F_{1x}\) | +0.065975227977 | Ha/Bohr |
| \(F_{2x}\) | −0.065975227977 | Ha/Bohr |

#### Off-equilibrium water (Å)

| Atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H | 1 | 1.05 | 0.00 | 0.00 |
| H | 1 | −0.26 | 1.00 | 0.00 |

| Quantity | Transcript | Unit |
| -------- | ---------- | ---- |
| Total energy | −4.0964033746 | Ha |
| \(F_\mathrm{O}\) | (0.039206443065, 0.036588999897, 0) | Ha/Bohr |
| \(F_\mathrm{H1}\) | (−0.048418613962, −0.000229747051, 0) | Ha/Bohr |
| \(F_\mathrm{H2}\) | (0.009212170898, −0.036359252846, 0) | Ha/Bohr |

#### Tilted CO (ppσ / ppπ and a transverse component)

| Atom | Z | x | y | z |
| ---- | - | - | - | - |
| C | 6 | 0.00 | 0.00 | 0.00 |
| O | 8 | 1.20 | 0.15 | 0.00 |

| Quantity | Transcript | Unit |
| -------- | ---------- | ---- |
| Total energy | −5.0270725023 | Ha |
| \(F_\mathrm{C}\) | (0.183901804021, 0.022987724205, 0) | Ha/Bohr |
| \(F_\mathrm{O}\) | (−0.183901804021, −0.022987724205, 0) | Ha/Bohr |

### Pins (hard-coded gates)

| Gate | Tol |
| ---- | --- |
| Total energy vs the Ha transcript × `kHartreeInEv` | \(1\times10^{-6}\) eV |
| Each force component vs the Ha/Bohr transcript × `kHartreePerBohrInEvPerAngstrom` | \(1\times10^{-5}\) eV/Å |
| Analytic \(F\) vs central FD of the same energy (\(h = 10^{-4}\) Å) | \(1\times10^{-5}\) eV/Å |
| \(\lvert\sum_i F_i\rvert\) | \(1\times10^{-8}\) eV/Å |
| Published H₂ 0.74 eV pin −18.36635052 | \(1\times10^{-6}\) eV |

**Tolerance rationale.** The transcript prints energy to \(10^{-10}\) Ha
(\(\approx 3\times10^{-9}\) eV) and forces to \(\sim10^{-12}\) Ha/Bohr
(\(\approx 5\times10^{-11}\) eV/Å). The \(10^{-6}\) eV / \(10^{-5}\) eV/Å
gates sit above that print precision and above the central-difference
floor of `MioSkTable::grid_derivative` (\(h = 10^{-5}\) Å on the hopping
interpolant). They are not fitted to an ATV residual. A factor-of-two or
a sign error in the Pulay contraction is eV/Å, not \(10^{-5}\).

A wrapped H₂ (atoms at 0.20 Å and 7.60 Å in an 8 Å cube) is an ATV
min-image identity only: DFTB+ `C` geometry does not wrap. It gates
\(F = -\nabla E\) and \(\sum F = 0\), not an absolute oracle number.

### Gaps

- No cell stress, no k-mesh, no lattice-sum images beyond the single
  minimum image the energy already uses.
- Closed shell, integer occupation, H/C/N/O only.
- The SK radial derivative is still a finite difference of the interpolant
  (`dftb-sk-analytic-derivative`); this slug does not change that.

### Perf

Warm DFTB+ 24.1 single point, `OMP_NUM_THREADS=1`, process wall
(five repeats, 2026-10-09): H₂ \(R=0.90\) Å median **0.03 s**, water
median **0.04 s**. Cold start of the binary was 0.92 s and is not the
comparison.

ATV `Evaluate::run`, same thread count, Release, in-process chrono in
`reg_dftb0-analytic-forces`: first H₂ launch **0.014 s** (runtime init),
later H₂ **< 0.1 ms**, water **0.0010 s**, CO **0.0003 s**.

\(T_\mathrm{atv}/T_\mathrm{dftb+} = 0.014/0.03 = 0.45\) using the cold
ATV launch against the warm DFTB+ process (the slower reading). Warm
repeats are ~0.03. **Within 10×.** Not `perf_exempt`.

### Observed (2026-10-09, Release, 1 thread)

Not a retune of the gates. Max |ΔE| vs the Ha transcript was
\(8.3\times10^{-7}\) eV (CO). Max |ΔF| stayed under the
\(1\times10^{-5}\) eV/Å gate (H₂ 0.74 Å, ~2×10⁻⁶ eV/Å). FD identity
max residual \(1.0\times10^{-6}\) eV/Å (wrapped H₂). H₂ 0.74 Å energy
−18.3663505230 eV, |Δ| 3×10⁻⁹ eV from the published pin.
