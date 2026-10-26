# SCC-DFTB3

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Case: `ctest -R '^semiempirical/dftb3$'` — sources and oracle record in [`regressions/semiempirical/dftb3/`](../../../regressions/semiempirical/dftb3/). One section of the case per `##` heading below: `atv_regressions --case semiempirical/dftb3 --section <name>`.

Self-consistent-charge **DFTB3** (Gaus, Cui, Elstner, *J. Chem. Theory
Comput.* **7**, 931 (2011)) extends Elstner SCC-DFTB2 by a third-order
charge-fluctuation term \(E^{(3)}\) built from the charge derivative of the
Elstner \(\gamma\) kernel (\(\Gamma_{AB}=\partial\gamma_{AB}/\partial q_A\),
on-site \(U^{\mathrm{d}}\)). The concrete method is
`cpu::semiempirical::DFTB3{cpu::semiempirical::DftbSkSet::MioSketch}` with
parameters as public members (`hubbard_du_ha`, `scc_tol`, `max_scc`,
`mix_beta`, `with_forces`; no `*Config` bag, no `make_*` / `build_*`), bound
into `cpu::Evaluate` at `evaluate::Phase::evaluate`. Diagnostics after a run:
`last_scc_residual()`, `last_scc_iterations()`, `last_e3()`. SK, Hubbard
\(U\) and repulsives are static in-tree transcribed mio-1-1 H/C/N/O tables
(`basis_data/dftb_mio_tables.h`, H–H via `internal::MioHHTable`); there is no
runtime `.skf` parser. Third-order Hubbard derivatives are **not** in that
table: they are explicit `hubbard_du_ha` input in Ha, set to the published
3ob values `basis_data::k3obHubbardDerivsHa`.

The case is CPU-only and has three sections: `dftb3-water-mio` (below, H₂O
SCC energy), `dftb3-analytic-forces` (H₂O closed-form forces vs re-SCC central
FD, tol 3×10⁻⁵ eV/Å) and `dftb3-periodic-forces` (Γ-PBC C-chain forces vs
re-SCC FD, tol 1×10⁻⁵ eV/Å). Their oracle records are in the case
`REFERENCE.md`.

## Oracle and parameter set

```text
oracle: literature
oracle_version: Gaus, Cui, Elstner, J. Chem. Theory Comput. 7, 931 (2011),
  DOI 10.1021/ct100684s — DFTB3 E^(3)/Γ formulation
family_oracle: dftb+
```

The absolute energies are **ATV twin snapshots**, not DFTB+ pins, for two
named reasons:

1. **Mixed parameter set.** mio-1-1 SK/repulsive paired with 3ob-class
   \(U^{\mathrm d}\) [Ha]: H −0.1857, C −0.1492, N −0.1535, O −0.1575
   (3ob-3-1 README, CC-BY-SA-4.0, Elstner/KIT; 3ob "supersedes the MIO
   parameter set"). The engine `ATV_WARN`s. This is a user-declared pairing,
   not a DFTB+-reproducible Hamiltonian.
2. **Missing \(\gamma^h\) X–H damping (\(\zeta = 4.00\), DFTB+
   `DampXHExponent`).** DFTB3/3ob is undefined without it.

The real gates are the homonuclear reduction to the DFTB+ 24.1 / mio-1-1 H–H
pin of [`dftb2-dftbplus-energy-pin`](semiempirical-dftb2.md#dftb2-dftbplus-energy-pin)
and \(E^{(3)}\) moving the energy by its own magnitude. Family lock: the
numerical oracle package is **DFTB+** only — never OpenMOPAC, never xtb
(GFN-xTB is not a DFTB3 successor); ctest never launches it. `perf_exempt:
true` for every section — no matched DFTB+ DFTB3 Hamiltonian to wall against.

Build: SE-only closure (`-DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_EVALUATE=ON
-DATV_BUILD_REGRESSIONS=ON`, every other domain and CUDA OFF) in the CPU
environment `buildenv-gcc/2026.03-mpich`.

## dftb3-water-mio — SCC-DFTB3 energy — H₂O (mio SK + 3ob \(U^{\mathrm d}\)) {#dftb3-water-mio}

### Method

#### Energy (closed shell, molecular)

\[
E = \mathrm{Tr}(\rho H^{0}) + E^{(2)} + E^{(3)} + E_{\mathrm{rep}},
\]

with second- and third-order SCC terms (Gaus/Cui/Elstner 2011 form)

\[
E^{(2)}=\tfrac12\sum_{AB}\gamma_{AB}\,\Delta q_A\Delta q_B,\qquad
E^{(3)}=\tfrac13\sum_{AB}\Delta q_A^{2}\,\Delta q_B\,\Gamma_{AB}.
\]

Here \(\gamma_{AB}\) is the DFTB+ Elstner charge kernel (on-site
\(\gamma_{AA}=U_A\)), \(\Gamma_{AB}=\partial\gamma_{AB}/\partial q_A\)
(\(\Gamma_{AB}\neq\Gamma_{BA}\); on-site \(\Gamma_{AA}=U^{\mathrm{d}}_A\)),
and \(\Delta q_A\) are Mulliken charge fluctuations from the neutral valence
reference. The SCC atomic potential is the **exact** derivative

\[
V^{(3)}_{C}=\tfrac23\,\Delta q_C\sum_B\Gamma_{CB}\Delta q_B
         +\tfrac13\sum_B\Delta q_B^{2}\,\Gamma_{BC},
\]

so the Euler identity \(\sum_A\Delta q_A V_A=2E^{(2)}+3E^{(3)}\) holds and a
band-structure energy built from \(V\) double-counts by \(E^{(2)}+2E^{(3)}\).
Both \(E^{(2/3)}\) and \(V\) live in private workspace
`internal/dftb3_scc.h`.

**Assembly route (binding).** Both DFTB2 and DFTB3 assemble
\(E = \mathrm{Tr}(\rho H^{0}) + E^{(2)} + E^{(3)} + E_{\mathrm{rep}}\), not
\(E_{\mathrm{band}}-E_{\mathrm{dc}}\) with \(\sum_A V_A\Delta q_A\) only.
Using gross Mulliken populations as \(\sum_A V_A q_A\) while subtracting only
\(E^{(2)}+2E^{(3)}\) would inject a spurious linear term
\(\sum_A V_A q^{0}_A\).

#### Pipeline

```text
ParameterSet → SK → H/S → eigen → P → SCC(Hubbard, γ, Γ, Uᵈ) → repulsive → E
```

1. **SK tables.** Ordered pair \((Z_A,Z_B)\) radial grids for \(H_{\mu\nu}(r)\),
   \(S_{\mu\nu}(r)\) plus short-range repulsive (static; length Bohr / energy Ha
   in-header; public surface eV/Å). Z ∈ {1, 6, 7, 8} only; other Z throws.
2. **Electronic structure.** Assemble \(H^{0}\), \(S\); generalized eigenproblem;
   density \(P\); Mulliken \(\Delta q_A\).
3. **SCC.** Build \(V^{(2)}+V^{(3)}\); Anderson charge mixing (`mix_beta`, default
   0.3) until \(\max|\Delta q_{\mathrm{out}}-\Delta q_{\mathrm{in}}|<\texttt{scc_tol}\).
   Exhausting `max_scc` **throws** (no silent budget-exhausted energy);
   undamped Picard is a period-2 limit cycle for polar systems.
4. **Total energy.** Trace + \(E^{(2)}\) + \(E^{(3)}\) + repulsive →
   `frame.meta.get(kEnergy)` [eV]. Forces are off in this section.

#### Settings

| Setting | Value |
| ------- | ----- |
| System | H₂O, \(Z=(8,1,1)\), closed shell, charge 0 |
| Geometry (Å) | O \((0,0,0)\); H \((0.96,0,0)\); H \((-0.24,0.93,0)\) — slightly off-eq |
| Box | cubic **20 Å** (isolation; molecular path) |
| Public energy unit | **eV** on `frame.meta.get(kEnergy)` |
| SCC | `scc_tol = 1e-10`, `max_scc = 50`, `mix_beta = 0.3` (default) |
| \(U^{\mathrm d}\) | `hubbard_du_ha = basis_data::k3obHubbardDerivsHa` |
| Forces | **off** (`with_forces = false`) |
| DFTB2 twin | same geometry, `DFTB2{MioSketch}`, `boundary = DftbBoundary::cluster`, `scc_tol = 1e-10` |

### Implementation

```cpp
cpu::Frame frame = make_water();   // geometry above, box cubic(20.0)

cpu::Evaluate eval;
cpu::semiempirical::DFTB3 dftb3{cpu::semiempirical::DftbSkSet::MioSketch};
dftb3.hubbard_du_ha = cpu::semiempirical::basis_data::k3obHubbardDerivsHa;
dftb3.with_forces = false;
dftb3.scc_tol = 1e-10;
eval.on(evaluate::Phase::evaluate).add_ref(dftb3);
eval.build();
eval.run(frame);

const double E_eV = frame.meta.get(kEnergy);
// dftb3.last_scc_residual() < 1e-10; dftb3.last_e3() for the ratio gate
```

| Role | Path |
| ---- | ---- |
| Method | `include/atomiverse/cpu/semiempirical/dftb3.h` |
| Impl | `src/cpu/semiempirical/dftb3.cpp` |
| SCC kernel (private) | `internal/dftb3_scc.h` |
| mio-1-1 SK | `…/basis_data/dftb_mio_tables.h`, `…/basis_data/dftb_mio_hh_sk.h` |

### Validation

#### Gates

| Quantity | Value | Check | Tol |
| -------- | ----: | ----- | --- |
| \(E_{\mathrm{tot}}\) | **−111.195830730395 eV** | ATV twin snapshot (`kPinEnergyEV`) | `kPinTolEV = 1e-6` eV |
| SCC residual | \(< 10^{-10}\,e\) | converged, not budget-exhausted | — |
| \(\lvert E_{\mathrm{DFTB3}}-E_{\mathrm{DFTB2}}\rvert\) | \(> 10^{-6}\) eV | third order active | — |
| \(\lvert E_{\mathrm{DFTB3}}-E_{\mathrm{DFTB2}}\rvert / \lvert E^{(3)}\rvert\) | 0.777 | within \([0.5, 2]\) | — |

Measured (Release, `OMP_NUM_THREADS=1`): DFTB2 on the same geometry
**−110.958967574695 eV**; \(E^{(3)}=-3.048297\times10^{-1}\) eV.

#### Homonuclear reduction (absolute oracle)

For homonuclear H₂ at \(r=0.74\) Å, \(\Delta q\equiv 0\) so
\(E^{(2)}=E^{(3)}=0\) and DFTB3 **must** equal DFTB2:

| Twin | \(E\) [eV] | Notes |
| ---- | ---------: | ----- |
| DFTB3 H₂ | **−18.366350522958** | shared `MioHHTable` |
| DFTB2 H₂ / DFTB+ 24.1 pin | **−18.36635052** | residual ~3×10⁻⁹ eV |

This is a recorded check, not a gate of this section (no standing unit test
covers it).

#### Absolute DFTB+ pin owed

An absolute DFTB+ DFTB3 pin needs a matched Hamiltonian (3ob SK set with
\(\gamma^h\) damping, `ThirdOrderFull = Yes` or diagonal DFTB3, same geometry
and SCC tol); only then does `kPinEnergyEV` become an oracle value.

### Limitations

| Item | Status |
| ---- | ------ |
| Third-order SCC energy + exact \(\partial E^{(3)}/\partial\Delta q\) | gated here |
| Anderson mix + throw-on-nonconvergence | gated here |
| Homonuclear reduction to DFTB2 / DFTB+ H–H pin | recorded, not gated |
| Absolute \(E\) vs DFTB+ for H₂O | owed (3ob SK + \(\gamma^h\)) |
| Spin / k-mesh / cell stress / D3 / TD-DFTB on DFTB3 | out of scope |
| Live oracle in CI | **forbidden** — offline pins only |

This section pins the DFTB3 energy on the Evaluate surface with a DFTB2
magnitude gate; it does **not** establish absolute DFTB+ water identity.

## References

- M. Gaus, Q. Cui, M. Elstner, *J. Chem. Theory Comput.* **7**, 931 (2011),
  DOI [10.1021/ct100684s](https://doi.org/10.1021/ct100684s)
- M. Elstner et al., *Phys. Rev. B* **58**, 7260 (1998),
  DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)
- B. Hourahine et al., *J. Chem. Phys.* **152**, 124101 (2020) (force-module boundary)
- DFTB.org / dftbparams mio-1-1 and 3ob-3-1 README
- Roadmap: [`docs/roadmap/semiempirical.md`](../../roadmap/semiempirical.md)
