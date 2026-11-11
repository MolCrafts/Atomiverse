# Method handbook pages

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

**Purpose: method reference, not textbook reading order.**
These technical reports own method-specific APIs, validation, performance and
limitations. The [tutorial chapters](../index.md) own the theory-first teaching
narrative. Do not duplicate that prose here or maintain a separate LaTeX
version. Browse by domain in the [method catalog](catalog.md); what is
implemented on which backend is the [roadmap](../../roadmap/index.md).

## One page per method

A page mirrors one regression case, `regressions/<domain>/<method>/`:

- **File:** `methods/<domain>-<method>.md` (`gpw-d3.md` for `gpw/d3`).
- **Intro:** what the method is, the public API that runs it, the case line
  `ctest -R '^<domain>/<method>$'` and a link to the case directory, whose
  `REFERENCE.md` is the oracle record.
- **Shared block:** what every section has in common — oracle program and
  version, settings, limitations, literature — stated once.
- **One `##` section per section of the case**, headed
  `## <section> — <system / quantity> {#<section>}`. The anchor is the section
  name, so other documents link as `<domain>-<method>.md#<section>`. A section
  runs as `atv_regressions --case <domain>/<method> --section <section>`.
- A variant documented on a page but not run by the case carries
  `*No section of the case runs this variant.*` under its heading. A case
  section with no page section is named in the page intro.

Template: [`_TEMPLATE.md`](_TEMPLATE.md).

## Oracles (do not invent)

Every number on a page is one an external program printed, a literature value
or an analytic identity, with its provenance in the case `REFERENCE.md`
(`.claude/notes/oracle-pin-provenance.md`). Oracles never run in ctest. By
domain:

- molecular Hartree–Fock, post-HF and response — **PySCF**
  (`docs/tutorial/appendix-pyscf-oracle.md`)
- GPW / plane-wave / periodic DFT — **CP2K**; **Quantum ESPRESSO** for DFPT
- classical and electrochemical MD — **LAMMPS** where the feature overlaps,
  **MetalWalls** for MetalWalls-only physics
- NDDO / DFTB / GFN-xTB — family-locked to **OpenMOPAC** / **DFTB+** / **xtb**;
  no cross-family energy pins, and GFN is not treated as a DFTB3 successor

Performance is the fair wall \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) on the
same settings class (gate \(r \le 10\)), or `perf_exempt` with the reason.

## Index

Grouped by domain; the [catalog](catalog.md) lists every section.

- **Gaussian-basis electronic structure** — [Unrestricted Hartree–Fock (UHF)](scf-uhf.md) · [Restricted open-shell Hartree–Fock (ROHF)](scf-rohf.md) · [Generalized Hartree–Fock (GHF)](scf-ghf.md) · [Restricted Kohn–Sham DFT (RKS)](scf-rks.md) · [Unrestricted Kohn–Sham DFT (UKS)](scf-uks.md) · [Restricted open-shell Kohn–Sham DFT (ROKS)](scf-roks.md) · [Molecular SCF properties](scf-properties.md) · [Møller–Plesset perturbation theory (MP2, MP3, MP4)](scf-mp2.md) · [Coupled cluster (CCSD, CCSD(T), CC2)](scf-coupled-cluster.md) · [Equation-of-motion coupled cluster (EOM-CC)](scf-eom-cc.md) · [Algebraic diagrammatic construction (ADC)](scf-adc.md) · [Configuration interaction and multireference methods](scf-multireference.md) · [Linear-response TDDFT and spin-flip response](scf-tddft.md) · [Green's-function and RPA methods](scf-green-function.md) · [Polarizable continuum solvation (PCM)](scf-solvation.md) · [Molecular DFT dispersion corrections (D3BJ, D4)](scf-dispersion.md) · [Periodic Hartree–Fock](scf-periodic-hf.md)
- **Gaussian and plane waves (GPW)** — [GPW closed-shell Kohn–Sham DFT](gpw-rks.md) · [GPW unrestricted Kohn–Sham (UKS)](gpw-uks.md) · [GPW orbital transformation (OT)](gpw-orbital-transformation.md) · [GPW Dudarev DFT+U](gpw-dft-plus-u.md) · [GPW DFT-D3 dispersion](gpw-d3.md) · [GPW DFT-D4 dispersion](gpw-d4.md) · [GPW Tkatchenko–Scheffler dispersion](gpw-tkatchenko-scheffler.md) · [GPW many-body dispersion (MBD@rsSCS)](gpw-mbd.md) · [GPW free-atom XDM dispersion](gpw-xdm.md) · [GPW Hirshfeld-scaled MBD@rsSCS](gpw-hirshfeld.md) · [GPW nonlocal vdW correlation (vdW-DF, vdW-DF2, rVV10, optB88-vdW)](gpw-vdw-df.md) · [GPW Berry-phase polarization](gpw-berry-phase.md) · [GPW maximally localized Wannier functions](gpw-wannier.md) · [Periodic correlated methods on GPW (MP2, RPA, GW)](gpw-periodic-mp2.md) · [GPW variable-cell relaxation (CELL_OPT)](gpw-variable-cell.md)
- **Plane-wave DFT** — [Plane-wave restricted Kohn–Sham DFT](pw-rks.md) · [Plane-wave collinear spin-polarized Kohn–Sham](pw-collinear-spin.md) · [Plane-wave metallic smearing](pw-smearing.md) · [Plane-wave band structure and density of states](pw-bands-dos.md) · [Plane-wave Berry-phase polarization](pw-berry-phase.md) · [Plane-wave periodic maximally localized Wannier functions](pw-wannier.md) · [Plane-wave variable-cell relaxation](pw-variable-cell.md)
- **Response and lattice dynamics** — [Plane-wave DFPT phonons](response-dfpt.md) · [Finite-displacement crystal phonons](response-phonons.md)
- **Semiempirical and tight binding** — [Non-SCC DFTB0](semiempirical-dftb0.md) · [SCC-DFTB2](semiempirical-dftb2.md) · [SCC-DFTB3](semiempirical-dftb3.md) · [Linear-response TD-DFTB (Casida)](semiempirical-td-dftb.md) · [GFN0-xTB](semiempirical-gfn0-xtb.md) · [GFN1-xTB](semiempirical-gfn1-xtb.md) · [GFN2-xTB](semiempirical-gfn2-xtb.md) · [IPEA-xTB](semiempirical-ipea-xtb.md)
- **Ab initio molecular dynamics** — [Born–Oppenheimer molecular dynamics](dynamics-bomd.md) · [Car–Parrinello molecular dynamics](dynamics-car-parrinello.md) · [Second-generation Car–Parrinello MD](dynamics-second-gen-cpmd.md) · [Extended-Lagrangian BOMD (XL-BOMD)](dynamics-xl-bomd.md)
- **Classical and electrochemical MD** — [Constant-pressure MD (NPT)](md-npt.md) · [Rigid water and virtual sites](md-constraints.md) · [Many-body classical potentials (EAM, Stillinger–Weber)](md-manybody-potentials.md) · [Constant-potential electrodes](md-constant-potential.md) · [Constant-charge electrodes](md-constant-charge.md) · [Thomas–Fermi electrode metallicity](md-thomas-fermi.md) · [Electrode observables](md-electrode-observables.md) · [External electric and displacement fields](md-electric-fields.md) · [Polarizable and fluctuating-charge models](md-polarizable.md)
- **Force fields** — [Classical short-range pair styles and walls](forcefield-pair-styles.md)
- **Long-range electrostatics** — [Ewald and PME electrostatics](longrange-ewald-pme.md)
- **Sampling and free energy** — [Free-energy methods](sampling-free-energy.md)
