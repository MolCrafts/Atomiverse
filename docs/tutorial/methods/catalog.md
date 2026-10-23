# Method catalog

One page per method; one `##` section per section of the method's regression
case (`regressions/<domain>/<method>/`). Each page states what the method is, the
public API that runs it, the external code its numbers were pinned against, and the
fair wall against that code where a twin exists. Sections that document a variant
no section of the case runs say so under their heading. These are reference pages,
not a reading order — the [tutorial index](../index.md) gives the reading order;
what is implemented on which backend is the [roadmap](../../roadmap/index.md).

Page policy and template: [methods/README](README.md).

## Gaussian-basis electronic structure

Hartree–Fock, Kohn–Sham, post-HF, excited states, molecular properties and environments. Oracle: **PySCF**.

| Page | Case | Page sections |
| ---- | ---- | -------- |
| [Unrestricted Hartree–Fock (UHF)](scf-uhf.md) | `scf/uhf` | [`delta-scf-n2`](scf-uhf.md#delta-scf-n2) |
| [Restricted open-shell Hartree–Fock (ROHF)](scf-rohf.md) | `scf/rohf` | [`rohf-hessian-o2`](scf-rohf.md#rohf-hessian-o2) |
| [Generalized Hartree–Fock (GHF)](scf-ghf.md) | `scf/ghf` | [`ghf-grad-h3-sto3g`](scf-ghf.md#ghf-grad-h3-sto3g) |
| [Restricted Kohn–Sham DFT (RKS)](scf-rks.md) | `scf/rks` | [`blyp-water-sto3g`](scf-rks.md#blyp-water-sto3g), [`b2plyp-h2o`](scf-rks.md#b2plyp-h2o), [`rks-hessian-h2o-sto3g`](scf-rks.md#rks-hessian-h2o-sto3g) |
| [Unrestricted Kohn–Sham DFT (UKS)](scf-uks.md) | `scf/uks` | [`uks-hessian-o2-sto3g`](scf-uks.md#uks-hessian-o2-sto3g) |
| [Restricted open-shell Kohn–Sham DFT (ROKS)](scf-roks.md) | `scf/roks` | [`roks-o2-sto3g`](scf-roks.md#roks-o2-sto3g), [`roks-grad-o2-sto3g`](scf-roks.md#roks-grad-o2-sto3g) |
| [Molecular SCF properties](scf-properties.md) | `scf/properties` | [`wannier-centers`](scf-properties.md#wannier-centers), [`static-polariz-h2o`](scf-properties.md#static-polariz-h2o) |
| [Møller–Plesset perturbation theory (MP2, MP3, MP4)](scf-mp2.md) | `scf/mp2` | [`frozen-core-mp2-h2o`](scf-mp2.md#frozen-core-mp2-h2o), [`scs-mp2-h2o-sto3g`](scf-mp2.md#scs-mp2-h2o-sto3g), [`ump2-o2-sto3g`](scf-mp2.md#ump2-o2-sto3g), [`ump2-grad-o2-sto3g`](scf-mp2.md#ump2-grad-o2-sto3g), [`romp2-o2-sto3g`](scf-mp2.md#romp2-o2-sto3g), [`mp3-h2-sto3g`](scf-mp2.md#mp3-h2-sto3g), [`mp4-h2-sto3g`](scf-mp2.md#mp4-h2-sto3g) |
| [Coupled cluster (CCSD, CCSD(T), CC2)](scf-coupled-cluster.md) | `scf/coupled-cluster` | [`ccsd-h2-sto3g`](scf-coupled-cluster.md#ccsd-h2-sto3g), [`ccsd-grad-h2o-sto3g`](scf-coupled-cluster.md#ccsd-grad-h2o-sto3g), [`ccsd-t-h2o-sto3g`](scf-coupled-cluster.md#ccsd-t-h2o-sto3g), [`cc2-h2-sto3g`](scf-coupled-cluster.md#cc2-h2-sto3g) |
| [Equation-of-motion coupled cluster (EOM-CC)](scf-eom-cc.md) | `scf/eom-cc` | [`eom-ee-ccsd`](scf-eom-cc.md#eom-ee-ccsd), [`eom-ip-ccsd`](scf-eom-cc.md#eom-ip-ccsd), [`eom-ea-ccsd`](scf-eom-cc.md#eom-ea-ccsd), [`eom-sf-ccsd-o2`](scf-eom-cc.md#eom-sf-ccsd-o2), [`eom-cc2-h2`](scf-eom-cc.md#eom-cc2-h2) |
| [Algebraic diagrammatic construction (ADC)](scf-adc.md) | `scf/adc` | [`adc2-h2co-sto3g`](scf-adc.md#adc2-h2co-sto3g), [`adc2x-h2co-sto3g`](scf-adc.md#adc2x-h2co-sto3g), [`ip-adc2-h2o-sto3g`](scf-adc.md#ip-adc2-h2o-sto3g), [`ea-adc2-h2-sto3g`](scf-adc.md#ea-adc2-h2-sto3g) |
| [Configuration interaction and multireference methods](scf-multireference.md) | `scf/multireference` | [`cisd-h2co-sto3g`](scf-multireference.md#cisd-h2co-sto3g), [`casscf-h2-sto3g`](scf-multireference.md#casscf-h2-sto3g), [`casscf-h2o-sto3g`](scf-multireference.md#casscf-h2o-sto3g), [`nevpt2-h2-sto3g`](scf-multireference.md#nevpt2-h2-sto3g), [`casci-h2-sto3g`](scf-multireference.md#casci-h2-sto3g) |
| [Linear-response TDDFT and spin-flip response](scf-tddft.md) | `scf/tddft` | [`h2-b3lyp-hybrid-tda`](scf-tddft.md#h2-b3lyp-hybrid-tda), [`h2-b3lyp-rpa-tddft`](scf-tddft.md#h2-b3lyp-rpa-tddft), [`full-lr-tddft-h2-pbe`](scf-tddft.md#full-lr-tddft-h2-pbe), [`uks-full-lr-o2-lsda`](scf-tddft.md#uks-full-lr-o2-lsda), [`sf-tddft-o2`](scf-tddft.md#sf-tddft-o2), [`spin-flip-tda-o2`](scf-tddft.md#spin-flip-tda-o2) |
| [Green's-function and RPA methods](scf-green-function.md) | `scf/green-function` | [`g0w0-h2-sto3g`](scf-green-function.md#g0w0-h2-sto3g), [`bse-h2-sto3g`](scf-green-function.md#bse-h2-sto3g), [`direct-rpa-h2-sto3g`](scf-green-function.md#direct-rpa-h2-sto3g), [`gf2-h2-sto3g`](scf-green-function.md#gf2-h2-sto3g) |
| [Polarizable continuum solvation (PCM)](scf-solvation.md) | `scf/solvation` | [`ief-pcm-h2o`](scf-solvation.md#ief-pcm-h2o), [`pcm-grad-h2o`](scf-solvation.md#pcm-grad-h2o) |
| [Molecular DFT dispersion corrections (D3BJ, D4)](scf-dispersion.md) | `scf/dispersion` | [`dft-d4-h2o-pbe`](scf-dispersion.md#dft-d4-h2o-pbe) |
| [Periodic Hartree–Fock](scf-periodic-hf.md) | `scf/periodic-hf` | [`periodic-kpoint-hf`](scf-periodic-hf.md#periodic-kpoint-hf) |

## Gaussian and plane waves (GPW)

Periodic and isolated Kohn–Sham DFT on the GPW path, dispersion corrections, OT, DFT+U, periodic correlated methods. Oracle: **CP2K**.

| Page | Case | Page sections |
| ---- | ---- | -------- |
| [GPW closed-shell Kohn–Sham DFT](gpw-rks.md) | `gpw/rks` | [`gpw-forces-water`](gpw-rks.md#gpw-forces-water), [`gpw-stress-water`](gpw-rks.md#gpw-stress-water), [`gpw-analytic-stress`](gpw-rks.md#gpw-analytic-stress), [`gpw-water-pbe-mt`](gpw-rks.md#gpw-water-pbe-mt), [`gpw-kpoint-stress`](gpw-rks.md#gpw-kpoint-stress), [`gpw-multigrid-kpoints`](gpw-rks.md#gpw-multigrid-kpoints), [`mpi-gpw-scaling-4rank`](gpw-rks.md#mpi-gpw-scaling-4rank), [`gpw-blyp-water`](gpw-rks.md#gpw-blyp-water), [`gpw-kpoint-forces`](gpw-rks.md#gpw-kpoint-forces), [`gpw-metallic-smearing`](gpw-rks.md#gpw-metallic-smearing), [`gpw-pbesol-si`](gpw-rks.md#gpw-pbesol-si), [`gpw-revpbe-si`](gpw-rks.md#gpw-revpbe-si), [`gpw-rpbe-water`](gpw-rks.md#gpw-rpbe-water) |
| [GPW unrestricted Kohn–Sham (UKS)](gpw-uks.md) | `gpw/uks` | [`gpw-uks-forces`](gpw-uks.md#gpw-uks-forces), [`gpw-uks-stress`](gpw-uks.md#gpw-uks-stress), [`gpw-uks-multigrid`](gpw-uks.md#gpw-uks-multigrid), [`gpw-uks-o2-gamma`](gpw-uks.md#gpw-uks-o2-gamma), [`gpw-uks-pbe`](gpw-uks.md#gpw-uks-pbe), [`gpw-kpoint-uks-forces`](gpw-uks.md#gpw-kpoint-uks-forces), [`gpw-kpoint-uks-stress`](gpw-uks.md#gpw-kpoint-uks-stress), [`gpw-kpoint-uks-stress-gga`](gpw-uks.md#gpw-kpoint-uks-stress-gga), [`gpw-kpoints-uks`](gpw-uks.md#gpw-kpoints-uks), [`gpw-uks-blyp`](gpw-uks.md#gpw-uks-blyp), [`gpw-uks-revpbe`](gpw-uks.md#gpw-uks-revpbe), [`gpw-uks-rpbe`](gpw-uks.md#gpw-uks-rpbe), [`gpw-uks-smearing`](gpw-uks.md#gpw-uks-smearing) |
| [GPW orbital transformation (OT)](gpw-orbital-transformation.md) | `gpw/orbital-transformation` | [`ot-minimizer-water`](gpw-orbital-transformation.md#ot-minimizer-water), [`ot-uks-o2`](gpw-orbital-transformation.md#ot-uks-o2), [`ot-dftu`](gpw-orbital-transformation.md#ot-dftu), [`ot-dftu-kpoints`](gpw-orbital-transformation.md#ot-dftu-kpoints), [`ot-dftu-smearing`](gpw-orbital-transformation.md#ot-dftu-smearing), [`ot-kpoints`](gpw-orbital-transformation.md#ot-kpoints), [`ot-smearing`](gpw-orbital-transformation.md#ot-smearing), [`ot-uks-dftu`](gpw-orbital-transformation.md#ot-uks-dftu), [`ot-uks-dftu-smearing`](gpw-orbital-transformation.md#ot-uks-dftu-smearing), [`ot-uks-kpoints`](gpw-orbital-transformation.md#ot-uks-kpoints), [`ot-uks-smearing`](gpw-orbital-transformation.md#ot-uks-smearing) |
| [GPW Dudarev DFT+U](gpw-dft-plus-u.md) | `gpw/dft-plus-u` | [`gpw-dftu-nio`](gpw-dft-plus-u.md#gpw-dftu-nio), [`gpw-dftu-stress`](gpw-dft-plus-u.md#gpw-dftu-stress), [`gpw-dftu-kpoint-forces`](gpw-dft-plus-u.md#gpw-dftu-kpoint-forces), [`gpw-dftu-kpoint-stress`](gpw-dft-plus-u.md#gpw-dftu-kpoint-stress), [`gpw-dftu-kpoints`](gpw-dft-plus-u.md#gpw-dftu-kpoints) |
| [GPW DFT-D3 dispersion](gpw-d3.md) | `gpw/d3` | [`gpw-d3-forces`](gpw-d3.md#gpw-d3-forces), [`gpw-d3-stress`](gpw-d3.md#gpw-d3-stress), [`gpw-d3-atm`](gpw-d3.md#gpw-d3-atm), [`gpw-d3-atm-forces`](gpw-d3.md#gpw-d3-atm-forces), [`gpw-d3-atm-stress`](gpw-d3.md#gpw-d3-atm-stress), [`gpw-d3bj-forces`](gpw-d3.md#gpw-d3bj-forces), [`gpw-d3bj-stress`](gpw-d3.md#gpw-d3bj-stress), [`gpw-pbe-d3bj-ice`](gpw-d3.md#gpw-pbe-d3bj-ice) |
| [GPW DFT-D4 dispersion](gpw-d4.md) | `gpw/d4` | [`gpw-d4-water`](gpw-d4.md#gpw-d4-water), [`gpw-d4-forces`](gpw-d4.md#gpw-d4-forces), [`gpw-d4-stress`](gpw-d4.md#gpw-d4-stress), [`gpw-d4-atm`](gpw-d4.md#gpw-d4-atm), [`gpw-d4-atm-forces`](gpw-d4.md#gpw-d4-atm-forces), [`gpw-d4-atm-stress`](gpw-d4.md#gpw-d4-atm-stress), [`gpw-d4-kpoint-forces`](gpw-d4.md#gpw-d4-kpoint-forces) |
| [GPW Tkatchenko–Scheffler dispersion](gpw-tkatchenko-scheffler.md) | `gpw/tkatchenko-scheffler` | [`gpw-ts-ice`](gpw-tkatchenko-scheffler.md#gpw-ts-ice), [`gpw-ts-forces`](gpw-tkatchenko-scheffler.md#gpw-ts-forces), [`gpw-ts-kpoint-forces`](gpw-tkatchenko-scheffler.md#gpw-ts-kpoint-forces) |
| [GPW many-body dispersion (MBD@rsSCS)](gpw-mbd.md) | `gpw/mbd` | [`gpw-mbd-ice`](gpw-mbd.md#gpw-mbd-ice), [`gpw-mbd-periodic`](gpw-mbd.md#gpw-mbd-periodic), [`gpw-mbd-hirshfeld`](gpw-mbd.md#gpw-mbd-hirshfeld), [`gpw-mbd-forces`](gpw-mbd.md#gpw-mbd-forces), [`gpw-mbd-stress`](gpw-mbd.md#gpw-mbd-stress), [`gpw-mbd-kpoint-forces`](gpw-mbd.md#gpw-mbd-kpoint-forces), [`gpw-mbd-kpoint-stress`](gpw-mbd.md#gpw-mbd-kpoint-stress) |
| [GPW free-atom XDM dispersion](gpw-xdm.md) | `gpw/xdm` | [`gpw-xdm-ice`](gpw-xdm.md#gpw-xdm-ice), [`gpw-xdm-forces`](gpw-xdm.md#gpw-xdm-forces) |
| [GPW Hirshfeld-scaled MBD@rsSCS](gpw-hirshfeld.md) | `gpw/hirshfeld` | [`gpw-hirshfeld-uks`](gpw-hirshfeld.md#gpw-hirshfeld-uks), [`gpw-hirshfeld-kpoints`](gpw-hirshfeld.md#gpw-hirshfeld-kpoints), [`gpw-hirshfeld-uks-kpoints`](gpw-hirshfeld.md#gpw-hirshfeld-uks-kpoints) |
| [GPW nonlocal vdW correlation (vdW-DF, vdW-DF2, rVV10, optB88-vdW)](gpw-vdw-df.md) | `gpw/vdw-df` | [`gpw-vdw-df-ice`](gpw-vdw-df.md#gpw-vdw-df-ice), [`gpw-rvv10-ice`](gpw-vdw-df.md#gpw-rvv10-ice), [`gpw-vdw-scf-potential`](gpw-vdw-df.md#gpw-vdw-scf-potential), [`gpw-vdw-multigrid`](gpw-vdw-df.md#gpw-vdw-multigrid), [`gpw-optb88-uks-kpoints`](gpw-vdw-df.md#gpw-optb88-uks-kpoints), [`gpw-optb88-vdw`](gpw-vdw-df.md#gpw-optb88-vdw), [`gpw-vdw-df2-ice`](gpw-vdw-df.md#gpw-vdw-df2-ice) |
| [GPW Berry-phase polarization](gpw-berry-phase.md) | `gpw/berry-phase` | [`periodic-berry-gpw-uks`](gpw-berry-phase.md#periodic-berry-gpw-uks) |
| [GPW maximally localized Wannier functions](gpw-wannier.md) | `gpw/wannier` | [`periodic-wannier-gpw`](gpw-wannier.md#periodic-wannier-gpw) |
| [Periodic correlated methods on GPW (MP2, RPA, GW)](gpw-periodic-mp2.md) | `gpw/periodic-mp2` | [`periodic-ri-mp2-si`](gpw-periodic-mp2.md#periodic-ri-mp2-si), [`periodic-scs-mp2-si`](gpw-periodic-mp2.md#periodic-scs-mp2-si), [`periodic-mp2-si`](gpw-periodic-mp2.md#periodic-mp2-si), [`periodic-mp2-kpoints`](gpw-periodic-mp2.md#periodic-mp2-kpoints), [`periodic-ump2-si`](gpw-periodic-mp2.md#periodic-ump2-si), [`periodic-ump2-kpoints`](gpw-periodic-mp2.md#periodic-ump2-kpoints), [`periodic-rpa-si`](gpw-periodic-mp2.md#periodic-rpa-si), [`periodic-rpa-kpoints`](gpw-periodic-mp2.md#periodic-rpa-kpoints), [`periodic-rpa-finite-q`](gpw-periodic-mp2.md#periodic-rpa-finite-q), [`periodic-urpa-si`](gpw-periodic-mp2.md#periodic-urpa-si), [`periodic-urpa-kpoints`](gpw-periodic-mp2.md#periodic-urpa-kpoints), [`periodic-gw-si`](gpw-periodic-mp2.md#periodic-gw-si), [`periodic-gw-kpoints`](gpw-periodic-mp2.md#periodic-gw-kpoints), [`periodic-gw-finite-q`](gpw-periodic-mp2.md#periodic-gw-finite-q), [`periodic-ugw-si`](gpw-periodic-mp2.md#periodic-ugw-si), [`periodic-ugw-kpoints`](gpw-periodic-mp2.md#periodic-ugw-kpoints), [`ugw-uks-rpbe`](gpw-periodic-mp2.md#ugw-uks-rpbe) |
| [GPW variable-cell relaxation (CELL_OPT)](gpw-variable-cell.md) | `gpw/variable-cell` | [`gpw-variable-cell-molopt`](gpw-variable-cell.md#gpw-variable-cell-molopt) |

## Plane-wave DFT

Norm-conserving plane-wave Kohn–Sham: energies, forces, stress, spin, smearing, band structure, polarization, Wannier functions, variable cell.

| Page | Case | Page sections |
| ---- | ---- | -------- |
| [Plane-wave restricted Kohn–Sham DFT](pw-rks.md) | `pw/rks` | [`pw-lda-si`](pw-rks.md#pw-lda-si), [`pw-pbe-gga-si`](pw-rks.md#pw-pbe-gga-si), [`pw-analytic-stress-si`](pw-rks.md#pw-analytic-stress-si), [`mpi-pw-scaling-4rank`](pw-rks.md#mpi-pw-scaling-4rank) |
| [Plane-wave collinear spin-polarized Kohn–Sham](pw-collinear-spin.md) | `pw/collinear-spin` | [`pw-spin-polarized-o2`](pw-collinear-spin.md#pw-spin-polarized-o2), [`pw-spin-gga`](pw-collinear-spin.md#pw-spin-gga), [`pw-uks-kmesh-fe`](pw-collinear-spin.md#pw-uks-kmesh-fe) |
| [Plane-wave metallic smearing](pw-smearing.md) | `pw/smearing` | [`pw-uks-fe-smearing`](pw-smearing.md#pw-uks-fe-smearing), [`pw-smearing-al`](pw-smearing.md#pw-smearing-al) |
| [Plane-wave band structure and density of states](pw-bands-dos.md) | `pw/bands-dos` | [`bands-dos-si`](pw-bands-dos.md#bands-dos-si), [`bands-metallic-si`](pw-bands-dos.md#bands-metallic-si), [`bands-spin-si`](pw-bands-dos.md#bands-spin-si), [`bands-spin-metallic-si`](pw-bands-dos.md#bands-spin-metallic-si), [`dos-metallic-si`](pw-bands-dos.md#dos-metallic-si), [`dos-spin-si`](pw-bands-dos.md#dos-spin-si), [`dos-spin-metallic-si`](pw-bands-dos.md#dos-spin-metallic-si) |
| [Plane-wave Berry-phase polarization](pw-berry-phase.md) | `pw/berry-phase` | [`berry-spin-si`](pw-berry-phase.md#berry-spin-si) |
| [Plane-wave periodic maximally localized Wannier functions](pw-wannier.md) | `pw/wannier` | [`periodic-wannier-si`](pw-wannier.md#periodic-wannier-si), [`periodic-wannier-uks`](pw-wannier.md#periodic-wannier-uks) |
| [Plane-wave variable-cell relaxation](pw-variable-cell.md) | `pw/variable-cell` | [`pw-variable-cell-si`](pw-variable-cell.md#pw-variable-cell-si) |

## Response and lattice dynamics

DFPT and finite-displacement phonons. Oracle: **Quantum ESPRESSO** for DFPT.

| Page | Case | Page sections |
| ---- | ---- | -------- |
| [Plane-wave DFPT phonons](response-dfpt.md) | `response/dfpt` | [`dfpt-phonon-si`](response-dfpt.md#dfpt-phonon-si), [`dfpt-gga-si`](response-dfpt.md#dfpt-gga-si), [`dfpt-spin-si`](response-dfpt.md#dfpt-spin-si), [`dfpt-gga-spin-si`](response-dfpt.md#dfpt-gga-spin-si), [`dfpt-metallic-si`](response-dfpt.md#dfpt-metallic-si), [`dfpt-spin-smearing`](response-dfpt.md#dfpt-spin-smearing), [`dfpt-gga-spin-smearing`](response-dfpt.md#dfpt-gga-spin-smearing), [`dfpt-incommensurate-si`](response-dfpt.md#dfpt-incommensurate-si), [`dfpt-spin-incommensurate`](response-dfpt.md#dfpt-spin-incommensurate), [`dfpt-finite-q-si`](response-dfpt.md#dfpt-finite-q-si), [`dfpt-gga-finite-q-si`](response-dfpt.md#dfpt-gga-finite-q-si), [`dfpt-finite-q-smearing`](response-dfpt.md#dfpt-finite-q-smearing), [`dfpt-finite-q-spin-smearing`](response-dfpt.md#dfpt-finite-q-spin-smearing), [`dfpt-gga-spin-finite-q-si`](response-dfpt.md#dfpt-gga-spin-finite-q-si), [`dfpt-gga-spin-finite-q-smearing`](response-dfpt.md#dfpt-gga-spin-finite-q-smearing) |
| [Finite-displacement crystal phonons](response-phonons.md) | `response/phonons` | [`phonon-dispersion-si`](response-phonons.md#phonon-dispersion-si), [`fd-phonon-supercell`](response-phonons.md#fd-phonon-supercell) |

## Semiempirical and tight binding

Family-locked oracles: NDDO → **OpenMOPAC**, DFTB → **DFTB+**, GFN-xTB → **xtb**; no cross-family pins.

| Page | Case | Page sections |
| ---- | ---- | -------- |
| [Non-SCC DFTB0](semiempirical-dftb0.md) | `semiempirical/dftb0` | [`dftb0-analytic-forces`](semiempirical-dftb0.md#dftb0-analytic-forces) |
| [SCC-DFTB2](semiempirical-dftb2.md) | `semiempirical/dftb2` | [`dftb2-dftbplus-energy-pin`](semiempirical-dftb2.md#dftb2-dftbplus-energy-pin), [`dftb2-3ob-organic`](semiempirical-dftb2.md#dftb2-3ob-organic), [`dftb2-analytic-forces`](semiempirical-dftb2.md#dftb2-analytic-forces), [`spin-dftb2-o2`](semiempirical-dftb2.md#spin-dftb2-o2), [`dftb2-spin-analytic-forces`](semiempirical-dftb2.md#dftb2-spin-analytic-forces), [`dftb2-kpoints`](semiempirical-dftb2.md#dftb2-kpoints), [`dftb2-periodic-forces`](semiempirical-dftb2.md#dftb2-periodic-forces), [`dftb2-periodic-gamma`](semiempirical-dftb2.md#dftb2-periodic-gamma) |
| [SCC-DFTB3](semiempirical-dftb3.md) | `semiempirical/dftb3` | [`dftb3-water-mio`](semiempirical-dftb3.md#dftb3-water-mio) |
| [Linear-response TD-DFTB (Casida)](semiempirical-td-dftb.md) | `semiempirical/td-dftb` | [`td-dftb-h2`](semiempirical-td-dftb.md#td-dftb-h2) |
| [GFN0-xTB](semiempirical-gfn0-xtb.md) | `semiempirical/gfn0-xtb` | [`gfn0-xtb-forces`](semiempirical-gfn0-xtb.md#gfn0-xtb-forces), [`gfn0-xtb-analytic-hessian`](semiempirical-gfn0-xtb.md#gfn0-xtb-analytic-hessian) |
| [GFN1-xTB](semiempirical-gfn1-xtb.md) | `semiempirical/gfn1-xtb` | [`gfn1-xtb-organic`](semiempirical-gfn1-xtb.md#gfn1-xtb-organic), [`gfn1-xtb-forces`](semiempirical-gfn1-xtb.md#gfn1-xtb-forces) |
| [GFN2-xTB](semiempirical-gfn2-xtb.md) | `semiempirical/gfn2-xtb` | [`oracle`](semiempirical-gfn2-xtb.md#oracle), [`gfn2-xtb-forces`](semiempirical-gfn2-xtb.md#gfn2-xtb-forces), [`gfn2-aes-fock`](semiempirical-gfn2-xtb.md#gfn2-aes-fock), [`gfn2-quadrupole-aes`](semiempirical-gfn2-xtb.md#gfn2-quadrupole-aes), [`gfn-xtb-analytic-hessian`](semiempirical-gfn2-xtb.md#gfn-xtb-analytic-hessian) |
| [IPEA-xTB](semiempirical-ipea-xtb.md) | `semiempirical/ipea-xtb` | [`oracle`](semiempirical-ipea-xtb.md#oracle), [`ipea-xtb-small`](semiempirical-ipea-xtb.md#ipea-xtb-small), [`ipea-xtb-forces`](semiempirical-ipea-xtb.md#ipea-xtb-forces), [`ipea-xtb-vip`](semiempirical-ipea-xtb.md#ipea-xtb-vip), [`ipea-xtb-analytic-hessian`](semiempirical-ipea-xtb.md#ipea-xtb-analytic-hessian) |

## Ab initio molecular dynamics

Born–Oppenheimer, Car–Parrinello, second-generation CPMD and extended-Lagrangian BOMD over the electronic-structure methods above.

| Page | Case | Page sections |
| ---- | ---- | -------- |
| [Born–Oppenheimer molecular dynamics](dynamics-bomd.md) | `dynamics/bomd` | [`gpw-bomd-water`](dynamics-bomd.md#gpw-bomd-water), [`pw-bomd-h`](dynamics-bomd.md#pw-bomd-h), [`gpw-bomd-kpoints`](dynamics-bomd.md#gpw-bomd-kpoints) |
| [Car–Parrinello molecular dynamics](dynamics-car-parrinello.md) | `dynamics/car-parrinello` | [`cpmd-water`](dynamics-car-parrinello.md#cpmd-water), [`cpmd-rhf-h2`](dynamics-car-parrinello.md#cpmd-rhf-h2), [`cpmd-uks`](dynamics-car-parrinello.md#cpmd-uks), [`cpmd-kpoints`](dynamics-car-parrinello.md#cpmd-kpoints), [`cpmd-uks-kpoints`](dynamics-car-parrinello.md#cpmd-uks-kpoints), [`cpmd-electronic-thermostat`](dynamics-car-parrinello.md#cpmd-electronic-thermostat), [`cpmd-rattle-water`](dynamics-car-parrinello.md#cpmd-rattle-water), [`cpmd-uks-thermostat-kpoints`](dynamics-car-parrinello.md#cpmd-uks-thermostat-kpoints) |
| [Second-generation Car–Parrinello MD](dynamics-second-gen-cpmd.md) | `dynamics/second-gen-cpmd` | [`second-gen-cpmd-water`](dynamics-second-gen-cpmd.md#second-gen-cpmd-water), [`second-gen-cpmd-uks`](dynamics-second-gen-cpmd.md#second-gen-cpmd-uks), [`second-gen-cpmd-ot`](dynamics-second-gen-cpmd.md#second-gen-cpmd-ot), [`second-gen-cpmd-uks-ot`](dynamics-second-gen-cpmd.md#second-gen-cpmd-uks-ot), [`second-gen-cpmd-kpoints`](dynamics-second-gen-cpmd.md#second-gen-cpmd-kpoints), [`second-gen-cpmd-ot-kpoints`](dynamics-second-gen-cpmd.md#second-gen-cpmd-ot-kpoints), [`second-gen-cpmd-ot-uks-kpoints`](dynamics-second-gen-cpmd.md#second-gen-cpmd-ot-uks-kpoints) |
| [Extended-Lagrangian BOMD (XL-BOMD)](dynamics-xl-bomd.md) | `dynamics/xl-bomd` | [`xl-bomd-water`](dynamics-xl-bomd.md#xl-bomd-water), [`xl-bomd-rhf-water`](dynamics-xl-bomd.md#xl-bomd-rhf-water), [`xl-bomd-uks`](dynamics-xl-bomd.md#xl-bomd-uks), [`xl-bomd-kpoints`](dynamics-xl-bomd.md#xl-bomd-kpoints), [`xl-bomd-linearized-forces`](dynamics-xl-bomd.md#xl-bomd-linearized-forces), [`xl-bomd-linearized-kpoints`](dynamics-xl-bomd.md#xl-bomd-linearized-kpoints), [`xl-bomd-linearized-uks`](dynamics-xl-bomd.md#xl-bomd-linearized-uks), [`xl-bomd-uks-kpoints`](dynamics-xl-bomd.md#xl-bomd-uks-kpoints) |

## Classical and electrochemical MD

Ensembles, constraints, many-body potentials, electrodes, fields and polarizable models. Oracle: **LAMMPS** where the feature overlaps, **MetalWalls** for MetalWalls-only physics, literature otherwise.

| Page | Case | Page sections |
| ---- | ---- | -------- |
| [Constant-pressure MD (NPT)](md-npt.md) | `md/npt` | [`surface-tension-coupling`](md-npt.md#surface-tension-coupling) |
| [Rigid water and virtual sites](md-constraints.md) | `md/constraints` | [`virtual-site-multi-topology`](md-constraints.md#virtual-site-multi-topology), [`four-site-tip4p-foothold`](md-constraints.md#four-site-tip4p-foothold) |
| [Many-body classical potentials (EAM, Stillinger–Weber)](md-manybody-potentials.md) | `md/manybody-potentials` | [`eam_cu`](md-manybody-potentials.md#eam_cu), [`sw_si`](md-manybody-potentials.md#sw_si) |
| [Constant-potential electrodes](md-constant-potential.md) | `md/constant-potential` | [`const-pot-bo-parallel-plate`](md-constant-potential.md#const-pot-bo-parallel-plate), [`const-pot-cg-large-electrode`](md-constant-potential.md#const-pot-cg-large-electrode), [`const-pot-charge-halo-mpi`](md-constant-potential.md#const-pot-charge-halo-mpi), [`const-pot-charge-observer`](md-constant-potential.md#const-pot-charge-observer), [`const-pot-il-edl-box`](md-constant-potential.md#const-pot-il-edl-box), [`const-pot-md-nvt-short`](md-constant-potential.md#const-pot-md-nvt-short), [`per-electrode-eta`](md-constant-potential.md#per-electrode-eta), [`gaussian-electrode-charge`](md-constant-potential.md#gaussian-electrode-charge), [`electrode-charge-predictor`](md-constant-potential.md#electrode-charge-predictor), [`freeze-electrode-atoms`](md-constant-potential.md#freeze-electrode-atoms), [`piston-electrode-pz`](md-constant-potential.md#piston-electrode-pz), [`dipole-electrode-joint`](md-constant-potential.md#dipole-electrode-joint) |
| [Constant-charge electrodes](md-constant-charge.md) | `md/constant-charge` | [`const-charge-electrode`](md-constant-charge.md#const-charge-electrode) |
| [Thomas–Fermi electrode metallicity](md-thomas-fermi.md) | `md/thomas-fermi` | [`thomas-fermi-electrode`](md-thomas-fermi.md#thomas-fermi-electrode), [`voronoi-volume-tf`](md-thomas-fermi.md#voronoi-volume-tf) |
| [Electrode observables](md-electrode-observables.md) | `md/electrode-observables` | [`capacitance-charge-fluctuation`](md-electrode-observables.md#capacitance-charge-fluctuation), [`charging-work-energy`](md-electrode-observables.md#charging-work-energy), [`potential-profile-observer`](md-electrode-observables.md#potential-profile-observer) |
| [External electric and displacement fields](md-electric-fields.md) | `md/electric-fields` | [`external-efield-md`](md-electric-fields.md#external-efield-md), [`oscillating-efield-md`](md-electric-fields.md#oscillating-efield-md), [`finite-field-single-electrode`](md-electric-fields.md#finite-field-single-electrode), [`constant-d-polarization-feedback`](md-electric-fields.md#constant-d-polarization-feedback) |
| [Polarizable and fluctuating-charge models](md-polarizable.md) | `md/polarizable` | [`pim-induced-dipole-bulk`](md-polarizable.md#pim-induced-dipole-bulk), [`pim-npt-polarizable-il`](md-polarizable.md#pim-npt-polarizable-il), [`extended-lagrangian-dipoles`](md-polarizable.md#extended-lagrangian-dipoles), [`qeq-fluctuating-charge`](md-polarizable.md#qeq-fluctuating-charge), [`maze-symplectic-charge-dynamics`](md-polarizable.md#maze-symplectic-charge-dynamics) |

## Force fields

Pair styles pinned against **LAMMPS**.

| Page | Case | Page sections |
| ---- | ---- | -------- |
| [Classical short-range pair styles and walls](forcefield-pair-styles.md) | `forcefield/pair-styles` | [`born-repulsion-foothold`](forcefield-pair-styles.md#born-repulsion-foothold), [`buckingham-pair-lammps`](forcefield-pair-styles.md#buckingham-pair-lammps), [`fumi-tosi-nacl`](forcefield-pair-styles.md#fumi-tosi-nacl), [`steele-wall-argon`](forcefield-pair-styles.md#steele-wall-argon) |

## Long-range electrostatics

Ewald / PME / slab corrections.

| Page | Case | Page sections |
| ---- | ---- | -------- |
| [Ewald and PME electrostatics](longrange-ewald-pme.md) | `longrange/ewald-pme` | [`slab-2d-ewald-foothold`](longrange-ewald-pme.md#slab-2d-ewald-foothold) |

## Sampling and free energy

| Page | Case | Page sections |
| ---- | ---- | -------- |
| [Free-energy methods](sampling-free-energy.md) | `sampling/free-energy` | [`constant-force-pull-ion`](sampling-free-energy.md#constant-force-pull-ion) |

## Cases without a method page

These cases pin their method in the regression oracle record only:

- scf: [`scf/rhf`](../../../regressions/scf/rhf/), [`scf/embedding`](../../../regressions/scf/embedding/), [`scf/relativistic`](../../../regressions/scf/relativistic/)
- gpw: [`gpw/bands-dos`](../../../regressions/gpw/bands-dos/)
- response: [`response/vibrations`](../../../regressions/response/vibrations/)
- semiempirical: [`semiempirical/am1`](../../../regressions/semiempirical/am1/), [`semiempirical/rm1`](../../../regressions/semiempirical/rm1/), [`semiempirical/am1-bcc`](../../../regressions/semiempirical/am1-bcc/)
- md: [`md/nve`](../../../regressions/md/nve/), [`md/langevin`](../../../regressions/md/langevin/), [`md/domain-decomposition`](../../../regressions/md/domain-decomposition/)
- forcefield: [`forcefield/gaff`](../../../regressions/forcefield/gaff/)
- longrange: [`longrange/pppm`](../../../regressions/longrange/pppm/), [`longrange/fmm`](../../../regressions/longrange/fmm/)
- sampling: [`sampling/umbrella`](../../../regressions/sampling/umbrella/), [`sampling/metadynamics`](../../../regressions/sampling/metadynamics/)
- path: [`path/chain-of-states`](../../../regressions/path/chain-of-states/), [`path/saddle-search`](../../../regressions/path/saddle-search/), [`path/pimd`](../../../regressions/path/pimd/), [`path/replica-exchange`](../../../regressions/path/replica-exchange/)
- mc: [`mc/metropolis`](../../../regressions/mc/metropolis/)
- multiscale: [`multiscale/qmmm`](../../../regressions/multiscale/qmmm/), [`multiscale/oniom`](../../../regressions/multiscale/oniom/)
- minimizer: [`minimizer/geometry`](../../../regressions/minimizer/geometry/), [`minimizer/cell-relax`](../../../regressions/minimizer/cell-relax/)
