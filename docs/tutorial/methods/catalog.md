# Method catalog

These inherited capability and benchmark tables are reference reports,
not a textbook reading sequence or a new verification of current support.

One page per delivered capability. Most pages map 1:1 to a capability case;
pages whose **Case** cell is `—` document API usage without a standing
end-to-end case. Every page is a technical report: what the method is, the
public API that runs it, the external code the number was pinned against, and
the measured wall against that code where a fair twin exists. These are
reference pages, not a reading order — the [tutorial index](../index.md) gives the reading
order.

Full index and template: [methods/README](README.md). What is
implemented on which backend is the [roadmap](../../roadmap/index.md).

### Cross-cutting support notes

- [Classical-model support notes](classical-models.md) owns the inherited
  EAM and Stillinger–Weber scope restrictions and the evidence needed to
  update them. Unlike an individual capability report, this note is a
  support checklist and does not claim a newly executed end-to-end case.

### Gaussian-basis electronic structure

Hartree–Fock, Kohn–Sham, post-HF, excited states and molecular properties. Oracle: **PySCF**.

| Method | Page | Case |
|--------|------|------------|
| ADC(2) | [adc2-h2co-sto3g](adc2-h2co-sto3g.md) | `adc2-h2co-sto3g` |
| ADC(2)-x | [adc2x-h2co-sto3g](adc2x-h2co-sto3g.md) | `adc2x-h2co-sto3g` |
| B2PLYP double hybrid | [b2plyp-h2o](b2plyp-h2o.md) | `b2plyp-h2o` |
| BLYP GGA | [blyp-water-sto3g](blyp-water-sto3g.md) | `blyp-water-sto3g` |
| BSE@G0W0 | [bse-h2-sto3g](bse-h2-sto3g.md) | `bse-h2-sto3g` |
| CASCI | [casci-h2-sto3g](casci-h2-sto3g.md) | — |
| CASSCF | [casscf-h2-sto3g](casscf-h2-sto3g.md) | `casscf-h2-sto3g` |
| CC2 | [cc2-h2-sto3g](cc2-h2-sto3g.md) | `cc2-h2-sto3g` |
| CCSD nuclear gradient | [ccsd-grad-h2o-sto3g](ccsd-grad-h2o-sto3g.md) | `ccsd-grad-h2o-sto3g` |
| CCSD | [ccsd-h2-sto3g](ccsd-h2-sto3g.md) | `ccsd-h2-sto3g` |
| CCSD(T) | [ccsd-t-h2o-sto3g](ccsd-t-h2o-sto3g.md) | `ccsd-t-h2o-sto3g` |
| CIS(D) | [cisd-h2co-sto3g](cisd-h2co-sto3g.md) | `cisd-h2co-sto3g` |
| ΔSCF (IMOM) | [delta-scf-n2](delta-scf-n2.md) | `delta-scf-n2` |
| DFT-D4 (PBE params) | [dft-d4-h2o-pbe](dft-d4-h2o-pbe.md) | `dft-d4-h2o-pbe` |
| Direct RPA | [direct-rpa-h2-sto3g](direct-rpa-h2-sto3g.md) | `direct-rpa-h2-sto3g` |
| EA-ADC(2) | [ea-adc2-h2-sto3g](ea-adc2-h2-sto3g.md) | `ea-adc2-h2-sto3g` |
| EOM-CC2 | [eom-cc2-h2](eom-cc2-h2.md) | `eom-cc2-h2` |
| EOM-EA-CCSD | [eom-ea-ccsd](eom-ea-ccsd.md) | `eom-ea-ccsd` |
| EOM-EE-CCSD | [eom-ee-ccsd](eom-ee-ccsd.md) | `eom-ee-ccsd` |
| EOM-IP-CCSD | [eom-ip-ccsd](eom-ip-ccsd.md) | `eom-ip-ccsd` |
| EOM-SF-CCSD | [eom-sf-ccsd-o2](eom-sf-ccsd-o2.md) | `eom-sf-ccsd-o2` |
| Frozen-core RMP2 | [frozen-core-mp2-h2o](frozen-core-mp2-h2o.md) | `frozen-core-mp2-h2o` |
| Full LR-TDDFT (RKS) | [full-lr-tddft-h2-pbe](full-lr-tddft-h2-pbe.md) | `full-lr-tddft-h2-pbe` |
| G0W0 | [g0w0-h2-sto3g](g0w0-h2-sto3g.md) | `g0w0-h2-sto3g` |
| One-shot GF2 (G0F2) | [gf2-h2-sto3g](gf2-h2-sto3g.md) | `gf2-h2-sto3g` |
| GHF analytic gradient | [ghf-grad-h3-sto3g](ghf-grad-h3-sto3g.md) | `ghf-grad-h3-sto3g` |
| Hybrid TDDFT-TDA (B3LYP) | [h2-b3lyp-hybrid-tda](h2-b3lyp-hybrid-tda.md) | `h2-b3lyp-hybrid-tda` |
| Hybrid full-LR TDDFT (B3LYP RPA) | [h2-b3lyp-rpa-tddft](h2-b3lyp-rpa-tddft.md) | `h2-b3lyp-rpa-tddft` |
| IEF-PCM | [ief-pcm-h2o](ief-pcm-h2o.md) | `ief-pcm-h2o` |
| IP-ADC(2) | [ip-adc2-h2o-sto3g](ip-adc2-h2o-sto3g.md) | `ip-adc2-h2o-sto3g` |
| RMP3 | [mp3-h2-sto3g](mp3-h2-sto3g.md) | `mp3-h2-sto3g` |
| RMP4(SDTQ) | [mp4-h2-sto3g](mp4-h2-sto3g.md) | `mp4-h2-sto3g` |
| SC-NEVPT2 | [nevpt2-h2-sto3g](nevpt2-h2-sto3g.md) | `nevpt2-h2-sto3g` |
| C-PCM free-energy forces | [pcm-grad-h2o](pcm-grad-h2o.md) | `pcm-grad-h2o` |
| RKS nuclear Hessian (FD of forces) | [rks-hessian-h2o-sto3g](rks-hessian-h2o-sto3g.md) | `rks-hessian-h2o-sto3g` |
| ROHF nuclear Hessian | [rohf-hessian-o2](rohf-hessian-o2.md) | `rohf-hessian-o2` |
| ROKS analytic gradient | [roks-grad-o2-sto3g](roks-grad-o2-sto3g.md) | `roks-grad-o2-sto3g` |
| ROKS | [roks-o2-sto3g](roks-o2-sto3g.md) | `roks-o2-sto3g` |
| ROMP2 | [romp2-o2-sto3g](romp2-o2-sto3g.md) | `romp2-o2-sto3g` |
| SCS-MP2 | [scs-mp2-h2o-sto3g](scs-mp2-h2o-sto3g.md) | `scs-mp2-h2o-sto3g` |
| Spin-flip full TDDFT (SF-RPA) | [sf-tddft-o2](sf-tddft-o2.md) | `sf-tddft-o2` |
| Spin-flip TDA (SF-CIS) | [spin-flip-tda-o2](spin-flip-tda-o2.md) | `spin-flip-tda-o2` |
| RHF static polarizability | [static-polariz-h2o](static-polariz-h2o.md) | `static-polariz-h2o` |
| UKS full-LR TDDFT (LSDA O₂) | [uks-full-lr-o2-lsda](uks-full-lr-o2-lsda.md) | `uks-full-lr-o2-lsda` |
| UKS nuclear Hessian (FD of forces) | [uks-hessian-o2-sto3g](uks-hessian-o2-sto3g.md) | `uks-hessian-o2-sto3g` |
| UMP2 analytic gradient | [ump2-grad-o2-sto3g](ump2-grad-o2-sto3g.md) | `ump2-grad-o2-sto3g` |
| UMP2 | [ump2-o2-sto3g](ump2-o2-sto3g.md) | `ump2-o2-sto3g` |
| Maximally localized Wannier functions (Foster–Boys) | [wannier-centers](wannier-centers.md) | `wannier-centers` |

### Linear response and phonons

DFPT across the spin / GGA / finite-q / smearing grid, and phonon dispersion. Oracle: **Quantum ESPRESSO**.

| Method | Page | Case |
|--------|------|------------|
| Finite-\(q\) DFPT phonons at X (diamond Si) | [dfpt-finite-q-si](dfpt-finite-q-si.md) | `dfpt-finite-q-si` |
| Finite-\(q\) DFPT + metallic smearing (Si at X) | [dfpt-finite-q-smearing](dfpt-finite-q-smearing.md) | `dfpt-finite-q-smearing` |
| Finite-\(q\) DFPT phonons at X, pure GGA kernel (Si) | [dfpt-gga-finite-q-si](dfpt-gga-finite-q-si.md) | `dfpt-gga-finite-q-si` |
| DFPT phonons at Γ, pure GGA kernel (Si) | [dfpt-gga-si](dfpt-gga-si.md) | `dfpt-gga-si` |
| Finite-\(q\) DFPT at X, joint spin × GGA (Si) | [dfpt-gga-spin-finite-q-si](dfpt-gga-spin-finite-q-si.md) | `dfpt-gga-spin-finite-q-si` |
| Finite-\(q\) DFPT at X, joint spin × GGA × smear (Si) | [dfpt-gga-spin-finite-q-smearing](dfpt-gga-spin-finite-q-smearing.md) | `dfpt-gga-spin-finite-q-smearing` |
| DFPT phonons at Γ, joint spin × GGA (Si) | [dfpt-gga-spin-si](dfpt-gga-spin-si.md) | `dfpt-gga-spin-si` |
| DFPT phonons at Γ, joint spin × GGA × smear (Si) | [dfpt-gga-spin-smearing](dfpt-gga-spin-smearing.md) | `dfpt-gga-spin-smearing` |
| Incommensurate-\(q\) DFPT phonons (Si) | [dfpt-incommensurate-si](dfpt-incommensurate-si.md) | `dfpt-incommensurate-si` |
| DFPT phonons + metallic smearing (Si) | [dfpt-metallic-si](dfpt-metallic-si.md) | `dfpt-metallic-si` |
| DFPT phonons at Γ (diamond Si, PW LSDA) | [dfpt-phonon-si](dfpt-phonon-si.md) | `dfpt-phonon-si` |
| Collinear-spin DFPT at incommensurate \(q\) (Si) | [dfpt-spin-incommensurate](dfpt-spin-incommensurate.md) | `dfpt-spin-incommensurate` |
| DFPT phonons at Γ, collinear spin (Si) | [dfpt-spin-si](dfpt-spin-si.md) | `dfpt-spin-si` |
| DFPT: collinear spin + metallic smearing (Si) | [dfpt-spin-smearing](dfpt-spin-smearing.md) | `dfpt-spin-smearing` |
| Phonon dispersion \(\omega(q)\) (diamond Si) | [phonon-dispersion-si](phonon-dispersion-si.md) | `phonon-dispersion-si` |

### Plane-wave DFT

Periodic Kohn–Sham in a plane-wave basis with norm-conserving pseudopotentials. Oracle: **Quantum ESPRESSO**.

| Method | Page | Case |
|--------|------|------------|
| PW bands + DOS bulk Si | [bands-dos-si](bands-dos-si.md) | `bands-dos-si` |
| PW metallic-occupation bands Si | [bands-metallic-si](bands-metallic-si.md) | `bands-metallic-si` |
| PW collinear spin band structure (bulk Si) | [bands-spin-si](bands-spin-si.md) | `bands-spin-si` |
| Collinear-spin Berry-phase polarization (Si) | [berry-spin-si](berry-spin-si.md) | `berry-spin-si` |
| Metallic-occupation DOS (bulk Si) | [dos-metallic-si](dos-metallic-si.md) | `dos-metallic-si` |
| FD supercell phonons diamond Si (Γ) | [fd-phonon-supercell](fd-phonon-supercell.md) | `fd-phonon-supercell` |
| MPI plane-wave k-mesh scaling (4 ranks) | [mpi-pw-scaling-4rank](mpi-pw-scaling-4rank.md) | `mpi-pw-scaling-4rank` |
| Periodic multi-k MLWF (bulk Si) | [periodic-wannier-si](periodic-wannier-si.md) | `periodic-wannier-si` |
| Collinear spin-polarized multi-k MLWF (Si, UKS) | [periodic-wannier-uks](periodic-wannier-uks.md) | `periodic-wannier-uks` |
| PW Nielsen–Martin analytic stress Si | [pw-analytic-stress-si](pw-analytic-stress-si.md) | `pw-analytic-stress-si` |
| PW BOMD H (NVE foothold) | [pw-bomd-h](pw-bomd-h.md) | `pw-bomd-h` |
| PW LDA Si (k-mesh) | [pw-lda-si](pw-lda-si.md) | `pw-lda-si` |
| PW true PBE GGA Si | [pw-pbe-gga-si](pw-pbe-gga-si.md) | `pw-pbe-gga-si` |
| Metallic Fermi–Dirac Al | [pw-smearing-al](pw-smearing-al.md) | — |
| PW collinear spin-GGA PBE (O₂ triplet, Γ) | [pw-spin-gga](pw-spin-gga.md) | `pw-spin-gga` |
| PW collinear spin O₂ | [pw-spin-polarized-o2](pw-spin-polarized-o2.md) | `pw-spin-polarized-o2` |
| PW collinear UKS + metallic smearing (BCC Fe) | [pw-uks-fe-smearing](pw-uks-fe-smearing.md) | `pw-uks-fe-smearing` |
| PW k-point collinear UKS O₂ | [pw-uks-kmesh-fe](pw-uks-kmesh-fe.md) | — |
| PW isotropic variable-cell relax (diamond Si) | [pw-variable-cell-si](pw-variable-cell-si.md) | `pw-variable-cell-si` |

### Gaussian and plane waves (GPW)

Periodic Kohn–Sham on a Gaussian basis collocated on a grid, its derivatives, dispersion set and correlated extensions. Oracle: **CP2K**.

| Method | Page | Case |
|--------|------|------------|
| GPW analytic fixed-\(P\) stress H₂O | [gpw-analytic-stress](gpw-analytic-stress.md) | `gpw-analytic-stress` |
| GPW Γ-RKS BLYP, isolated H₂O (MT) | [gpw-blyp-water](gpw-blyp-water.md) | — |
| GPW BOMD H₂O NVE energy-drift foothold | [gpw-bomd-water](gpw-bomd-water.md) | `gpw-bomd-water` |
| GPW PBE-D3 + ATM (C9) isolated dimer | [gpw-d3-atm](gpw-d3-atm.md) | `gpw-d3-atm` |
| GPW PBE-D3 analytic ionic forces | [gpw-d3-forces](gpw-d3-forces.md) | `gpw-d3-forces` |
| GPW PBE-D3 (zero damping) analytic cell stress | [gpw-d3-stress](gpw-d3-stress.md) | `gpw-d3-stress` |
| GPW PBE-D3(BJ) analytic ionic forces | [gpw-d3bj-forces](gpw-d3bj-forces.md) | `gpw-d3bj-forces` |
| GPW PBE-D3(BJ) analytic cell stress | [gpw-d3bj-stress](gpw-d3bj-stress.md) | `gpw-d3bj-stress` |
| GPW DFT-D4 three-body ATM (\(s_9=1\)) analytic cell stress | [gpw-d4-atm-stress](gpw-d4-atm-stress.md) | `gpw-d4-atm-stress` |
| GPW PBE-D4 analytic ionic forces | [gpw-d4-forces](gpw-d4-forces.md) | `gpw-d4-forces` |
| GPW PBE-D4 analytic cell stress | [gpw-d4-stress](gpw-d4-stress.md) | `gpw-d4-stress` |
| GPW PBE-D4 periodic water dimer | [gpw-d4-water](gpw-d4-water.md) | `gpw-d4-water` |
| Multi-k DFT+U analytic cell stress (H₂ UKS) | [gpw-dftu-kpoint-stress](gpw-dftu-kpoint-stress.md) | `gpw-dftu-kpoint-stress` |
| Multi-k Dudarev DFT+U (GPW UKS) | [gpw-dftu-kpoints](gpw-dftu-kpoints.md) | — |
| Dudarev DFT+U NiO mol (Γ-UKS GPW) | [gpw-dftu-nio](gpw-dftu-nio.md) | `gpw-dftu-nio` |
| GPW analytic cell stress + DFT+U (NiO) | [gpw-dftu-stress](gpw-dftu-stress.md) | `gpw-dftu-stress` |
| GPW analytic forces H₂O | [gpw-forces-water](gpw-forces-water.md) | `gpw-forces-water` |
| GPW multi-k Hirshfeld volumes | [gpw-hirshfeld-kpoints](gpw-hirshfeld-kpoints.md) | `gpw-hirshfeld-kpoints` |
| Γ-UKS Hirshfeld-scaled MBD@rsSCS (O₂) | [gpw-hirshfeld-uks](gpw-hirshfeld-uks.md) | `gpw-hirshfeld-uks` |
| Multi-k UKS Hirshfeld-scaled MBD@rsSCS (O₂) | [gpw-hirshfeld-uks-kpoints](gpw-hirshfeld-uks-kpoints.md) | `gpw-hirshfeld-uks-kpoints` |
| GPW analytic forces at k-points (H₂) | [gpw-kpoint-forces](gpw-kpoint-forces.md) | — |
| GPW analytic cell stress at k-points (Si) | [gpw-kpoint-stress](gpw-kpoint-stress.md) | `gpw-kpoint-stress` |
| Multi-k UKS GPW analytic forces | [gpw-kpoint-uks-forces](gpw-kpoint-uks-forces.md) | — |
| Multi-k UKS GPW analytic cell stress | [gpw-kpoint-uks-stress](gpw-kpoint-uks-stress.md) | — |
| GPW collinear multi-k UKS (H₂ + O₂ twin) | [gpw-kpoints-uks](gpw-kpoints-uks.md) | — |
| GPW MBD@rsSCS ionic forces | [gpw-mbd-forces](gpw-mbd-forces.md) | `gpw-mbd-forces` |
| GPW PBE-MBD@rsSCS + Hirshfeld volumes | [gpw-mbd-hirshfeld](gpw-mbd-hirshfeld.md) | `gpw-mbd-hirshfeld` |
| GPW PBE-MBD@rsSCS, free-atom water dimer | [gpw-mbd-ice](gpw-mbd-ice.md) | `gpw-mbd-ice` |
| GPW multi-k analytic forces + MBD@rsSCS | [gpw-mbd-kpoint-forces](gpw-mbd-kpoint-forces.md) | `gpw-mbd-kpoint-forces` |
| GPW PBE-MBD@rsSCS, periodic Ewald + k-mesh | [gpw-mbd-periodic](gpw-mbd-periodic.md) | `gpw-mbd-periodic` |
| GPW Γ cell stress with periodic MBD@rsSCS | [gpw-mbd-stress](gpw-mbd-stress.md) | `gpw-mbd-stress` |
| Metallic FD smearing (BCC Na, GPW multi-k) | [gpw-metallic-smearing](gpw-metallic-smearing.md) | — |
| GPW optB88-vdW (periodic water dimer) | [gpw-optb88-vdw](gpw-optb88-vdw.md) | — |
| GPW PBE-D3(BJ) ice dimer | [gpw-pbe-d3bj-ice](gpw-pbe-d3bj-ice.md) | `gpw-pbe-d3bj-ice` |
| GPW Γ-RKS PBEsol Si₂ (MT) | [gpw-pbesol-si](gpw-pbesol-si.md) | — |
| GPW Γ-RKS revPBE (Si₂ fragment, MT) | [gpw-revpbe-si](gpw-revpbe-si.md) | — |
| GPW Γ-RKS RPBE, isolated H₂O (MT) | [gpw-rpbe-water](gpw-rpbe-water.md) | — |
| GPW PBE + rVV10 (periodic water dimer) | [gpw-rvv10-ice](gpw-rvv10-ice.md) | `gpw-rvv10-ice` |
| GPW cell stress strain-FD H₂O (NUMERICAL pin) | [gpw-stress-water](gpw-stress-water.md) | `gpw-stress-water` |
| GPW PBE-TS (periodic water dimer) | [gpw-ts-ice](gpw-ts-ice.md) | `gpw-ts-ice` |
| Multi-k free-atom TS analytic ionic forces (H₂) | [gpw-ts-kpoint-forces](gpw-ts-kpoint-forces.md) | — |
| Γ-UKS GPW BLYP (O₂) | [gpw-uks-blyp](gpw-uks-blyp.md) | — |
| Γ-UKS GPW analytic forces (O₂ / PBE) | [gpw-uks-forces](gpw-uks-forces.md) | `gpw-uks-forces` |
| Γ-UKS GPW multigrid (O₂ / PADE) | [gpw-uks-multigrid](gpw-uks-multigrid.md) | `gpw-uks-multigrid` |
| Γ-UKS GPW O₂ | [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md) | `gpw-uks-o2-gamma` |
| Γ-UKS GPW PBE (O₂) | [gpw-uks-pbe](gpw-uks-pbe.md) | `gpw-uks-pbe` |
| Γ-UKS GPW revPBE (O₂) | [gpw-uks-revpbe](gpw-uks-revpbe.md) | — |
| Γ-UKS GPW RPBE (O₂) | [gpw-uks-rpbe](gpw-uks-rpbe.md) | — |
| Collinear UKS + FD smearing (BCC Na) | [gpw-uks-smearing](gpw-uks-smearing.md) | — |
| Γ-UKS GPW analytic cell stress (O₂ / PBE) | [gpw-uks-stress](gpw-uks-stress.md) | `gpw-uks-stress` |
| GPW `CELL_OPT` MOLOPT re-twin (water chain) | [gpw-variable-cell-molopt](gpw-variable-cell-molopt.md) | `gpw-variable-cell-molopt` |
| GPW PBE + vdW-DF (DRSLL) | [gpw-vdw-df-ice](gpw-vdw-df-ice.md) | `gpw-vdw-df-ice` |
| GPW PBE + vdW-DF2 (LMKLL) | [gpw-vdw-df2-ice](gpw-vdw-df2-ice.md) | — |
| GPW nonlocal vdW on the REL_CUTOFF multigrid | [gpw-vdw-multigrid](gpw-vdw-multigrid.md) | `gpw-vdw-multigrid` |
| GPW nonlocal vdW: SC \(V_{\mathrm{nl}}\) + forces | [gpw-vdw-scf-potential](gpw-vdw-scf-potential.md) | `gpw-vdw-scf-potential` |
| GPW Γ-RKS H₂O PBE (MT) | [gpw-water-pbe-mt](gpw-water-pbe-mt.md) | `gpw-water-pbe-mt` |
| GPW free-atom XDM (periodic water dimer) | [gpw-xdm-ice](gpw-xdm-ice.md) | `gpw-xdm-ice` |
| MPI GPW k-point RR (n=1 vs n=4) | [mpi-gpw-scaling-4rank](mpi-gpw-scaling-4rank.md) | `mpi-gpw-scaling-4rank` |
| OT + Dudarev DFT+U, Γ-UKS GPW (NiO) | [ot-dftu](ot-dftu.md) | `ot-dftu` |
| Multi-k OT + Dudarev DFT+U (H₂) | [ot-dftu-kpoints](ot-dftu-kpoints.md) | — |
| Multi-k OT + DFT+U + FD smearing (H₂) | [ot-dftu-smearing](ot-dftu-smearing.md) | — |
| Multi-k Orbital Transformation SCF (H₂) | [ot-kpoints](ot-kpoints.md) | — |
| OT SCF water (GPW PBE) | [ot-minimizer-water](ot-minimizer-water.md) | `ot-minimizer-water` |
| OT + metallic FD smearing (BCC Na) | [ot-smearing](ot-smearing.md) | — |
| Multi-k UKS OT + Dudarev DFT+U (H₂) | [ot-uks-dftu](ot-uks-dftu.md) | — |
| Multi-k UKS OT + DFT+U + smearing (H₂) | [ot-uks-dftu-smearing](ot-uks-dftu-smearing.md) | — |
| Multi-k collinear UKS OT (high-spin H₂) | [ot-uks-kpoints](ot-uks-kpoints.md) | — |
| OT orbital minimizer for Γ-UKS (O₂) | [ot-uks-o2](ot-uks-o2.md) | `ot-uks-o2` |
| UKS OT + metallic FD smearing (BCC Na) | [ot-uks-smearing](ot-uks-smearing.md) | — |
| Periodic finite-\(q\) inter-\(k\) G0W0 (Si) | [periodic-gw-finite-q](periodic-gw-finite-q.md) | — |
| Multi-k spectral G0W0 (diamond Si, bct) | [periodic-gw-kpoints](periodic-gw-kpoints.md) | — |
| Periodic Γ-point spectral G0W0 (sc-Si₂) | [periodic-gw-si](periodic-gw-si.md) | — |
| Multi-k RI-GPW MP2 (diamond Si, bct) | [periodic-mp2-kpoints](periodic-mp2-kpoints.md) | — |
| Periodic Γ-point GPW MP2, canonical (sc-Si₂) | [periodic-mp2-si](periodic-mp2-si.md) | — |
| Periodic Γ-point RI-GPW MP2 (sc-Si₂) | [periodic-ri-mp2-si](periodic-ri-mp2-si.md) | `periodic-ri-mp2-si` |
| Finite-\(q\) inter-\(k\) RI-GPW dRPA (Si) | [periodic-rpa-finite-q](periodic-rpa-finite-q.md) | — |
| Multi-k RI-GPW direct RPA (Si₄, bct) | [periodic-rpa-kpoints](periodic-rpa-kpoints.md) | — |
| Periodic Γ-point GPW direct RPA (sc-Si₂) | [periodic-rpa-si](periodic-rpa-si.md) | — |
| Multi-k unrestricted spectral G0W0 (UGW) | [periodic-ugw-kpoints](periodic-ugw-kpoints.md) | — |
| Periodic Γ-point UGW@UKS (sc-Si₂) | [periodic-ugw-si](periodic-ugw-si.md) | — |
| Multi-k unrestricted GPW MP2 (UMP2) | [periodic-ump2-kpoints](periodic-ump2-kpoints.md) | — |
| Periodic Γ-point GPW UMP2 (sc-Si₂) | [periodic-ump2-si](periodic-ump2-si.md) | — |
| Multi-k unrestricted direct RPA (URPA@UKS) | [periodic-urpa-kpoints](periodic-urpa-kpoints.md) | — |
| Periodic Γ-point URPA@UKS (sc-Si₂) | [periodic-urpa-si](periodic-urpa-si.md) | — |
| Multi-k MLWF on GPW (Marzari–Vanderbilt) | [periodic-wannier-gpw](periodic-wannier-gpw.md) | `periodic-wannier-gpw` |

### Ab-initio molecular dynamics

Born–Oppenheimer, Car–Parrinello, second-generation CPMD and XL-BOMD over the electronic methods above. Oracle: **CP2K**.

| Method | Page | Case |
|--------|------|------------|
| Multi-k Car–Parrinello MD on Bloch \(C_k\) H₂ NVE | [cpmd-kpoints](cpmd-kpoints.md) | `cpmd-kpoints` |
| Car–Parrinello MD + electronic RATTLE H₂O NVE | [cpmd-rattle-water](cpmd-rattle-water.md) | `cpmd-rattle-water` |
| AO-basis CPMD on the SCF methods (RHF / RKS / UHF) | [cpmd-rhf-h2](cpmd-rhf-h2.md) | `cpmd-rhf-h2` |
| Car–Parrinello MD on Γ-UKS (O₂ triplet NVE) | [cpmd-uks](cpmd-uks.md) | `cpmd-uks` |
| Multi-k UKS CPMD on dual-spin Bloch \(C_k^\sigma\) (H₂ NVE) | [cpmd-uks-kpoints](cpmd-uks-kpoints.md) | `cpmd-uks-kpoints` |
| Multi-k UKS CPMD + electronic Nosé–Hoover per (k, σ) (H₂ NVE) | [cpmd-uks-thermostat-kpoints](cpmd-uks-thermostat-kpoints.md) | — |
| Car–Parrinello MD H₂O NVE (fictitious e⁻ DOF) | [cpmd-water](cpmd-water.md) | `cpmd-water` |
| Multi-k second-gen CP (ASPC on \(P_k\)) H₂ NVE | [second-gen-cpmd-kpoints](second-gen-cpmd-kpoints.md) | `second-gen-cpmd-kpoints` |
| Second-generation CP with OT corrector (H₂O) | [second-gen-cpmd-ot](second-gen-cpmd-ot.md) | `second-gen-cpmd-ot` |
| Multi-k second-gen CP + OT corrector H₂ NVE | [second-gen-cpmd-ot-kpoints](second-gen-cpmd-ot-kpoints.md) | `second-gen-cpmd-ot-kpoints` |
| Multi-k UKS second-gen CP + OT corrector H₂ NVE | [second-gen-cpmd-ot-uks-kpoints](second-gen-cpmd-ot-uks-kpoints.md) | — |
| Second-generation CP on Γ-UKS (O₂ triplet) | [second-gen-cpmd-uks](second-gen-cpmd-uks.md) | `second-gen-cpmd-uks` |
| Second-generation CP + OT on Γ-UKS (O₂) | [second-gen-cpmd-uks-ot](second-gen-cpmd-uks-ot.md) | `second-gen-cpmd-uks-ot` |
| Second-generation CP: ASPC + partial SCF (H₂O) | [second-gen-cpmd-water](second-gen-cpmd-water.md) | `second-gen-cpmd-water` |
| Multi-k XL-BOMD (H₂ NVE, GPW + k-mesh) | [xl-bomd-kpoints](xl-bomd-kpoints.md) | `xl-bomd-kpoints` |
| XL-BOMD on the Gaussian SCF methods (RHF H₂O NVE) | [xl-bomd-rhf-water](xl-bomd-rhf-water.md) | `xl-bomd-rhf-water` |
| XL-BOMD on Γ-UKS (O₂ triplet NVE, GPW) | [xl-bomd-uks](xl-bomd-uks.md) | `xl-bomd-uks` |
| XL-BOMD, extended-Lagrangian BOMD (H₂O NVE) | [xl-bomd-water](xl-bomd-water.md) | `xl-bomd-water` |

### Semiempirical and tight binding

NDDO, DFTB and extended tight binding. Family-locked oracles: NDDO → **OpenMOPAC**, DFTB → **DFTB+**, GFN-xTB → **xtb**. Narrative: [Semiempirical Models](../ch17-semiempirical-am1.md).

| Method | Page | Case |
|--------|------|------------|
| Non-SCC DFTB0 analytic nuclear forces (H₂, H₂O, CO) | [dftb0-analytic-forces](dftb0-analytic-forces.md) | `dftb0-analytic-forces` |
| SCC-DFTB2 CH₄ 3ob-3-1 organic SK | [dftb2-3ob-organic](dftb2-3ob-organic.md) | `dftb2-3ob-organic` |
| SCC-DFTB2 analytic nuclear forces H₂O | [dftb2-analytic-forces](dftb2-analytic-forces.md) | `dftb2-analytic-forces` |
| SCC-DFTB2 H₂ total \(E\) vs mio-1-1 | [dftb2-dftbplus-energy-pin](dftb2-dftbplus-energy-pin.md) | `dftb2-dftbplus-energy-pin` |
| Multi-k SCC-DFTB2 (4×1×1 C chain) | [dftb2-kpoints](dftb2-kpoints.md) | — |
| Γ-PBC analytic nuclear forces (off-eq C chain) | [dftb2-periodic-forces](dftb2-periodic-forces.md) | — |
| Γ-point periodic SCC-DFTB2 (1-D C chain) | [dftb2-periodic-gamma](dftb2-periodic-gamma.md) | — |
| SCC-DFTB3 H₂O (mio-style; third-order \(E\)) | [dftb3-water-mio](dftb3-water-mio.md) | `dftb3-water-mio` |
| GFN1/GFN2-xTB FD Hessian of analytic \(G\) (off-eq H₂O) | [gfn-xtb-analytic-hessian](gfn-xtb-analytic-hessian.md) | `gfn-xtb-analytic-hessian` |
| GFN0-xTB FD Hessian of analytic \(G\) (off-eq H₂O) | [gfn0-xtb-analytic-hessian](gfn0-xtb-analytic-hessian.md) | `gfn0-xtb-analytic-hessian` |
| GFN0-xTB analytic nuclear forces (off-eq H₂O) | [gfn0-xtb-forces](gfn0-xtb-forces.md) | `gfn0-xtb-forces` |
| GFN1-xTB analytic nuclear forces (off-eq H₂O) | [gfn1-xtb-forces](gfn1-xtb-forces.md) | `gfn1-xtb-forces` |
| GFN1-xTB organic energy (CH₄, H/C/N/O) | [gfn1-xtb-organic](gfn1-xtb-organic.md) | `gfn1-xtb-organic` |
| GFN2-xTB AES in the SCC Fock (off-eq H₂O) | [gfn2-aes-fock](gfn2-aes-fock.md) | `gfn2-aes-fock` |
| GFN2-xTB charge–quadrupole AES + \(q_{\mathrm{kernel}}\) AXC (H₂O) | [gfn2-quadrupole-aes](gfn2-quadrupole-aes.md) | `gfn2-quadrupole-aes` |
| GFN2-xTB analytic nuclear forces (off-eq H₂O) | [gfn2-xtb-forces](gfn2-xtb-forces.md) | `gfn2-xtb-forces` |
| IPEA-xTB FD Hessian of analytic \(G\) (Td CH₄) | [ipea-xtb-analytic-hessian](ipea-xtb-analytic-hessian.md) | `ipea-xtb-analytic-hessian` |
| IPEA-xTB analytic nuclear forces (off-eq CH₄) | [ipea-xtb-forces](ipea-xtb-forces.md) | `ipea-xtb-forces` |
| IPEA-xTB organic energy (CH₄, extra \((n+1)s\)) | [ipea-xtb-small](ipea-xtb-small.md) | `ipea-xtb-small` |
| IPEA-xTB ΔSCC vertical IP/EA (Td CH₄, `ipeashift`) | [ipea-xtb-vip](ipea-xtb-vip.md) | `ipea-xtb-vip` |
| Collinear spin-DFTB2 O₂ triplet | [spin-dftb2-o2](spin-dftb2-o2.md) | `spin-dftb2-o2` |
| Linear-response TD-DFTB (Casida) H₂ singlet/triplet | [td-dftb-h2](td-dftb-h2.md) | `td-dftb-h2` |

### Classical MD and electrochemistry

Force fields, constraints, non-equilibrium drives, and constant-potential electrodes. Oracles: **LAMMPS** and **MetalWalls**. Narrative: [Molecular Dynamics](../ch10-classical-md.md).

| Method | Page | Case |
|--------|------|------------|
| Capacitance from electrode charge fluctuations | [capacitance-charge-fluctuation](capacitance-charge-fluctuation.md) | `capacitance-charge-fluctuation` |
| Charging work \(-\sum_i \Psi_i Q_i\) | [charging-work-energy](charging-work-energy.md) | `charging-work-energy` |
| Constant-charge electrode control (conq) | [const-charge-electrode](const-charge-electrode.md) | `const-charge-electrode` |
| Const-pot BO parallel plate (madelung) | [const-pot-bo-parallel-plate](const-pot-bo-parallel-plate.md) | `const-pot-bo-parallel-plate` |
| Const-pot CG large multi-site electrode | [const-pot-cg-large-electrode](const-pot-cg-large-electrode.md) | `const-pot-cg-large-electrode` |
| Electrode total-charge observer | [const-pot-charge-observer](const-pot-charge-observer.md) | `const-pot-charge-observer` |
| Const-pot MD NVT short trajectory | [const-pot-md-nvt-short](const-pot-md-nvt-short.md) | `const-pot-md-nvt-short` |
| Constant-force ion pull (`fix addforce`) | [constant-force-pull-ion](constant-force-pull-ion.md) | `constant-force-pull-ion` |
| Coupled PIM dipoles ⊕ electrode \(Q\) | [dipole-electrode-joint](dipole-electrode-joint.md) | `dipole-electrode-joint` |
| Electrode charge predictor (Kolafa ASPC) | [electrode-charge-predictor](electrode-charge-predictor.md) | `electrode-charge-predictor` |
| External uniform E-field MD | [external-efield-md](external-efield-md.md) | `external-efield-md` |
| Finite-field + single electrode (Dufils 3D) | [finite-field-single-electrode](finite-field-single-electrode.md) | `finite-field-single-electrode` |
| Four-site TIP4P foothold | [four-site-tip4p-foothold](four-site-tip4p-foothold.md) | `four-site-tip4p-foothold` |
| Freeze electrode atoms (`setforce 0`) | [freeze-electrode-atoms](freeze-electrode-atoms.md) | `freeze-electrode-atoms` |
| Gaussian electrode charge kernels | [gaussian-electrode-charge](gaussian-electrode-charge.md) | `gaussian-electrode-charge` |
| Oscillating external E-field MD | [oscillating-efield-md](oscillating-efield-md.md) | `oscillating-efield-md` |
| Bulk PIM self-consistent induced dipoles | [pim-induced-dipole-bulk](pim-induced-dipole-bulk.md) | `pim-induced-dipole-bulk` |
| Rigid piston electrode constant \(P_z\) | [piston-electrode-pz](piston-electrode-pz.md) | `piston-electrode-pz` |
| Slab \(\Psi(z)\) planar Poisson observer | [potential-profile-observer](potential-profile-observer.md) | `potential-profile-observer` |
| Slab 2D Ewald foothold (Yeh–Berkowitz) | [slab-2d-ewald-foothold](slab-2d-ewald-foothold.md) | `slab-2d-ewald-foothold` |
| Surface-tension / NPγT coupling | [surface-tension-coupling](surface-tension-coupling.md) | `surface-tension-coupling` |
| Thomas–Fermi finite metallicity electrode | [thomas-fermi-electrode](thomas-fermi-electrode.md) | `thomas-fermi-electrode` |
| Multi-topology virtual sites TIP4P+TIP5P | [virtual-site-multi-topology](virtual-site-multi-topology.md) | `virtual-site-multi-topology` |

### Force-field styles

Individual pair and bonded styles pinned against **LAMMPS**.

| Method | Page | Case |
|--------|------|------------|
| Born–Mayer–Huggins pair foothold | [born-repulsion-foothold](born-repulsion-foothold.md) | `born-repulsion-foothold` |
| Buckingham pair (LAMMPS `buck`) | [buckingham-pair-lammps](buckingham-pair-lammps.md) | `buckingham-pair-lammps` |
| Fumi–Tosi multi-type Born + Coulomb NaCl | [fumi-tosi-nacl](fumi-tosi-nacl.md) | `fumi-tosi-nacl` |
| Steele 10-4-3 wall (Ar single-point) | [steele-wall-argon](steele-wall-argon.md) | `steele-wall-argon` |

SCF reading map: [User Guide → SCF Learning Path](../../user-guide/scf-roadmap.md).
