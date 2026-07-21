# Excited States and Many-Body Methods

Legend and caveats: [Roadmap index](index.md).

**Nothing in this chapter is implemented.** Atomiverse is a ground-state engine:
there is no response solver, no excited-state root finder, and no self-energy.

This has knock-on effects elsewhere in the roadmap — nonadiabatic dynamics
(surface hopping, Ehrenfest) in [Molecular Dynamics](molecular-dynamics.md) is
blocked on the same gap, since it needs excited-state surfaces and couplings.

The nearest existing footholds, if these are ever built: the SCF solver already
exposes orbitals and the density matrix as Capabilities, and the ERI tensor can
be materialized and transformed to the MO basis (the MP2 `ovov` transform),
which is the same machinery a CIS/TDHF/TDDFT kernel would consume.

## 1. SCF and Linear-Response Methods

| Method                             | CPU | CUDA |
| ---------------------------------- | :-: | :--: |
| Delta-SCF excited states           |  ○  |   ○  |
| MOM excited states                 |  ○  |   ○  |
| IMOM excited states                |  ○  |   ○  |
| ROKS excited states                |  ○  |   ○  |
| CIS excitation energies            |  ○  |   ○  |
| CIS(D)                             |  ○  |   ○  |
| SCS-CIS(D)                         |  ○  |   ○  |
| SOS-CIS(D)                         |  ○  |   ○  |
| Time-dependent Hartree–Fock, TDHF  |  ○  |   ○  |
| Tamm–Dancoff TDDFT                 |  ○  |   ○  |
| Full linear-response TDDFT         |  ○  |   ○  |
| Spin-flip TDA                      |  ○  |   ○  |
| Spin-flip TDDFT                    |  ○  |   ○  |
| Restricted excitation-window TDDFT |  ○  |   ○  |
| Core–valence-separation TDDFT      |  ○  |   ○  |
| Real-time TDDFT                    |  ○  |   ○  |
| TDDFT analytic gradient            |  ○  |   ○  |
| TDDFT nonadiabatic coupling        |  ○  |   ○  |

## 2. Algebraic-Diagrammatic Construction

| Method                | CPU | CUDA |
| --------------------- | :-: | :--: |
| ADC(1)                |  ○  |   ○  |
| ADC(2)                |  ○  |   ○  |
| ADC(2)-x              |  ○  |   ○  |
| ADC(3)                |  ○  |   ○  |
| CVS-ADC(2)            |  ○  |   ○  |
| CVS-ADC(2)-x          |  ○  |   ○  |
| IP-ADC(2)             |  ○  |   ○  |
| EA-ADC(2)             |  ○  |   ○  |
| ADC analytic gradient |  ○  |   ○  |

## 3. Equation-of-Motion Coupled Cluster

| Method                     | CPU | CUDA |
| -------------------------- | :-: | :--: |
| EOM-EE-CCSD                |  ○  |   ○  |
| EOM-IP-CCSD                |  ○  |   ○  |
| EOM-EA-CCSD                |  ○  |   ○  |
| EOM-SF-CCSD                |  ○  |   ○  |
| EOM-DIP-CCSD               |  ○  |   ○  |
| EOM-DEA-CCSD               |  ○  |   ○  |
| CVS-EOM-CCSD               |  ○  |   ○  |
| EOM-CC2                    |  ○  |   ○  |
| EOM-MP2                    |  ○  |   ○  |
| STEOM-CCSD                 |  ○  |   ○  |
| EOM-CCSD analytic gradient |  ○  |   ○  |

Blocked upstream: there is no coupled cluster to build EOM on
(see [Post-Hartree–Fock](post-hf.md#3-coupled-cluster)).

## 4. Simplified Excited-State Methods

| Method                       | CPU | CUDA |
| ---------------------------- | :-: | :--: |
| Simplified TDA, sTDA         |  ○  |   ○  |
| Simplified TDDFT, sTDDFT     |  ○  |   ○  |
| sTDA-xTB                     |  ○  |   ○  |
| TD-DFTB                      |  ○  |   ○  |
| Long-range-corrected TD-DFTB |  ○  |   ○  |

Blocked upstream: no DFTB, no xTB
(see [Semiempirical and Tight Binding](semiempirical.md)).

## 5. Green's-Function and Many-Body Methods

| Method                                        | CPU | CUDA |
| --------------------------------------------- | :-: | :--: |
| Second-order Green's function, GF2            |  ○  |   ○  |
| Self-consistent GF2                           |  ○  |   ○  |
| G0W0                                          |  ○  |   ○  |
| Eigenvalue-self-consistent GW                 |  ○  |   ○  |
| Quasiparticle self-consistent GW              |  ○  |   ○  |
| Fully self-consistent GW                      |  ○  |   ○  |
| Bethe–Salpeter equation, BSE                  |  ○  |   ○  |
| Direct random-phase approximation             |  ○  |   ○  |
| RPA with exchange                             |  ○  |   ○  |
| SOSEX                                         |  ○  |   ○  |
| Renormalized second-order perturbation theory |  ○  |   ○  |
| ACFDT-RPA                                     |  ○  |   ○  |
