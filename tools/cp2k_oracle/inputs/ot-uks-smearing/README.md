# Oracle inputs — ot-uks-smearing

Offline CP2K twin class for multi-k collinear **UKS + FERMI_DIRAC**.

| File | Role |
| ---- | ---- |
| `na_bcc_uks_fermi_dirac.inp` | DIIS + LSD + SMEAR (same as `gpw-uks-smearing`) |

**CP2K 2026.1:** OT + k-points is hard-aborted at SCF init
(`OT not possible with kpoint calculations`). There is **no** fair
OT+KPOINTS+UKS+SMEAR energy/wall twin. Product validation is
**internal exactness** (UKS OT+FD ≡ UKS DIIS+FD) + frozen first-green pin;
wall class uses the DIIS LSD twin (~89.5 s class as RKS FD).

Never launch CP2K from ctest.
