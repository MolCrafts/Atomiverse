# DFTB+ 24.1 twin — `dftb2-spin-analytic-stress`

```text
oracle: dftb+
oracle_version: 24.1 (energy + forces; no cluster stress)
```

- **Binary:** DFTB+ 24.1 (`source tools/tb_oracle/setup_dftbplus_env.sh`)
- **SK:** mio-1-1 `O-O.skf`
- **Geometry:** cluster (`C`) O₂, tilted \(|R|=1.30\) Å along
  \((0.6, 0.7, \sqrt{0.15})\). ATV's regression uses the same cluster
  Hamiltonian plus a cubic 20 Å box only as \(\Omega\) for the intensive
  virial. A DFTB+ `S` geometry would be a **different** (Ewald) Hamiltonian
  ATV does not use here, so it is not archived.
- **Spin:** same block as `spin-dftb2-o2`
- **Archived output:** `dftb.out.txt` = `detailed.out` verbatim
- **Total energy:** −6.4963341184 H = −176.774256878772 eV
  (same \(|R|\) as the linear 1.30 Å force twin)
- **Forces (Ha/Bohr):**
  \(F_1=(0.057307719745,\ 0.066859006948,\ 0.036991973513)\),
  \(F_2=-F_1\). No stress tensor is printed for a cluster geometry.

Re-run:

```bash
source tools/tb_oracle/setup_dftbplus_env.sh
tools/tb_oracle/run_dftbplus_pin.sh \
  tools/tb_oracle/inputs/dftbplus/dftb2-spin-analytic-stress mio
```
