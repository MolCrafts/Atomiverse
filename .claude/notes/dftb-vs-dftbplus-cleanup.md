# DFTB family vs DFTB+

**Family oracle (binding): every DFTB-family number pins to DFTB+.**

## State

- mio-1-1 H/C/N/O static SK tables are generated
  (`tools/gen_dftb_mio_tables.py` → `dftb_mio_sk_tables.h`); hetero SKF
  poly-header parse supported.
- γ is DFTB+'s Elstner form (`internal/dftb_gamma.h`), not Ohno–Klopman.
- Product paths match DFTB+ on mio/3ob for H₂ / H₂O / CH₄; DFTB3 Γ-PBC analytic
  forces, multi-k collinear open shell and TD-DFTB oscillator strengths are in.

## Open (roadmap deepen, no slug)

- 3ob multi-k; TD excited-state gradients / LC-TD; DFTB3 k-mesh / open shell /
  stress / D3.
- Noncollinear / SOC: blocked — [tb-blocked-capabilities.md](tb-blocked-capabilities.md).
- NDDO and GFN phases use different oracle families (OpenMOPAC, xtb/tblite).

## Pin host

| Item | Path |
| --- | --- |
| DFTB+ 24.1 | `…/software/dftbplus-24.1/bin/dftb+` |
| mio / 3ob | `…/software/dftb-params/{mio-1-1,3ob-3-1}` |
| Env | `source tools/tb_oracle/setup_dftbplus_env.sh` |

Related: [dftb-plus-u.md](dftb-plus-u.md), `docs/developer/tb-method-queue.md`.
