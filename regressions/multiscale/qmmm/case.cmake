# multiscale/qmmm: one case, one section per pinned system or quantity.

atv_regression(SECTION qmmm_es_h2o_pyscf
  REQUIRES SEMIEMPIRICAL HF)

atv_regression(SECTION qmmm_link_ch3oh
  REQUIRES SEMIEMPIRICAL HF)

atv_regression(SECTION qmmm_md_conservation
  REQUIRES SEMIEMPIRICAL HF MD)

atv_regression(SECTION qmmm_pol_h2o
  REQUIRES SEMIEMPIRICAL HF)

atv_regression(SECTION lj_rhf_coupled
  REQUIRES HF POTENTIALS NEIGHBOR)
