# md/npt: one case, one section per pinned system or quantity.

atv_regression(SECTION npt_lj_mtk
  REQUIRES MD
  DEVICES both)

atv_regression(SECTION nph_lj
  REQUIRES MD)

atv_regression(SECTION pr_npt_lj
  REQUIRES MD)

atv_regression(SECTION npt_pme_nacl
  REQUIRES KSPACE MD)

atv_regression(SECTION cuda-npt-pme-nacl
  REQUIRES MD POTENTIALS NEIGHBOR KSPACE EVALUATE
  DEVICES cuda)

atv_regression(SECTION surface-tension-coupling
  REQUIRES MD)
