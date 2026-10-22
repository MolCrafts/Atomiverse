# minimizer/geometry: one case, one section per pinned system or quantity.

atv_regression(SECTION lj_dimer
  REQUIRES MINIMIZER)

atv_regression(SECTION fire_lj_dimer
  REQUIRES MINIMIZER)

atv_regression(SECTION rhf_h2_linesearch
  REQUIRES MINIMIZER HF)
