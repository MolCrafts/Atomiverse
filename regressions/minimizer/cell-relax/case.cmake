# minimizer/cell-relax: one case, one section per pinned system or quantity.

atv_regression(SECTION cell_lj_pressure
  REQUIRES MINIMIZER)

atv_regression(SECTION joint_lj_zero_p
  REQUIRES MINIMIZER)
