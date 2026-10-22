# mc/metropolis: one case, one section per pinned system or quantity.

atv_regression(SECTION npt_lj
  REQUIRES MC)

atv_regression(SECTION widom_hs
  REQUIRES MC)

atv_regression(SECTION gcmc_lj
  REQUIRES MC)
