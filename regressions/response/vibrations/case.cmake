# response/vibrations: one case, one section per pinned system or quantity.

atv_regression(SECTION h2o_rhf_freq
  REQUIRES HF VIB)

atv_regression(SECTION h2o_rhf_ir
  REQUIRES HF)

atv_regression(SECTION harmonic_driver_matrix
  REQUIRES VIB MD MINIMIZER EVALUATE)
