# scf/coupled-cluster: one case, one section per pinned system or quantity.

atv_regression(SECTION ccsd-h2-sto3g
  REQUIRES HF)

atv_regression(SECTION ccsd-grad-h2o-sto3g
  REQUIRES HF)

atv_regression(SECTION ccsd-t-h2o-sto3g
  REQUIRES HF)

atv_regression(SECTION ccsd-spatial-so-parity
  REQUIRES HF)

atv_regression(SECTION cc2-h2-sto3g
  REQUIRES HF)

atv_regression(SECTION cc2-h2o-sto3g
  REQUIRES HF)
