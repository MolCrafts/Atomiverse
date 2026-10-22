# scf/rks: one case, one section per pinned system or quantity.

atv_regression(SECTION blyp-water-sto3g
  REQUIRES HF)

atv_regression(SECTION b3lyp-fused-jk-h2o
  REQUIRES HF)

atv_regression(SECTION cam-b3lyp-h2-sto3g
  REQUIRES HF)

atv_regression(SECTION b2plyp-h2o
  REQUIRES HF)

atv_regression(SECTION rks-hessian-h2o-sto3g
  REQUIRES HF)
