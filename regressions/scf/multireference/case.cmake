# scf/multireference: one case, one section per pinned system or quantity.

atv_regression(SECTION cisd-h2co-sto3g
  REQUIRES HF)

atv_regression(SECTION ci-engine-equiv-h2-sto3g
  REQUIRES HF)

atv_regression(SECTION casscf-h2-sto3g
  REQUIRES HF)

atv_regression(SECTION casscf-h2o-sto3g
  REQUIRES HF)

atv_regression(SECTION nevpt2-h2-sto3g
  REQUIRES HF)
