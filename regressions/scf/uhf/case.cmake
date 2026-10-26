# scf/uhf: one case, one section per pinned system or quantity.

atv_regression(SECTION h2_uhf_sto3g
  REQUIRES HF
  DEVICES both)

atv_regression(SECTION cuda-uhf
  REQUIRES HF
  DEVICES both)

atv_regression(SECTION uhf-hessian-o2
  REQUIRES HF)

atv_regression(SECTION delta-scf-n2
  REQUIRES HF)

atv_regression(SECTION n2_ionization
  REQUIRES HF
  THREADS 4)
