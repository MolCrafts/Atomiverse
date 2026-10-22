# scf/tddft: one case, one section per pinned system or quantity.

atv_regression(SECTION ch2o_tdhf
  REQUIRES HF)

atv_regression(SECTION h2-b3lyp-hybrid-tda
  REQUIRES HF)

atv_regression(SECTION h2-b3lyp-rpa-tddft
  REQUIRES HF)

atv_regression(SECTION full-lr-tddft-h2-pbe
  REQUIRES HF)

atv_regression(SECTION h2_full_grid_alda
  REQUIRES HF)

atv_regression(SECTION uks-tda-o2-lsda
  REQUIRES HF)

atv_regression(SECTION uks-full-lr-o2-lsda
  REQUIRES HF)

atv_regression(SECTION sf-tddft-o2
  REQUIRES HF)

atv_regression(SECTION spin-flip-tda-o2
  REQUIRES HF)
