# scf/solvation: one case, one section per pinned system or quantity.

atv_regression(SECTION h2o_rhf_631gs_cpcm
  REQUIRES HF)

atv_regression(SECTION ief-pcm-h2o
  REQUIRES HF)

atv_regression(SECTION pcm-grad-h2o
  REQUIRES HF)
