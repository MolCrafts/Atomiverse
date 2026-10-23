# scf/rhf: one case, one section per pinned system or quantity.

atv_regression(SECTION h2o_rhf_sto3g
  REQUIRES HF
  DEVICES both)

atv_regression(SECTION sto3g_const
  REQUIRES HF
  DEVICES both)

atv_regression(SECTION h2o_rhf_sto3g_grad
  REQUIRES HF
  DEVICES both)

atv_regression(SECTION h2o_rhf_def2svp_grad
  REQUIRES HF
  DEVICES both)

atv_regression(SECTION h2o_rhf_def2svp_sph
  REQUIRES HF)

atv_regression(SECTION diffuse_anion
  REQUIRES HF)

atv_regression(SECTION h2_rhf_stability
  REQUIRES HF)

atv_regression(SECTION h2o_scf_convergence
  REQUIRES HF)

atv_regression(SECTION soscf_hard_case
  REQUIRES HF)

atv_regression(SECTION es-cuda-rhf-driver
  REQUIRES HF
  DEVICES both)

atv_regression(SECTION hf-cuda-lgt1
  REQUIRES HF
  DEVICES both)

atv_regression(SECTION h2o_dimer_cp_hf631gs
  REQUIRES HF)

atv_regression(SECTION h2o_ri_jk
  REQUIRES HF)

atv_regression(SECTION c10h22_cfmm_j
  REQUIRES HF
  THREADS 8)
