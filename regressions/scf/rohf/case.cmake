# scf/rohf: one case, one section per pinned system or quantity.

atv_regression(SECTION o2_rohf_sto3g
  REQUIRES HF)

atv_regression(SECTION o2_rohf_grad
  REQUIRES HF)

atv_regression(SECTION rohf-hessian-o2
  REQUIRES HF)
