# scf/relativistic: one case, one section per pinned system or quantity.

atv_regression(SECTION x2c_argon
  REQUIRES HF)

atv_regression(SECTION x2c_hcl_grad
  REQUIRES HF)

atv_regression(SECTION hf-x2c-forces
  REQUIRES HF)

atv_regression(SECTION ecp_heavy_atom
  REQUIRES HF)

atv_regression(SECTION ecp_i2_grad
  REQUIRES HF)
