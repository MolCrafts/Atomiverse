# scf/embedding: one case, one section per pinned system or quantity.

atv_regression(SECTION embed_h2o_grad
  REQUIRES HF)

atv_regression(SECTION embed_h2o_mm_forces
  REQUIRES HF)

atv_regression(SECTION field_h2o_grad
  REQUIRES HF)

atv_regression(SECTION h2o_embedding_field
  REQUIRES HF)
