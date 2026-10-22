# gpw/uks: one case, one section per pinned system or quantity.

atv_regression(SECTION gpw-uks-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-uks-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-uks-multigrid
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-uks-o2-gamma
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-uks-pbe
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-uks-metallic-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION cuda-gpw-uks-o2-forces
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft CUDA::cublas CUDA::cusolver)

atv_regression(SECTION cuda-gpw-uks-o2-gate
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft)
