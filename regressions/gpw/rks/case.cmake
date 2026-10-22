# gpw/rks: one case, one section per pinned system or quantity.

atv_regression(SECTION gpw-forces-water
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-stress-water
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-analytic-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-water-pbe-mt
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-kpoints-si
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-kpoint-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-multigrid-kpoints
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-metallic-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-metallic-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION cuda-gpw-device-forces
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft)

atv_regression(SECTION cuda-gpw-full-product
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft)

atv_regression(SECTION cuda-gpw-kpoint-forces
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft CUDA::cublas CUDA::cusolver)

atv_regression(SECTION cuda-gpw-kpoints
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft)

atv_regression(SECTION cuda-gpw-multigrid-si-energy-path
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft)

atv_regression(SECTION cuda-gpw-pbe-product-gate
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft)

atv_regression(SECTION cuda-gpw-product-h2o-gate
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft)

atv_regression(SECTION cuda-gpw-product-h2o-stress
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft)

atv_regression(SECTION cuda-gpw-product-si-gate
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft)

atv_regression(SECTION cuda-gpw-stress
  REQUIRES GPW
  DEVICES both
  LIBS CUDA::cufft)

atv_regression(SECTION mpi-gpw-scaling-4rank
  REQUIRES GPW
  THREADS 8
  MPIEXEC)
