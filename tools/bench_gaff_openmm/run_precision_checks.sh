#!/bin/bash
# GPU batch entry point: focused units and scientific pins, then GAFF scaling.
set -euo pipefail
BUILD="${1:?build directory}"
ctest --test-dir "$BUILD" --output-on-failure -R '^test_cuda_(frame|materialize|backend|pme_cell_guard|accumulator_reset|motion_precision|langevin_samples|position_images|minimizer_motion_precision|async_capture|recorder_snapshot|observer_rdf_molrs|atom_cell_sort|md_zero_momentum|nose_hoover_npt|cluster_mask|harmonic_coefficients|packed_tiles|hilbert|timer)$'
ctest --test-dir "$BUILD" --output-on-failure -R '^(test_cpu_md_initial_forces|test_npt_initial_kinetic|test_langevin_force_samples)$'
if [[ "${RUN_MPI_CHECKS:-0}" == "1" ]]; then
    ctest --test-dir "$BUILD" --output-on-failure -R '^test_npt_initial_kinetic_mpi_n2$'
    timeout 10s mpiexec -n 2 "$BUILD/tests/test_mpi_comm_brick" \
        --gtest_filter=TestCommBrick.N2BrickGoldens
fi
for pin in cpu-velocity-verlet-initial-force cuda-velocity-verlet-initial-force cuda-npt-volume-parity langevin-initial-force-samples cuda-gaff-fragment cuda-pme-nacl cuda-pme-half-spectrum cuda-ewald-nacl; do
    timeout 10s "$BUILD/regressions/reg_$pin"
done
if [[ "${RUN_MULTI_GPU:-0}" == "1" ]]; then
    # This entry point requests two GPUs; the independent four-device layout
    # test belongs in a four-GPU allocation.
    timeout 30s "$BUILD/tests/test_cuda_multi_gpu" \
        --gtest_filter=-MultiGpu.LjForceParityVsSingleGpu4Way
    for pin in NveTrajectoryParity NveTrajectoryParityNlist NveMigrationParity MigratePreservesBondedTopology; do
        timeout 10s "$BUILD/regressions/reg_cuda-multi-gpu-trajectory" "$pin"
    done
fi
if [[ "${SKIP_SCALING:-0}" != "1" ]]; then
    bash tools/bench_gaff_openmm/run_scaling.sh "$BUILD"
fi
