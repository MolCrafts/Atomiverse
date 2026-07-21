# tests/cuda/async/CMakeLists.txt — AsyncRecordBase / AsyncBridge guards.
#
# Benchmark-independent guards (always registered):
#   * test_hot_path_no_sync      — 0 cuda::stream_synchronize on the hot path.
#   * test_async_bridge_tearing  — AsyncBridge device-ring freeze delivers only
#                                  single-step, correctly-stamped snapshots
#                                  (async-analysis-snapshot-ring-02-device-ring).
#
# Benchmark-dependent guard (needs the polyethylene_tension binary, so gated on
# ATV_BUILD_BENCHMARKS):
#   * test_async_record_correctness — async vs sync recorder field parity.
# (The 25× recorder-frequency wall-clock gate is a PERFORMANCE assertion and
# lives in benchmarks/md/pe/performance/bench_async_record_overhead.cpp.)
#
# All carry the `gpu_required` ctest label so CI can skip them on GPU-less
# runners; they also carry `cuda;async` for partial selection.

atv_add_test(test_hot_path_no_sync        test_hot_path_no_sync.cu)
atv_add_test(test_async_bridge_tearing    test_async_bridge_tearing.cu)
atv_add_test(test_async_rhf_headers       test_async_rhf_headers.cu)

set_tests_properties(
    test_hot_path_no_sync
    test_async_bridge_tearing
    test_async_rhf_headers
    PROPERTIES LABELS "cuda;async;gpu_required"
)
# The tearing test races a lagging host worker against the MD loop on the
# single GPU; serialize it so parallel ctest runs don't oversubscribe.
set_tests_properties(test_async_bridge_tearing PROPERTIES RUN_SERIAL TRUE)

if(ATV_BUILD_BENCHMARKS)
    atv_add_test(test_async_record_correctness test_async_record_correctness.cu)

    set_tests_properties(
        test_async_record_correctness
        PROPERTIES LABELS "cuda;async;gpu_required"
    )

    # The correctness test shells out to the polyethylene_tension benchmark
    # and contends on the single GPU; serialize it so parallel ctest runs
    # don't oversubscribe the device or race on the shared /tmp output dirs.
    set_tests_properties(
        test_async_record_correctness
        PROPERTIES RUN_SERIAL TRUE
    )

    # Surface the benchmark binary's absolute path so the test does not
    # depend on cwd; add_dependencies so it exists before the test runs.
    target_compile_definitions(test_async_record_correctness PRIVATE
        ATV_PE_TENSION_BIN="$<TARGET_FILE:polyethylene_tension>")
    add_dependencies(test_async_record_correctness polyethylene_tension)
endif()
