# tests/cuda/asynchronous/CMakeLists.txt — AsyncRecordBase / AsyncBridge unit guards.
#
# Unit-only (one component / one property). Multi-step MD pipelines that used
# to live here (hot_path_no_sync over 100 steps, async_bridge_tearing over a
# full MD race) were deleted — they are e2e, not ctest material.
#
# test_async_record_correctness.cu shells out to a full MD binary — kept as
# a source reference only; NOT registered with ctest.
#
# Labels: gpu_required so CI can skip GPU-less runners; cuda;async for
# partial selection.

atv_add_test(test_async_rhf_headers       test_async_rhf_headers.cu)

set_tests_properties(
    test_async_rhf_headers
    PROPERTIES LABELS "cuda;async;gpu_required"
)
