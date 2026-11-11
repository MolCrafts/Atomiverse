# tests/cuda/asynchronous/CMakeLists.txt — the shared asynchronous capture
# buffer (cuda::CaptureBuffer / cuda::AsyncConsumer).
#
# Unit-only: one property per test on a 10-atom LJ run. Labels: gpu_required
# so CI can skip GPU-less runners; cuda;async for partial selection.

atv_add_unit_exe(test_async_rhf_headers       test_async_rhf_headers.cu)
