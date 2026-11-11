/// Single-component NPT construction and capture contract.
#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include <cmath>
#include <gtest/gtest.h>

TEST(NoseHooverNPT, CtorMirrorsCpu) {
    ATV::cuda::md::NoseHooverNPT npt{300, 100, 0, 1000, 0.5};
    EXPECT_EQ(npt.name(), "nose_hoover_npt");
    EXPECT_DOUBLE_EQ(npt.T_target(), 300);
}

TEST(NoseHooverNPT, LaunchCanBeCapturedWithoutHostReadback) {
    ATV::cpu::Frame host(2);
    host.box = ATV::cpu::Box::cubic(10);
    host.span(ATV::kAtom_mass)[0] = host.span(ATV::kAtom_mass)[1] = 1;
    auto device = ATV::cpu::to_cuda(host);
    cudaStream_t stream;
    ATV_CUDA_CHECK(cudaStreamCreate(&stream));
    ATV::cuda::Context context{};
    context.stream = stream;
    context.md.n_dof_global = ATV::md::degrees_of_freedom(2);
    ATV::cuda::md::NoseHooverNPT npt{300, 100, 0, 1000, 0.5};
    npt.build({context, device});
    npt.initialize_forces(
        {context, device, ATV::slot_of(ATV::md::Phase::end_of_step), 0, true});
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph;
    ATV_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    npt.launch(
        {context, device, ATV::slot_of(ATV::md::Phase::initial_integrate), 0});
    ATV_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t executable;
    ATV_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    ATV_CUDA_CHECK(cudaGraphLaunch(executable, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    ATV::cuda::to_cpu(device, host);
    EXPECT_TRUE(std::isfinite(host.box.volume()));
    npt.destroy({context, device});
    ATV_CUDA_CHECK(cudaGraphExecDestroy(executable));
    ATV_CUDA_CHECK(cudaGraphDestroy(graph));
    ATV::cuda::destroy(device);
    ATV_CUDA_CHECK(cudaStreamDestroy(stream));
}
