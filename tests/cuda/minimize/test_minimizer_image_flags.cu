/// @file tests/cuda/minimize/test_minimizer_image_flags.cu
/// @brief CUDA minimizer steps wrap atoms with image bookkeeping (LAMMPS
///        Domain::pbc): r + H·image follows the step, and an L-BFGS Armijo
///        rollback restores the image flags the rejected trial changed.
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/minimize/cg.cuh"
#include "atomiverse/cuda/minimize/lbfgs.cuh"
#include "atomiverse/cuda/minimize/sd.cuh"

using namespace ATV;

namespace {

constexpr double kL = 10.0;
constexpr double kTol = std::is_same_v<ATV::cuda::pos_t, float> ? 1e-5 : 1e-12;

// One atom just inside +x with a pre-existing image count past int8 range,
// pushed out through +x by a unit force.
cpu::Frame edge_atom(double x0) {
    cpu::Frame f(1);
    f.box = cpu::Box::cubic(kL);
    f.span(kAtom_x)[0] = x0;
    f.span(kAtom_y)[0] = 5.0;
    f.span(kAtom_z)[0] = 5.0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_image)(0, 0) = 127;
    f.span(kAtom_image)(0, 1) = -3;
    return f;
}

struct Device {
    ATV::cuda::Frame frame;
    ATV::cuda::Context context;
    explicit Device(const cpu::Frame& host) : frame(cpu::to_cuda(host)) {
        ATV::cuda::geom_t cell[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell[3 * i + j] = host.box.cell()(i, j);
        context.host_box.assign_cell(cell);
        // Upload resets forces; set the unit +x force directly.
        ATV::cuda::fixed_t fx = 0;
        if (!ATV::cuda::encode_force(1.0, fx))
            throw std::logic_error("unit force not representable");
        ATV_CUDA_CHECK(cudaMemcpy(frame.atoms.fx, &fx, sizeof(fx),
                                  cudaMemcpyHostToDevice));
    }
    ~Device() { ATV::cuda::destroy(frame); }
    void set_energy(double e) {
        ATV_CUDA_CHECK(cudaMemcpy(&frame.scalars->energy, &e, sizeof(e),
                                  cudaMemcpyHostToDevice));
    }
    ATV::cuda::LaunchContext at(minimizer::Phase phase) {
        return {context, frame, slot_of(phase), 0};
    }
    void retrieve(cpu::Frame& host) {
        ATV_CUDA_CHECK(cudaDeviceSynchronize());
        ATV::cuda::to_cpu(frame, host);
    }
};

double unwrapped_x(const cpu::Frame& f) {
    return f.span(kAtom_x)[0] + kL * f.span(kAtom_image)(0, 0);
}

void expect_crossed_once(const cpu::Frame& host, double expected_unwrapped) {
    EXPECT_EQ(host.span(kAtom_image)(0, 0), 128);
    EXPECT_EQ(host.span(kAtom_image)(0, 1), -3);
    EXPECT_EQ(host.span(kAtom_image)(0, 2), 0);
    EXPECT_GE(host.span(kAtom_x)[0], 0.0);
    EXPECT_LT(host.span(kAtom_x)[0], kL);
    EXPECT_NEAR(unwrapped_x(host), expected_unwrapped, kTol);
}

} // namespace

TEST(CudaMinimizerImageFlags, SteepestDescentCountsTheCrossing) {
    cpu::Frame host = edge_atom(9.95);
    const double u0 = unwrapped_x(host);
    {
        Device dev(host);
        // Δx = α F / m · force_to_accel = 0.2 Å.
        ATV::cuda::minimizer::SteepestDescent sd(0.2 / units::md_ftm2v);
        sd.build({dev.context, dev.frame});
        sd.launch(dev.at(minimizer::Phase::search_direction));
        dev.retrieve(host);
        sd.destroy({dev.context, dev.frame});
    }
    expect_crossed_once(host, u0 + 0.2);
}

TEST(CudaMinimizerImageFlags, ConjugateGradientCountsTheCrossing) {
    cpu::Frame host = edge_atom(9.95);
    const double u0 = unwrapped_x(host);
    {
        Device dev(host);
        // First CG step is steepest descent: Δx = α0 F = 0.2 Å.
        ATV::cuda::minimizer::CG cg(0.2);
        cg.build({dev.context, dev.frame});
        cg.launch(dev.at(minimizer::Phase::search_direction));
        dev.retrieve(host);
        cg.destroy({dev.context, dev.frame});
    }
    expect_crossed_once(host, u0 + 0.2);
}

TEST(CudaMinimizerImageFlags, LbfgsStepCountsTheCrossing) {
    cpu::Frame host = edge_atom(9.95);
    const double u0 = unwrapped_x(host);
    {
        Device dev(host);
        // Max-displacement cap: α = min(α0, 0.1 / max|d|) = 0.1 Å.
        ATV::cuda::minimizer::LBFGS lbfgs(5, 1.0);
        lbfgs.build({dev.context, dev.frame});
        lbfgs.launch(dev.at(minimizer::Phase::search_direction));
        dev.retrieve(host);
        lbfgs.destroy({dev.context, dev.frame});
    }
    expect_crossed_once(host, u0 + 0.1);
}

// The first trial (α = 0.1) crosses +x and is rejected; the contracted trial
// (α = 0.05) stays inside, so its image must be the ORIGIN's, not the
// rejected trial's.
TEST(CudaMinimizerImageFlags, LbfgsRollbackRestoresTheRejectedCrossing) {
    cpu::Frame host = edge_atom(9.93);
    const double u0 = unwrapped_x(host);
    {
        Device dev(host);
        ATV::cuda::minimizer::LBFGS lbfgs(5, 1.0);
        lbfgs.line_search(1e-4, 0.5, 1e-6);
        lbfgs.build({dev.context, dev.frame});
        lbfgs.launch(dev.at(minimizer::Phase::search_direction));
        int trials = 0;
        dev.context.opt.reeval_forces = [&](bool) {
            // Reject the first trial (energy rises), accept the second.
            dev.set_energy(++trials == 1 ? 1.0e3 : -1.0e3);
        };
        lbfgs.launch(dev.at(minimizer::Phase::line_search));
        EXPECT_EQ(trials, 2);
        dev.retrieve(host);
        lbfgs.destroy({dev.context, dev.frame});
    }
    EXPECT_EQ(host.span(kAtom_image)(0, 0), 127);
    EXPECT_EQ(host.span(kAtom_image)(0, 1), -3);
    EXPECT_NEAR(host.span(kAtom_x)[0], 9.98, kTol);
    EXPECT_NEAR(unwrapped_x(host), u0 + 0.05, kTol);
}

// An exhausted search restores the origin: position and image both.
TEST(CudaMinimizerImageFlags, LbfgsExhaustedSearchRestoresOriginImage) {
    cpu::Frame host = edge_atom(9.95);
    {
        Device dev(host);
        ATV::cuda::minimizer::LBFGS lbfgs(5, 1.0, /*max_ls=*/3);
        lbfgs.line_search(1e-4, 0.5, 1e-6);
        lbfgs.build({dev.context, dev.frame});
        lbfgs.launch(dev.at(minimizer::Phase::search_direction));
        dev.context.opt.reeval_forces = [&](bool) { dev.set_energy(1.0e3); };
        EXPECT_THROW(lbfgs.launch(dev.at(minimizer::Phase::line_search)),
                     std::runtime_error);
        dev.retrieve(host);
        lbfgs.destroy({dev.context, dev.frame});
    }
    EXPECT_EQ(host.span(kAtom_image)(0, 0), 127);
    EXPECT_EQ(host.span(kAtom_image)(0, 1), -3);
    EXPECT_NEAR(host.span(kAtom_x)[0], 9.95, kTol);
}
