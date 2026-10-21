/// @file tests/cuda/md/test_zero_momentum.cu
/// @brief cuda::md::ZeroMomentum — net COM linear momentum removal.
///
/// Covers `msu-pe-cuda-components-01-fidelity-and-momentum` ac-002:
/// after one `end_of_step` invocation, the device-side Σ m_i v_i must
/// be zero per axis to FP tolerance; with `freq>1` the alternate steps
/// must leave momentum un-touched.

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/dynamics/init.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"

using namespace ATV;

namespace {

/// @brief Build an LJ cubic lattice with a uniform non-zero velocity bias.
///
/// The lattice spacing is well past 2·sigma so LJ forces are negligible —
/// integration over a tiny `dt` keeps velocities close to the initial
/// bias, and any deviation of Σ m v from zero after `ZeroMomentum::launch`
/// is attributable to the component, not to integrator dynamics.
cpu::Frame make_drifting_lj(int n_side, double vx, double vy, double vz) {
    int n = n_side * n_side * n_side;
    const double spacing = 4.0; // > 2·sigma=2.0 in Lennard-Jones natural units
    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.span(kAtom_x)[idx] = (double(ix) + 0.5) * spacing;
                frame.span(kAtom_y)[idx] = (double(iy) + 0.5) * spacing;
                frame.span(kAtom_z)[idx] = (double(iz) + 0.5) * spacing;
                frame.span(kAtom_vx)[idx] = vx;
                frame.span(kAtom_vy)[idx] = vy;
                frame.span(kAtom_vz)[idx] = vz;
                frame.span(kAtom_mass)[idx] = 1.0;
                frame.span(kAtom_type)[idx] = 0;
                ++idx;
            }
    frame.box = cpu::Box::cubic(double(n_side) * spacing);
    return frame;
}

void com_momentum(const cpu::Frame& f, double p[3]) {
    p[0] = p[1] = p[2] = 0.0;
    for (int i = 0; i < f.nlocal(); ++i) {
        const double m = f.span(kAtom_mass)[i];
        p[0] += m * f.span(kAtom_vx)[i];
        p[1] += m * f.span(kAtom_vy)[i];
        p[2] += m * f.span(kAtom_vz)[i];
    }
}

} // namespace

TEST(ZeroMomentum, RoleIsStructuralNone) {
    cuda::md::ZeroMomentum zm{1};
    EXPECT_EQ(zm.role(), 0);
    EXPECT_EQ(zm.name(), "zero_momentum");
    auto bindings = zm.default_bindings();
    ASSERT_EQ(bindings.size(), 1u);
    EXPECT_EQ(bindings[0].phase,
              ATV::make_binding(ATV::md::Phase::end_of_step).phase);
    EXPECT_EQ(bindings[0].order, 25);
}

TEST(ZeroMomentum, FrequencyZeroTreatedAsOne) {
    cuda::md::ZeroMomentum zm{0};
    EXPECT_EQ(zm.frequency_steps(), 1);
}

TEST(ZeroMomentum, ZeroesUniformCOMDrift) {
    // Test the momentum-removal method alone, with a known uniform drift.
    auto frame = make_drifting_lj(4, 1.0, 2.0, 3.0);
    double p0[3];
    com_momentum(frame, p0);
    ASSERT_GT(std::fabs(p0[0]), 1.0);
    ASSERT_GT(std::fabs(p0[1]), 1.0);
    ASSERT_GT(std::fabs(p0[2]), 1.0);

    auto device = cpu::to_cuda(frame);
    cuda::Context runtime;
    cuda::md::ZeroMomentum component{1};
    component.build({runtime, device});
    component.launch({runtime, device, slot_of(md::Phase::end_of_step), 0});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    cuda::to_cpu(device, frame);
    component.destroy({runtime, device});
    cuda::destroy(device);

    double p1[3];
    com_momentum(frame, p1);
    // Per ac-002: |Σ m v_α| < 1e-5 (FP32) / < 1e-10 (FP64). cuda::work_t
    // defaults to double in this build; 1e-8 (eV·fs/Å) leaves headroom.
    EXPECT_LT(std::fabs(p1[0]), 1e-8);
    EXPECT_LT(std::fabs(p1[1]), 1e-8);
    EXPECT_LT(std::fabs(p1[2]), 1e-8);
}

TEST(ZeroMomentum, FrequencyLeavesAlternateCaptureUnchanged) {
    auto frame = make_drifting_lj(4, 1.0, 2.0, 3.0);
    auto device = cpu::to_cuda(frame);
    cuda::Context runtime;
    cuda::md::ZeroMomentum component{2};
    component.build({runtime, device});
    component.launch({runtime, device, slot_of(md::Phase::end_of_step), 0});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<cuda::motion_t> bias(frame.nlocal(), cuda::motion_t(4));
    ATV_CUDA_CHECK(cudaMemcpy(device.atoms.vx, bias.data(),
                              bias.size() * sizeof(cuda::motion_t),
                              cudaMemcpyHostToDevice));
    component.launch({runtime, device, slot_of(md::Phase::end_of_step), 1});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    cuda::to_cpu(device, frame);
    for (int i = 0; i < frame.nlocal(); ++i)
        EXPECT_EQ(frame.span(kAtom_vx)[i], 4);
    component.destroy({runtime, device});
    cuda::destroy(device);
}
