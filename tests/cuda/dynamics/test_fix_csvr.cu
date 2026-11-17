/// @file tests/cuda/dynamics/test_fix_csvr.cu
/// @brief cuda::md::FixCSVR against cpu::md::FixCSVR: same randoms, same
///        rescale, same group / COM-bias semantics and bookkeeping.
#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

#include "atomiverse/core/csvr.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_csvr.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/dynamics/fix_csvr.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"

using namespace ATV;
using ATV::md::TemperatureBias;

namespace {

// Velocities are FP64 in the double and mixed modes, FP32 in single.
constexpr double kTol =
    std::is_same_v<ATV::cuda::mixed_t, float> ? 2e-6 : 1e-12;

/// 300 free atoms (lj units), types 1/2/3 in turn, masses 1..3, a drift
/// along x on type 3; deterministic pseudo-random velocities.
cpu::Frame mixture() {
    constexpr int n = 300;
    cpu::Frame f(n);
    std::uint64_t s = 0x1234;
    auto next = [&s] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(s >> 11) * 1.1102230246251565e-16 - 0.5;
    };
    for (int i = 0; i < n; ++i) {
        const int t = 1 + i % 3;
        f.span(kAtom_x)[i] = 1.0 + 0.1 * i;
        f.span(kAtom_y)[i] = 2.0;
        f.span(kAtom_z)[i] = 3.0;
        f.span(kAtom_mass)[i] = static_cast<double>(t);
        f.span(kAtom_type)[i] = t;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = next() + (t == 3 ? 0.9 : 0.0);
        f.span(kAtom_vy)[i] = next();
        f.span(kAtom_vz)[i] = next();
    }
    f.box = cpu::Box::cubic(100.0);
    f.meta.get(kEnergy) = -2.0;
    return f;
}

struct DeviceInput {
    ATV::cuda::Frame frame;
    ATV::cuda::Context context;
    explicit DeviceInput(const cpu::Frame& source)
        : frame(cpu::to_cuda(source)) {
        ATV::cuda::geom_t cell[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell[3 * i + j] = source.box.cell()(i, j);
        context.host_box.assign_cell(cell);
    }
    ~DeviceInput() { ATV::cuda::destroy(frame); }
};

/// Run `steps` end-of-step launches of every thermostat on both devices
/// and compare the velocities and the bookkeeping.
void compare(std::vector<cpu::md::FixCSVR> cpu_th,
             std::vector<ATV::cuda::md::FixCSVR> gpu_th,
             const std::vector<BigInt>& by_type, int n_dof_global, int steps) {
    cpu::Frame ref = mixture();
    cpu::Frame host = mixture();
    cpu::Context cpu_ctx;
    cpu_ctx.md.n_dof_global = n_dof_global;
    cpu_ctx.md.n_constraints_by_type =
        by_type.empty() ? nullptr : by_type.data();
    cpu_ctx.md.n_constraint_types = static_cast<int>(by_type.size());
    cpu_ctx.md.constraints_by_type_known = true;

    DeviceInput input(host);
    input.context.md.n_dof_global = n_dof_global;
    input.context.md.n_constraints_by_type =
        by_type.empty() ? nullptr : by_type.data();
    input.context.md.n_constraint_types = static_cast<int>(by_type.size());
    input.context.md.constraints_by_type_known = true;

    for (auto& th : cpu_th)
        th.build({cpu_ctx, ref});
    for (auto& th : gpu_th)
        th.build({input.context, input.frame});
    for (std::size_t k = 0; k < cpu_th.size(); ++k)
        EXPECT_EQ(cpu_th[k].n_dof(), gpu_th[k].n_dof());

    const int slot = ATV::slot_of(md::Phase::end_of_step);
    for (int s = 0; s < steps; ++s) {
        for (auto& th : cpu_th)
            th.launch({cpu_ctx, ref, slot, s});
        for (auto& th : gpu_th)
            th.launch({input.context, input.frame, slot, s});
    }
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV::cuda::to_cpu(input.frame, host);
    for (auto& th : gpu_th)
        th.destroy({input.context, input.frame});

    double moved = 0;
    const cpu::Frame start = mixture();
    for (int i = 0; i < ref.nlocal(); ++i) {
        EXPECT_NEAR(host.span(kAtom_vx)[i], ref.span(kAtom_vx)[i], kTol);
        EXPECT_NEAR(host.span(kAtom_vy)[i], ref.span(kAtom_vy)[i], kTol);
        EXPECT_NEAR(host.span(kAtom_vz)[i], ref.span(kAtom_vz)[i], kTol);
        moved += std::fabs(ref.span(kAtom_vx)[i] - start.span(kAtom_vx)[i]);
    }
    EXPECT_GT(moved, 1e-3); // the thermostat did act
    EXPECT_NEAR(host.meta.get(kThermostatEnergy),
                ref.meta.get(kThermostatEnergy), 1e3 * kTol);
    EXPECT_NEAR(host.meta.get(kConserved), ref.meta.get(kConserved),
                1e3 * kTol);
}

} // namespace

TEST(CudaFixCsvr, WholeSystemMatchesCpu) {
    std::vector<cpu::md::FixCSVR> c;
    c.emplace_back(0.6, 2.0, 1.0, 7u);
    std::vector<ATV::cuda::md::FixCSVR> g;
    g.emplace_back(0.6, 2.0, 1.0, 7u);
    compare(std::move(c), std::move(g), {}, 3 * 300 - 3, 25);
}

TEST(CudaFixCsvr, SpeciesGroupsWithComBiasMatchCpu) {
    // Three instances, one per type, own seeds and setpoints; type 2 owns
    // five constraints (N_f = 3·100 − 3 − 5).
    const std::vector<BigInt> by_type = {0, 0, 5, 0};
    std::vector<cpu::md::FixCSVR> c;
    std::vector<ATV::cuda::md::FixCSVR> g;
    const double T[3] = {0.4, 0.6, 0.9};
    for (int t = 1; t <= 3; ++t) {
        c.emplace_back(T[t - 1], 3.0, 1.0, 100u + t, std::vector<int>{t},
                       TemperatureBias::Com);
        g.emplace_back(T[t - 1], 3.0, 1.0, 100u + t, std::vector<int>{t},
                       TemperatureBias::Com);
    }
    compare(std::move(c), std::move(g), by_type, 3 * 300 - 3 - 5, 25);
}

TEST(CudaFixCsvr, ComBiasLeavesTheGroupMomentum) {
    cpu::Frame host = mixture();
    DeviceInput input(host);
    input.context.md.n_dof_global = 3 * 300 - 3;
    input.context.md.constraints_by_type_known = true;
    ATV::cuda::md::FixCSVR th(0.1, 0.5, 1.0, 3u, {3}, TemperatureBias::Com);
    th.build({input.context, input.frame});
    EXPECT_EQ(th.n_dof(), 3 * 100 - 3);
    const int slot = ATV::slot_of(md::Phase::end_of_step);
    for (int s = 0; s < 10; ++s)
        th.launch({input.context, input.frame, slot, s});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    cpu::Frame out = mixture();
    ATV::cuda::to_cpu(input.frame, out);
    th.destroy({input.context, input.frame});
    const cpu::Frame start = mixture();
    double p0 = 0, p1 = 0, other = 0;
    for (int i = 0; i < out.nlocal(); ++i) {
        if (start.span(kAtom_type)[i] == 3) {
            p0 += start.span(kAtom_mass)[i] * start.span(kAtom_vx)[i];
            p1 += out.span(kAtom_mass)[i] * out.span(kAtom_vx)[i];
        } else {
            other += std::fabs(out.span(kAtom_vx)[i] - start.span(kAtom_vx)[i]);
        }
    }
    EXPECT_NEAR(p1, p0, 1e3 * kTol);
    EXPECT_EQ(other, 0.0);
}

TEST(CudaFixCsvr, RefusesAGroupThatSelectsNothing) {
    cpu::Frame host = mixture();
    DeviceInput input(host);
    input.context.md.constraints_by_type_known = true;
    ATV::cuda::md::FixCSVR th(0.1, 0.5, 1.0, 3u, {8}, TemperatureBias::None);
    EXPECT_THROW(th.build({input.context, input.frame}), std::runtime_error);
}
