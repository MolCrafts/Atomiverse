/// @file tests/cuda/potentials/test_bond_fene.cu
/// @brief cuda::BondFene energy, forces and virial against the analytic
///        Kremer–Grest FENE and the same-settings CPU BondFene; an
///        overstretched or untyped bond raises the bond force fault.
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cpu/potentials/bond_fene.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/potentials/bond_fene.cuh"

using namespace ATV;

namespace {

constexpr double kTolerance =
    std::is_same_v<ATV::cuda::force_t, float> ? 5e-5 : 1e-11;
constexpr double kBox = 20.0;
const std::vector<double> kK = {30.0, 20.0}; // eV/Å²
const std::vector<double> kR0 = {1.5, 1.6};  // Å

struct Bond {
    int i, j, type;
    double d[3]; ///< r_j − r_i [Å] (minimum image).
};

// Three bonds, two types; the third crosses x = L so min-image is exercised.
const Bond kBonds[3] = {
    {0, 1, 0, {1.0, 0.0, 0.0}},
    {2, 3, 1, {0.6, -0.7, 0.5}},
    {4, 5, 1, {0.5, 0.3, -0.6}},
};

cpu::Frame bonds_frame(double stretch = 1.0) {
    constexpr int n = 6;
    cpu::Frame f(n);
    const double origin[3][3] = {
        {5.0, 5.0, 5.0}, {12.0, 8.0, 3.0}, {19.8, 14.0, 9.0}};
    for (int b = 0; b < 3; ++b) {
        const Bond& bond = kBonds[b];
        for (int a = 0; a < 3; ++a) {
            const double ri = origin[b][a];
            const double rj = ri + stretch * bond.d[a];
            const ColKey<StorageReal>& key =
                a == 0 ? kAtom_x : (a == 1 ? kAtom_y : kAtom_z);
            f.span(key)[bond.i] = std::fmod(ri + kBox, kBox);
            f.span(key)[bond.j] = std::fmod(rj + kBox, kBox);
        }
    }
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(kBox);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    for (const Bond& bond : kBonds)
        f.add_bond(bond.i, bond.j, bond.type);
    f.meta.get(kEnergy) = 0;
    f.meta.get(kStress).fill(0);
    return f;
}

struct DeviceInput {
    ATV::cuda::Frame frame;
    ATV::cuda::Context context;
    explicit DeviceInput(const cpu::Frame& source)
        : frame(cpu::to_cuda(source)) {
        ATV::cuda::geom_t cell[9] = {kBox, 0, 0, 0, kBox, 0, 0, 0, kBox};
        context.host_box.assign_cell(cell);
    }
    ~DeviceInput() { ATV::cuda::destroy(frame); }
};

} // namespace

TEST(CudaBondFene, MatchesAnalyticEnergyForcesVirialAndCpu) {
    // Analytic Kremer–Grest: U = −½ K R0² ln(1 − (r/R0)²); the force on j is
    // fpair · d with fpair = −K / (1 − (r/R0)²); W_ab = Σ d_a F_j,b.
    double energy = 0;
    double force[6][3] = {};
    double virial[6] = {};
    for (const Bond& bond : kBonds) {
        const double K = kK[bond.type], R0 = kR0[bond.type];
        const double r2 = bond.d[0] * bond.d[0] + bond.d[1] * bond.d[1]
                          + bond.d[2] * bond.d[2];
        const double one_m = 1.0 - r2 / (R0 * R0);
        energy += -0.5 * K * R0 * R0 * std::log(one_m);
        const double fpair = -K / one_m;
        for (int a = 0; a < 3; ++a) {
            force[bond.j][a] += fpair * bond.d[a];
            force[bond.i][a] -= fpair * bond.d[a];
        }
        const int pairs[6][2] = {{0, 0}, {1, 1}, {2, 2},
                                 {0, 1}, {0, 2}, {1, 2}};
        for (int c = 0; c < 6; ++c)
            virial[c] += bond.d[pairs[c][0]] * fpair * bond.d[pairs[c][1]];
    }

    cpu::Frame cpu_frame = bonds_frame();
    cpu::Context cpu_ctx;
    cpu::BondFene cpu_bond(kK, kR0);
    cpu_bond.build({cpu_ctx, cpu_frame});
    cpu_bond.launch({cpu_ctx, cpu_frame, 0, 0, true, true});
    cpu_bond.destroy({cpu_ctx, cpu_frame});

    cpu::Frame host = bonds_frame();
    DeviceInput input(host);
    ATV::cuda::BondFene bond(kK, kR0);
    bond.build({input.context, input.frame});
    bond.launch({input.context, input.frame, 0, 0});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV::cuda::to_cpu(input.frame, host);
    bond.destroy({input.context, input.frame});

    auto tol = [](double ref) { return kTolerance * (1.0 + std::abs(ref)); };
    EXPECT_NEAR(host.meta.get(kEnergy), energy, tol(energy));
    EXPECT_NEAR(cpu_frame.meta.get(kEnergy), energy, 1e-12 * (1 + energy));
    const ColKey<AccReal>* fkeys[3] = {&kAtom_fx, &kAtom_fy, &kAtom_fz};
    for (int i = 0; i < 6; ++i)
        for (int a = 0; a < 3; ++a) {
            EXPECT_NEAR(host.span(*fkeys[a])[i], force[i][a], tol(force[i][a]))
                << "row " << i << " axis " << a;
            EXPECT_NEAR(cpu_frame.span(*fkeys[a])[i], force[i][a],
                        1e-12 * (1 + std::abs(force[i][a])))
                << "cpu row " << i << " axis " << a;
        }
    for (int c = 0; c < 6; ++c) {
        EXPECT_NEAR(host.meta.get(kStress)[c], virial[c], tol(virial[c]))
            << "virial component " << c;
        EXPECT_NEAR(cpu_frame.meta.get(kStress)[c], host.meta.get(kStress)[c],
                    tol(virial[c]))
            << "cpu virial component " << c;
    }
    // Not vacuous: the attractive FENE virial is negative on the diagonal.
    EXPECT_LT(virial[0], -1.0);
    EXPECT_GT(std::abs(force[1][0]), 1.0);
}

TEST(CudaBondFene, UnrequestedEnergyAndVirialAreNotTouched) {
    cpu::Frame host = bonds_frame();
    DeviceInput input(host);
    ATV::cuda::BondFene bond(kK, kR0);
    bond.build({input.context, input.frame});
    ATV::cuda::LaunchContext ctx{input.context, input.frame, 0, 0};
    ctx.energy_requested = false;
    ctx.stress_requested = false;
    bond.launch(ctx);
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV::cuda::to_cpu(input.frame, host);
    bond.destroy({input.context, input.frame});
    EXPECT_EQ(host.meta.get(kEnergy), 0.0);
    for (int c = 0; c < 6; ++c)
        EXPECT_EQ(host.meta.get(kStress)[c], 0.0);
    EXPECT_GT(std::abs(host.span(kAtom_fx)[1]), 1.0);
}

// r ≥ R0 has no finite FENE energy: the kernel raises ForceFault::kBond and
// the retrieve refuses the faulted Frame instead of returning garbage.
TEST(CudaBondFene, OverstretchedBondRaisesTheBondFault) {
    // Only bond 0 overstretches: r = 1.51 ≥ R0 = 1.5 (bond 1: 1.584 < 1.6).
    cpu::Frame host = bonds_frame(1.51);
    DeviceInput input(host);
    ATV::cuda::BondFene bond(kK, kR0);
    bond.build({input.context, input.frame});
    bond.launch({input.context, input.frame, 0, 0});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV::cuda::ForceFault fault{};
    ATV_CUDA_CHECK(cudaMemcpy(&fault, input.frame.force_fault, sizeof(fault),
                              cudaMemcpyDeviceToHost));
    EXPECT_TRUE(fault.kinds & ATV::cuda::ForceFault::kBond);
    EXPECT_EQ(fault.row, 0); // first atom of the overstretched bond
    try {
        ATV::cuda::to_cpu(input.frame, host);
        ADD_FAILURE() << "to_cpu accepted a faulted Frame";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("FENE"), std::string::npos)
            << e.what();
    }
    bond.destroy({input.context, input.frame});
}

TEST(CudaBondFene, UntypedBondRaisesTheBondFault) {
    cpu::Frame host = bonds_frame();
    DeviceInput input(host);
    ATV::cuda::BondFene bond({30.0}, {1.5}); // type 1 has no entry
    bond.build({input.context, input.frame});
    bond.launch({input.context, input.frame, 0, 0});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV::cuda::ForceFault fault{};
    ATV_CUDA_CHECK(cudaMemcpy(&fault, input.frame.force_fault, sizeof(fault),
                              cudaMemcpyDeviceToHost));
    EXPECT_TRUE(fault.kinds & ATV::cuda::ForceFault::kBond);
    EXPECT_THROW(ATV::cuda::to_cpu(input.frame, host), std::runtime_error);
    bond.destroy({input.context, input.frame});
}

TEST(CudaBondFene, ConstructorRejectsInvalidTables) {
    EXPECT_THROW(ATV::cuda::BondFene({}, {}), std::invalid_argument);
    EXPECT_THROW(ATV::cuda::BondFene({30.0}, {1.5, 1.6}),
                 std::invalid_argument);
    EXPECT_THROW(ATV::cuda::BondFene({0.0}, {1.5}), std::invalid_argument);
    EXPECT_THROW(ATV::cuda::BondFene({30.0}, {-1.5}), std::invalid_argument);
}
