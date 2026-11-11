/// @file tests/cuda/gpw/test_gpw_stress.cu
/// @brief CUDA GPW Frame and explicit-state stress guards (no SCF/device work).
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#include "atomiverse/cuda/gpw/internal/gpw_access.cuh"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::xc::XCType;

namespace {

std::vector<ATV::cpu::gpw::internal::GpwAtom> h2_atoms() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double a[3] = {5.5, 6.0, 6.0}, b[3] = {6.5, 6.0, 6.0};
    return {ATV::cpu::gpw::internal::GpwAtom(a, basis, pot, 1),
            ATV::cpu::gpw::internal::GpwAtom(b, basis, pot, 1)};
}

ATV::cpu::Frame h2_frame() {
    using namespace ATV;
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(12.0 * units::kBohrInAngstrom);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_atomic_number)[i] = 1;
        frame.span(kAtom_x)[i] = (5.5 + i) * units::kBohrInAngstrom;
        frame.span(kAtom_y)[i] = 6.0 * units::kBohrInAngstrom;
        frame.span(kAtom_z)[i] = 6.0 * units::kBohrInAngstrom;
        frame.span(kAtom_fx)[i] = 3.0 + i;
        frame.span(kAtom_fy)[i] = 5.0 + i;
        frame.span(kAtom_fz)[i] = 7.0 + i;
    }
    frame.meta.get(kEnergy) = 17.5;
    frame.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
    return frame;
}

ATV::cuda::gpw::GpwDft frame_method() {
    return {
        {{molrs::Element::H, GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
          GthPotential(molrs::Element::H, "GTH-PADE-q1")}},
        80.0,
        XCType::PADE};
}

void expect_frame_outputs_preserved(const ATV::cpu::Frame& frame) {
    using namespace ATV;
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 17.5);
    const MetaF64x6 expected_virial = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(frame.meta.get(kStress), expected_virial);
    for (int i = 0; i < 2; ++i) {
        EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[i], 3.0 + i);
        EXPECT_DOUBLE_EQ(frame.span(kAtom_fy)[i], 5.0 + i);
        EXPECT_DOUBLE_EQ(frame.span(kAtom_fz)[i], 7.0 + i);
    }
}

template <typename Exception>
void expect_frame_entries_rejected(ATV::cuda::gpw::GpwDft& method,
                                   ATV::cpu::Frame& frame) {
    EXPECT_THROW(method.run(std::as_const(frame)), Exception);
    expect_frame_outputs_preserved(frame);
    EXPECT_THROW(method.run(frame), Exception);
    expect_frame_outputs_preserved(frame);
    EXPECT_THROW(method.forces(frame), Exception);
    expect_frame_outputs_preserved(frame);
    EXPECT_THROW(method.stress(frame), Exception);
    expect_frame_outputs_preserved(frame);
}

} // namespace

TEST(CudaGpwStress, FrameEntriesRequireBuildWithoutPublishingResults) {
    auto frame = h2_frame();
    auto method = frame_method();
    expect_frame_entries_rejected<std::logic_error>(method, frame);
}

TEST(CudaGpwStress, FrameEntriesRejectCellShearedAfterBuild) {
    auto frame = h2_frame();
    auto method = frame_method();
    method.build(frame);
    auto cell = frame.box.cell();
    cell(0, 1) = 0.1;
    frame.box.set_cell(cell);
    expect_frame_entries_rejected<std::invalid_argument>(method, frame);
}

TEST(CudaGpwStress, FrameEntriesRejectUnsupportedSettingsAfterBuild) {
    auto frame = h2_frame();
    auto method = frame_method();
    method.build(frame);
    method.use_ot = true;
    expect_frame_entries_rejected<std::invalid_argument>(method, frame);
}

TEST(CudaGpwStress, RejectsInvalidStressStepBeforeEvaluatingFrame) {
    auto frame = h2_frame();
    auto method = frame_method();
    // Probe validation precedes the unbuilt-method check and every CUDA call.
    for (double h :
         {0.0, -1.0, 0.1, 0.2, std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN()}) {
        method.stress_strain_h = h;
        EXPECT_THROW(method.stress(frame), std::invalid_argument) << "h=" << h;
        expect_frame_outputs_preserved(frame);
    }
}

TEST(CudaGpwStress, RejectsMaskedStressProbeBeforeEvaluatingFrame) {
    auto frame = h2_frame();
    auto method = frame_method();
    method.stress_axes = {false, false, false};
    EXPECT_THROW(method.stress(frame), std::invalid_argument);
    expect_frame_outputs_preserved(frame);
}

TEST(CudaGpwStress, MultiKBuildIsOnProductPath) {
    auto frame = h2_frame();
    auto method = frame_method();
    method.n_occ = 1;
    method.k_mesh[0] = 2;
    method.k_time_reversal = false;
    EXPECT_NO_THROW(method.build(frame));
    ATV::cpu::gpw::GpwResult fake;
    fake.converged = true;
    // Unit-only access: a full multi-k payload must come from a real solve.
    using Access = ATV::cuda::gpw::internal::GpwDftAccess;
    const auto geometry = Access::assemble_geometry(method, h2_atoms());
    EXPECT_THROW(Access::stress_from_state(method, geometry, fake, {}),
                 std::exception);
}

TEST(CudaGpwStress, FailClosedUnconvergedState) {
    auto frame = h2_frame();
    auto method = frame_method();
    method.build(frame);
    ATV::cpu::gpw::GpwResult fake;
    fake.converged = false;
    using Access = ATV::cuda::gpw::internal::GpwDftAccess;
    const auto geometry = Access::assemble_geometry(method, h2_atoms());
    EXPECT_THROW(Access::stress_from_state(method, geometry, fake, {}),
                 std::invalid_argument);
    expect_frame_outputs_preserved(frame);
}

TEST(CudaGpwStress, FrameEntriesRejectInvalidResidualToleranceBeforeCuda) {
    for (double invalid : {0., -1., std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
        auto frame = h2_frame();
        auto method = frame_method();
        method.build(frame);
        method.residual_tol = invalid;
        EXPECT_THROW(method.build(frame), std::invalid_argument);
        expect_frame_outputs_preserved(frame);
        expect_frame_entries_rejected<std::invalid_argument>(method, frame);
    }
}
