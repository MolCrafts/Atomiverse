/// @file tests/cpu/gpw/test_gpw_evaluator.cpp
/// @brief Unit tests for cpu::gpw::GPWEvaluator's cell precondition.
///
/// GPW carries the cell as three edge lengths and builds its lattice images
/// from them, so a sheared frame would be solved on `diag(h)` — a different
/// Bravais lattice. Every case here fires before any SCF work, so the file
/// stays inside the seconds-scale unit budget.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/electronic/car_parrinello.h"
#include "atomiverse/cpu/electronic/second_gen_cp.h"
#include "atomiverse/cpu/electronic/xl_bomd_force.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/gpw_atom.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/gpw/internal/gpw_geometry.h"
#include <limits>
#include <xtensor/generators/xbuilder.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ATV;

namespace {

/// H2 in a cubic 12 Bohr cell — the cheapest frame GPW accepts.
cpu::Frame make_h2() {
    cpu::Frame f(2);
    constexpr double bohr2ang = 0.52917720859;
    const double x[2] = {5.5 * bohr2ang, 6.5 * bohr2ang};
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_atomic_number)[i] = 1;
        f.span(kAtom_mass)[i] = 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_x)[i] = x[i];
        f.span(kAtom_y)[i] = 6.0 * bohr2ang;
        f.span(kAtom_z)[i] = 6.0 * bohr2ang;
    }
    f.box = cpu::Box::cubic(12.0 * bohr2ang);
    return f;
}

/// @param cutoff_ry Density-grid cutoff [Ry].
cpu::gpw::GPWEvaluator h2_eval(double cutoff_ry = 60.0) {
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};
    cpu::gpw::GpwDft dft{{H}, cutoff_ry, cpu::xc::XCType::PADE};
    dft.max_iter = 1;
    dft.tol = 1e-6;
    dft.mix = 0.3;
    return cpu::gpw::GPWEvaluator(std::move(dft), /*compute_forces=*/false);
}

} // namespace

TEST(GpwEvaluatorCellGuard, BuildAcceptsOrthorhombic) {
    auto frame = make_h2();
    auto eval = h2_eval();
    cpu::Context rt;
    cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorCellGuard, BuildRefusesShearedCell) {
    auto frame = make_h2();
    Mat3 h = frame.box.cell();
    h(0, 1) = 1.0; // shear the a-vector into y
    frame.box.set_cell(h);

    auto eval = h2_eval();
    cpu::Context rt;
    cpu::BuildContext bc{rt, frame};
    try {
        eval.build(bc);
        FAIL() << "expected std::invalid_argument for a sheared cell";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("orthorhombic"), std::string::npos) << msg;
        EXPECT_NE(msg.find("h(0,1)"), std::string::npos) << msg;
    }
}

/// The variable-cell path is the reason a build-time-only guard is unsound:
/// TriclinicCellRelax and ParrinelloRahmanNPT write a general cell matrix back
/// to the Frame between launches.
TEST(GpwEvaluatorCellGuard, LaunchRefusesCellShearedAfterBuild) {
    auto frame = make_h2();
    auto eval = h2_eval();
    cpu::Context rt;
    cpu::BuildContext bc{rt, frame};
    ASSERT_NO_THROW(eval.build(bc));

    Mat3 h = frame.box.cell();
    h(1, 2) = 0.5;
    frame.box.set_cell(h);

    EXPECT_THROW(eval.launch({rt, frame, 0, 0}), std::invalid_argument);
}

TEST(GpwEvaluatorTrialGuard, PropagatingModesRejectTrialsBeforeColdSeed) {
    cpu::Frame frame(0); // refusal precedes geometry/SCF construction
    cpu::Context ctx;
    ctx.exec.trial_evaluation = true;
    cpu::electronic::CarParrinello cp(h2_eval(), 400.0);
    cpu::electronic::SecondGenCp second(h2_eval());
    cpu::electronic::XlBomdForce shadow(h2_eval(), 5,
                                        cpu::electronic::XlBomdLinearized{});
    EXPECT_THROW(cp.launch({ctx, frame, 0, 0}), std::runtime_error);
    EXPECT_THROW(second.launch({ctx, frame, 0, 0}), std::runtime_error);
    EXPECT_THROW(shadow.launch({ctx, frame, 0, 0}), std::runtime_error);
}

TEST(GpwEvaluatorCellGuard, RejectsExplicitIndependentMpiReplicasBeforeBuild) {
    cpu::Frame frame; // Guard must precede even method geometry validation.
    auto eval = h2_eval();
    cpu::Context context;
    context.exec.serial_on_replica = true;
    context.mpi_size = 2; // Runtime publishes the actual WORLD size here.
    cpu::BuildContext build{context, frame};
    try {
        eval.build(build);
        FAIL() << "independent replica execution was accepted";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("serial replicas"),
                  std::string::npos);
    }
    cpu::LaunchContext launch{context, frame, 0, 0};
    try {
        eval.launch(launch);
        FAIL() << "independent replica launch was accepted";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("serial replicas"),
                  std::string::npos);
    }
    // The same explicit replica policy is supported on a single process.
    auto valid = make_h2();
    auto configured = h2_eval();
    context.mpi_size = 1;
    cpu::BuildContext serial{context, valid};
    EXPECT_NO_THROW(configured.build(serial));
}

TEST(GpwEvaluatorForceReuse,
     GeometryCacheDoesNotReuseAuxiliaryDensityPotentials) {
    using namespace cpu::gpw;
    using cpu::linalg::RealMatrix;
    auto frame = make_h2();
    // Small direct force-kernel fixture, no SCF.
    auto eval = h2_eval(/*cutoff_ry=*/5.0);
    auto& method = eval.method();
    method.n_alpha = method.n_beta = 1;
    method.build(frame);
    const auto atoms = internal::GpwDftAccess::atoms_from_frame(method, frame);
    const internal::GpwGeometry geometry(
        atoms, method.cell_bohr(), method.cutoff_ry, method.eps_default,
        !method.nonperiodic, internal::GpwGridTables::compact);
    const std::size_t nb = geometry.ao.nb();
    ASSERT_GT(nb, 1u);
    RealMatrix c = xt::zeros<double>({nb, std::size_t{1}});
    RealMatrix h = xt::zeros<double>({nb, nb});
    c(0, 0) = 1.0;
    for (std::size_t i = 0; i < nb; ++i)
        h(i, i) = 0.2;
    GpwResult cache;
    cache.P = xt::zeros<double>({nb, nb});
    cache.P_beta = xt::zeros<double>({nb, nb});
    cache.P(0, 0) = cache.P_beta(0, 0) = 1.0;
    // These potentials belong to an auxiliary density, not the shadow.
    internal::GpwWorkspace workspace;
    workspace.veff_a.assign(geometry.grid.npts(),
                            std::numeric_limits<double>::quiet_NaN());
    workspace.veff_b = workspace.veff_a;
    workspace.v_es = workspace.veff_a;
    RealMatrix shadow_a = cache.P, shadow_b = cache.P_beta;
    shadow_a(0, 0) = 0.6;
    shadow_a(1, 1) = 0.4;
    const auto expected = internal::gpw_forces_at_density(
        geometry, method, shadow_a, c, h, shadow_b, c, h);
    for (int repeat = 0; repeat < 2; ++repeat) {
#ifdef _OPENMP
        const int original_threads = omp_get_max_threads();
        omp_set_num_threads(8);
#endif
        const auto actual = internal::gpw_forces_at_density(
            geometry, method, shadow_a, c, h, shadow_b, c, h);
#ifdef _OPENMP
        omp_set_num_threads(original_threads);
#endif
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t a = 0; a < actual.size(); ++a)
            for (int k = 0; k < 3; ++k) {
                EXPECT_TRUE(std::isfinite(actual[a][k]));
                EXPECT_NEAR(actual[a][k], expected[a][k], 1e-12);
            }
    }
    EXPECT_THROW(internal::gpw_forces_at_density(geometry, method, shadow_a, c,
                                                 h, shadow_b, c, h, &cache,
                                                 &workspace),
                 std::invalid_argument);
}
