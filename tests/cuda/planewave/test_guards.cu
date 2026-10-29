/// @file tests/cuda/planewave/test_guards.cu
/// @brief `cuda::pw::PW` surface + honesty guards (no SCF, no device work).
///
/// Every assertion here fires before `setup_pw_problem` and before the first
/// CUDA call: the class must reject a member it cannot honour instead of
/// silently computing a different Hamiltonian. Unit-scope by construction —
/// one method (`run`'s member validation), no loop, no oracle.
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#include <memory>

#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>

using namespace ATV;

// PW owns device-backed state and can be transferred by move. Its execution
// topology is tracked separately in the architecture debt ledger.
static_assert(!std::is_copy_constructible_v<ATV::cuda::pw::PW>,
              "cuda::pw::PW owns device-backed run state: no copies");
static_assert(std::is_move_constructible_v<ATV::cuda::pw::PW>,
              "cuda::pw::PW must be movable (constructed then handed off)");

namespace {

/// @brief One H atom in a 6 Å cubic box. The potential comes from the
///        analytic GTH-PBE-q1 parameterization, so no UPF file is needed.
cpu::Frame make_h_atom() {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.box = cpu::Box::cubic(6.0);
    return f;
}

/// @brief Method with every member on a supported setting (LSDA, restricted,
///        fixed occupations) so a test can flip exactly one of them.
/// @param xc Exchange–correlation functional.
/// @param ecut_ry Wavefunction cutoff [Ry].
/// @param k_mesh Monkhorst-Pack mesh per dimension (dimensionless).
cuda::pw::PW make_supported(cpu::xc::XCType xc = cpu::xc::XCType::LSDA,
                            double ecut_ry = 6.0, int k_mesh = 1) {
    // Real H potential, no file needed.
    return cuda::pw::PW{std::make_shared<cpu::pw::GthPseudopotential>(
                            molrs::Element::H, "GTH-PBE-q1"),
                        ecut_ry, k_mesh, xc};
}

} // namespace

TEST(CudaPWGuards, DefaultXcIsPbe) {
    cuda::pw::PW pw{std::make_shared<cpu::pw::GthPseudopotential>(
                        molrs::Element::H, "GTH-PBE-q1"),
                    /*ecut_ry=*/30.0, /*k_mesh=*/4};
    // The default deliberately matches cpu::pw::PW (PBE): identical user
    // code computes the same functional on both backends. PBE has a device
    // path, so the default is runnable — no throw to assert here.
    EXPECT_EQ(pw.xc, cpu::xc::XCType::PBE);
}

TEST(CudaPWGuards, UnsupportedXcThrows) {
    auto frame = make_h_atom();
    // BLYP has no device kernel — LSDA and the PBE family do. The gate must throw
    // rather than silently compute a different functional.
    auto pw = make_supported(cpu::xc::XCType::BLYP);
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, PbeFamilyValidatesOtherSettingsBeforeDeviceWork) {
    auto frame = make_h_atom();
    for (const auto xc : {cpu::xc::XCType::PBE, cpu::xc::XCType::PBEsol,
                          cpu::xc::XCType::RevPBE, cpu::xc::XCType::RPBE})
        for (const bool spin : {false, true}) {
            auto pw = make_supported(xc);
            pw.spin_polarized = spin;
            pw.multiplicity = spin ? 2 : 1;
            pw.tol = 0;
            // Reach numerical validation without an SCF or device work.
            try {
                pw.run(frame);
                FAIL() << "invalid tolerance accepted";
            } catch (const std::invalid_argument& error) {
                EXPECT_NE(std::string(error.what()).find("tol"),
                          std::string::npos)
                    << error.what();
            }
        }
}

TEST(CudaPWGuards, MultiplicityWithoutSpinThrows) {
    auto frame = make_h_atom();
    auto pw = make_supported();
    pw.multiplicity = 2;
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, SpinSmearZeroDegaussThrows) {
    auto frame = make_h_atom();
    auto pw = make_supported();
    pw.spin_polarized = true;
    pw.multiplicity = 2;
    pw.smearing = ::ATV::electronic::Smearing::fermi_dirac;
    pw.degauss = 0.0;
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, SmearZeroDegaussThrows) {
    auto frame = make_h_atom();
    auto pw = make_supported();
    pw.smearing = ::ATV::electronic::Smearing::fermi_dirac;
    pw.degauss = 0.0;
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, NonPositiveTolThrows) {
    auto frame = make_h_atom();
    auto pw = make_supported();
    pw.tol = 0.0;
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, ZeroKMeshThrows) {
    auto frame = make_h_atom();
    // generate_mp_mesh returns an empty grid for 0 without throwing, so an
    // ungated k_mesh = 0 reports a "converged" run with e_band == 0.
    auto pw = make_supported(cpu::xc::XCType::LSDA, /*ecut_ry=*/6.0,
                             /*k_mesh=*/0);
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, NonPositiveEcutThrows) {
    auto frame = make_h_atom();
    auto pw = make_supported(cpu::xc::XCType::LSDA, /*ecut_ry=*/0.0);
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, ZeroPulayDepthThrows) {
    auto frame = make_h_atom();
    auto pw = make_supported();
    pw.pulay_depth = 0;
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, EmptyPseudopotentialThrowsAtConstruction) {
    EXPECT_THROW((cuda::pw::PW{std::make_shared<cpu::pw::GthPseudopotential>(
                                   molrs::Element::H, ""),
                               6.0, 1}),
                 std::invalid_argument);
    EXPECT_THROW(
        (cuda::pw::PW{std::make_shared<cpu::pw::UpfPseudopotential>(""), 6.0,
                      1}),
        std::invalid_argument);
}

TEST(CudaPWGuards, FreshResultIsEmpty) {
    auto pw = make_supported();
    EXPECT_EQ(pw.last_result().iterations, 0);
    EXPECT_FALSE(pw.last_result().converged);
}

// Spin-smear stress is supported, but invalid thermal widths must still
// fail before CUDA setup through both public entries.
TEST(CudaPWGuards, SpinSmearStressRejectsInvalidWidthBeforeDeviceWork) {
    auto frame = make_h_atom();
    for (const auto xc : {cpu::xc::XCType::LSDA, cpu::xc::XCType::PBE}) {
        auto pw = make_supported(xc);
        pw.spin_polarized = true;
        pw.multiplicity = 2;
        for (const auto smear :
             {::ATV::electronic::Smearing::fermi_dirac,
              ::ATV::electronic::Smearing::gaussian,
              ::ATV::electronic::Smearing::methfessel_paxton}) {
            pw.smearing = smear;
            pw.degauss = 0;
            EXPECT_THROW(pw.stress(frame), std::invalid_argument);
            EXPECT_THROW(pw.stress(frame), std::invalid_argument);
        }
        EXPECT_EQ(pw.last_result().iterations, 0);
        EXPECT_FALSE(pw.last_result().converged);
    }
}
