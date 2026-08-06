/// @file tests/cuda/planewave/test_guards.cu
/// @brief `cuda::pw::PW` surface + honesty guards (no SCF, no device work).
///
/// Every assertion here fires before `setup_pw_problem` and before the first
/// CUDA call: the class must reject a member it cannot honour instead of
/// silently computing a different Hamiltonian. Unit-scope by construction —
/// one method (`run`'s member validation), no loop, no oracle.
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>

using namespace ATV;

// Value semantics of the method object. ("PW is a concrete method, not a
// Component / Driver node" is an architectural rule — it lives in CLAUDE.md
// and is enforced by review, never by a test artefact.)
static_assert(!std::is_copy_constructible_v<ATV::cuda::pw::PW>,
              "cuda::pw::PW owns device-backed run state: no copies");
static_assert(std::is_move_constructible_v<ATV::cuda::pw::PW>,
              "cuda::pw::PW must be movable (member-set then handed off)");

namespace {

/// @brief One H atom in a 6 Å cubic box, toy V_loc (no UPF file needed).
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
cuda::pw::PW make_supported() {
    cuda::pw::PW pw;
    pw.upf_path = "";
    pw.ecut_ry = 6.0;
    pw.k_mesh = 1;
    pw.xc = cpu::scf::XCType::LSDA;
    return pw;
}

} // namespace

TEST(CudaPWGuards, DefaultXcIsPbeAndThrows) {
    auto frame = make_h_atom();
    cuda::pw::PW pw;
    // The default deliberately matches cpu::pw::PW (PBE): a CUDA-only LSDA
    // default would make identical user code compute a different functional
    // on the two backends.
    EXPECT_EQ(pw.xc, cpu::scf::XCType::PBE);
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, SpinPolarizedThrows) {
    auto frame = make_h_atom();
    auto pw = make_supported();
    pw.spin_polarized = true;
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, MultiplicityThrows) {
    auto frame = make_h_atom();
    auto pw = make_supported();
    pw.multiplicity = 2;
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, SmearingThrows) {
    auto frame = make_h_atom();
    auto pw = make_supported();
    pw.smearing = cuda::pw::Smearing::fermi_dirac;
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
    auto pw = make_supported();
    // generate_mp_mesh returns an empty grid for 0 without throwing, so an
    // ungated k_mesh = 0 reports a "converged" run with e_band == 0.
    pw.k_mesh = 0;
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, NonPositiveEcutThrows) {
    auto frame = make_h_atom();
    auto pw = make_supported();
    pw.ecut_ry = 0.0;
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, ZeroPulayDepthThrows) {
    auto frame = make_h_atom();
    auto pw = make_supported();
    pw.pulay_depth = 0;
    EXPECT_THROW(pw.run(frame), std::invalid_argument);
}

TEST(CudaPWGuards, FreshResultIsEmpty) {
    cuda::pw::PW pw;
    EXPECT_EQ(pw.last_result().iterations, 0);
    EXPECT_FALSE(pw.last_result().converged);
}
