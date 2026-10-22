/// @file test_rhf.cu
/// @brief Integration tests for H2/STO-3G/RHF: integrals, CPU solver, CUDA solver, parity.

#ifdef ATV_HAS_HF

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/core/device.h"
#include "atomiverse/core/driver_template.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/basis_data/accessors.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cuda/driver/driver.cuh"
#include "atomiverse/cuda/driver/driver_node.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/evaluate/single_point.cuh"
#include "atomiverse/cuda/scf/internal/rhf_scf_cycle.cuh"
#include "atomiverse/cuda/scf/rhf.cuh"
#include "atomiverse/cuda/scf/scf_evaluator.cuh"

using namespace ATV;

namespace {

cpu::Frame make_h2() {
    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = double(0);
    frame.span(kAtom_y)[0] = double(0);
    frame.span(kAtom_z)[0] = double(0);
    frame.span(kAtom_x)[1] = double(0.74);
    frame.span(kAtom_y)[1] = double(0);
    frame.span(kAtom_z)[1] = double(0);
    frame.span(kAtom_mass)[0] = double(1.008);
    frame.span(kAtom_mass)[1] = double(1.008);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_type)[1] = 1; // H: Z=1
    frame.box = cpu::Box::cubic(100.0);
    return frame;
}

// Lone hydrogen atom — one electron, odd count (open shell).
cpu::Frame make_h_atom() {
    cpu::Frame frame(1);
    frame.span(kAtom_x)[0] = double(0);
    frame.span(kAtom_y)[0] = double(0);
    frame.span(kAtom_z)[0] = double(0);
    frame.span(kAtom_mass)[0] = double(1.008);
    frame.span(kAtom_type)[0] = 1; // H: Z=1, 1 electron
    frame.box = cpu::Box::cubic(100.0);
    return frame;
}

// Water (Å) — same geometry as CPU SCF H2O pins; s+p STO-3G force twin.
cpu::Frame make_h2o() {
    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = double(0);
    frame.span(kAtom_y)[0] = double(0);
    frame.span(kAtom_z)[0] = double(0.1173);
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = double(0);
    frame.span(kAtom_y)[1] = double(0.7572);
    frame.span(kAtom_z)[1] = double(-0.4692);
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = double(0);
    frame.span(kAtom_y)[2] = double(-0.7572);
    frame.span(kAtom_z)[2] = double(-0.4692);
    frame.span(kAtom_mass)[0] = double(15.999);
    frame.span(kAtom_mass)[1] = double(1.008);
    frame.span(kAtom_mass)[2] = double(1.008);
    frame.box = cpu::Box::cubic(100.0);
    return frame;
}

// Textbook H2/STO-3G/RHF reference values
constexpr double kH2Energy = -1.1167; // approx total energy [Hartree]
constexpr double kH2Vnuc = 0.7151;    // approx nuclear repulsion [Hartree]
// Public Frame energy is eV (iron law); convert pins when comparing meta.
inline double ha_to_ev(double ha) { return ha * units::kHartreeInEv; }
inline double meta_ha(const cpu::Frame& f) {
    return static_cast<double>(f.meta.get(kEnergy)) / units::kHartreeInEv;
}

// CPU RHF reference from the concrete cpu::scf::RHF method (Frame in/out).
// Replaces the DELETED free CPU RHF oracle function. This is the project's
// own CPU RHF — not a third-party oracle — used to ground the CPU-side checks;
// the CUDA parity checks below compare against the hard-coded textbook
// constants (kH2Energy, kHeEnergy, ...) directly.
struct CpuRhfRef {
    double energy = 0;      // total SCF energy [Hartree]
    bool converged = false; // loop convergence verdict
};

CpuRhfRef cpu_rhf_reference(cpu::Frame& frame, int charge = 0) {
    const double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, charge, tol, 100};
    rhf.run(frame);
    return {rhf.energy(), rhf.converged()};
}

} // anonymous namespace

// ── Integral tests ────────────────────────────────────────────────────

TEST(RHF, IntegralSymmetry) {
    auto frame = make_h2();
    auto integrals =
        cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};

    int n = integrals.n_basis;
    EXPECT_EQ(n, 2);
    EXPECT_EQ(integrals.n_electrons, 2);
    EXPECT_EQ(integrals.n_occ(), 1);

    // S, Hcore should be symmetric
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(integrals.S(i, j), integrals.S(j, i), 1e-12);
            EXPECT_NEAR(integrals.Hcore(i, j), integrals.Hcore(j, i), 1e-12);
        }

    // Overlap diagonal should be ~1 for normalized basis
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(static_cast<double>(integrals.S(i, i)), 1.0, 0.01);
}

TEST(RHF, NuclearRepulsion) {
    auto frame = make_h2();
    auto integrals =
        cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};
    // V_nuc = Z_A * Z_B / R_AB (in Bohr) = 1*1 / (0.74 * 1.8897) ~ 0.7151
    EXPECT_NEAR(static_cast<double>(integrals.V_nuc), kH2Vnuc, 0.01);
}

// ── Orthogonalization test ────────────────────────────────────────────

TEST(RHF, Orthogonalization) {
    auto frame = make_h2();
    auto integrals =
        cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};
    int n = integrals.n_basis;

    // Compute X = S^{-1/2} the same way the RHF orthogonalization does
    using RealMatrix = xt::xtensor<double, 2, xt::layout_type::column_major>;
    RealMatrix S_cm = integrals.S;
    auto [s_eig, U] = xt::linalg::eigh(S_cm);
    RealMatrix s_inv_sqrt = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        s_inv_sqrt(i, i) = double(1) / std::sqrt(s_eig(i));
    RealMatrix X =
        xt::linalg::dot(xt::linalg::dot(U, s_inv_sqrt), xt::transpose(U));

    // Verify X^T S X = I
    double ortho_tol = (sizeof(double) == sizeof(float)) ? 1e-6 : 1e-10;
    RealMatrix XtSX =
        xt::linalg::dot(xt::transpose(X), xt::linalg::dot(S_cm, X));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(static_cast<double>(XtSX(i, j)), expected, ortho_tol)
                << "X^T S X not identity at (" << i << "," << j << ")";
        }
}

// ── Fock build test ──────────────────────────────────────────────────

// ── CPU RHF tests ─────────────────────────────────────────────────────

// ── Occupation guard (hf-robust-01 CUDA mirror) ──────────────────────
// Guards live on the private RhfScfCycle (owned by RHF). Host-matrix path
// throws before device work; device path rejects n_occ < 0.

TEST(RHF, OddElectronThrowsHostIntegrals) {
    // Lone H atom / STO-3G: 1 electron — closed-shell RHF rejects odd count.
    auto frame = make_h_atom();
    xt::xtensor<double, 2> S, Hcore;
    xt::xtensor<double, 4> ERI;
    double V_nuc = 0;
    int n_basis = 0, n_electrons = 0;
    cpu::scf::internal::IntegralWorkspace::fill_rhf_matrices(
        frame, cpu::scf::BasisSet::STO3G, 0, true, S, Hcore, ERI, V_nuc,
        n_basis, n_electrons);
    ASSERT_EQ(n_electrons, 1);

    EXPECT_THROW((cuda::scf::internal::RhfScfCycle{
                     std::move(S), std::move(Hcore), std::move(ERI), V_nuc,
                     n_electrons, 1e-10, 100}),
                 std::invalid_argument);
}

TEST(RHF, EvenElectronConstructsHostIntegrals) {
    // Neutral H2 (2 electrons) constructs without throwing.
    auto frame = make_h2();
    xt::xtensor<double, 2> S, Hcore;
    xt::xtensor<double, 4> ERI;
    double V_nuc = 0;
    int n_basis = 0, n_electrons = 0;
    cpu::scf::internal::IntegralWorkspace::fill_rhf_matrices(
        frame, cpu::scf::BasisSet::STO3G, 0, true, S, Hcore, ERI, V_nuc,
        n_basis, n_electrons);
    ASSERT_EQ(n_electrons, 2);

    EXPECT_NO_THROW((cuda::scf::internal::RhfScfCycle{
        std::move(S), std::move(Hcore), std::move(ERI), V_nuc, n_electrons,
        1e-10, 100}));
}

TEST(RHF, NegativeNoccThrowsDeviceIntegrals) {
    // Device-integrals path guards only non-negativity of n_occ.
    const auto& h1s = cpu::scf::basis_data::pure_s_sto3g_shell(/*Z=*/1);
    EXPECT_THROW(
        (cuda::scf::internal::RhfScfCycle{static_cast<int>(h1s.alpha.size()),
                                          h1s.alpha.data(), h1s.coeff.data(),
                                          /*n_occ=*/-1, 1e-10, 100}),
        std::invalid_argument);
}

// ── CUDA RHF single-point tests (cuda::scf::RHF + Evaluate) ───────

// ── CPU/CUDA parity ───────────────────────────────────────────────────

// ── Public CUDA RHF (Frame only; private workspace inside the method) ─

// ── Multi-system tests ───────────────────────────────────────────────

static cpu::Frame make_he() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_mass)[0] = double(4.003);
    f.span(kAtom_type)[0] = 2;
    f.box = cpu::Box::cubic(100.0);
    return f;
}

static cpu::Frame make_heh_plus() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = double(0);
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = double(0.772);
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_mass)[0] = double(4.003);
    f.span(kAtom_mass)[1] = double(1.008);
    f.span(kAtom_type)[0] = 2;
    f.span(kAtom_type)[1] = 1; // HeH+ has 2 electrons
    f.box = cpu::Box::cubic(100.0);
    return f;
}

static cpu::Frame make_lih() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = double(0);
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = double(1.595);
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_mass)[0] = double(6.941);
    f.span(kAtom_mass)[1] = double(1.008);
    f.span(kAtom_type)[0] = 3;
    f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(100.0);
    return f;
}

constexpr double kHeEnergy = -2.8078;  // He/STO-3G/RHF [Hartree]
constexpr double kHeHEnergy = -2.8609; // HeH+/STO-3G/RHF [Hartree]
constexpr double kLiHEnergy = -7.8634; // LiH/STO-3G/RHF [Hartree]

// ── Multi-system CUDA parity ─────────────────────────────────────────

// ── Structural / discipline tests ────────────────────────────────────

TEST(RHF, PublicRHFIsScfDriver) {
    // Concrete RHF is the SCF loop node (Driver + HF), not a Component.
    // Outer Evaluate/MD nest it with set_evaluator / on(phase).add.
    static_assert(
        std::is_base_of_v<cuda::Driver<ATV::scf::Phase>, cuda::scf::RHF>,
        "cuda::scf::RHF must be cuda::Driver<scf::Phase>");
    static_assert(std::is_base_of_v<cuda::DriverNode, cuda::scf::RHF>,
                  "cuda::scf::RHF must be a nestable DriverNode");
    static_assert(std::is_base_of_v<cuda::HF, cuda::scf::RHF>,
                  "cuda::scf::RHF must inherit cuda::HF");
    static_assert(!std::is_base_of_v<cuda::Component, cuda::scf::RHF>,
                  "cuda::scf::RHF must not inherit cuda::Component");
    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    EXPECT_EQ(rhf.default_budget(), rhf.max_iter);
    EXPECT_TRUE(rhf.provides_energy());
}

TEST(RHF, PrivateCycleIsNotADriver) {
    // The workspace helper is not the loop. The loop is RHF.
    static_assert(
        !std::is_base_of_v<Driver<ATV::evaluate::Phase, cuda::device_tag>,
                           cuda::scf::internal::RhfScfCycle>,
        "RhfScfCycle must not be a Driver<> subclass");
    static_assert(!std::is_base_of_v<cuda::Driver<ATV::scf::Phase>,
                                     cuda::scf::internal::RhfScfCycle>,
                  "RhfScfCycle must not be cuda::Driver<scf::Phase>");
    static_assert(
        !std::is_base_of_v<cuda::DriverNode, cuda::scf::internal::RhfScfCycle>,
        "RhfScfCycle must not be a DriverNode");
    SUCCEED();
}

TEST(RHF, ScfPhaseOrderMatchesMeanField) {
    // Driver-like phase order frozen for the private cycle.
    EXPECT_EQ(ATV::slot_of(ATV::scf::Phase::fock), 0);
    EXPECT_EQ(ATV::slot_of(ATV::scf::Phase::mix), 1);
    EXPECT_EQ(ATV::slot_of(ATV::scf::Phase::diagonalize), 2);
    EXPECT_EQ(ATV::slot_of(ATV::scf::Phase::density), 3);
    EXPECT_EQ(ATV::slot_of(ATV::scf::Phase::check), 4);
    EXPECT_EQ(ATV::slot_count<ATV::scf::Phase>(), 5);
}

TEST(RHF, EvaluateRunAcceptsCPUFrameOnly) {
    // cuda::Evaluate::run() takes cpu::Frame& -- compile-time guarantee.
    static_assert(std::is_invocable_v<decltype(&cuda::Evaluate::run),
                                      cuda::Evaluate*, cpu::Frame&>,
                  "cuda::Evaluate::run() must accept cpu::Frame&");
}

TEST(RHF, PrivateIntegralWorkspaceIsHostOnly) {
    // Internal integral workspace (not a public API type) is host xtensor
    // storage — unit tests may include internal/integrals.h; users must not.
    static_assert(
        std::is_same_v<
            decltype(std::declval<cpu::scf::internal::IntegralWorkspace>().S),
            xt::xtensor<double, 2>>,
        "private integral workspace S must be host xtensor");
}

TEST(RHF, DeterminantDensityHasNoCUDA) {
    // The converged wavefunction is host-side dense linear algebra: the CPU
    // RHF density (density_total()) is a column-major xtensor, not a device
    // pointer. (Replaces the deleted cpu::scf::RHFResult contract check.)
    static_assert(
        std::is_same_v<decltype(std::declval<cpu::scf::SlaterDeterminant>()
                                    .density_total()),
                       xt::xtensor<double, 2, xt::layout_type::column_major>>,
        "SlaterDeterminant::density_total() must be a host xtensor");
}

// ── Force milestone (wave-2): H2/STO-3G unit coverage ─────────────────
// Fail-closed before water e2e: CPU RHF force parity + central FD of run().

TEST(RHF, WithGradientChain) {
    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    EXPECT_TRUE(rhf.compute_force);
    rhf.with_gradient(false);
    EXPECT_FALSE(rhf.compute_force);
    rhf.with_gradient(true);
    EXPECT_TRUE(rhf.compute_force);
}

TEST(RHF, OutOfCoverageForcesThrowHowto) {
    // def2-SVP is outside the device s/p STO-3G envelope. Must throw a
    // how-to, not silently grade cpu::scf::RHF::forces as CUDA.
    auto frame = make_h2o();
    cuda::scf::RHF rhf{cpu::scf::BasisSet::Def2SVP, 0, 1e-8, 20};
    try {
        (void)rhf.forces(frame);
        FAIL() << "expected invalid_argument for def2-SVP device forces";
    } catch (const std::invalid_argument& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("cpu::scf::RHF"), std::string::npos) << msg;
        EXPECT_NE(msg.find("STO-3G"), std::string::npos) << msg;
    }
}

// ── AIMD milestone (wave-3): multi-step MD + per-launch geometry rebuild ─
// CPU twin: MethodDriverWire.NestedMultiStepAimd / MdAcceptsNestedRhfForce.
// No AimdDriver; composition is cuda::md::MD + set_evaluator(RHF) + VV.

#else // !ATV_HAS_HF

#include <gtest/gtest.h>
TEST(RHF, Disabled) { GTEST_SKIP() << "RHF tests require -DATV_BUILD_HF=ON"; }

#endif // ATV_HAS_HF
