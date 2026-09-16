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

TEST(RHF, FockBuild) {
    auto frame = make_h2();
    auto integrals = cpu::scf::internal::IntegralWorkspace{
        frame, cpu::scf::BasisSet::STO3G, 0, true};
    int n = integrals.n_basis;

    // Use zero density -> G(0) = 0 -> F = Hcore
    using RealMatrix = xt::xtensor<double, 2, xt::layout_type::column_major>;
    RealMatrix P_zero = xt::zeros<double>({n, n});

    // G_ij = sum_kl P_kl * [(ij|kl) - 0.5*(ik|jl)]
    // With P=0, G=0, so F = Hcore
    RealMatrix G = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double val = 0;
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    val += P_zero(k, l)
                           * (integrals.ERI(i, j, k, l)
                              - double(0.5) * integrals.ERI(i, k, j, l));
            G(i, j) = val;
        }

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(static_cast<double>(G(i, j)), 0.0, 1e-15)
                << "G(P=0) should be zero at (" << i << "," << j << ")";

    // Now use converged density and verify G is nonzero and symmetric.
    // The converged density comes from the concrete cpu::scf::RHF method;
    // density_total() is the factor-2 closed-shell P (same convention the
    // deleted free CPU solver used), and its column-major RealMatrix type is
    // identical to the local one, so the ERI contraction below is unchanged.
    const double tol_cpu =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol_cpu, 100};
    rhf.run(frame);
    RealMatrix P_conv = rhf.determinant().density_total();

    RealMatrix G_conv = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double val = 0;
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    val += P_conv(k, l)
                           * (integrals.ERI(i, j, k, l)
                              - double(0.5) * integrals.ERI(i, k, j, l));
            G_conv(i, j) = val;
        }

    // G must be symmetric
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(static_cast<double>(G_conv(i, j)),
                        static_cast<double>(G_conv(j, i)), 1e-10)
                << "G not symmetric at (" << i << "," << j << ")";

    // F = Hcore + G should give self-consistent energy
    RealMatrix F_conv = integrals.Hcore + G_conv;
    double E_elec =
        double(0.5) * xt::sum((integrals.Hcore + F_conv) * P_conv)();
    double E_total = E_elec + integrals.V_nuc;
    EXPECT_NEAR(static_cast<double>(E_total), kH2Energy, 0.01)
        << "Reconstructed energy from Fock build does not match";
}

// ── CPU RHF tests ─────────────────────────────────────────────────────

TEST(RHF, CPUConvergence) {
    auto frame = make_h2();
    auto ref = cpu_rhf_reference(frame);

    EXPECT_TRUE(ref.converged);
    EXPECT_TRUE(std::isfinite(ref.energy));

    // Energy should match textbook within 0.01 Hartree
    EXPECT_NEAR(ref.energy, kH2Energy, 0.01)
        << "CPU RHF energy " << ref.energy << " does not match textbook "
        << kH2Energy;
}

TEST(RHF, CPUDensityTrace) {
    auto frame = make_h2();
    auto integrals =
        cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};

    // Converged density from the concrete cpu::scf::RHF method.
    const double tol_cpu =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol_cpu, 100};
    rhf.run(frame);
    auto P = rhf.determinant().density_total();

    // Tr(P * S) should equal n_electrons
    double trace_PS = xt::sum(P * integrals.S)();
    EXPECT_NEAR(static_cast<double>(trace_PS),
                static_cast<double>(integrals.n_electrons), 0.01);
}

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

TEST(RHF, CUDANonDefaultStreamEnergyPath) {
    // cuda-rhf-nondefault-stream: Evaluate owns a non-default main stream
    // (cudaStreamCreate in driver.cu). Residual/V_nuc D2H after device s/p
    // SCF is bound to that stream (not the null default stream). Physics
    // tols unchanged from CUDASinglePointEnergy.
    auto frame = make_h2();
    double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);
    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
    rhf.with_gradient(false);
    cuda::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(rhf);
    eval.build();
    eval.run(frame);
    EXPECT_TRUE(rhf.converged());
    EXPECT_TRUE(rhf.last_scf_on_device());
    EXPECT_NEAR(meta_ha(frame), kH2Energy, 0.01);
}

TEST(RHF, CUDASinglePointEnergy) {
    // ac-006: single-point H2/STO-3G RHF through concrete cuda::scf::RHF
    // reproduces the textbook reference -1.1167 Ha (meta kEnergy is eV).
    auto frame = make_h2();
    double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);

    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
    rhf.with_gradient(false); // energy-only — force kernel is a separate path
    rhf.run(frame);

    EXPECT_TRUE(rhf.converged());
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.meta.get(kEnergy))));
    EXPECT_NEAR(meta_ha(frame), kH2Energy, 0.01)
        << "CUDA RHF single-point energy " << meta_ha(frame)
        << " Ha does not match textbook";
    EXPECT_NEAR(rhf.energy(), kH2Energy, 0.01);
    // Public meta is eV.
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergy)),
                ha_to_ev(kH2Energy), ha_to_ev(0.01));
}

TEST(RHF, CUDALevelShiftEnergyInvariant) {
    // hf-robust-03 CUDA mirror: the virtual-orbital level shift is a pure
    // convergence stabilizer — the converged energy must be λ-invariant.
    // Shifted s/p STO-3G must stay on the device integral path (not host-fill).
    auto frame0 = make_h2();
    auto frame5 = make_h2();
    auto frame_cpu = make_h2();
    double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);

    cuda::scf::RHF rhf0{cpu::scf::BasisSet::STO3G, 0, tol, 100,
                        /*level_shift=*/0.0};
    rhf0.with_gradient(false);
    rhf0.run(frame0);

    cuda::scf::RHF rhf5{cpu::scf::BasisSet::STO3G, 0, tol, 100,
                        /*level_shift=*/0.5};
    rhf5.with_gradient(false);
    rhf5.run(frame5);

    ASSERT_TRUE(rhf0.converged());
    ASSERT_TRUE(rhf5.converged());
    // run() destroys the private solver; use sticky path flag (not workspace()).
    EXPECT_TRUE(rhf0.last_scf_on_device())
        << "λ=0 H₂/STO-3G must use device s/p RHF path";
    EXPECT_TRUE(rhf5.last_scf_on_device())
        << "level_shift=0.5 must use device s/p RHF path, not host-fill SCF";
    ASSERT_TRUE(std::isfinite(static_cast<double>(frame0.meta.get(kEnergy))));
    ASSERT_TRUE(std::isfinite(static_cast<double>(frame5.meta.get(kEnergy))));
    double bar = (sizeof(double) == sizeof(float)) ? 1e-4 : 1e-8;
    EXPECT_NEAR(meta_ha(frame5), meta_ha(frame0), bar)
        << "level_shift=0.5 energy " << meta_ha(frame5)
        << " must match level_shift=0 energy " << meta_ha(frame0);

    // Same-settings CPU oracle (λ=0.5, H₂/STO-3G).
    {
        cpu::scf::RHF cpu_rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
        cpu_rhf.level_shift = 0.5;
        cpu_rhf.run(frame_cpu);
        ASSERT_TRUE(cpu_rhf.converged());
        EXPECT_NEAR(meta_ha(frame5), meta_ha(frame_cpu), bar)
            << "CUDA level_shift=0.5 energy " << meta_ha(frame5)
            << " must match CPU " << meta_ha(frame_cpu);
    }
}

TEST(RHF, CUDASinglePointHelper) {
    // cuda::single_point produces the identical result as run()/Evaluate.
    auto frame_direct = make_h2();
    auto frame_helper = make_h2();
    double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);

    {
        cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
        cuda::Evaluate eval;
        eval.on(ATV::evaluate::Phase::evaluate).add(std::move(rhf));
        eval.build();
        eval.run(frame_direct);
    }
    {
        cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
        cuda::single_point(frame_helper, rhf);
    }

    EXPECT_NEAR(static_cast<double>(frame_direct.meta.get(kEnergy)),
                static_cast<double>(frame_helper.meta.get(kEnergy)), 1e-9);
}

// ── CPU/CUDA parity ───────────────────────────────────────────────────

TEST(RHF, CUDAMatchesTextbookH2) {
    // Physics: cuda::scf::RHF single point reproduces the hard-coded
    // H2/STO-3G/RHF textbook energy (-1.1167 Ha).
    auto frame = make_h2();
    double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);

    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
    rhf.run(frame);

    EXPECT_NEAR(meta_ha(frame), kH2Energy, 0.01)
        << "CUDA H2 energy: " << meta_ha(frame) << " vs textbook " << kH2Energy;
}

// ── Public CUDA RHF (Frame only; private workspace inside the method) ─

TEST(RHF, PublicRhfMatchesTextbook) {
    auto frame = make_h2();
    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, double(1e-5), 100};
    rhf.run(frame);

    EXPECT_NEAR(meta_ha(frame), kH2Energy, 0.01)
        << "CUDA RHF energy " << meta_ha(frame)
        << " Ha does not match textbook";
}

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

TEST(RHF, HeAtom) {
    auto frame = make_he();
    auto integrals =
        cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};
    EXPECT_EQ(integrals.n_basis, 1);
    EXPECT_EQ(integrals.n_electrons, 2);
    auto ref = cpu_rhf_reference(frame);
    EXPECT_TRUE(ref.converged);
    EXPECT_NEAR(ref.energy, kHeEnergy, 0.01) << "He energy: " << ref.energy;
}

TEST(RHF, HeHPlus) {
    auto frame = make_heh_plus();
    auto integrals = cpu::scf::internal::IntegralWorkspace{
        frame, cpu::scf::BasisSet::STO3G, 1}; // charge = +1
    EXPECT_EQ(integrals.n_basis, 2);
    EXPECT_EQ(integrals.n_electrons, 2); // He(2) + H(1) - 1 = 2
    EXPECT_EQ(integrals.n_occ(), 1);
    auto ref = cpu_rhf_reference(frame, /*charge=*/1);
    EXPECT_TRUE(ref.converged);
    EXPECT_NEAR(ref.energy, kHeHEnergy, 0.02) << "HeH+ energy: " << ref.energy;
}

TEST(RHF, LiH) {
    auto frame = make_lih();
    auto integrals =
        cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};
    EXPECT_EQ(integrals.n_basis,
              6); // Li: 1s+2s+2p(x,y,z)=5, H: 1s=1 -> total 6
    EXPECT_EQ(integrals.n_electrons, 4);
    auto ref = cpu_rhf_reference(frame);
    EXPECT_TRUE(ref.converged);
    double lih_tol = (sizeof(double) == sizeof(float)) ? 0.1 : 0.01;
    EXPECT_NEAR(ref.energy, kLiHEnergy, lih_tol)
        << "LiH energy: " << ref.energy;
}

TEST(RHF, H2PESScan) {
    double min_energy = 1e10;
    double min_r = 0;
    for (int i = 0; i < 8; ++i) {
        double r = 0.5 + i * 0.2; // 0.5 to 1.9 A
        cpu::Frame f(2);
        f.span(kAtom_x)[0] = double(0);
        f.span(kAtom_y)[0] = 0;
        f.span(kAtom_z)[0] = 0;
        f.span(kAtom_x)[1] = double(r);
        f.span(kAtom_y)[1] = 0;
        f.span(kAtom_z)[1] = 0;
        f.span(kAtom_mass)[0] = double(1.008);
        f.span(kAtom_mass)[1] = double(1.008);
        f.span(kAtom_type)[0] = 1;
        f.span(kAtom_type)[1] = 1;
        f.box = cpu::Box::cubic(100.0);
        auto ref = cpu_rhf_reference(f);
        EXPECT_TRUE(ref.converged) << "H2 scan failed at R=" << r;
        EXPECT_TRUE(std::isfinite(ref.energy));
        if (ref.energy < min_energy) {
            min_energy = ref.energy;
            min_r = r;
        }
    }
    // Minimum should be near 0.7 A (textbook: 0.74 A)
    EXPECT_NEAR(min_r, 0.7, 0.15) << "PES minimum at " << min_r << " A";
}

// ── Multi-system CUDA parity ─────────────────────────────────────────

TEST(RHF, CUDAHeAtom) {
    // He/STO-3G/RHF: cuda::scf::RHF reproduces the textbook energy.
    auto frame = make_he();
    double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);

    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
    rhf.run(frame);

    EXPECT_NEAR(meta_ha(frame), kHeEnergy, 0.01)
        << "He CUDA energy: " << meta_ha(frame);
}

TEST(RHF, CUDALiH) {
    // LiH/STO-3G/RHF: cuda::scf::RHF reproduces the textbook energy.
    auto frame = make_lih();
    double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);

    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
    rhf.run(frame);

    double lih_tol = (sizeof(double) == sizeof(float)) ? 0.1 : 0.01;
    EXPECT_NEAR(meta_ha(frame), kLiHEnergy, lih_tol)
        << "LiH CUDA energy: " << meta_ha(frame);
}

TEST(RHF, CUDAHeHPlus) {
    // HeH+/STO-3G/RHF: charge is a public member on cuda::scf::RHF.
    auto frame = make_heh_plus();
    double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);

    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/1, tol, 100};
    rhf.run(frame);

    EXPECT_NEAR(meta_ha(frame), kHeHEnergy, 0.02)
        << "HeH+ CUDA energy: " << meta_ha(frame);
}

// ── Structural / discipline tests ────────────────────────────────────

TEST(RHF, PublicRHFIsHFComponent) {
    // Concrete RHF is the public energy source (Component + HF).
    static_assert(std::is_base_of_v<cuda::Component, cuda::scf::RHF>,
                  "cuda::scf::RHF must inherit cuda::Component");
    static_assert(std::is_base_of_v<cuda::HF, cuda::scf::RHF>,
                  "cuda::scf::RHF must inherit cuda::HF");
    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    EXPECT_EQ(rhf.role(), 0);
}

TEST(RHF, PrivateCycleIsNotADriver) {
    // RhfScfCycle is private machinery under RHF, not a Driver node.
    // Full cuda::Driver<scf::Phase> for RHF waits on nested CUDA DriverNode
    // (RHF stays Component for Evaluate/MD).
    static_assert(
        !std::is_base_of_v<Driver<ATV::evaluate::Phase, cuda::device_tag>,
                           cuda::scf::internal::RhfScfCycle>,
        "RhfScfCycle must not be a Driver<> subclass");
    static_assert(!std::is_base_of_v<cuda::Driver<ATV::scf::Phase>,
                                     cuda::scf::internal::RhfScfCycle>,
                  "RhfScfCycle must not be cuda::Driver<scf::Phase>");
    static_assert(std::is_base_of_v<cuda::Component, cuda::scf::RHF>,
                  "RHF remains Component for outer nesting");
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

TEST(RHF, CUDAForcesMatchCPU_H2_STO3G) {
    // cuda::scf::RHF.forces vs cpu::scf::RHF.forces on H2 R=0.74 Å STO-3G.
    auto frame_cpu = make_h2();
    auto frame_cuda = make_h2();
    const double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);

    cpu::scf::RHF cpu_rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
    const cpu::scf::GradientResult g_cpu = cpu_rhf.forces(frame_cpu);

    cuda::scf::RHF cuda_rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
    const cpu::scf::GradientResult g_cuda = cuda_rhf.forces(frame_cuda);

    ASSERT_TRUE(cuda_rhf.converged());
    ASSERT_EQ(g_cpu.forces.shape()[0], 2u);
    ASSERT_EQ(g_cuda.forces.shape()[0], 2u);
    EXPECT_TRUE(cuda_rhf.last_force_on_device());

    // Energy: meta eV and GradientResult energy must agree; match CPU.
    const double e_bar =
        (sizeof(double) == sizeof(float)) ? ha_to_ev(1e-4) : ha_to_ev(1e-6);
    EXPECT_NEAR(g_cuda.energy, g_cpu.energy, e_bar);
    EXPECT_NEAR(static_cast<double>(frame_cuda.meta.get(kEnergy)),
                g_cuda.energy, 1e-9);
    EXPECT_NEAR(static_cast<double>(frame_cuda.meta.get(kEnergy)),
                static_cast<double>(frame_cpu.meta.get(kEnergy)), e_bar);

    // Forces [eV/Å]: element-wise |ΔF| tight after matching SCF tol.
    const double f_bar =
        (sizeof(double) == sizeof(float)) ? 1e-3 : 1e-5; // eV/Å
    for (int A = 0; A < 2; ++A) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(g_cuda.forces(A, d), g_cpu.forces(A, d), f_bar)
                << "F[" << A << "][" << d << "] CUDA=" << g_cuda.forces(A, d)
                << " CPU=" << g_cpu.forces(A, d);
            // Frame columns overwritten like CPU standalone forces.
            if (d == 0)
                EXPECT_NEAR(static_cast<double>(frame_cuda.span(kAtom_fx)[A]),
                            g_cuda.forces(A, 0), 1e-12);
            if (d == 1)
                EXPECT_NEAR(static_cast<double>(frame_cuda.span(kAtom_fy)[A]),
                            g_cuda.forces(A, 1), 1e-12);
            if (d == 2)
                EXPECT_NEAR(static_cast<double>(frame_cuda.span(kAtom_fz)[A]),
                            g_cuda.forces(A, 2), 1e-12);
        }
    }
}

TEST(RHF, CUDAForcesMatchFD_H2_STO3G) {
    // Central FD of cuda::scf::RHF.run energy vs analytic forces (H2/STO-3G).
    const double h = 1e-3; // Å
    const double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);

    auto frame0 = make_h2();
    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
    const cpu::scf::GradientResult g = rhf.forces(frame0);
    ASSERT_TRUE(rhf.converged());

    auto energy_at = [&](int atom, int axis, double delta) {
        auto f = make_h2();
        if (axis == 0)
            f.span(kAtom_x)[atom] += delta;
        else if (axis == 1)
            f.span(kAtom_y)[atom] += delta;
        else
            f.span(kAtom_z)[atom] += delta;
        cuda::scf::RHF r{cpu::scf::BasisSet::STO3G, 0, tol, 100};
        r.run(f);
        return static_cast<double>(f.meta.get(kEnergy)); // eV
    };

    // F = −∂E/∂R; central FD of energy in eV/Å.
    // Looser than CPU parity: FD truncation + SCF residual.
    const double fd_bar =
        (sizeof(double) == sizeof(float)) ? 0.5 : 5e-2; // eV/Å
    for (int A = 0; A < 2; ++A) {
        for (int d = 0; d < 3; ++d) {
            const double ep = energy_at(A, d, +h);
            const double em = energy_at(A, d, -h);
            const double f_fd = -(ep - em) / (2.0 * h);
            EXPECT_NEAR(g.forces(A, d), f_fd, fd_bar)
                << "F[" << A << "][" << d << "] analytic=" << g.forces(A, d)
                << " FD=" << f_fd;
        }
    }
}

// Wave 6: device analytic forces for s+p (end host force fallback on H2O).
TEST(RHF, CUDAForcesMatchCPU_H2O_STO3G) {
    auto frame_cpu = make_h2o();
    auto frame_cuda = make_h2o();
    const double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);

    cpu::scf::RHF cpu_rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
    const cpu::scf::GradientResult g_cpu = cpu_rhf.forces(frame_cpu);

    cuda::scf::RHF cuda_rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
    const cpu::scf::GradientResult g_cuda = cuda_rhf.forces(frame_cuda);

    ASSERT_TRUE(cuda_rhf.converged());
    ASSERT_EQ(g_cpu.forces.shape()[0], 3u);
    ASSERT_EQ(g_cuda.forces.shape()[0], 3u);

    // Fail-closed: device s/p force path, not host cpu::scf::RHF::forces.
    EXPECT_TRUE(cuda_rhf.last_force_on_device());

    const double e_bar =
        (sizeof(double) == sizeof(float)) ? ha_to_ev(1e-4) : ha_to_ev(1e-6);
    EXPECT_NEAR(g_cuda.energy, g_cpu.energy, e_bar);

    // Forces [eV/Å]: stated CUDA↔CPU tol (non-bit-exact FP accumulation).
    const double f_bar =
        (sizeof(double) == sizeof(float)) ? 1e-2 : 1e-4; // eV/Å
    for (int A = 0; A < 3; ++A) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(g_cuda.forces(A, d), g_cpu.forces(A, d), f_bar)
                << "F[" << A << "][" << d << "] CUDA=" << g_cuda.forces(A, d)
                << " CPU=" << g_cpu.forces(A, d);
        }
    }
}

TEST(RHF, CUDAForcesMatchFD_H2O_STO3G) {
    // Unit-budget FD spot-check (O, z only) — full 3×3×2 scan is regression-
    // scale; CUDA↔CPU force twin is the primary gate above.
    const double h = 1e-3; // Å
    const double tol =
        (sizeof(double) == sizeof(float)) ? double(1e-5) : double(1e-10);

    auto frame0 = make_h2o();
    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
    const cpu::scf::GradientResult g = rhf.forces(frame0);
    ASSERT_TRUE(rhf.converged());
    EXPECT_TRUE(rhf.last_force_on_device());

    auto energy_at = [&](double delta) {
        auto f = make_h2o();
        f.span(kAtom_z)[0] += delta; // O z
        cuda::scf::RHF r{cpu::scf::BasisSet::STO3G, 0, tol, 100};
        r.run(f);
        return static_cast<double>(f.meta.get(kEnergy)); // eV
    };

    const double fd_bar =
        (sizeof(double) == sizeof(float)) ? 1.0 : 1e-1; // eV/Å
    const double ep = energy_at(+h);
    const double em = energy_at(-h);
    const double f_fd = -(ep - em) / (2.0 * h);
    EXPECT_NEAR(g.forces(0, 2), f_fd, fd_bar)
        << "F[O][z] analytic=" << g.forces(0, 2) << " FD=" << f_fd;
}

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

TEST(RHF, NestedMultiStepAimd) {
    // H2 R=0.74 Å STO-3G, small ±vx, VV 0.5 fs × 3: finite PE < 0.
    auto frame = make_h2();
    frame.span(kAtom_vx)[0] = 0.001;
    frame.span(kAtom_vx)[1] = -0.001;

    cuda::md::MD sim;
    sim.set_integrator(cuda::md::VelocityVerlet{0.5});
    sim.set_evaluator(cuda::scf::RHF{cpu::scf::BasisSet::STO3G, 0, 1e-9, 100,
                                     0.0,
                                     /*compute_force=*/true});
    EXPECT_NO_THROW(sim.build());
    sim.run(frame, 3);

    const double pe = static_cast<double>(frame.meta.get(kEnergy));
    EXPECT_TRUE(std::isfinite(pe)) << "PE after 3 BOMD steps: " << pe;
    EXPECT_LT(pe, 0.0) << "H2 RHF PE should be bound (eV): " << pe;
    // Weak textbook band ~[-32, -29] eV (~-1.117 Ha class); not a golden.
    EXPECT_LT(pe, -29.0);
    EXPECT_GT(pe, -32.0);
}

TEST(RHF, MdAcceptsRhfForceOneStep) {
    auto frame = make_h2();
    cuda::md::MD sim;
    sim.set_integrator(cuda::md::VelocityVerlet{0.5});
    sim.set_evaluator(cuda::scf::RHF{cpu::scf::BasisSet::STO3G});
    EXPECT_NO_THROW(sim.build());
    sim.run(frame, 1);
    const double pe = static_cast<double>(frame.meta.get(kEnergy));
    EXPECT_TRUE(std::isfinite(pe));
    EXPECT_LT(pe, -29.0);
    EXPECT_GT(pe, -32.0);
}

TEST(RHF, LaunchGeometrySensitivity) {
    // Prove per-launch geometry rebuild: energy at stretched R must match
    // standalone run() at that R, and differ from equilibrium energy.
    // Fail-closed if SCF stays on build-time geometry.
    auto frame_eq = make_h2(); // R = 0.74 Å
    cuda::scf::RHF rhf_eq{cpu::scf::BasisSet::STO3G, 0, 1e-10, 100};
    rhf_eq.run(frame_eq);
    const double e_eq = static_cast<double>(frame_eq.meta.get(kEnergy));
    ASSERT_TRUE(std::isfinite(e_eq));
    ASSERT_LT(e_eq, 0.0);

    // Stretched H2 R = 1.20 Å — V_nuc and electronic energy both change.
    cpu::Frame frame_st(2);
    frame_st.span(kAtom_x)[0] = 0.0;
    frame_st.span(kAtom_y)[0] = 0.0;
    frame_st.span(kAtom_z)[0] = 0.0;
    frame_st.span(kAtom_x)[1] = 1.20;
    frame_st.span(kAtom_y)[1] = 0.0;
    frame_st.span(kAtom_z)[1] = 0.0;
    frame_st.span(kAtom_mass)[0] = 1.008;
    frame_st.span(kAtom_mass)[1] = 1.008;
    frame_st.span(kAtom_type)[0] = 1;
    frame_st.span(kAtom_type)[1] = 1;
    frame_st.box = cpu::Box::cubic(100.0);

    cuda::scf::RHF rhf_st{cpu::scf::BasisSet::STO3G, 0, 1e-10, 100};
    rhf_st.run(frame_st);
    const double e_st = static_cast<double>(frame_st.meta.get(kEnergy));
    ASSERT_TRUE(std::isfinite(e_st));

    // Energies at different geometries must differ (not frozen-build PE).
    EXPECT_GT(std::abs(e_st - e_eq), 0.05) // eV — well above SCF noise
        << "eq=" << e_eq << " stretched=" << e_st
        << " (geometry rebuild failed: energy frozen?)";

    // Standalone forces() energy must match run() at the same geometry.
    auto frame_f = frame_st;
    cuda::scf::RHF rhf_f{cpu::scf::BasisSet::STO3G, 0, 1e-10, 100};
    const auto gr = rhf_f.forces(frame_f);
    EXPECT_NEAR(gr.energy, e_st, ha_to_ev(1e-5));
    EXPECT_NEAR(static_cast<double>(frame_f.meta.get(kEnergy)), e_st,
                ha_to_ev(1e-5));
}

TEST(RHF, MultiStepMdGeometryTracksNuclei) {
    // After several MD steps with nonzero initial velocity, PE must remain
    // finite and the H–H separation must have moved (forces drive motion).
    auto frame = make_h2();
    const double r0 = frame.span(kAtom_x)[1] - frame.span(kAtom_x)[0];
    frame.span(kAtom_vx)[0] = 0.002;
    frame.span(kAtom_vx)[1] = -0.002;

    cuda::md::MD sim;
    sim.set_integrator(cuda::md::VelocityVerlet{0.5});
    sim.set_evaluator(cuda::scf::RHF{cpu::scf::BasisSet::STO3G});
    sim.build();
    sim.run(frame, 5);

    const double pe = static_cast<double>(frame.meta.get(kEnergy));
    EXPECT_TRUE(std::isfinite(pe));
    EXPECT_LT(pe, 0.0);
    const double r1 = frame.span(kAtom_x)[1] - frame.span(kAtom_x)[0];
    // Nuclei must have moved; PE still physical (rebuild, not frozen geometry).
    EXPECT_GT(std::abs(r1 - r0), 1e-8)
        << "r0=" << r0 << " r1=" << r1 << " (no nuclear motion)";
}

#else // !ATV_HAS_HF

#include <gtest/gtest.h>
TEST(RHF, Disabled) { GTEST_SKIP() << "RHF tests require -DATV_BUILD_HF=ON"; }

#endif // ATV_HAS_HF
