/// @file tests/cpu/scf/test_xc_response_kernel.cpp
/// @brief Unit checks for the closed-shell f^xc response kernel and the KS
///        (exact-exchange fraction + kernel) configuration of TDHFPack, plus
///        the KS response admission guards.
///
/// NO SCF loop runs here: the reference is the orthonormal core-Hamiltonian
/// determinant of water/STO-3G (one generalized eigensolve). Kernel and
/// solver identities hold for any orthonormal integer-occupation reference.

#ifdef ATV_HAS_HF

#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/internal/tdhf_pack.h"
#include "atomiverse/cpu/scf/internal/xc_response_kernel.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/scf/xc_grid_engine.h"

using namespace ATV;
using cpu::scf::RealMatrix;
using cpu::scf::RealVector;
using cpu::scf::SlaterDeterminant;
using cpu::scf::internal::Channel;
using cpu::scf::internal::IntegralWorkspace;
using cpu::scf::internal::OrbitalHessian;
using cpu::scf::internal::TDHFPack;
using cpu::scf::internal::XcResponseKernel;

namespace {

cpu::Frame make_water() {
    cpu::Frame f(3);
    const double y[3] = {0.0, 0.7572, -0.7572};
    const double z[3] = {0.1173, -0.4692, -0.4692};
    const int Z[3] = {8, 1, 1};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = y[i];
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_atomic_number)[i] = Z[i];
    }
    f.box = cpu::Box::cubic(30.0);
    return f;
}

/// Core-Hamiltonian determinant: Hcore C = S C ε, five doubly occupied.
SlaterDeterminant core_determinant(const IntegralWorkspace& pack) {
    auto [s, U] = xt::linalg::eigh(pack.S);
    RealMatrix X = U;
    for (std::size_t c = 0; c < s.size(); ++c)
        xt::view(X, xt::all(), c) /= std::sqrt(s(c));
    const RealMatrix Hp = xt::linalg::dot(
        xt::transpose(X), RealMatrix(xt::linalg::dot(pack.Hcore, X)));
    auto [eps, V] = xt::linalg::eigh(Hp);
    const RealMatrix C = xt::linalg::dot(X, V);
    RealVector occ = xt::zeros<double>({eps.size()});
    for (int i = 0; i < 5; ++i)
        occ(i) = 1.0;
    return SlaterDeterminant{C, RealVector(eps), occ, 5};
}

struct Water {
    cpu::Frame frame = make_water();
    IntegralWorkspace pack{frame, cpu::scf::BasisSet::STO3G, 0,
                           /*compute_eri=*/true};
    SlaterDeterminant det = core_determinant(pack);
    cpu::scf::MolecularGrid grid{pack.view().atoms.data(), 3,
                                 cpu::scf::GridLevel::SG0, 50, 110};
};

const Water& water() {
    static const Water w;
    return w;
}

} // namespace

TEST(XcResponseKernel, DenseIsSymmetricAndColumnsMatchApply) {
    const Water& w = water();
    for (Channel ch : {Channel::singlet, Channel::triplet}) {
        const XcResponseKernel K{w.det, w.pack.view(), cpu::xc::XCType::PBE,
                                 w.grid, ch};
        ASSERT_EQ(K.n_ov(), 10);
        const RealMatrix M = K.dense();
        RealVector v = xt::zeros<double>({std::size_t(10)});
        for (int p = 0; p < 10; ++p)
            v(p) = 0.1 * (p + 1) - 0.4;
        const RealVector Kv = K.apply(v);
        const RealVector Mv = xt::linalg::dot(M, v);
        for (int p = 0; p < 10; ++p) {
            EXPECT_NEAR(Kv(p), Mv(p), 1e-7);
            for (int q = 0; q < 10; ++q)
                EXPECT_NEAR(M(p, q), M(q, p), 1e-7);
        }
        // Linear in v: the step is normalized to the trial direction.
        const RealVector K3v = K.apply(RealVector(3.0 * v));
        for (int p = 0; p < 10; ++p)
            EXPECT_NEAR(K3v(p), 3.0 * Kv(p), 1e-7);
    }
}

TEST(XcResponseKernel, MatchesDirectionalDerivativeOfVxc) {
    // (K v)_jb = C_jᵀ [dV_xc^α/dε along δP_α = D, δP_β = ±D] C_b, checked
    // against an independent coarse central difference of the grid engine.
    const Water& w = water();
    const auto view = w.pack.view();
    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& C = w.det.coefficients(Spin::alpha);
    const RealMatrix P0 = w.det.density(Spin::alpha);
    const int no = 5, nv = 2;
    RealVector v = xt::zeros<double>({std::size_t(no * nv)});
    v(0) = 0.6;
    v(7) = -0.8; // (a=1, i=2)
    RealMatrix D = xt::zeros<double>({std::size_t(7), std::size_t(7)});
    for (int a = 0; a < nv; ++a)
        for (int i = 0; i < no; ++i)
            for (int mu = 0; mu < 7; ++mu)
                for (int nu = 0; nu < 7; ++nu)
                    D(mu, nu) +=
                        0.5 * v(a * no + i)
                        * (C(mu, i) * C(nu, no + a) + C(mu, no + a) * C(nu, i));
    const cpu::xc::BuiltinXc pbe(cpu::xc::XCType::PBE);
    for (Channel ch : {Channel::singlet, Channel::triplet}) {
        const double sb = ch == Channel::singlet ? 1.0 : -1.0;
        const auto central = [&](double h) {
            const auto xp = cpu::scf::evaluate_xc_batched(
                w.grid, view.shells, 7, RealMatrix(P0 + h * D),
                RealMatrix(P0 + sb * h * D), pbe);
            const auto xm = cpu::scf::evaluate_xc_batched(
                w.grid, view.shells, 7, RealMatrix(P0 - h * D),
                RealMatrix(P0 - sb * h * D), pbe);
            return RealMatrix((xp.Vxc_alpha - xm.Vxc_alpha) / (2 * h));
        };
        // Richardson-extrapolated O(h^4) reference with steps unlike the
        // kernel's own.
        const RealMatrix dV = (4.0 * central(1e-3) - central(2e-3)) / 3.0;
        const XcResponseKernel K{w.det, view, cpu::xc::XCType::PBE, w.grid, ch};
        const RealVector Kv = K.apply(v);
        for (int b = 0; b < nv; ++b)
            for (int j = 0; j < no; ++j) {
                double ref = 0.0;
                for (int mu = 0; mu < 7; ++mu)
                    for (int nu = 0; nu < 7; ++nu)
                        ref += C(mu, j) * dV(mu, nu) * C(nu, no + b);
                EXPECT_NEAR(Kv(b * no + j), ref, 1e-6);
            }
    }
}

TEST(TDHFPack, KsKernelDavidsonMatchesDense) {
    const Water& w = water();
    const auto eri = w.pack.ERI;
    const OrbitalHessian h{w.det, eri};
    for (Channel ch : {Channel::singlet, Channel::triplet}) {
        const XcResponseKernel K{w.det, w.pack.view(), cpu::xc::XCType::B3LYP,
                                 w.grid, ch};
        const TDHFPack dense{h, ch, 0.2, &K};
        TDHFPack davidson{h, ch, 0.2, &K};
        davidson.dense_threshold = 0;
        const auto a = dense.tda(3), b = davidson.tda(3);
        const auto c = dense.rpa(3), d = davidson.rpa(3);
        for (int s = 0; s < 3; ++s) {
            EXPECT_NEAR(a.omega(s), b.omega(s), 1e-7);
            EXPECT_NEAR(c.omega(s), d.omega(s), 1e-7);
            EXPECT_LE(c.omega(s), a.omega(s) + 1e-9);
        }
        // Dense TDA root equals the eigenvalue of A(α) + K.
        RealMatrix A = h.dense_A(ch, 0.2) + K.dense();
        auto [e, U] = xt::linalg::eigh(A);
        (void)U;
        EXPECT_NEAR(a.omega(0), e(0), 1e-10);
    }
}

TEST(TDDFTAdmission, RangeSeparatedHybridRejectsBeforeReference) {
    // Without the guard an unrun method throws runtime_error (no result).
    cpu::scf::RKS method(cpu::scf::BasisSet::STO3G, cpu::xc::XCType::CAMB3LYP);
    using cpu::scf::ExcitationChannel;
    for (ExcitationChannel ch :
         {ExcitationChannel::singlet, ExcitationChannel::triplet}) {
        try {
            (void)cpu::scf::TDA<cpu::scf::RKS>{method}.run(1, ch);
            FAIL() << "TDA<RKS> accepted CAM-B3LYP";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("range-separated"),
                      std::string::npos);
        }
        EXPECT_THROW((cpu::scf::TDDFT<cpu::scf::RKS>{method}.run(1, ch)),
                     std::invalid_argument);
    }
    cpu::scf::RKS pbe(cpu::scf::BasisSet::STO3G, cpu::xc::XCType::PBE);
    EXPECT_THROW((cpu::scf::TDA<cpu::scf::RKS>{pbe}.run(1)),
                 std::runtime_error);
}

#else
#include <gtest/gtest.h>
TEST(XcResponseKernel, SkippedWithoutHF) { GTEST_SKIP() << "ATV_HAS_HF off"; }
#endif
