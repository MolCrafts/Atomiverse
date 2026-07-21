/// @file tests/cpu/scf/test_cphf.cpp
/// @brief Unit checks for the CPHF building blocks (hf-capability-gaps-13):
///        second/skeleton derivative integrals against finite differences,
///        the B^x right-hand side sum rules, and the CphfPack solve — all on
///        micro systems (HeH⁺, milliseconds).
///
/// FD gates use the 5-point central stencil
///   f'(x) ≈ [8(f(h) − f(−h)) − (f(2h) − f(−2h))] / (12h),  O(h⁴),
/// at h = 1e-3 Å over rebuilt IntegralPack / IntegralDerivPack objects,
/// so the classic bug sources (skeleton derivatives, B^x inhomogeneity)
/// are each pinned independently before any Hessian is assembled.
///
/// The CPHF LHS is asserted to be the link-08 OrbitalHessian singlet σ₊
/// (the kernel's third configuration — no reimplementation), and the
/// iterative (preconditioned CG) response is checked against the dense
/// ¹(A+B) direct solve to 1e-10.

#ifdef ATV_HAS_HF

#include <cmath>
#include <functional>

#include <gtest/gtest.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/cphf_pack.h"
#include "atomiverse/cpu/scf/internal/integral_derivatives.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/core/units.h"

using namespace ATV;
using cpu::scf::BasisSet;
using cpu::scf::IntegralDerivPack;
using cpu::scf::IntegralPack;
using cpu::scf::IntegralSecondDerivPack;
using cpu::scf::RealMatrix;
using cpu::scf::RealVector;
using cpu::scf::SlaterDeterminant;
using cpu::scf::internal::Channel;
using cpu::scf::internal::CphfPack;
using cpu::scf::internal::OrbitalHessian;

namespace {

/// HeH⁺ at the Szabo-Ostlund distance R = 1.4632 Bohr (≈ 0.7743 Å),
/// deliberately along a skew axis so every Cartesian component is nonzero.
cpu::Frame make_hehp() {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 2; // He
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1; // H
    f.span(kAtom_x)[1] = 0.40;
    f.span(kAtom_y)[1] = 0.31;
    f.span(kAtom_z)[1] = 0.55;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

/// Copy of `frame` with atom `A`'s axis-`d` coordinate displaced by `dx` Å.
cpu::Frame displaced(const cpu::Frame& frame, int A, int d, double dx) {
    cpu::Frame f = frame;
    if (d == 0)
        f.span(kAtom_x)[A] += static_cast<StorageReal>(dx);
    else if (d == 1)
        f.span(kAtom_y)[A] += static_cast<StorageReal>(dx);
    else
        f.span(kAtom_z)[A] += static_cast<StorageReal>(dx);
    return f;
}

/// 5-point central difference d/dR_{A,d} of a matrix-valued function of the
/// geometry, returned in per-Bohr units (the displacement is applied in Å).
RealMatrix fd5(const std::function<RealMatrix(const cpu::Frame&)>& eval,
               const cpu::Frame& frame, int A, int d, double h_ang) {
    const RealMatrix f_p1 = eval(displaced(frame, A, d, h_ang));
    const RealMatrix f_m1 = eval(displaced(frame, A, d, -h_ang));
    const RealMatrix f_p2 = eval(displaced(frame, A, d, 2.0 * h_ang));
    const RealMatrix f_m2 = eval(displaced(frame, A, d, -2.0 * h_ang));
    const double h_bohr = h_ang * units::kAngstromInBohr;
    return (8.0 * (f_p1 - f_m1) - (f_p2 - f_m2)) / (12.0 * h_bohr);
}

double max_abs(const RealMatrix& m) {
    double v = 0.0;
    for (std::size_t i = 0; i < m.size(); ++i)
        v = std::max(v, std::abs(m.data()[i]));
    return v;
}

} // namespace

// ── d²S / d²T / d²V vs central FD of dS / dT / dV (HeH⁺ / STO-3G) ─────

TEST(Cphf, SecondDerivativeIntegralsMatchFdOfFirst) {
    const auto frame = make_hehp();
    const IntegralPack pack{frame, BasisSet::STO3G, +1};
    const IntegralSecondDerivPack d2{pack};
    const int natm = 2;
    const double h = 1e-3; // Å

    for (int tensor = 0; tensor < 3; ++tensor) {
        for (int B = 0; B < natm; ++B) {
            for (int dj = 0; dj < 3; ++dj) {
                for (int A = 0; A < natm; ++A) {
                    for (int di = 0; di < 3; ++di) {
                        // FD over the FIRST-derivative pack: bump (B, dj),
                        // read slice (A, di).
                        const auto eval =
                            [&](const cpu::Frame& fr) -> RealMatrix {
                            const IntegralPack p{fr, BasisSet::STO3G, +1};
                            const IntegralDerivPack dp{p};
                            const auto& t = (tensor == 0)   ? dp.dS
                                            : (tensor == 1) ? dp.dT
                                                            : dp.dV;
                            return xt::view(t, A, di, xt::all(), xt::all());
                        };
                        const RealMatrix fd = fd5(eval, frame, B, dj, h);
                        const auto& t2 = (tensor == 0)   ? d2.d2S
                                         : (tensor == 1) ? d2.d2T
                                                         : d2.d2V;
                        const RealMatrix an = xt::view(
                            t2, A, di, B, dj, xt::all(), xt::all());
                        EXPECT_LT(max_abs(an - fd), 1e-7)
                            << "tensor=" << tensor << " A=" << A
                            << " di=" << di << " B=" << B << " dj=" << dj;
                    }
                }
            }
        }
    }
}

TEST(Cphf, NuclearRepulsionSecondDerivativeMatchesFd) {
    const auto frame = make_hehp();
    const IntegralPack pack{frame, BasisSet::STO3G, +1};
    const IntegralSecondDerivPack d2{pack};
    const double h = 1e-3; // Å

    for (int B = 0; B < 2; ++B) {
        for (int dj = 0; dj < 3; ++dj) {
            const auto eval = [&](const cpu::Frame& fr) -> RealMatrix {
                const IntegralPack p{fr, BasisSet::STO3G, +1};
                const IntegralDerivPack dp{p};
                RealMatrix out = xt::zeros<double>({2, 3});
                for (int A = 0; A < 2; ++A)
                    for (int di = 0; di < 3; ++di)
                        out(A, di) = dp.dV_nuc(A, di);
                return out;
            };
            const RealMatrix fd = fd5(eval, frame, B, dj, h);
            for (int A = 0; A < 2; ++A)
                for (int di = 0; di < 3; ++di)
                    EXPECT_NEAR(d2.d2V_nuc(A, di, B, dj), fd(A, di), 1e-9);
        }
    }
}

// ── Skeleton MO S^(x) / F^(x) vs FD of S / F at frozen C, P ────────────

TEST(Cphf, SkeletonMoDerivativesMatchFdOfSAndF) {
    const auto frame = make_hehp();

    // Converged reference: C and P are FROZEN at the base geometry — the
    // FD then probes exactly the skeleton (core) derivative.
    cpu::scf::RHF rhf;
    rhf.basis = BasisSet::STO3G;
    rhf.charge = +1;
    rhf.tol = 1e-12;
    rhf.conv_tol_grad = 1e-10; // density-level convergence for FD gates
    rhf.run(frame);
    ASSERT_TRUE(rhf.converged());
    const SlaterDeterminant& det = rhf.determinant();
    const RealMatrix C = det.coefficients(SlaterDeterminant::Spin::alpha);
    const RealMatrix P = det.density_total();

    const IntegralPack pack{frame, BasisSet::STO3G, +1};
    const IntegralDerivPack dpack{pack};
    xt::xtensor<double, 4> dF = dpack.dT + dpack.dV;
    cpu::scf::accumulate_two_electron_fock_derivative(dF, pack, P);

    const double h = 1e-3; // Å
    for (int A = 0; A < 2; ++A) {
        for (int d = 0; d < 3; ++d) {
            // S^(x) in MO: Cᵀ dS C  vs  FD of Cᵀ S(R) C.
            const auto eval_S = [&](const cpu::Frame& fr) -> RealMatrix {
                const IntegralPack p{fr, BasisSet::STO3G, +1};
                return xt::linalg::dot(xt::transpose(C),
                                       xt::linalg::dot(p.S, C));
            };
            RealMatrix dS = xt::view(dpack.dS, A, d, xt::all(), xt::all());
            const RealMatrix an_S = xt::linalg::dot(
                xt::transpose(C), xt::linalg::dot(dS, C));
            EXPECT_LT(max_abs(an_S - fd5(eval_S, frame, A, d, h)), 1e-6)
                << "S^(x) A=" << A << " d=" << d;

            // F^(x) in MO: skeleton Fock at frozen density,
            // F(R; P₀) = Hcore(R) + G(R)[P₀] via coulomb_exchange_fused.
            const auto eval_F = [&](const cpu::Frame& fr) -> RealMatrix {
                const IntegralPack p{fr, BasisSet::STO3G, +1,
                                     /*compute_eri=*/true};
                RealMatrix G = xt::zeros<double>({p.n_basis, p.n_basis});
                p.coulomb_exchange_fused(P, G);
                const RealMatrix F = p.Hcore + G;
                return xt::linalg::dot(xt::transpose(C),
                                       xt::linalg::dot(F, C));
            };
            RealMatrix dFx = xt::view(dF, A, d, xt::all(), xt::all());
            const RealMatrix an_F = xt::linalg::dot(
                xt::transpose(C), xt::linalg::dot(dFx, C));
            EXPECT_LT(max_abs(an_F - fd5(eval_F, frame, A, d, h)), 1e-6)
                << "F^(x) A=" << A << " d=" << d;
        }
    }
}

// ── CphfPack: B^x sum rule, LHS identity, iterative vs dense ───────────

namespace {

struct CphfFixture {
    cpu::scf::RHF rhf;
    IntegralPack pack;
    xt::xtensor<double, 4> eri;
    IntegralDerivPack dpack;
    xt::xtensor<double, 4> dF;
    OrbitalHessian oh;

    explicit CphfFixture(const cpu::Frame& frame, BasisSet basis, int charge)
        : rhf(), pack((run_scf(frame, basis, charge),
                       IntegralPack{frame, basis, charge,
                                    /*compute_eri=*/true})),
          eri(pack.ERI), dpack(pack), dF(dpack.dT + dpack.dV),
          oh(rhf.determinant(), eri) {
        cpu::scf::accumulate_two_electron_fock_derivative(
            dF, pack, rhf.determinant().density_total());
    }

    void run_scf(const cpu::Frame& frame, BasisSet basis, int charge) {
        rhf.basis = basis;
        rhf.charge = charge;
        rhf.tol = 1e-12;
        rhf.conv_tol_grad = 1e-10; // density-level convergence for FD gates
        rhf.run(frame);
        if (!rhf.converged())
            throw std::runtime_error("CphfFixture: SCF did not converge");
    }

    CphfPack make() const {
        return CphfPack{rhf.determinant(), oh, pack, eri, dpack.dS, dF};
    }
};

} // namespace

TEST(Cphf, RhsSumsToZeroUnderRigidTranslation) {
    const auto frame = make_hehp();
    CphfFixture fx{frame, BasisSet::STO3G, +1};
    CphfPack cphf = fx.make();

    const auto& B = cphf.rhs();
    const int nov = static_cast<int>(B.shape()[1]);
    for (int d = 0; d < 3; ++d)
        for (int p = 0; p < nov; ++p)
            EXPECT_NEAR(B(0 * 3 + d, p) + B(1 * 3 + d, p), 0.0, 1e-10)
                << "axis " << d << " element " << p;
}

TEST(Cphf, LhsApplicationIsTheLink08SigmaPlusSinglet) {
    const auto frame = make_hehp();
    CphfFixture fx{frame, BasisSet::Pople631G, +1};
    CphfPack cphf = fx.make();

    RealVector v = xt::zeros<double>({static_cast<std::size_t>(fx.oh.n_ov())});
    for (int p = 0; p < fx.oh.n_ov(); ++p)
        v(p) = 0.1 * (p + 1) * ((p % 2 == 0) ? 1.0 : -1.0);

    // Same OrbitalHessian instance → identical arithmetic path.
    const RealVector lhs = cphf.apply_lhs(v);
    const RealVector sigma = fx.oh.sigma_plus(v, Channel::singlet);
    for (int p = 0; p < fx.oh.n_ov(); ++p)
        EXPECT_LE(std::abs(lhs(p) - sigma(p)), 1e-14);
}

TEST(Cphf, IterativeSolveMatchesDenseDirectSolve) {
    const auto frame = make_hehp();
    CphfFixture fx{frame, BasisSet::Pople631G, +1}; // n_ov = 3 (4 AOs)

    CphfPack cg = fx.make();
    cg.dense_threshold = 0; // force preconditioned CG
    cg.solve();
    EXPECT_GT(cg.iterations(), 0);

    CphfPack dense = fx.make();
    dense.dense_threshold = 1 << 20; // force the dense direct path
    dense.solve();
    EXPECT_EQ(dense.iterations(), 0);

    const auto& Ucg = cg.response();
    const auto& Ud = dense.response();
    for (std::size_t x = 0; x < Ucg.shape()[0]; ++x)
        for (std::size_t p = 0; p < Ucg.shape()[1]; ++p)
            EXPECT_NEAR(Ucg(x, p), Ud(x, p), 1e-10);
}

// ── Density response ∂P/∂x vs FD of the converged density ─────────────
//
// The density response is convention-free, so this pins the CPHF sign and
// factor conventions independently of any Hessian assembly.

TEST(Cphf, DensityResponseMatchesFdOfConvergedDensity) {
    const auto frame = make_hehp();
    CphfFixture fx{frame, BasisSet::STO3G, +1};
    CphfPack cphf = fx.make();
    cphf.solve();

    const double h = 1e-3; // Å
    const auto eval_P = [&](const cpu::Frame& fr) -> RealMatrix {
        cpu::scf::RHF r;
        r.basis = BasisSet::STO3G;
        r.charge = +1;
        r.tol = 1e-13;
        r.conv_tol_grad = 1e-11; // FD of P needs density-level convergence
        r.run(fr);
        return r.determinant().density_total();
    };
    for (int A = 0; A < 2; ++A)
        for (int d = 0; d < 3; ++d) {
            const RealMatrix fd = fd5(eval_P, frame, A, d, h);
            const RealMatrix an = cphf.density_response(3 * A + d);
            EXPECT_LT(max_abs(an - fd), 1e-6) << "A=" << A << " d=" << d;
        }
}

#endif // ATV_HAS_HF
