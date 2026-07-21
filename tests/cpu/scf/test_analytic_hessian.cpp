/// @file tests/cpu/scf/test_analytic_hessian.cpp
/// @brief RHF::hessian micro first-gate (hf-capability-gaps-13-cphf):
///        H₂/STO-3G analytic Hessian vs the central finite difference of the
///        ANALYTIC forces — both in-house, so the CPHF response coupling and
///        every skeleton piece are pinned before any external oracle. Plus
///        the structural invariants (symmetry, translational sum rules) and
///        the no-virtuals throw (He/STO-3G).
///
/// FD uses the 5-point central stencil at h = 2e-3 Å over `rhf.forces()`
/// (H = −∂F/∂R), each displacement a fresh milliseconds H₂/STO-3G SCF at
/// tol 1e-13.

#ifdef ATV_HAS_HF

#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/vib/hessian_result.h"

using namespace ATV;
using cpu::scf::BasisSet;

namespace {

/// H₂ near equilibrium, along a skew axis so all Cartesian blocks engage.
cpu::Frame make_h2() {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.30;
    f.span(kAtom_y)[1] = 0.40;
    f.span(kAtom_z)[1] = 0.50;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

cpu::Frame displaced(const cpu::Frame& frame, int coord, double dx) {
    cpu::Frame f = frame;
    const int A = coord / 3, d = coord % 3;
    if (d == 0)
        f.span(kAtom_x)[A] += static_cast<StorageReal>(dx);
    else if (d == 1)
        f.span(kAtom_y)[A] += static_cast<StorageReal>(dx);
    else
        f.span(kAtom_z)[A] += static_cast<StorageReal>(dx);
    return f;
}

/// Flattened analytic forces [eV/Å] at a geometry (fresh tight SCF).
xt::xtensor<double, 1> forces_at(const cpu::Frame& frame) {
    cpu::scf::RHF rhf;
    rhf.basis = BasisSet::STO3G;
    rhf.tol = 1e-13;
    cpu::Frame f = frame;
    const auto g = rhf.forces(f);
    const int natm = static_cast<int>(g.forces.shape()[0]);
    xt::xtensor<double, 1> out =
        xt::zeros<double>({static_cast<std::size_t>(3 * natm)});
    for (int A = 0; A < natm; ++A)
        for (int d = 0; d < 3; ++d)
            out(3 * A + d) = g.forces(A, d);
    return out;
}

} // namespace

TEST(AnalyticHessian, H2MatchesFdOfAnalyticForces) {
    const auto frame = make_h2();

    cpu::scf::RHF rhf;
    rhf.basis = BasisSet::STO3G;
    rhf.tol = 1e-13;
    cpu::Frame f = frame;
    const cpu::vib::HessianResult res = rhf.hessian(f);
    ASSERT_EQ(res.n_atoms, 2);
    ASSERT_EQ(res.hessian.shape()[0], 6u);
    ASSERT_EQ(res.units, "eV/Angstrom^2");

    // H(x, y) = ∂²E/∂x∂y = −∂F_y/∂x — 5-point central FD per coordinate.
    const double h = 2e-3; // Å
    for (int x = 0; x < 6; ++x) {
        const auto fp1 = forces_at(displaced(frame, x, h));
        const auto fm1 = forces_at(displaced(frame, x, -h));
        const auto fp2 = forces_at(displaced(frame, x, 2.0 * h));
        const auto fm2 = forces_at(displaced(frame, x, -2.0 * h));
        for (int y = 0; y < 6; ++y) {
            const double dF =
                (8.0 * (fp1(y) - fm1(y)) - (fp2(y) - fm2(y))) / (12.0 * h);
            EXPECT_NEAR(res.hessian(x, y), -dF, 1e-6)
                << "element (" << x << ", " << y << ")";
        }
    }
}

TEST(AnalyticHessian, H2SymmetricAndTranslationClean) {
    auto frame = make_h2();
    cpu::scf::RHF rhf;
    rhf.basis = BasisSet::STO3G;
    rhf.tol = 1e-13;
    const cpu::vib::HessianResult res = rhf.hessian(frame);

    for (int x = 0; x < 6; ++x)
        for (int y = 0; y < 6; ++y)
            EXPECT_NEAR(res.hessian(x, y), res.hessian(y, x), 1e-10);

    // Rigid translation leaves E unchanged: Σ_B H[(A,i),(B,i')δ...] — the
    // row sum over same-axis columns of every atom vanishes.
    for (int x = 0; x < 6; ++x)
        for (int d = 0; d < 3; ++d) {
            double sum = 0.0;
            for (int B = 0; B < 2; ++B)
                sum += res.hessian(x, 3 * B + d);
            EXPECT_NEAR(sum, 0.0, 1e-6) << "row " << x << " axis " << d;
        }
}

TEST(AnalyticHessian, HeliumNoVirtualsThrows) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 2; // He/STO-3G: one AO, zero virtuals
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.simbox = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = BasisSet::STO3G;
    EXPECT_THROW(rhf.hessian(f), std::invalid_argument);
}

#endif // ATV_HAS_HF
