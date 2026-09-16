/// @file tests/cpu/scf/test_fd_hessian_template.cpp
/// @brief scf-review-04-meanfield-dedup (Item 3): the central-difference
///        nuclear-Hessian skeleton — three byte-identical copies in
///        rks.cpp / uks.cpp / rohf.cpp — is hoisted to a protected CRTP
///        member template
///        MeanFieldMethod<Derived>::central_difference_nuclear_hessian(
///            Frame&, const std::function<GradientResult(Frame&)>& forces_at).
///        This RED-first unit pins that skeleton in ISOLATION with an analytic
///        quadratic mock force field passed as forces_at — no SCF, microseconds.
///
/// PINNED SHAPE — protected CRTP member template (direct, not a free core).
///   The FD template is reached through a test-local subclass of a concrete
///   method (RHF, whose ctor is trivial member init — no SCF machinery, no
///   build()/run()) that re-exposes the protected member via a using-decl.
///   MeanFieldMethod<RHF> is already explicitly instantiated in
///   mean_field_method.cpp, so no template link fork is introduced. A
///   test-local subclass of MeanFieldMethod<TestType> would instead be
///   unlinkable (the member definitions live in the .cpp behind explicit
///   instantiation for RHF/UHF/RKS/UKS/ROHF only); that is why we wrap a real
///   concrete method rather than a synthetic Derived.
///
/// Mock physics: F(R) = -K.(R - R0), forces in eV/Angstrom, R in Angstrom,
/// K in eV/Angstrom^2. This is the force field of E = 1/2 (R-R0)^T K (R-R0),
/// whose analytic Hessian is K. The template differentiates F by central FD
/// (internally in Ha/Bohr) and converts back to eV/Angstrom^2; for a quadratic
/// the recovery is exact and independent of the (frozen) step h, so the
/// template must return the symmetrized 1/2 (K + K^T) to FD/float tolerance
/// 1e-8. Distinct diagonal entries pin the atom-major col = 3A+d index layout;
/// asymmetric off-diagonals pin the 1/2 (K + K^T) symmetrization.

#ifdef ATV_HAS_HF

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/mean_field_method.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

using namespace ATV;

namespace {

/// @brief Test hook: re-expose the protected FD-Hessian template as public by
///        subclassing a concrete method with a trivial ctor.
class HessianProbe : public cpu::scf::RHF {
  public:
    using cpu::scf::MeanFieldMethod<
        cpu::scf::RHF>::central_difference_nuclear_hessian;
};

/// @brief Quadratic mock force field F(R) = -K.(R - R0) [eV/Angstrom].
///        Atom-major flatten: coordinate index = 3*atom + axis.
std::function<cpu::scf::GradientResult(cpu::Frame&)>
mock_forces(xt::xtensor<double, 2> K, std::vector<double> r0) {
    return [K = std::move(K),
            r0 = std::move(r0)](cpu::Frame& f) -> cpu::scf::GradientResult {
        const int natm = f.nlocal();
        const int n3 = 3 * natm;
        auto x = f.span(kAtom_x);
        auto y = f.span(kAtom_y);
        auto z = f.span(kAtom_z);
        std::vector<double> R(static_cast<std::size_t>(n3));
        for (int a = 0; a < natm; ++a) {
            R[static_cast<std::size_t>(3 * a + 0)] = static_cast<double>(x[a]);
            R[static_cast<std::size_t>(3 * a + 1)] = static_cast<double>(y[a]);
            R[static_cast<std::size_t>(3 * a + 2)] = static_cast<double>(z[a]);
        }
        xt::xtensor<double, 2> forces = xt::zeros<double>({natm, 3});
        for (int row = 0; row < n3; ++row) {
            double frow = 0.0;
            for (int col = 0; col < n3; ++col)
                frow += -K(row, col)
                        * (R[static_cast<std::size_t>(col)]
                           - r0[static_cast<std::size_t>(col)]);
            forces(row / 3, row % 3) = frow;
        }
        cpu::scf::GradientResult g;
        g.forces = std::move(forces);
        g.energy = 0.0;
        return g;
    };
}

/// @brief Build an all-H frame at reference geometry r0 (Angstrom, atom-major).
cpu::Frame make_frame(const std::vector<double>& r0) {
    const int natm = static_cast<int>(r0.size() / 3);
    cpu::Frame f(natm);
    for (int a = 0; a < natm; ++a) {
        f.span(kAtom_type)[a] = 1;
        f.span(kAtom_x)[a] =
            static_cast<StorageReal>(r0[static_cast<std::size_t>(3 * a + 0)]);
        f.span(kAtom_y)[a] =
            static_cast<StorageReal>(r0[static_cast<std::size_t>(3 * a + 1)]);
        f.span(kAtom_z)[a] =
            static_cast<StorageReal>(r0[static_cast<std::size_t>(3 * a + 2)]);
    }
    f.box = cpu::Box::cubic(20.0);
    return f;
}

// Happy path + ordering: asymmetric 6x6 K with distinct diagonals -> the
// template returns 1/2 (K + K^T) with atom-major (col = 3A+d) index layout.
TEST(TestFdHessianTemplate, RecoversSymmetrizedQuadraticStiffness) {
    // Exactly-representable reference geometry (Angstrom), atom-major.
    const std::vector<double> r0 = {0.0, 0.5,   1.25,  // atom 0 (x,y,z)
                                    2.0, -0.75, 0.25}; // atom 1 (x,y,z)

    // Hard-coded stiffness K [eV/Angstrom^2]: distinct diagonal (pins the
    // col = 3A+d ordering — a wrong axis-major layout would permute the
    // diagonal) and asymmetric off-diagonals (pins 1/2 (K + K^T)).
    xt::xtensor<double, 2> K = xt::zeros<double>({6, 6});
    const double diag[6] = {10.0, 13.0, 17.0, 23.0, 29.0, 31.0};
    for (int i = 0; i < 6; ++i)
        K(i, i) = diag[i];
    K(0, 3) = 2.0;
    K(3, 0) = 5.0; // symmetric part 3.5
    K(1, 4) = -1.0;
    K(4, 1) = 4.0; // symmetric part 1.5
    K(2, 5) = 7.0;
    K(5, 2) = -3.0; // symmetric part 2.0
    K(0, 1) = 1.0;
    K(1, 0) = -1.0; // symmetric part 0.0

    cpu::Frame frame = make_frame(r0);
    HessianProbe probe;
    cpu::vib::HessianResult H =
        probe.central_difference_nuclear_hessian(frame, mock_forces(K, r0));

    EXPECT_EQ(H.n_atoms, 2);
    ASSERT_EQ(H.hessian.shape()[0], 6u);
    ASSERT_EQ(H.hessian.shape()[1], 6u);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            EXPECT_NEAR(H.hessian(i, j), 0.5 * (K(i, j) + K(j, i)), 1e-8)
                << "mismatch at (" << i << "," << j << ")";
}

// Edge: geometry is restored to the reference r0 exactly at exit.
TEST(TestFdHessianTemplate, RestoresGeometryToReference) {
    const std::vector<double> r0 = {0.0, 0.5, 1.25, 2.0, -0.75, 0.25};
    xt::xtensor<double, 2> K = xt::zeros<double>({6, 6});
    for (int i = 0; i < 6; ++i)
        K(i, i) = 3.0;

    cpu::Frame frame = make_frame(r0);
    HessianProbe probe;
    probe.central_difference_nuclear_hessian(frame, mock_forces(K, r0));

    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    for (int a = 0; a < 2; ++a) {
        EXPECT_DOUBLE_EQ(static_cast<double>(x[a]),
                         r0[static_cast<std::size_t>(3 * a + 0)]);
        EXPECT_DOUBLE_EQ(static_cast<double>(y[a]),
                         r0[static_cast<std::size_t>(3 * a + 1)]);
        EXPECT_DOUBLE_EQ(static_cast<double>(z[a]),
                         r0[static_cast<std::size_t>(3 * a + 2)]);
    }
}

// Edge: single atom (3x3) with a symmetric K -> 1/2 (K + K^T) == K
// (symmetrization is idempotent on symmetric input).
TEST(TestFdHessianTemplate, SingleAtomSymmetricInputIsIdempotent) {
    const std::vector<double> r0 = {0.0, 0.0, 0.0};
    xt::xtensor<double, 2> K = xt::zeros<double>({3, 3});
    K(0, 0) = 11.0;
    K(1, 1) = 19.0;
    K(2, 2) = 23.0;
    K(0, 1) = K(1, 0) = 2.0;
    K(0, 2) = K(2, 0) = -3.0;
    K(1, 2) = K(2, 1) = 5.0;

    cpu::Frame frame = make_frame(r0);
    HessianProbe probe;
    cpu::vib::HessianResult H =
        probe.central_difference_nuclear_hessian(frame, mock_forces(K, r0));

    EXPECT_EQ(H.n_atoms, 1);
    ASSERT_EQ(H.hessian.shape()[0], 3u);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(H.hessian(i, j), K(i, j), 1e-8);
}

} // namespace

#endif
