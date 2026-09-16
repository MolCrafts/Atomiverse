/// @file tests/cpu/observer/test_multipole.cpp
/// @brief Unit: MultipoleObserver on synthetic determinants — no SCF run.
///
/// The raw translation law μ′ = μ − q_tot·d is pinned at the kernel level
/// (test_multipole_integrals). Here the observer's convention is pinned:
/// the origin is the center of nuclear charge and is REPORTED, so the
/// published dipole is invariant under a global translation of the system
/// (the origin co-moves) — for charged systems too, which is exactly why
/// the origin must accompany the number.

#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/multipole.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

using namespace ATV;
using cpu::MultipoleObserver;
using cpu::scf::MeanFieldOutputs;
using cpu::scf::SlaterDeterminant;
using cpu::scf::native::Shell;

namespace {

constexpr double kPi = 3.14159265358979323846;

/// One normalized single-primitive s AO at (x0,y0,z0) Å-independent (Bohr).
struct OneAtomSystem {
    std::vector<double> storage; // alpha, coeff
    std::shared_ptr<std::vector<Shell>> shells;
    std::shared_ptr<SlaterDeterminant> det;
    std::shared_ptr<bool> ready;

    explicit OneAtomSystem(const double* center_bohr) {
        const double alpha = 1.0;
        const double norm = std::pow(2.0 * alpha / kPi, 0.75);
        storage = {alpha, norm};
        shells = std::make_shared<std::vector<Shell>>();
        shells->push_back(
            Shell{1,
                  &storage[0],
                  &storage[1],
                  {center_bohr[0], center_bohr[1], center_bohr[2]},
                  0,
                  0});
        cpu::linalg::RealMatrix C =
            xt::ones<double>({std::size_t(1), std::size_t(1)});
        cpu::linalg::RealVector eps = xt::zeros<double>({std::size_t(1)});
        cpu::linalg::RealVector occ = xt::ones<double>({std::size_t(1)});
        det = std::make_shared<SlaterDeterminant>(C, eps, occ, 1);
        ready = std::make_shared<bool>(true);
    }

    MeanFieldOutputs handle() const {
        return {det, shells, ready,
                std::make_shared<const xt::xtensor<double, 2>>()};
    }
};

cpu::Frame one_atom_frame(double x_ang, double y_ang, double z_ang, int z) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = z;
    f.span(kAtom_x)[0] = x_ang;
    f.span(kAtom_y)[0] = y_ang;
    f.span(kAtom_z)[0] = z_ang;
    f.box = cpu::Box::cubic(50.0);
    return f;
}

} // namespace

/// A symmetric one-atom system has zero dipole and zero traceless quadrupole,
/// the origin is the nuclear position, and the meta keys are written.
TEST(MultipoleObserver, OneAtomSymmetric) {
    const double c[3] = {0.0, 0.0, 0.0};
    OneAtomSystem sys{c};
    MultipoleObserver obs{sys.handle()};

    auto frame = one_atom_frame(0.0, 0.0, 0.0, 2); // He, 2 e⁻ in the s AO
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    obs.launch(lc);

    for (int a = 0; a < 3; ++a)
        EXPECT_NEAR(obs.dipole_au()[a], 0.0, 1e-12);
    const auto q = obs.quadrupole_buckingham();
    EXPECT_NEAR(q[0] + q[3] + q[5], 0.0, 1e-12); // traceless
    for (int a = 0; a < 3; ++a)
        EXPECT_NEAR(obs.origin_angstrom()[a], 0.0, 1e-12);
    const auto d = frame.meta.get(kDipole);
    for (int a = 0; a < 3; ++a)
        EXPECT_NEAR(d[a], 0.0, 1e-12);
}

/// Global translation: the origin co-moves with the center of nuclear
/// charge, so the reported dipole is unchanged — even for a net-charged
/// system (He⁺-like: Z=2, one electron). This is the convention the origin
/// report exists for.
TEST(MultipoleObserver, GlobalTranslationInvariance) {
    const double shift_ang = 0.7431;
    const double c0[3] = {0.0, 0.0, 0.0};
    const double c1[3] = {shift_ang * units::kAngstromInBohr, 0.0, 0.0};

    OneAtomSystem s0{c0}, s1{c1};
    MultipoleObserver o0{s0.handle()}, o1{s1.handle()};

    auto f0 = one_atom_frame(0.0, 0.0, 0.0, 2);
    auto f1 = one_atom_frame(shift_ang, 0.0, 0.0, 2);
    cpu::Context ctx;
    cpu::LaunchContext l0{ctx, f0, 0, 0};
    cpu::LaunchContext l1{ctx, f1, 0, 0};
    o0.launch(l0);
    o1.launch(l1);

    for (int a = 0; a < 3; ++a)
        EXPECT_NEAR(o0.dipole_au()[a], o1.dipole_au()[a], 1e-12);
    EXPECT_NEAR(o1.origin_angstrom()[0] - o0.origin_angstrom()[0], shift_ang,
                1e-10);
}

/// Not ready ⇒ no-op (no meta writes, no throw).
TEST(MultipoleObserver, NotReadyIsNoOp) {
    const double c[3] = {0.0, 0.0, 0.0};
    OneAtomSystem sys{c};
    *sys.ready = false;
    MultipoleObserver obs{sys.handle()};
    auto frame = one_atom_frame(0.0, 0.0, 0.0, 1);
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    EXPECT_NO_THROW(obs.launch(lc));
    EXPECT_FALSE(frame.meta.contains(kDipole.key));
}
