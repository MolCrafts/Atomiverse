/// @file tests/cpu/md/test_pe_fd_force.cpp
/// @brief Finite-difference consistency check for each bonded force kernel.
///
/// For a perturbed C4 backbone (so every kernel sees a non-trivial geometry),
/// compare the analytic force returned by cpu::BondHarmonic, cpu::AngleHarmonic and
/// cpu::DihedralMultiHarmonic against a central-difference gradient of the same
/// kernel's energy. Mismatch pinpoints which kernel's force formula is bad.

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/improper_harmonic.h"

using namespace ATV;

namespace {

cpu::Frame make_perturbed_butane() {
    cpu::Frame frame(4);
    // Off-trans geometry so cos(phi) != -1, sin(phi) != 0.
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = 1.5;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.0;
    frame.span(ATV::kAtom_x)[2] = 2.1;
    frame.span(ATV::kAtom_y)[2] = 1.3;
    frame.span(ATV::kAtom_z)[2] = 0.0;
    frame.span(ATV::kAtom_x)[3] = 3.5;
    frame.span(ATV::kAtom_y)[3] = 1.1;
    frame.span(ATV::kAtom_z)[3] = 0.4;
    for (int i = 0; i < 4; ++i) {
        frame.span(ATV::kAtom_mass)[i] = 14.0;
        frame.span(ATV::kAtom_type)[i] = 0;
    }
    frame.simbox = ATV::cpu::Box::cubic(50.0);

    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int k = 0; k < 3; ++k) {
        frame.add_bond(k, k + 1, 0);
    }
    for (int k = 0; k < 2; ++k) {
        frame.add_angle(k, k + 1, k + 2, 0);
    }
    frame.add_dihedral(0, 1, 2, 3, 0);
    frame.add_improper(0, 1, 2, 3, 0);
    frame.build_special_list();
    return frame;
}

// Return the analytic force on atom `atom_idx`, axis `d` (0=x,1=y,2=z),
// after running `launch` on a freshly-zeroed-force copy of `frame`.
double analytic_force(cpu::Frame frame, std::function<void(cpu::Frame&)> launch,
                      int atom_idx, int axis) {
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(ATV::kAtom_fx)[i] = 0;
        frame.span(ATV::kAtom_fy)[i] = 0;
        frame.span(ATV::kAtom_fz)[i] = 0;
    }
    frame.meta.get(ATV::kEnergy) = 0;
    launch(frame);
    if (axis == 0)
        return frame.span(ATV::kAtom_fx)[atom_idx];
    if (axis == 1)
        return frame.span(ATV::kAtom_fy)[atom_idx];
    return frame.span(ATV::kAtom_fz)[atom_idx];
}

// Central-difference estimate of dE/dx[atom, axis].
double fd_gradient(cpu::Frame frame, std::function<void(cpu::Frame&)> launch,
                   int atom_idx, int axis, double h) {
    auto eval = [&](cpu::Frame& f) {
        for (int i = 0; i < f.nlocal(); ++i) {
            f.span(ATV::kAtom_fx)[i] = 0;
            f.span(ATV::kAtom_fy)[i] = 0;
            f.span(ATV::kAtom_fz)[i] = 0;
        }
        f.meta.get(ATV::kEnergy) = 0;
        launch(f);
        return f.meta.get(ATV::kEnergy);
    };
    auto pos = (axis == 0)   ? frame.span(ATV::kAtom_x)
               : (axis == 1) ? frame.span(ATV::kAtom_y)
                             : frame.span(ATV::kAtom_z);
    const double orig = pos[atom_idx];
    pos[atom_idx] = orig + h;
    const double e_plus = eval(frame);
    pos[atom_idx] = orig - h;
    const double e_minus = eval(frame);
    pos[atom_idx] = orig;
    return (e_plus - e_minus) / (2.0 * h);
}

void check_kernel(const std::string& label,
                  std::function<void(cpu::Frame&)> launch, double tol) {
    auto frame = make_perturbed_butane();
    const double h = 1.0e-4;
    for (int a = 0; a < frame.nlocal(); ++a) {
        for (int d = 0; d < 3; ++d) {
            const double f_analytic = analytic_force(frame, launch, a, d);
            const double grad = fd_gradient(frame, launch, a, d, h);
            // Analytic force = -grad; compare.
            EXPECT_NEAR(f_analytic, -grad, tol)
                << label << " atom " << a << " axis " << d
                << "  analytic=" << f_analytic << "  -grad=" << -grad;
        }
    }
}

} // namespace

TEST(PeFdForce, BondHarmonic) {
    check_kernel(
        "cpu::BondHarmonic",
        [](cpu::Frame& f) {
            cpu::Context ctx;
            cpu::BuildContext bctx{ctx, f};
            cpu::LaunchContext lctx{ctx, f, 0, 0};
            cpu::BondHarmonic c({1.0}, {1.5});
            c.build(bctx);
            c.launch(lctx);
            c.destroy({ctx, f});
        },
        1e-6);
}

TEST(PeFdForce, AngleHarmonic) {
    check_kernel(
        "cpu::AngleHarmonic",
        [](cpu::Frame& f) {
            cpu::Context ctx;
            cpu::BuildContext bctx{ctx, f};
            cpu::LaunchContext lctx{ctx, f, 0, 0};
            cpu::AngleHarmonic c({1.0}, {2.0}); // K=1, theta0=2 rad
            c.build(bctx);
            c.launch(lctx);
            c.destroy({ctx, f});
        },
        1e-6);
}

TEST(PeFdForce, DihedralMultiHarmonic) {
    check_kernel(
        "cpu::DihedralMultiHarmonic",
        [](cpu::Frame& f) {
            cpu::Context ctx;
            cpu::BuildContext bctx{ctx, f};
            cpu::LaunchContext lctx{ctx, f, 0, 0};
            cpu::DihedralMultiHarmonic c({{0.1, -0.2, 0.05, 0.3, -0.1}});
            c.build(bctx);
            c.launch(lctx);
            c.destroy({ctx, f});
        },
        1e-6);
}

TEST(PeFdForce, DihedralCharmm) {
    check_kernel(
        "cpu::DihedralCharmm",
        [](cpu::Frame& f) {
            cpu::Context ctx;
            cpu::BuildContext bctx{ctx, f};
            cpu::LaunchContext lctx{ctx, f, 0, 0};
            // K=0.2 eV, n=2, delta=0 — non-zero gradient on the perturbed butane.
            cpu::DihedralCharmm c({0.2}, {2}, {0.0});
            c.build(bctx);
            c.launch(lctx);
            c.destroy({ctx, f});
        },
        1e-6);
}

TEST(PeFdForce, DihedralCharmmPhasePi) {
    check_kernel(
        "cpu::DihedralCharmm phase=pi",
        [](cpu::Frame& f) {
            cpu::Context ctx;
            cpu::BuildContext bctx{ctx, f};
            cpu::LaunchContext lctx{ctx, f, 0, 0};
            cpu::DihedralCharmm c({0.15}, {1}, {3.141592653589793});
            c.build(bctx);
            c.launch(lctx);
            c.destroy({ctx, f});
        },
        1e-6);
}

TEST(PeFdForce, ImproperHarmonic) {
    check_kernel(
        "cpu::ImproperHarmonic",
        [](cpu::Frame& f) {
            cpu::Context ctx;
            cpu::BuildContext bctx{ctx, f};
            cpu::LaunchContext lctx{ctx, f, 0, 0};
            cpu::ImproperHarmonic c({0.3}, {0.0});
            c.build(bctx);
            c.launch(lctx);
            c.destroy({ctx, f});
        },
        1e-6);
}
