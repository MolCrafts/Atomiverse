/// @file regressions/semiempirical/dftb3-analytic-forces/reg_dftb3-analytic-forces.cpp
/// @brief H2O SCC-DFTB3 analytic nuclear forces vs re-SCC central FD.
///
/// Domain: Gaus, Cui, Elstner, J. Chem. Theory Comput. 7, 931 (2011)
/// (DOI 10.1021/ct100684s) third-order SCC-DFTB; second-order backbone
/// Elstner et al., Phys. Rev. B 58, 7260 (1998). Force-module boundary:
/// DFTB+ 24.1 (Hourahine et al., J. Chem. Phys. 152, 124101 (2020)).
/// In-tree mio-style SK tables — not absolute DFTB.org mio-1-1 numbers.
///
/// Exit non-zero on failure. Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
#include "atomiverse/cpu/semiempirical/dftb3.h"

using namespace ATV;

namespace {

// ATV twin pins after published 3ob U^d (dftb-data-00-honesty). Physics
// oracle is the FD identity below (not DFTB+ absolute forces). Twin snapshot,
// not DFTB+ absolute. Supersedes sketch-U^d -110.972004227931 /
// 2.651279044727 / -0.171473609620.
constexpr double kPinEnergyEV = -110.961689188230;
constexpr double kPinFoxEVA = 2.662706377721;
constexpr double kPinFh1yEVA = -0.175866697553;
constexpr double kPinTol = 1e-6;

cpu::Frame make_water() {
    // Off-equilibrium water (Å) so |F| != 0 and the Mulliken Δq != 0.
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 1.05;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = -0.26;
    f.span(kAtom_y)[2] = 1.0;
    f.span(kAtom_z)[2] = 0.0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

/// Converged DFTB3 energy [eV] at a displaced geometry — a FULL new SCC solve,
/// so the finite difference differentiates the functional the solver actually
/// converged (not a frozen-density surrogate).
double energy_only(cpu::Frame g) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < 3; ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Evaluate ev;
    cpu::semiempirical::DFTB3 d;
    d.sk_set = cpu::semiempirical::DftbSkSet::MioSketch;
    d.hubbard_du_ha = cpu::semiempirical::basis_data::k3obHubbardDerivsHa;
    d.with_forces = false;
    d.scc_tol = 1e-10;
    ev.on(evaluate::Phase::evaluate).add(std::move(d));
    ev.build();
    ev.run(g);
    return g.meta.get(kEnergy);
}

/// Central FD of `energy_only` on one Cartesian component [eV/Å].
double fd_force(int atom, int k, double h) {
    auto shifted = [&](double s) {
        cpu::Frame g = make_water();
        StorageReal* col = (k == 0)   ? g.span(kAtom_x)
                           : (k == 1) ? g.span(kAtom_y)
                                      : g.span(kAtom_z);
        col[atom] += static_cast<StorageReal>(s);
        return energy_only(g);
    };
    return -(shifted(h) - shifted(-h)) / (2.0 * h);
}

} // namespace

int main() {
    auto frame = make_water();

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB3 dftb3;
    dftb3.sk_set = cpu::semiempirical::DftbSkSet::MioSketch;
    dftb3.hubbard_du_ha = cpu::semiempirical::basis_data::k3obHubbardDerivsHa;
    dftb3.with_forces = true;
    dftb3.scc_tol = 1e-10;
    eval.on(evaluate::Phase::evaluate).add_ref(dftb3);
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("DFTB3 analytic F H2O: E=%.12f eV  E3=%.6e eV\n", e,
                dftb3.last_e3());
    if (!std::isfinite(e) || std::fabs(e) < 1e-3) {
        std::printf("FAIL energy not finite/bound\n");
        return 1;
    }
    if (!(dftb3.last_scc_residual() < 1e-10)) {
        std::printf("FAIL SCC residual %.3e above scc_tol\n",
                    dftb3.last_scc_residual());
        return 1;
    }

    double f2 = 0;
    double net[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        const double fxi = static_cast<double>(frame.span(kAtom_fx)[i]);
        const double fyi = static_cast<double>(frame.span(kAtom_fy)[i]);
        const double fzi = static_cast<double>(frame.span(kAtom_fz)[i]);
        std::printf("  F[%d]=(%.12f, %.12f, %.12f) eV/A\n", i, fxi, fyi, fzi);
        f2 += fxi * fxi + fyi * fyi + fzi * fzi;
        net[0] += fxi;
        net[1] += fyi;
        net[2] += fzi;
        if (!std::isfinite(fxi) || !std::isfinite(fyi) || !std::isfinite(fzi)) {
            std::printf("FAIL non-finite force\n");
            return 1;
        }
    }
    if (f2 < 1e-8) {
        std::printf("FAIL all forces ~0 at off-eq geometry\n");
        return 1;
    }
    // Every gradient term is a pair force (F_B = -F_A): translational
    // invariance is exact by construction, not a tolerance.
    for (int d = 0; d < 3; ++d) {
        if (std::fabs(net[d]) > 1e-10) {
            std::printf("FAIL net force[%d]=%.3e eV/A\n", d, net[d]);
            return 1;
        }
    }

    // A converged SCC-DFTB3 gradient is stationary (H = dE/dP exactly, with the
    // Euler-correct third-order potential), so the closed-form force is FD-exact
    // to the central-difference truncation error.
    constexpr double h = 1e-4;   // A
    constexpr double tol = 3e-5; // eV/A (lattice-walk residual vs re-SCC FD)
    const double fx0 = static_cast<double>(frame.span(kAtom_fx)[0]);
    const double fy1 = static_cast<double>(frame.span(kAtom_fy)[1]);
    {
        const double fd = fd_force(0, 0, h);
        std::printf("  F_O_x analytic=%.12f  fd=%.12f  |d|=%.3e eV/A\n", fx0,
                    fd, std::fabs(fx0 - fd));
        if (std::fabs(fx0 - fd) > tol) {
            std::printf("FAIL F_O_x vs re-SCC FD (tol %g eV/A)\n", tol);
            return 1;
        }
    }
    {
        const double fd = fd_force(1, 1, h);
        std::printf("  F_H1_y analytic=%.12f  fd=%.12f  |d|=%.3e eV/A\n", fy1,
                    fd, std::fabs(fy1 - fd));
        if (std::fabs(fy1 - fd) > tol) {
            std::printf("FAIL F_H1_y vs re-SCC FD (tol %g eV/A)\n", tol);
            return 1;
        }
    }

    // Absolute residual floor vs re-SCC FD. A missing/mis-signed term leaves
    // an h-independent offset of ~0.01–1 eV/Å; the lattice-sum + Elstner-γ
    // product path sits at ~1e-5 eV/Å (below the absolute FD tol). Strict
    // O(h²) shrink is not diagnostic once the floor dominates truncation.
    {
        const double err_coarse = std::fabs(fx0 - fd_force(0, 0, 8e-4));
        const double err_fine = std::fabs(fx0 - fd_force(0, 0, 2e-4));
        std::printf(
            "  F_O_x FD residual: h=8e-4 -> %.3e, h=2e-4 -> %.3e eV/A\n",
            err_coarse, err_fine);
        if (err_fine > tol || err_coarse > tol) {
            std::printf("FAIL FD residual above absolute tol %g eV/A\n", tol);
            return 1;
        }
    }

    // Third order must actually move the force, not only the energy.
    auto frame2 = make_water();
    cpu::Evaluate eval2;
    cpu::semiempirical::DFTB2 d2;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    d2.boundary = cpu::semiempirical::DftbBoundary::cluster;
    d2.with_forces = true;
    d2.scc_tol = 1e-10;
    eval2.on(evaluate::Phase::evaluate).add(std::move(d2));
    eval2.build();
    eval2.run(frame2);
    const double dfx = fx0 - static_cast<double>(frame2.span(kAtom_fx)[0]);
    std::printf("  F_O_x DFTB3-DFTB2 = %.6e eV/A\n", dfx);
    if (std::fabs(dfx) < 1e-6) {
        std::printf(
            "FAIL DFTB3 force identical to DFTB2 (third order inert)\n");
        return 1;
    }

    // Provisional ATV-twin determinism pins (see REFERENCE.md Gaps).
    if (std::fabs(e - kPinEnergyEV) > kPinTol
        || std::fabs(fx0 - kPinFoxEVA) > kPinTol
        || std::fabs(fy1 - kPinFh1yEVA) > kPinTol) {
        std::printf("FAIL pin: E=%.12f (exp %.12f) F_O_x=%.12f (exp %.12f) "
                    "F_H1_y=%.12f (exp %.12f) tol %g\n",
                    e, kPinEnergyEV, fx0, kPinFoxEVA, fy1, kPinFh1yEVA,
                    kPinTol);
        return 1;
    }
    std::printf("PIN match (tol %g)\n", kPinTol);

    std::printf("PASS\n");
    return 0;
}
