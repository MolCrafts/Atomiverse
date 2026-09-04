/// @file regressions/semiempirical/dftb2-analytic-forces-threeob/reg_dftb2-analytic-forces-threeob.cpp
/// @brief H2O SCC-DFTB2 ThreeOb analytic forces vs re-SCC central FD.
///
/// Domain: Elstner / DFTB+ force module; 3ob-3-1 SK (Gaus 2013). Closed-form
/// HF+Pulay+γ+rep on DftbSkSet::ThreeOb (not re-SCC FD).
///
/// Exit non-zero on failure. Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

cpu::Frame make_water() {
    // Off-equilibrium water (Å) so |F| ≠ 0.
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

double energy_only(cpu::Frame g) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < 3; ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Evaluate ev;
    cpu::semiempirical::DFTB2 d;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    d.boundary = cpu::semiempirical::DftbBoundary::cluster;
    d.sk_set = cpu::semiempirical::DftbSkSet::ThreeOb;
    d.with_forces = false;
    d.scc_tol = 1e-10;
    d.max_scc = 80;
    ev.on(evaluate::Phase::evaluate).add(std::move(d));
    ev.build();
    ev.run(g);
    return g.meta.get(kEnergy);
}

} // namespace

int main() {
    auto frame = make_water();

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
    dftb.sk_set = cpu::semiempirical::DftbSkSet::ThreeOb;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-10;
    dftb.max_scc = 80;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("DFTB2 ThreeOb analytic F H2O: E=%.8f eV\n", e);
    if (!std::isfinite(e) || std::fabs(e) < 1e-3) {
        std::printf("FAIL energy not finite/bound\n");
        return 1;
    }

    double f2 = 0;
    double net[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        const double fxi = static_cast<double>(frame.span(kAtom_fx)[i]);
        const double fyi = static_cast<double>(frame.span(kAtom_fy)[i]);
        const double fzi = static_cast<double>(frame.span(kAtom_fz)[i]);
        std::printf("  F[%d]=(%.8f, %.8f, %.8f) eV/A\n", i, fxi, fyi, fzi);
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
    for (int d = 0; d < 3; ++d) {
        if (std::fabs(net[d]) > 1e-6) {
            std::printf("FAIL net force[%d]=%.3e eV/A\n", d, net[d]);
            return 1;
        }
    }

    constexpr double h = 1e-4;
    // Fixed-Δq HF vs re-SCC FD residual (charge response).
    // A converged SCC-DFTB2 gradient is stationary ⇒ fixed-Δq analytic is
    // FD-exact to the central-difference truncation error. Measured residual
    // 4.4e-07 / 2.0e-09 eV/Å (2026-07-28); the old 1.0 eV/Å "charge-response"
    // gate was masking the γ-gradient sign bug.
    constexpr double tol = 1e-5; // eV/Å

    // F_O,x
    {
        auto fp = make_water();
        fp.span(kAtom_x)[0] += static_cast<StorageReal>(h);
        auto fm = make_water();
        fm.span(kAtom_x)[0] -= static_cast<StorageReal>(h);
        const double fd = -(energy_only(fp) - energy_only(fm)) / (2.0 * h);
        const double fx0 = static_cast<double>(frame.span(kAtom_fx)[0]);
        std::printf("  F_O_x analytic=%.8f  fd=%.8f  |d|=%.3e eV/A\n", fx0, fd,
                    std::fabs(fx0 - fd));
        if (std::fabs(fx0 - fd) > tol) {
            std::printf("FAIL F_O_x vs FD (tol %.3f eV/A)\n", tol);
            return 1;
        }
    }
    // F_H1,y
    {
        auto fp = make_water();
        fp.span(kAtom_y)[1] += static_cast<StorageReal>(h);
        auto fm = make_water();
        fm.span(kAtom_y)[1] -= static_cast<StorageReal>(h);
        const double fd = -(energy_only(fp) - energy_only(fm)) / (2.0 * h);
        const double fy1 = static_cast<double>(frame.span(kAtom_fy)[1]);
        std::printf("  F_H1_y analytic=%.8f  fd=%.8f  |d|=%.3e eV/A\n", fy1, fd,
                    std::fabs(fy1 - fd));
        if (std::fabs(fy1 - fd) > tol) {
            std::printf("FAIL F_H1_y vs FD (tol %.3f eV/A)\n", tol);
            return 1;
        }
    }

    std::printf("PASS\n");
    return 0;
}
