/// @file regressions/semiempirical/h2o_dftb2_mio/reg_h2o_dftb2_mio.cpp
/// @brief H2O SCC-DFTB2 with transcribed mio-1-1 SK — energy + force gate.
///
/// Domain: Elstner et al., Phys. Rev. B 58, 7260 (1998). Static H/C/N/O
/// mio-1-1 tables (`dftb_mio_sk_tables.h`). Absolute energy pin vs DFTB+ 24.1
/// Matches DFTB+ to ~5e-8 eV; see REFERENCE.md.
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
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.96;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = -0.24;
    f.span(kAtom_y)[2] = 0.93;
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

} // namespace

int main() {
    auto frame = make_water();

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-10;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("DFTB2 H2O: E=%.8f eV\n", e);
    if (!std::isfinite(e)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    // DFTB+ mio-1-1 SCC at this geometry, at the precision DFTB+ prints:
    //   Total energy = −4.0776668457 Ha × 27.211386245988 eV/Ha
    // (tools/tb_oracle/inputs/dftbplus/h2o_mio_scc/). Identical on 24.1 and on
    // 25.1 (commit a23bfb2), so the pin is not version-bound.
    constexpr double kDftbPlusEV = -110.9589675208;
    constexpr double kAbsTolEV = 1e-4;
    if (std::fabs(e - kDftbPlusEV) > kAbsTolEV) {
        std::printf("FAIL |E-DFTB+| = %.3e > tol %.1e eV (got %.10f)\n",
                    std::fabs(e - kDftbPlusEV), kAbsTolEV, e);
        return 1;
    }
    std::printf("  vs DFTB+: pin %.10f  |dE|=%.3e eV (tol %.1e)\n", kDftbPlusEV,
                std::fabs(e - kDftbPlusEV), kAbsTolEV);

    double f2 = 0;
    for (int i = 0; i < 3; ++i) {
        const double fxi = static_cast<double>(frame.span(kAtom_fx)[i]);
        const double fyi = static_cast<double>(frame.span(kAtom_fy)[i]);
        const double fzi = static_cast<double>(frame.span(kAtom_fz)[i]);
        std::printf("  F[%d]=(%.6f, %.6f, %.6f) eV/A\n", i, fxi, fyi, fzi);
        f2 += fxi * fxi + fyi * fyi + fzi * fzi;
        if (!std::isfinite(fxi) || !std::isfinite(fyi) || !std::isfinite(fzi)) {
            std::printf("FAIL non-finite force\n");
            return 1;
        }
    }
    if (f2 < 1e-8) {
        std::printf("FAIL all forces ~0 at off-eq geometry\n");
        return 1;
    }

    // Central FD on O_x vs analytic (loose window for product gate).
    const double h = 1e-4;
    auto energy_only = [](cpu::Frame g) {
        g.meta.get(kEnergy) = 0;
        for (int i = 0; i < 3; ++i)
            g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
        cpu::Evaluate ev;
        cpu::semiempirical::DFTB2 d;
        // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
        // geometry, so say so rather than letting a cutoff hide the images.
        d.boundary = cpu::semiempirical::DftbBoundary::cluster;
        d.with_forces = false;
        d.scc_tol = 1e-10;
        ev.on(evaluate::Phase::evaluate).add(std::move(d));
        ev.build();
        ev.run(g);
        return g.meta.get(kEnergy);
    };
    auto fp = make_water();
    fp.span(kAtom_x)[0] += static_cast<StorageReal>(h);
    auto fm = make_water();
    fm.span(kAtom_x)[0] -= static_cast<StorageReal>(h);
    const double fd = -(energy_only(fp) - energy_only(fm)) / (2.0 * h);
    const double fx0 = static_cast<double>(frame.span(kAtom_fx)[0]);
    std::printf("  F_O_x analytic=%.6f  fd=%.6f eV/A\n", fx0, fd);
    // A converged SCC-DFTB2 gradient is stationary, so fixed-Δq analytic is
    // FD-exact to the central-difference truncation error. Measured residual
    // 4.5e-08 eV/Å (2026-07-28); the old 1.5 eV/Å window was masking the
    // γ-gradient sign bug and the double-counting error.
    constexpr double kForceTol = 1e-5;
    if (std::fabs(fx0 - fd) > kForceTol) {
        std::printf("FAIL force vs FD (tol %.1f eV/A)\n", kForceTol);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
