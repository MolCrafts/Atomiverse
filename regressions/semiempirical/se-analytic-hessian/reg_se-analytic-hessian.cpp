/// @file regressions/semiempirical/se-analytic-hessian/reg_se-analytic-hessian.cpp
/// @brief FD nuclear Hessian of H₂ via `cpu::vib::Hessian` + DFTB2 forces.
///
/// Unblocks the SE-only loop's VIB policy gate: vib::Hessian binds a
/// Semiempirical force kind (DFTB2) and returns a finite 3N×3N Hessian plus
/// one positive stretch frequency for linear H₂ (3N−5 = 1 mode).
///
/// Physics oracle: structural — Hessian symmetric, frequencies finite,
/// stretch ω > 0. Absolute cm⁻¹ not pinned (sketch SK for multi-element;
/// H–H is mio-1-1 but FD step and thermo defaults make a literature twin
/// unnecessary for this foothold).
///
/// Requires ATV_BUILD_VIB=ON and ATV_BUILD_SEMIEMPIRICAL=ON.
/// Exit non-zero on failure. Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
#include "atomiverse/cpu/vibration/hessian.h"

using namespace ATV;

namespace {

cpu::Frame make_h2() {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

} // namespace

int main() {
    auto frame = make_h2();

    cpu::semiempirical::DFTB2 dftb;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
    dftb.with_forces = true;
    dftb.with_stress = false;
    dftb.scc_tol = 1e-10;

    cpu::vib::Hessian hess;
    hess.on(vib::Phase::force).add(std::move(dftb));
    hess.build();
    const cpu::vib::HessianResult res = hess.run(frame);

    if (res.n_atoms != 2) {
        std::printf("FAIL n_atoms=%d\n", res.n_atoms);
        return 1;
    }
    if (res.hessian.shape(0) != 6 || res.hessian.shape(1) != 6) {
        std::printf("FAIL hessian shape\n");
        return 1;
    }

    // Symmetry of the FD Hessian (central difference + symmetrize).
    double max_asym = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            const double a = res.hessian(i, j);
            const double b = res.hessian(j, i);
            if (!std::isfinite(a)) {
                std::printf("FAIL non-finite H(%d,%d)\n", i, j);
                return 1;
            }
            max_asym = std::max(max_asym, std::fabs(a - b));
        }
    }
    if (max_asym > 1e-8) {
        std::printf("FAIL Hessian asymmetry %.3e eV/A^2\n", max_asym);
        return 1;
    }

    if (!res.is_linear) {
        std::printf("FAIL H2 not classified linear\n");
        return 1;
    }
    if (res.frequencies.size() != 1) {
        std::printf("FAIL expected 1 vib mode for linear H2, got %zu\n",
                    res.frequencies.size());
        return 1;
    }
    const double w = res.frequencies(0);
    std::printf(
        "SE Hessian H2 (DFTB2): omega=%.4f cm^-1  n_imag=%d  max|H-H^T|=%.3e\n",
        w, res.n_imaginary, max_asym);
    if (!(w > 100.0) || !std::isfinite(w)) {
        // mio-1-1 H–H stretch is thousands of cm^-1; 100 is a soft floor.
        std::printf("FAIL stretch frequency not positive/physical: %g\n", w);
        return 1;
    }
    if (res.n_imaginary != 0) {
        std::printf("FAIL imaginary modes at near-eq H2\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
