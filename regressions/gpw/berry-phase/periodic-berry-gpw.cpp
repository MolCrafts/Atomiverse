/// @file regressions/gpw/periodic-berry-gpw/reg_periodic-berry-gpw.cpp
/// @brief END-TO-END: KSV Berry-phase polarization of bulk Si (GPW) vs
///        symmetry pins.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + ATV_BUILD_PW.
/// Pins (cited literature, not snapshots of our own output):
///  1. Centrosymmetric diamond Si: every total p_i sits on the
///     symmetry-allowed lattice {0, Q/2} (Spaldin, J. Solid State Chem.
///     195, 2 (2012)).
///  2. Born effective charge of an elemental crystal is exactly zero
///     (acoustic sum rule; Pick, Cohen & Martin, Phys. Rev. B 1, 910
///     (1970)): a sublattice displacement leaves the total polarization
///     unchanged — the electronic Berry phase must cancel the bare ionic
///     Z_val * delta term.
/// Exit non-zero on failure with got / want / tol.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/berry_polarization.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molpy.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_periodic_berry_gpw {

using namespace ATV;
using namespace ATV::cpu::gpw;

namespace {

cpu::Frame si_bct(double a_ang, double delta_frac_z) {
    const double at = a_ang / std::sqrt(2.0);
    // Diamond BCT fractional sites; atom 1 optionally displaced along c.
    double frac[4][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.5 + delta_frac_z},
        {0.5, 0.0, 0.25},
        {0.0, 0.5, 0.75},
    };
    // Keep fractional z in [0,1) for the displaced site.
    frac[1][2] -= std::floor(frac[1][2]);
    cpu::Frame frame(4);
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_type)[i] = 14;
        frame.span(kAtom_x)[i] = frac[i][0] * at;
        frame.span(kAtom_y)[i] = frac[i][1] * at;
        frame.span(kAtom_z)[i] = frac[i][2] * a_ang;
    }
    frame.box = cpu::Box::orthorhombic(at, at, a_ang);
    return frame;
}

} // namespace

int run() {
    const auto start = std::chrono::steady_clock::now();
    constexpr double kA = 5.43;        // lattice constant [Angstrom]
    constexpr double kCutoffRy = 80.0; // density grid [Ry]
    constexpr double kDelta = 0.02;    // sublattice displacement [frac of c]
    constexpr double kZval = 4.0;      // Si valence (GTH-q4)

    // Pins / tolerances (REFERENCE.md).
    constexpr double kIonTol = 1e-9; // ionic bookkeeping is exact
    constexpr double kSymTol = 0.08; // distance to {0, Q/2} lattice [e]
    constexpr double kBornTol = 0.5; // |Z*| upper bound [e]

    cpu::Frame frame0 = si_bct(kA, 0.0);

    GpwDft method{{Kind{molrs::Element::Si,
                        GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
                        GthPotential(molrs::Element::Si, "GTH-PBE-q4")}},
                  kCutoffRy,
                  XcKind::PADE};
    method.n_occ = 8; // 4 Si × 4 valence e⁻ / 2
    method.max_iter = 60;
    method.tol = 1e-7;
    method.eps_default = 1e-8;
    method.build(frame0);

    BerryPolarization berry;
    // One transverse string is enough for centrosymmetric Si (phase is
    // quantized); J=4 matches the PW twin discretization along the string.
    berry.k_perp = 1;
    berry.k_string = 4;

    // ── Ideal centrosymmetric diamond Si (BCT) ─────────────────────────
    const auto r0 = berry.compute(method, frame0);
    if (!r0.scf_converged) {
        std::printf("FAIL: ideal-Si SCF did not converge (E=%g Ha)\n",
                    r0.energy_ha);
        return 1;
    }
    std::printf("ideal Si: E=%.6f Ha\n", r0.energy_ha);

    // Ionic bookkeeping: 4 atoms × Z=4
    //   p_ion_x = 4*(0+0.5+0.5+0) = 4
    //   p_ion_y = 4*(0+0.5+0+0.5) = 4
    //   p_ion_z = 4*(0+0.5+0.25+0.75) = 6
    const double ion_want[3] = {4.0, 4.0, 6.0};
    for (int d = 0; d < 3; ++d) {
        std::printf("  dir %d: phase=%+.6f  p_el=%+.6f  p_ion=%+.6f  p=%+.6f\n",
                    d, r0.phase_el[d], r0.p_el_frac[d], r0.p_ion_frac[d],
                    r0.p_frac[d]);
        if (std::fabs(r0.p_ion_frac[d] - ion_want[d]) > kIonTol) {
            std::printf("FAIL: p_ion[%d]=%.12f want %.1f tol %g\n", d,
                        r0.p_ion_frac[d], ion_want[d], kIonTol);
            return 1;
        }
        if (!r0.computed[static_cast<std::size_t>(d)]) {
            std::printf("FAIL: direction %d not computed\n", d);
            return 1;
        }
        // Centrosymmetric pin: p on the {0, Q/2} lattice (Spaldin 2012)
        // — distance to the nearest multiple of 0.5 below kSymTol.
        // Quantum for closed shell is 2; half-quantum lattice uses 0.5.
        const double dist = std::fabs(wrap_centered(r0.p_frac[d], 0.5));
        if (dist > kSymTol) {
            std::printf(
                "FAIL: centrosymmetry: p[%d]=%.6f is %.4f from the {0,1/2} "
                "lattice (tol %.3f)\n",
                d, r0.p_frac[d], dist, kSymTol);
            return 1;
        }
    }

    // ── Displaced sublattice along c: Born charge must vanish (ASR) ────
    BerryPolarization berry_z;
    berry_z.k_perp = 1;
    berry_z.k_string = 4;
    berry_z.directions = {false, false, true};

    cpu::Frame frame1 = si_bct(kA, kDelta);
    // Positions changed → run re-reads the geometry from the Frame (build
    // only validates structure once).
    const auto r1 = berry_z.compute(method, frame1);
    if (!r1.scf_converged) {
        std::printf("FAIL: displaced-Si SCF did not converge (E=%g Ha)\n",
                    r1.energy_ha);
        return 1;
    }

    // Ionic: Delta p_ion,z = Z_val * delta for the one moved atom.
    const double d_ion = r1.p_ion_frac[2] - r0.p_ion_frac[2];
    if (std::fabs(d_ion - kZval * kDelta) > kIonTol) {
        std::printf("FAIL: Delta p_ion,z=%.12f want %.12f tol %g\n", d_ion,
                    kZval * kDelta, kIonTol);
        return 1;
    }

    // Born-charge pin: Z* = (Delta p_z)/delta = 0 for an elemental crystal.
    const double dp = wrap_centered(r1.p_frac[2] - r0.p_frac[2], 2.0);
    const double z_born = dp / kDelta;
    std::printf("displaced Si: E=%.6f Ha  p_z %.6f -> %.6f  Delta p_z=%+.6f  "
                "Z*=%+.4f\n",
                r1.energy_ha, r0.p_frac[2], r1.p_frac[2], dp, z_born);
    if (std::fabs(z_born) > kBornTol) {
        std::printf("FAIL: |Z*|=%.4f exceeds %.2f (ASR: Z*(elemental)=0)\n",
                    std::fabs(z_born), kBornTol);
        return 1;
    }

    std::printf("PASS: periodic-berry-gpw (centrosymmetric lattice pin + "
                "Z*(Si)=0 ASR pin)\n");
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::printf("whole section got=%.6f want<=60 seconds\n", wall);
    if (wall > 60.0)
        return 1;
    return 0;
}

} // namespace atv_reg_gpw_periodic_berry_gpw

ATV_REGRESSION("gpw/berry-phase", "periodic-berry-gpw",
               atv_reg_gpw_periodic_berry_gpw::run);
