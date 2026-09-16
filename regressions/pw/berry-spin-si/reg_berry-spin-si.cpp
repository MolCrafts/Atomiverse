/// @file regressions/pw/berry-spin-si/reg_berry-spin-si.cpp
/// @brief END-TO-END: collinear-spin Berry-phase polarization of bulk Si.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.
/// Pins (cited literature, not snapshots of our own output):
///  1. Centrosymmetric diamond Si: every total p_i sits on the
///     symmetry-allowed lattice {0, Q/2} (Spaldin, J. Solid State Chem.
///     195, 2 (2012)) — same basis-independent pin as closed-shell Berry.
///  2. Born effective charge of an elemental crystal is exactly zero
///     (Pick, Cohen & Martin, Phys. Rev. B 1, 910 (1970)).
/// Collinear path: spin_polarized=true, multiplicity=1 (N↑=N↓=4).
/// Exit non-zero on failure with got / want / tol.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/berry_polarization.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

cpu::Frame make_si_diamond(double a_ang, double delta_frac_a3) {
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
    const double h = a_ang / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    frame.box.set_cell(cell);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = a_ang / 4.0 + delta_frac_a3 * cell(2, 0);
    frame.span(kAtom_y)[1] = a_ang / 4.0 + delta_frac_a3 * cell(2, 1);
    frame.span(kAtom_z)[1] = a_ang / 4.0 + delta_frac_a3 * cell(2, 2);
    return frame;
}

} // namespace

int main() {
    constexpr double kA = 5.43;
    constexpr double kEcutRy = 15.0;
    constexpr int kMeshScf = 2;
    constexpr double kDelta = 0.02;
    constexpr double kZval = 4.0;
    // nspin=2 polarization quantum is e (wrap interval (-0.5, 0.5]).
    constexpr double kQuantum = 1.0;

    constexpr double kIonTol = 1e-9;
    constexpr double kSymTol = 0.05;
    constexpr double kBornTol = 0.5;

    PW method;
    method.upf_path = ATV_PW_UPF_PATH;
    method.ecut_ry = kEcutRy;
    method.k_mesh = kMeshScf;
    method.tol = 1e-6;
    method.max_iter = 50;
    method.xc = cpu::scf::XCType::LSDA;
    method.spin_polarized = true;
    method.multiplicity = 1; // N↑ = N↓ = 4 for 8 valence electrons

    BerryPolarization berry;
    berry.k_perp = 2;
    berry.k_string = 4;

    const auto ideal = make_si_diamond(kA, 0.0);
    const auto r0 = berry.compute(method, ideal);
    if (!r0.scf_converged) {
        std::printf("FAIL: ideal-Si collinear SCF did not converge (E=%g Ry)\n",
                    r0.energy_ry);
        return 1;
    }
    std::printf("ideal Si (collinear): E=%.6f Ry\n", r0.energy_ry);
    for (int d = 0; d < 3; ++d) {
        std::printf("  dir %d: phase=%+.6f  p_el=%+.6f  p_ion=%+.6f  p=%+.6f\n",
                    d, r0.phase_el[d], r0.p_el_frac[d], r0.p_ion_frac[d],
                    r0.p_frac[d]);
        if (std::fabs(r0.p_ion_frac[d] - 1.0) > kIonTol) {
            std::printf("FAIL: p_ion[%d]=%.12f want 1.0 tol %g\n", d,
                        r0.p_ion_frac[d], kIonTol);
            return 1;
        }
        const double dist = std::fabs(wrap_centered(r0.p_frac[d], 0.5));
        if (dist > kSymTol) {
            std::printf(
                "FAIL: centrosymmetry: p[%d]=%.6f is %.4f from the {0,1/2} "
                "lattice (tol %.3f)\n",
                d, r0.p_frac[d], dist, kSymTol);
            return 1;
        }
    }

    BerryPolarization berry_z;
    berry_z.k_perp = 2;
    berry_z.k_string = 4;
    berry_z.directions = {false, false, true};
    const auto disp = make_si_diamond(kA, kDelta);
    const auto r1 = berry_z.compute(method, disp);
    if (!r1.scf_converged) {
        std::printf(
            "FAIL: displaced-Si collinear SCF did not converge (E=%g Ry)\n",
            r1.energy_ry);
        return 1;
    }

    const double d_ion = r1.p_ion_frac[2] - r0.p_ion_frac[2];
    if (std::fabs(d_ion - kZval * kDelta) > kIonTol) {
        std::printf("FAIL: Delta p_ion,3=%.12f want %.12f tol %g\n", d_ion,
                    kZval * kDelta, kIonTol);
        return 1;
    }

    const double dp = wrap_centered(r1.p_frac[2] - r0.p_frac[2], kQuantum);
    const double z_born = dp / kDelta;
    std::printf("displaced Si (collinear): E=%.6f Ry  p3 %.6f -> %.6f  "
                "Delta p3=%+.6f  Z*=%+.4f\n",
                r1.energy_ry, r0.p_frac[2], r1.p_frac[2], dp, z_born);
    if (std::fabs(z_born) > kBornTol) {
        std::printf("FAIL: |Z*|=%.4f exceeds %.2f (ASR: Z*(elemental)=0)\n",
                    std::fabs(z_born), kBornTol);
        return 1;
    }

    std::printf("PASS: berry-spin-si (collinear KSV strings; centrosymmetric "
                "lattice pin + Z*(Si)=0 ASR pin)\n");
    return 0;
}
