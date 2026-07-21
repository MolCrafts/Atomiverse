/// @file cpu/pw/internal/pw_stress.cpp
/// @brief Plane-wave cell stress via central FD of SCF energy under strain.
///
/// v1 delivers a correct stress oracle (same strategy as early PW ionic FD
/// forces). Strain is applied to the cell matrix at fixed fractional
/// coordinates. Nielsen–Martin analytic decomposition is a follow-on.

#include "atomiverse/cpu/pw/internal/pw_stress.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/pw/pw_solver.h"
#include "atomiverse/cpu/pw/qe_units.h"

#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {
namespace {

double scf_energy_ry(const cpu::Frame& frame, const PWConfig& cfg) {
    PWSolver solver{cfg};
    solver.build(frame);
    return solver.run(frame).energy; // Ry
}

/// Apply symmetric strain ε: cell' = cell · (I + ε), positions scaled in
/// Cartesian by the same map (fixed fractional coords).
void apply_strain(cpu::Frame& frame, const Mat3& strain) {
    Mat3 F{};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            F(a, b) = (a == b ? AccReal(1) : AccReal(0)) + strain(a, b);

    Mat3 cell = frame.simbox.cell();
    Mat3 cell_new;
    // cell_new = F * cell  (rows of cell are lattice vectors in Mat3 layout)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            AccReal s = 0;
            for (int k = 0; k < 3; ++k)
                s += F(i, k) * cell(k, j);
            cell_new(i, j) = s;
        }
    frame.simbox.set_cell(cell_new);

    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    const int n = frame.nlocal();
    for (int a = 0; a < n; ++a) {
        AccReal rx = x[a], ry = y[a], rz = z[a];
        x[a] = F(0, 0) * rx + F(0, 1) * ry + F(0, 2) * rz;
        y[a] = F(1, 0) * rx + F(1, 1) * ry + F(1, 2) * rz;
        z[a] = F(2, 0) * rx + F(2, 1) * ry + F(2, 2) * rz;
    }
}

/// Voigt index -> (a,b) with xy=3,xz=4,yz=5 (symmetric).
void voigt_pair(int v, int& a, int& b) {
    static const int A[6] = {0, 1, 2, 0, 0, 1};
    static const int B[6] = {0, 1, 2, 1, 2, 2};
    a = A[v];
    b = B[v];
}

} // namespace

std::array<double, 6> pw_stress_analytic_impl(const cpu::Frame& frame,
                                              const PWConfig& cfg) {
    if (frame.nlocal() <= 0)
        throw std::invalid_argument("pw_stress: frame has no atoms");
    if (frame.simbox.volume() <= 0)
        throw std::invalid_argument("pw_stress: non-positive cell volume");

    // Strain step (dimensionless).
    const double h = 2e-3;
    // Energy in Ry; volume in Å³; convert stress to eV/Å³.
    // σ_ab = (1/Ω) ∂E/∂ε_ab with E in eV, Ω in Å³.
    const double vol = static_cast<double>(frame.simbox.volume()); // Å³
    const double ry_to_ev = qe_units::rydberg_to_ev;

    std::array<double, 6> stress{};
    for (int v = 0; v < 6; ++v) {
        int a = 0, b = 0;
        voigt_pair(v, a, b);
        const double fac = (a == b) ? 1.0 : 0.5; // off-diag: ε and ε^T share

        Mat3 ep = Mat3{};
        Mat3 em = Mat3{};
        ep(a, b) += h * fac;
        ep(b, a) += h * fac;
        em(a, b) -= h * fac;
        em(b, a) -= h * fac;
        // diagonal: fac=1 so ep(a,a)=h; off-diag: each of ab,ba gets h/2 so
        // symmetric strain has ε_ab = h/2 + h/2 = h for the free component...
        // For Voigt engineering strain, use unit probe on free components:
        // Reset to cleaner definition: ε_ab = ε_ba = h for off-diagonal Voigt.
        ep = Mat3{};
        em = Mat3{};
        if (a == b) {
            ep(a, a) = h;
            em(a, a) = -h;
        } else {
            ep(a, b) = ep(b, a) = h;
            em(a, b) = em(b, a) = -h;
        }

        cpu::Frame fp = frame;
        cpu::Frame fm = frame;
        apply_strain(fp, ep);
        apply_strain(fm, em);
        const double ep_ry = scf_energy_ry(fp, cfg);
        const double em_ry = scf_energy_ry(fm, cfg);
        // dE/dε with E in eV; for off-diagonal, the free parameter is ε_ab=ε_ba=ε,
        // and ∂/∂ε of the energy uses 2h step in that parameter.
        const double dE_dev = (ep_ry - em_ry) * ry_to_ev / (2.0 * h);
        stress[static_cast<std::size_t>(v)] = dE_dev / vol;
    }
    return stress;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
