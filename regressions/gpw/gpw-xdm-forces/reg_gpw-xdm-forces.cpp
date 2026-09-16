/// @file regressions/gpw/gpw-xdm-forces/reg_gpw-xdm-forces.cpp
/// @brief Water-dimer free-atom XDM E_disp + analytic force dual-gate.
///
/// Out-of-library consumer: public GPW headers + internal XDM kernel (same
/// pattern as gpw-mbd-forces). Geometry and pins in REFERENCE.md (dual
/// pure-Python offline; no live oracle in ctest).

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/xdm_dispersion.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwAtom;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::xdm_pbe;
    using cpu::gpw::XdmResult;
    using cpu::gpw::XcKind;

    constexpr double kA2B = 1.0 / 0.52917720859;
    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000},  // O
        {0.758602, 0.000000, 0.504284},  // H
        {0.758602, 0.000000, -0.504284}, // H
        {3.000000, 0.000000, 0.000000},  // O
        {3.758602, 0.000000, 0.504284},  // H
        {3.758602, 0.000000, -0.504284}, // H
    };
    const int Zarr[6] = {8, 1, 1, 8, 1, 1};
    std::vector<int> Z(Zarr, Zarr + 6);
    std::vector<std::array<double, 3>> pos(6);
    for (int i = 0; i < 6; ++i)
        for (int d = 0; d < 3; ++d)
            pos[static_cast<std::size_t>(i)][d] = ang[i][d] * kA2B;

    // Offline pins (REFERENCE.md).
    constexpr double kEIsoRef = -2.1020435998516220e-03; // Ha
    constexpr double kEPerRef = -2.1148157612879029e-03; // Ha
    constexpr double kETol = 1.0e-8;                     // Ha
    constexpr double kFTol = 1.0e-6;                     // Ha/Bohr
    constexpr double kSumFTol = 1.0e-8;                  // Ha/Bohr
    constexpr double kPinFTol = 1.0e-10;                 // Ha/Bohr vs dual-Python
    // Isolated free-atom force sample (dual pure-Python ≡ ATV tables).
    constexpr double kFIsoPin[6][3] = {
        {1.6875829936003448e-04, 0.0, 0.0},
        {6.9736686381575340e-05, 0.0, -1.3172272218329521e-05},
        {6.9736686381575340e-05, 0.0, 1.3172272218329521e-05},
        {-1.1378113807647229e-04, 0.0, 0.0},
        {-9.7225267023356434e-05, 0.0, -1.5448899346549854e-05},
        {-9.7225267023356434e-05, 0.0, 1.5448899346549854e-05},
    };

    const double cell0[3] = {0.0, 0.0, 0.0};
    const XdmResult xdm_iso = xdm_pbe(Z, pos, cell0, /*periodic=*/false);
    if (!std::isfinite(xdm_iso.energy)
        || std::abs(xdm_iso.energy - kEIsoRef) > kETol) {
        std::fprintf(stderr,
                     "FAIL: E_disp isolated XDM got=%.14e want=%.14e tol=%.1e\n",
                     xdm_iso.energy, kEIsoRef, kETol);
        return 1;
    }

    constexpr double kRcut = 50.0 * kA2B;
    const double L = 12.0 * kA2B;
    const double cellp[3] = {L, L, L};
    const XdmResult xdm_per =
        xdm_pbe(Z, pos, cellp, /*periodic=*/true, kRcut);
    if (!std::isfinite(xdm_per.energy)
        || std::abs(xdm_per.energy - kEPerRef) > kETol) {
        std::fprintf(stderr,
                     "FAIL: E_disp periodic XDM got=%.14e want=%.14e tol=%.1e\n",
                     xdm_per.energy, kEPerRef, kETol);
        return 1;
    }

    if (xdm_iso.forces.size() != 6 || xdm_per.forces.size() != 6) {
        std::fprintf(stderr, "FAIL: forces size (iso=%zu per=%zu, want 6)\n",
                     xdm_iso.forces.size(), xdm_per.forces.size());
        return 1;
    }

    // Hard pin: isolated force components vs dual-Python sample.
    double max_pin = 0.0;
    for (std::size_t i = 0; i < 6; ++i)
        for (int d = 0; d < 3; ++d)
            max_pin = std::max(
                max_pin,
                std::abs(xdm_iso.forces[i][d] - kFIsoPin[i][d]));
    if (max_pin > kPinFTol) {
        std::fprintf(stderr,
                     "FAIL: max|F_iso-pin|=%.3e > tol=%.1e Ha/Bohr\n", max_pin,
                     kPinFTol);
        return 1;
    }

    // Dual-gate: independent central FD of energy vs product forces.
    const double h = 1.0e-5;
    auto dual_gate = [&](const XdmResult& xdm, bool periodic,
                         const double cell[3], const char* tag) -> int {
        double max_err = 0.0;
        double sum[3] = {0.0, 0.0, 0.0};
        auto pos_fd = pos;
        for (std::size_t i = 0; i < 6; ++i) {
            for (int d = 0; d < 3; ++d) {
                const double save = pos_fd[i][d];
                pos_fd[i][d] = save + h;
                const double ep =
                    xdm_pbe(Z, pos_fd, cell, periodic, kRcut).energy;
                pos_fd[i][d] = save - h;
                const double em =
                    xdm_pbe(Z, pos_fd, cell, periodic, kRcut).energy;
                pos_fd[i][d] = save;
                const double fd = -(ep - em) / (2.0 * h);
                max_err =
                    std::max(max_err, std::abs(fd - xdm.forces[i][d]));
                sum[d] += xdm.forces[i][d];
            }
        }
        if (max_err > kFTol) {
            std::fprintf(stderr,
                         "FAIL: %s max |F_product-F_fd|=%.3e > tol=%.1e "
                         "Ha/Bohr\n",
                         tag, max_err, kFTol);
            return 1;
        }
        const double sum_norm =
            std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
        if (sum_norm > kSumFTol) {
            std::fprintf(stderr, "FAIL: %s |sum F|=%.3e > tol=%.1e Ha/Bohr\n",
                         tag, sum_norm, kSumFTol);
            return 1;
        }
        std::printf("%s: max|F-F_fd|=%.2e  |sum F|=%.2e Ha/Bohr\n", tag,
                    max_err, sum_norm);
        return 0;
    };

    if (dual_gate(xdm_iso, false, cell0, "iso") != 0)
        return 1;
    if (dual_gate(xdm_per, true, cellp, "per") != 0)
        return 1;

    // Product path: short GPW SCF with xdm=true reports same periodic E_disp.
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    std::vector<GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double p[3] = {pos[static_cast<std::size_t>(i)][0],
                             pos[static_cast<std::size_t>(i)][1],
                             pos[static_cast<std::size_t>(i)][2]};
        if (Zarr[i] == 8)
            atoms.emplace_back(p, bO, pO, 8);
        else
            atoms.emplace_back(p, bH, pH, 1);
    }

    GpwDft gpw;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 200.0;
    gpw.n_occ = 8;
    gpw.max_iter = 8;
    gpw.tol = 1e-6;
    gpw.mix = 0.4;
    gpw.xc = XcKind::PBE;
    gpw.nonperiodic = false;
    gpw.xdm = true;

    gpw.build(atoms);
    const auto& res = gpw.run(atoms);
    if (!std::isfinite(res.e_disp)
        || std::abs(res.e_disp - kEPerRef) > kETol) {
        std::fprintf(stderr,
                     "FAIL: GpwDft e_disp got=%.14e want=%.14e\n", res.e_disp,
                     kEPerRef);
        return 1;
    }
    if (!std::isfinite(res.e_total)) {
        std::fprintf(stderr, "FAIL: e_total non-finite\n");
        return 1;
    }

    std::printf(
        "PASS: free-atom XDM water dimer E_iso=%.14e E_per=%.14e Ha "
        "(dual-Python pins); max|F_iso-pin|=%.2e; "
        "F_O0=(%.6e, %.6e, %.6e) Ha/Bohr; e_total=%.8f Ha (iters=%d)\n",
        xdm_iso.energy, xdm_per.energy, max_pin, xdm_iso.forces[0][0],
        xdm_iso.forces[0][1], xdm_iso.forces[0][2], res.e_total, res.iters);
    return 0;
}
