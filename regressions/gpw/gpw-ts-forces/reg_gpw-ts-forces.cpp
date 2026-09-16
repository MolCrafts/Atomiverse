/// @file regressions/gpw/gpw-ts-forces/reg_gpw-ts-forces.cpp
/// @brief Water-dimer free-atom TS E_disp + force dual-gate + product wiring.
///
/// Out-of-library consumer: public GPW headers only. Geometry and pins in
/// REFERENCE.md (libMBD / pure-RS offline; no live oracle in ctest).

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwAtom;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::gpw_forces_from_scf;
    using cpu::gpw::ts_pbe;
    using cpu::gpw::TsResult;
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
    constexpr double kEIsolated = -5.9764872295816794e-04; // Ha libMBD
    constexpr double kEPeriodic = -6.099630773386013e-04;  // Ha pure-RS
    constexpr double kETol = 1.0e-8;                       // Ha
    constexpr double kFTol = 1.0e-6;                       // Ha/Bohr
    constexpr double kSumFTol = 1.0e-8;                    // Ha/Bohr
    constexpr double kWiringTol = 1.0e-7;                  // Ha/Bohr

    // ── Isolated free-atom TS energy + force dual-gate ────────────────────
    const double cell0[3] = {0.0, 0.0, 0.0};
    const TsResult ts_iso = ts_pbe(Z, pos, cell0, /*periodic=*/false);
    if (!std::isfinite(ts_iso.energy)
        || std::abs(ts_iso.energy - kEIsolated) > kETol) {
        std::fprintf(stderr,
                     "FAIL: E_disp TS isolated got=%.14e want=%.14e tol=%.1e Ha\n",
                     ts_iso.energy, kEIsolated, kETol);
        return 1;
    }
    if (ts_iso.forces.size() != 6) {
        std::fprintf(stderr, "FAIL: forces size %zu (want 6)\n",
                     ts_iso.forces.size());
        return 1;
    }

    const double h = 1.0e-5;
    double max_err = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    auto pos_fd = pos;
    for (std::size_t i = 0; i < 6; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos_fd[i][d];
            pos_fd[i][d] = save + h;
            const double ep = ts_pbe(Z, pos_fd, cell0, false).energy;
            pos_fd[i][d] = save - h;
            const double em = ts_pbe(Z, pos_fd, cell0, false).energy;
            pos_fd[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - ts_iso.forces[i][d]));
            sum[d] += ts_iso.forces[i][d];
        }
    }
    if (max_err > kFTol) {
        std::fprintf(stderr,
                     "FAIL: max |F_product-F_fd|=%.3e > tol=%.1e Ha/Bohr\n",
                     max_err, kFTol);
        return 1;
    }
    const double sum_norm =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    if (sum_norm > kSumFTol) {
        std::fprintf(stderr, "FAIL: |sum F|=%.3e > tol=%.1e Ha/Bohr\n",
                     sum_norm, kSumFTol);
        return 1;
    }

    // ── Periodic pure-RS E_disp pin ───────────────────────────────────────
    const double L = 12.0 * kA2B;
    const double cell[3] = {L, L, L};
    constexpr double kRcut = 50.0 * kA2B;
    const TsResult ts_per = ts_pbe(Z, pos, cell, /*periodic=*/true, kRcut);
    if (!std::isfinite(ts_per.energy)
        || std::abs(ts_per.energy - kEPeriodic) > kETol) {
        std::fprintf(stderr,
                     "FAIL: E_disp TS periodic got=%.14e want=%.14e\n",
                     ts_per.energy, kEPeriodic);
        return 1;
    }

    // ── Product path: GPW SCF with ts=true + force wiring ─────────────────
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

    auto make_cfg = [&](bool with_ts) {
        GpwDft cfg;
        cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = L;
        cfg.cutoff_ry = 200.0;
        cfg.n_occ = 8;
        cfg.max_iter = 80;
        cfg.tol = 1e-6;
        cfg.mix = 0.4;
        cfg.xc = XcKind::PBE;
        cfg.nonperiodic = false;
        cfg.local_pp_on_grid = true;
        cfg.ts = with_ts;
        cfg.build(atoms);
        return cfg;
    };

    GpwDft gpw = make_cfg(/*with_ts=*/true);
    const auto& res = gpw.run(atoms);
    if (!std::isfinite(res.e_disp)
        || std::abs(res.e_disp - kEPeriodic) > kETol) {
        std::fprintf(stderr,
                     "FAIL: GpwDft e_disp got=%.14e want=%.14e\n", res.e_disp,
                     kEPeriodic);
        return 1;
    }
    if (!std::isfinite(res.e_total)) {
        std::fprintf(stderr, "FAIL: e_total non-finite\n");
        return 1;
    }
    if (!res.converged) {
        std::fprintf(stderr, "FAIL: GPW SCF did not converge (iters=%d)\n",
                     res.iters);
        return 1;
    }

    // Wiring: F(ts) − F(plain) on the same density matches standalone F_TS.
    const auto F_ts = gpw_forces_from_scf(atoms, gpw, res);
    GpwDft plain = make_cfg(/*with_ts=*/false);
    const auto F_plain = gpw_forces_from_scf(atoms, plain, res);

    double max_wiring = 0.0;
    for (int a = 0; a < 6; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double delta =
                F_ts[static_cast<std::size_t>(a)][k]
                - F_plain[static_cast<std::size_t>(a)][k]
                - ts_per.forces[static_cast<std::size_t>(a)][k];
            max_wiring = std::max(max_wiring, std::abs(delta));
        }
    }
    if (max_wiring > kWiringTol) {
        std::fprintf(stderr,
                     "FAIL: max|F(ts)-F(plain)-F_geom|=%.3e > %.1e Ha/Bohr\n",
                     max_wiring, kWiringTol);
        return 1;
    }

    std::printf(
        "PASS: TS free-atom water dimer E_iso=%.14e Ha (libMBD); "
        "E_per=%.14e Ha (pure-RS); max|F-F_fd|=%.2e Ha/Bohr; |sum F|=%.2e; "
        "max|wiring|=%.2e; F_O0=(%.6e, %.6e, %.6e) Ha/Bohr; "
        "e_total=%.8f Ha (iters=%d)\n",
        ts_iso.energy, ts_per.energy, max_err, sum_norm, max_wiring,
        ts_iso.forces[0][0], ts_iso.forces[0][1], ts_iso.forces[0][2],
        res.e_total, res.iters);
    return 0;
}
