/// @file regressions/gpw/gpw-hirshfeld-forces/reg_gpw-hirshfeld-forces.cpp
/// @brief Water-dimer Hirshfeld-scaled TS/MBD forces + product wiring dual-gate.
///
/// Found rot: energy folded post-SCF Hirshfeld volume ratios into TS/MBD, but
/// `gpw_forces` / `gpw_stress` called free-atom kernels with empty volumes.
/// Gates (offline pins in REFERENCE.md; no live oracle in ctest):
///   A. Isolated MBD@rsSCS energy with fixed molecular volumes (dual kernel).
///   B. Force dual-gate: product \(F\) vs independent central FD of the same
///      fixed-\(v\) energy for both TS and MBD (\(h=10^{-5}\) Bohr).
///   C. Product wiring: \(F(\mathrm{ts}+\mathrm{hirsh})-F(\mathrm{plain})\)
///      equals standalone `ts_pbe(…, volume_ratios)` from the SCF payload.
///   D. Free-atom forces differ from Hirshfeld-scaled (ratios move F).
///   E. Stress fold: fixed-\(P\) Lagrangian at the SCF state rebuilds
///      \(E_{\mathrm{disp}}\) with the same volume ratios (Hirshfeld σ shift).
///
/// CP2K 2026.1 has no stock TS/MBD pair potential — oracle is libMBD /
/// literature dual (same policy as `gpw-mbd-hirshfeld` / `gpw-ts-forces`).

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV;
    using clock = std::chrono::steady_clock;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwAtom;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::gpw_forces_from_scf;
    using cpu::gpw::gpw_stress;
    using cpu::gpw::mbd_rsscs_pbe;
    using cpu::gpw::MbdResult;
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

    // Representative Hirshfeld volumes (same order of magnitude as product
    // water-dimer ratios from reg_gpw-mbd-hirshfeld REFERENCE.md).
    const std::vector<double> kVol = {0.7403, 0.6529, 0.6529,
                                      0.7153, 0.5784, 0.5784};

    // Offline dual pin: isolated MBD@rsSCS with fixed kVol (product kernel =
    // independent dual; measured on this tree — see REFERENCE.md).
    constexpr double kEMbdHirsh = -1.1479717148086300e-03; // Ha (dual @ kVol)
    constexpr double kETol = 1.0e-8;                       // Ha
    constexpr double kFTol = 1.0e-6;                       // Ha/Bohr
    constexpr double kSumFTol = 1.0e-8;                    // Ha/Bohr
    constexpr double kWiringTol = 1.0e-7;                  // Ha/Bohr
    constexpr double kWallBudget = 60.0;                   // s

    const auto t0 = clock::now();
    int nfail = 0;
    auto fail = [&](const char* msg) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++nfail;
    };

    // ── A/B: isolated volume-scaled MBD energy + force dual-gate ──────────
    const double cell0[3] = {0.0, 0.0, 0.0};
    const MbdResult mbd =
        mbd_rsscs_pbe(Z, pos, cell0, /*periodic=*/false, 0.83, 6.0, kVol);
    if (!std::isfinite(mbd.energy)
        || std::abs(mbd.energy - kEMbdHirsh) > kETol) {
        std::fprintf(stderr,
                     "FAIL: E_disp MBD+Hirsh volumes got=%.14e want=%.14e\n",
                     mbd.energy, kEMbdHirsh);
        ++nfail;
    }
    if (mbd.forces.size() != 6) {
        fail("MBD forces size != 6");
    } else {
        const double h = 1.0e-5;
        double max_err = 0.0;
        double sum[3] = {0.0, 0.0, 0.0};
        auto pos_fd = pos;
        for (std::size_t i = 0; i < 6; ++i) {
            for (int d = 0; d < 3; ++d) {
                const double save = pos_fd[i][d];
                pos_fd[i][d] = save + h;
                const double ep =
                    mbd_rsscs_pbe(Z, pos_fd, cell0, false, 0.83, 6.0, kVol)
                        .energy;
                pos_fd[i][d] = save - h;
                const double em =
                    mbd_rsscs_pbe(Z, pos_fd, cell0, false, 0.83, 6.0, kVol)
                        .energy;
                pos_fd[i][d] = save;
                const double fd = -(ep - em) / (2.0 * h);
                max_err =
                    std::max(max_err, std::abs(fd - mbd.forces[i][d]));
                sum[d] += mbd.forces[i][d];
            }
        }
        const double sum_norm =
            std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
        if (max_err > kFTol) {
            std::fprintf(stderr,
                         "FAIL: MBD max|F-F_fd|=%.3e > tol=%.1e Ha/Bohr\n",
                         max_err, kFTol);
            ++nfail;
        }
        if (sum_norm > kSumFTol) {
            std::fprintf(stderr, "FAIL: MBD |sum F|=%.3e > tol\n", sum_norm);
            ++nfail;
        }
        std::printf("  MBD+Hirsh isolated: E=%.14e  max|F-F_fd|=%.2e  "
                    "|sum F|=%.2e\n",
                    mbd.energy, max_err, sum_norm);
    }

    // ── B: isolated volume-scaled TS force dual-gate ──────────────────────
    const TsResult ts =
        ts_pbe(Z, pos, cell0, /*periodic=*/false, 94.2, 0.94, 20.0, kVol);
    if (!std::isfinite(ts.energy) || !(ts.energy < 0.0)) {
        fail("TS Hirsh energy non-finite or non-attractive");
    } else {
        const double h = 1.0e-5;
        double max_err = 0.0;
        auto pos_fd = pos;
        for (std::size_t i = 0; i < 6; ++i) {
            for (int d = 0; d < 3; ++d) {
                const double save = pos_fd[i][d];
                pos_fd[i][d] = save + h;
                const double ep =
                    ts_pbe(Z, pos_fd, cell0, false, 94.2, 0.94, 20.0, kVol)
                        .energy;
                pos_fd[i][d] = save - h;
                const double em =
                    ts_pbe(Z, pos_fd, cell0, false, 94.2, 0.94, 20.0, kVol)
                        .energy;
                pos_fd[i][d] = save;
                const double fd = -(ep - em) / (2.0 * h);
                max_err = std::max(max_err, std::abs(fd - ts.forces[i][d]));
            }
        }
        if (max_err > kFTol) {
            std::fprintf(stderr,
                         "FAIL: TS max|F-F_fd|=%.3e > tol=%.1e Ha/Bohr\n",
                         max_err, kFTol);
            ++nfail;
        }
        // Free-atom TS forces must differ (Hirshfeld moves the gradient).
        const TsResult ts_free =
            ts_pbe(Z, pos, cell0, /*periodic=*/false, 94.2, 0.94, 20.0);
        double max_df = 0.0;
        for (std::size_t i = 0; i < 6; ++i)
            for (int d = 0; d < 3; ++d)
                max_df = std::max(
                    max_df, std::abs(ts.forces[i][d] - ts_free.forces[i][d]));
        if (max_df < 1e-8) {
            std::fprintf(stderr,
                         "FAIL: Hirshfeld volumes must move TS forces "
                         "(max|dF|=%.3e)\n",
                         max_df);
            ++nfail;
        }
        std::printf("  TS+Hirsh isolated: E=%.14e  max|F-F_fd|=%.2e  "
                    "max|F_h-F_free|=%.2e\n",
                    ts.energy, max_err, max_df);
    }

    // ── C/D: product GPW SCF (ts+hirshfeld) force wiring ──────────────────
    // One SCF only: stress gate reuses the same payload via at-state overload.
    // CUTOFF 120 Ry (sibling TS-stress class) keeps wall under 60 s.
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

    const double L = 12.0 * kA2B;
    auto make_cfg = [&](bool with_ts, bool with_hirsh) {
        GpwDft cfg;
        cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = L;
        cfg.cutoff_ry = 120.0;
        cfg.n_occ = 8;
        cfg.max_iter = 80;
        cfg.tol = 1e-6;
        cfg.mix = 0.4;
        cfg.xc = XcKind::PBE;
        cfg.nonperiodic = false;
        cfg.local_pp_on_grid = true;
        cfg.ts = with_ts;
        cfg.hirshfeld = with_hirsh;
        cfg.build(atoms);
        return cfg;
    };

    GpwDft gpw = make_cfg(/*ts=*/true, /*hirsh=*/true);
    const auto& res = gpw.run(atoms);
    if (!res.converged) {
        fail("GPW SCF (ts+hirshfeld) did not converge");
    } else if (res.volume_ratios.size() != 6) {
        fail("volume_ratios size != 6");
    } else if (!std::isfinite(res.e_disp) || !(res.e_disp < 0.0)) {
        fail("product e_disp non-finite or non-attractive");
    } else {
        // Energy must match volume-scaled TS at SCF ratios (periodic pure-RS).
        constexpr double kRcut = 50.0 * kA2B;
        const double cell[3] = {L, L, L};
        const TsResult geom_h =
            ts_pbe(Z, pos, cell, /*periodic=*/true, kRcut, 0.94, 20.0,
                   res.volume_ratios);
        const TsResult geom_free =
            ts_pbe(Z, pos, cell, /*periodic=*/true, kRcut);
        if (std::abs(res.e_disp - geom_h.energy) > kETol) {
            std::fprintf(stderr,
                         "FAIL: product e_disp=%.14e vs geom_h=%.14e\n",
                         res.e_disp, geom_h.energy);
            ++nfail;
        }
        if (std::abs(res.e_disp - geom_free.energy) < 1e-10) {
            fail("product e_disp matches free-atom (hirshfeld not applied)");
        }

        const auto F_h = gpw_forces_from_scf(atoms, gpw, res);
        GpwDft plain = make_cfg(false, false);
        // Reuse the same electronic state (plain cfg only changes disp flags).
        const auto F_plain = gpw_forces_from_scf(atoms, plain, res);
        if (F_h.size() != 6 || F_plain.size() != 6) {
            fail("force size mismatch");
        } else {
            double max_wire = 0.0;
            double max_wire_free = 0.0;
            for (std::size_t a = 0; a < 6; ++a) {
                for (int d = 0; d < 3; ++d) {
                    const double dF = F_h[a][d] - F_plain[a][d];
                    max_wire = std::max(
                        max_wire, std::abs(dF - geom_h.forces[a][d]));
                    max_wire_free = std::max(
                        max_wire_free,
                        std::abs(dF - geom_free.forces[a][d]));
                }
            }
            if (max_wire > kWiringTol) {
                std::fprintf(stderr,
                             "FAIL: product TS+Hirsh wiring max|d|=%.3e "
                             "> tol=%.1e\n",
                             max_wire, kWiringTol);
                ++nfail;
            }
            if (max_wire_free < 1e-9) {
                fail("forces still match free-atom TS (rot not fixed)");
            }
            std::printf("  product TS+Hirsh: e_disp=%.14e  max|wire_h|=%.2e  "
                        "max|wire_free|=%.2e  v0=%.4f\n",
                        res.e_disp, max_wire, max_wire_free,
                        res.volume_ratios[0]);
        }

        // ── E: stress fold at fixed SCF state (no second full SCF) ────────
        // σ(ts+hirsh) vs σ(ts free-atom) on the same P — electronic cancels;
        // only E_disp rebuild under strain differs via volume ratios.
        GpwDft cfg_h = make_cfg(true, true);
        GpwDft cfg_free = make_cfg(true, false);
        // One diagonal axis is enough to prove the fold moves σ.
        const std::array<bool, 3> axes = {true, false, false};
        const auto sig_h = gpw_stress(atoms, cfg_h, res, /*h=*/2e-3, axes);
        const auto sig_free =
            gpw_stress(atoms, cfg_free, res, /*h=*/2e-3, axes);
        const double dsig = std::abs(sig_h[0] - sig_free[0]);
        if (!(dsig > 1e-12)) {
            std::fprintf(stderr,
                         "FAIL: Hirshfeld must move analytic stress "
                         "(max|dσ|=%.3e)\n",
                         dsig);
            ++nfail;
        }
        // Masked axes and off-diagonals exactly zero.
        for (int i = 1; i < 6; ++i) {
            if (std::abs(sig_h[static_cast<std::size_t>(i)]) > 1e-12) {
                fail("non-zero stress on masked/off-diagonal component");
                break;
            }
        }
        std::printf("  stress fold (at-state): |σ_xx(h)-σ_xx(free)|=%.3e "
                    "Ha/Bohr³  σ_xx(h)=%.6e\n",
                    dsig, sig_h[0]);
    }

    const double wall = std::chrono::duration<double>(clock::now() - t0).count();
    if (wall > kWallBudget) {
        std::fprintf(stderr, "FAIL: wall %.2f s > budget %.1f s\n", wall,
                     kWallBudget);
        ++nfail;
    }

    if (nfail != 0) {
        std::fprintf(stderr, "FAIL: %d gate(s) failed (wall=%.2f s)\n", nfail,
                     wall);
        return 1;
    }
    std::printf("PASS: gpw-hirshfeld-forces (MBD/TS fixed-v dual + product "
                "TS+Hirsh wiring + stress fold; wall=%.2f s)\n",
                wall);
    return 0;
}
