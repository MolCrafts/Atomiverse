/// @file regressions/response/phonon-dispersion-si/reg_phonon-dispersion-si.cpp
/// @brief END-TO-END: phonon dispersion ω(q) for diamond Si (SW IFC Fourier).
///
/// Gates (pins in REFERENCE.md):
///   1. Γ acoustic |ω| below bound (no ASR imposition),
///   2. Γ optical max inside the SW Si optical window,
///   3. X and L points: finite branches, max ω in the SW band,
///   4. frequencies finite and sorted ascending at every path q.
///
/// Public API only. Exit non-zero on failure. No live CP2K.
///
/// Geometry note: the unit cell is the **conventional cubic** 8-atom diamond
/// cell (orthorhombic). A triclinic 2-atom primitive path hangs in the
/// classical neighbor cell list when `Box::length(d)` returns the diagonal
/// entry (0 for diamond primitive) — that is a separate infra bug. The
/// Fourier product surface is independent of the cell choice; conventional
/// cubic is the same SW twin family as `fd-phonon-supercell`.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/sw.h"
#include "atomiverse/cpu/vibration/phonon.h"
#include "atomiverse/cpu/vibration/phonon_dispersion.h"
#include "atomiverse/cpu/vibration/supercell.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

/// @brief Conventional cubic diamond Si: n_cell³ of 8-atom cells.
cpu::Frame diamond_conventional(double a, int n_cell) {
    const double frac[8][3] = {
        {0, 0, 0},     {0.25, 0.25, 0.25}, {0.5, 0.5, 0}, {0.75, 0.75, 0.25},
        {0.5, 0, 0.5}, {0.75, 0.25, 0.75}, {0, 0.5, 0.5}, {0.25, 0.75, 0.75}};
    const int n = 8 * n_cell * n_cell * n_cell;
    cpu::Frame f(n);
    int idx = 0;
    for (int ix = 0; ix < n_cell; ++ix)
        for (int iy = 0; iy < n_cell; ++iy)
            for (int iz = 0; iz < n_cell; ++iz)
                for (int k = 0; k < 8; ++k) {
                    f.span(kAtom_x)[idx] =
                        static_cast<StorageReal>((double(ix) + frac[k][0]) * a);
                    f.span(kAtom_y)[idx] =
                        static_cast<StorageReal>((double(iy) + frac[k][1]) * a);
                    f.span(kAtom_z)[idx] =
                        static_cast<StorageReal>((double(iz) + frac[k][2]) * a);
                    f.span(kAtom_type)[idx] = 14;
                    f.span(kAtom_mass)[idx] = 28.0855;
                    f.span(kAtom_tag)[idx] = static_cast<TagInt>(idx + 1);
                    ++idx;
                }
    f.box = cpu::Box::cubic(a * n_cell);
    return f;
}

bool near(const std::array<double, 3>& q, double x, double y, double z) {
    return std::abs(q[0] - x) < 1e-12 && std::abs(q[1] - y) < 1e-12
           && std::abs(q[2] - z) < 1e-12;
}

} // namespace

int main() {
    // Offline pins — see REFERENCE.md.
    constexpr double kA0 = 5.431; // Å
    // Unit = 1 conventional cubic (8 atoms); IFC supercell 2×2×2 → 64 atoms.
    constexpr int kNa = 2, kNb = 2, kNc = 2;
    constexpr int kNUnit = 8;
    constexpr int kBranches = 3 * kNUnit; // 24
    constexpr int kNSc = kNUnit * kNa * kNb * kNc;
    constexpr double kAcousticTol = 30.0; // cm^-1 at Γ
    constexpr double kOptLo = 500.0;      // cm^-1 Γ optical max
    constexpr double kOptHi = 900.0;
    constexpr double kZoneMaxLo = 300.0; // cm^-1 max at X/L
    constexpr double kZoneMaxHi = 900.0;

    cpu::Frame unit = diamond_conventional(kA0, /*n_cell=*/1);
    cpu::SWPotential pot(molrs::Element::Si, "stillinger-weber");
    const double cutoff = pot.cutoff();

    cpu::Frame sc = cpu::vib::expand_supercell(unit, kNa, kNb, kNc);
    if (sc.nlocal() != kNSc) {
        std::printf("FAIL: expand_supercell n=%d want %d\n", sc.nlocal(), kNSc);
        return 1;
    }
    if (sc.box.length(0) <= 2.0 * cutoff) {
        std::printf("FAIL: box L=%.4f must exceed 2*cutoff=%.4f\n",
                    sc.box.length(0), 2.0 * cutoff);
        return 1;
    }

    // Proven SW FD path (same as fd-phonon-supercell / unit test).
    cpu::vib::Phonon phonon(/*na=*/1, /*nb=*/1, /*nc=*/1);
    phonon.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, false});
    phonon.on(ATV::vib::Phase::force).add(cpu::StillingerWeber{pot});
    phonon.build();
    const cpu::vib::PhononResult pr = phonon.run(sc);
    if (pr.n_atoms != kNSc) {
        std::printf("FAIL: n_sc=%d want %d\n", pr.n_atoms, kNSc);
        return 1;
    }

    // Conventional-cubic high-symmetry path Γ–X–L (fractional of cubic).
    cpu::vib::PhononDispersion path;
    path.qfrac = {
        {0.0, 0.0, 0.0},                    // Γ
        {0.25, 0.0, 0.0},  {0.5, 0.0, 0.0}, // X
        {0.5, 0.25, 0.25}, {0.5, 0.5, 0.5}, // L
    };
    const cpu::vib::PhononDispersionResult r =
        path.compute(pr.hessian, unit, kNa, kNb, kNc);

    if (r.n_branches != kBranches || r.n_atoms_unit != kNUnit) {
        std::printf("FAIL: branches=%d atoms_unit=%d want %d / %d\n",
                    r.n_branches, r.n_atoms_unit, kBranches, kNUnit);
        return 1;
    }
    if (r.frequencies_cm.size() != r.qfrac.size()) {
        std::printf("FAIL: path size mismatch\n");
        return 1;
    }

    bool saw_G = false, saw_X = false, saw_L = false;
    double max_ac_G = 0.0, wmax_G = 0.0, wmax_X = 0.0, wmax_L = 0.0;

    for (std::size_t iq = 0; iq < r.qfrac.size(); ++iq) {
        const auto& w = r.frequencies_cm[iq];
        if (static_cast<int>(w.size()) != kBranches) {
            std::printf("FAIL: q[%zu] has %zu branches want %d\n", iq, w.size(),
                        kBranches);
            return 1;
        }
        for (int b = 0; b < kBranches; ++b) {
            if (!std::isfinite(w[static_cast<std::size_t>(b)])) {
                std::printf("FAIL: non-finite omega at q[%zu] branch %d\n", iq,
                            b);
                return 1;
            }
            if (b > 0
                && w[static_cast<std::size_t>(b)] + 1e-9
                       < w[static_cast<std::size_t>(b - 1)]) {
                std::printf("FAIL: unsorted branches at q[%zu]\n", iq);
                return 1;
            }
        }
        const double wmax = w[static_cast<std::size_t>(kBranches - 1)];

        if (near(r.qfrac[iq], 0, 0, 0)) {
            saw_G = true;
            for (int b = 0; b < 3; ++b)
                max_ac_G = std::max(max_ac_G,
                                    std::abs(w[static_cast<std::size_t>(b)]));
            wmax_G = wmax;
        }
        if (near(r.qfrac[iq], 0.5, 0, 0)) {
            saw_X = true;
            wmax_X = wmax;
        }
        if (near(r.qfrac[iq], 0.5, 0.5, 0.5)) {
            saw_L = true;
            wmax_L = wmax;
        }
    }

    if (!saw_G || !saw_X || !saw_L) {
        std::printf("FAIL: path missing G/X/L  G=%d X=%d L=%d\n", saw_G ? 1 : 0,
                    saw_X ? 1 : 0, saw_L ? 1 : 0);
        return 1;
    }
    if (!(max_ac_G < kAcousticTol)) {
        std::printf("FAIL: Gamma acoustic |ω|_max=%.3f cm^-1 want < %.1f\n",
                    max_ac_G, kAcousticTol);
        return 1;
    }
    if (!(wmax_G >= kOptLo && wmax_G <= kOptHi)) {
        std::printf("FAIL: Gamma optical max=%.3f cm^-1 want [%.0f, %.0f]\n",
                    wmax_G, kOptLo, kOptHi);
        return 1;
    }
    if (!(wmax_X >= kZoneMaxLo && wmax_X <= kZoneMaxHi)) {
        std::printf("FAIL: X max omega=%.3f cm^-1 want [%.0f, %.0f]\n", wmax_X,
                    kZoneMaxLo, kZoneMaxHi);
        return 1;
    }
    if (!(wmax_L >= kZoneMaxLo && wmax_L <= kZoneMaxHi)) {
        std::printf("FAIL: L max omega=%.3f cm^-1 want [%.0f, %.0f]\n", wmax_L,
                    kZoneMaxLo, kZoneMaxHi);
        return 1;
    }

    std::printf(
        "PASS: phonon dispersion Si SW  a=%.3f A  unit=8-atom cubic  "
        "sc=%dx%dx%d  branches=%d  path_n=%zu  "
        "Gamma |ac|_max=%.2f  Gamma wmax=%.2f  X wmax=%.2f  L wmax=%.2f "
        "cm^-1\n",
        kA0, kNa, kNb, kNc, kBranches, r.qfrac.size(), max_ac_G, wmax_G, wmax_X,
        wmax_L);
    return 0;
}
