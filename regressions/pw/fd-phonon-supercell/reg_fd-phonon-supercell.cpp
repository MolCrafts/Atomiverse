/// @file regressions/pw/fd-phonon-supercell/reg_fd-phonon-supercell.cpp
/// @brief END-TO-END: Γ-point FD phonons on diamond Si (SW force twin).
///
/// See REFERENCE.md. Public API only. Exit non-zero with got/want/tol.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/sw.h"
#include "atomiverse/cpu/vibration/phonon.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

/// @brief Conventional cubic diamond Si supercell: n_cell³ of 8-atom cells.
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

} // namespace

int main() {
    // Offline pin — see REFERENCE.md.
    // 2×2×2 conventional (64 atoms): L = 10.862 Å > 2 × SW cutoff.
    constexpr double kA0 = 5.431; // Å
    constexpr int kNCell = 2;     // conventional repeats
    constexpr int kNAtoms = 8 * kNCell * kNCell * kNCell;
    constexpr int kNModes = 3 * kNAtoms;  // 192
    constexpr double kAcousticTol = 30.0; // cm^-1
    constexpr double kOptLo = 500.0;      // cm^-1
    constexpr double kOptHi = 900.0;      // cm^-1

    cpu::Frame unit = diamond_conventional(kA0, kNCell);
    cpu::SWPotential pot(molrs::Element::Si, "stillinger-weber");
    const double cutoff = pot.cutoff();
    if (unit.box.length(0) <= 2.0 * cutoff) {
        std::printf("FAIL: box L=%.4f must exceed 2*cutoff=%.4f\n",
                    unit.box.length(0), 2.0 * cutoff);
        return 1;
    }

    // Phonon(1,1,1): the Frame already *is* the supercell (Γ of that cell).
    // expand_supercell path is unit-tested separately.
    cpu::vib::Phonon phonon(/*na=*/1, /*nb=*/1, /*nc=*/1);
    phonon.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, false});
    phonon.on(ATV::vib::Phase::force).add(cpu::StillingerWeber{pot});
    phonon.build();
    const cpu::vib::PhononResult r = phonon.run(unit);

    if (r.n_atoms != kNAtoms) {
        std::printf("FAIL: n_atoms got=%d want=%d\n", r.n_atoms, kNAtoms);
        return 1;
    }
    if (static_cast<int>(r.frequencies.size()) != kNModes) {
        std::printf("FAIL: n_modes got=%zu want=%d\n",
                    static_cast<std::size_t>(r.frequencies.size()), kNModes);
        return 1;
    }

    double max_ac = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double w = std::abs(r.frequencies(k));
        if (w > max_ac)
            max_ac = w;
    }
    if (!(max_ac < kAcousticTol)) {
        std::printf("FAIL: acoustic |ω|_max got=%.4f cm^-1 want < %.1f cm^-1 "
                    "(Γ acoustic sum rule; see REFERENCE.md)\n",
                    max_ac, kAcousticTol);
        for (int k = 0; k < 6; ++k)
            std::printf("  ω[%d]=%.4f\n", k, r.frequencies(k));
        return 1;
    }

    const double w_max = r.frequencies(kNModes - 1);
    if (!(w_max >= kOptLo && w_max <= kOptHi)) {
        std::printf(
            "FAIL: highest optical ω got=%.4f cm^-1 want band [%.1f, %.1f] "
            "(SW Si optical band; see REFERENCE.md)\n",
            w_max, kOptLo, kOptHi);
        return 1;
    }

    std::printf("PASS: FD phonon supercell Si SW  a=%.3f A  n_cell=%d  N=%d  "
                "acoustic|ω|_max=%.3f cm^-1  ω_max=%.3f cm^-1  modes=%d\n",
                kA0, kNCell, kNAtoms, max_ac, w_max, kNModes);
    return 0;
}
