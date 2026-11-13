/// @file tests/cuda/dynamics/nacl_parity.h
/// @brief The Tosi–Fumi NaCl model of `tests/cpu/dynamics/nacl_fixture.h`
///        mounted on the CPU and the CUDA MD drivers with the same
///        settings, and tag-keyed comparison helpers. The CPU run is the
///        oracle of the CUDA NEMD tests.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/fix_efield.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_born.h"
#include "atomiverse/cuda/dynamics/fix_efield.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/longrange/ewald.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/observe/stress.cuh"
#include "atomiverse/cuda/potentials/coul_long.cuh"
#include "atomiverse/cuda/potentials/pair_born.cuh"

#include "../../cpu/dynamics/nacl_fixture.h"

namespace nacl_parity {

using namespace ATV;

inline constexpr bool kDouble = std::is_same_v<cuda::force_t, double>
                                && std::is_same_v<cuda::pos_t, double>;
inline constexpr double kSkin = 0.5;      ///< [Å]; list cutoff 6.0 > L/2
inline constexpr double kAccuracy = 1e-6; ///< Ewald k-space accuracy

/// Which parts of the model a run mounts.
struct Terms {
    bool born = true;
    bool coulomb = true; ///< CoulLong + KSpaceEwald
    bool tail = false;   ///< Born `pair_modify tail`
    double field = 0.0;  ///< E along x [V/Å]; 0: no FixEfield
};

inline std::vector<cuda::BornTypeCoeff> cuda_tosi_fumi() {
    std::vector<cuda::BornTypeCoeff> rows;
    for (const cpu::BornTypeCoeff& c : nacl_fixture::tosi_fumi())
        rows.push_back({c.type_i, c.type_j, c.A, c.rho, c.sigma, c.C, c.D});
    return rows;
}

inline void mount(cpu::md::MD& sim, const Terms& t) {
    using namespace nacl_fixture;
    sim.set_units("metal");
    sim.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin});
    sim.on(md::Phase::post_force).add(cpu::md::Virial{}, -50);
    if (t.born)
        sim.set_evaluator(cpu::PairBorn{tosi_fumi(), kCut}
                              .set_energy_shift(false)
                              .set_tail_correction(t.tail));
    if (t.coulomb) {
        sim.set_evaluator(cpu::CoulLong{kAlpha, kCut});
        sim.set_evaluator(cpu::KSpaceEwald{kAlpha, kAccuracy});
    }
    // The force phase, where the Python `set_evaluator` puts it.
    if (t.field != 0.0)
        sim.set_evaluator(cpu::md::FixEfield{t.field, 0, 0});
}

inline void mount(cuda::md::MD& sim, const Terms& t) {
    using namespace nacl_fixture;
    sim.set_units("metal");
    sim.set_neighbor(cuda::NeighborList{kCut, kSkin});
    sim.on(md::Phase::post_force).add(cuda::Virial{}, -50);
    if (t.born)
        sim.set_evaluator(cuda::PairBorn{cuda_tosi_fumi(), kCut}
                              .set_energy_shift(false)
                              .set_tail_correction(t.tail));
    if (t.coulomb) {
        sim.set_evaluator(cuda::CoulLong{kAlpha, kCut});
        sim.set_evaluator(cuda::KSpaceEwald{kAlpha, kAccuracy});
    }
    if (t.field != 0.0)
        sim.set_evaluator(cuda::md::FixEfield{t.field, 0, 0});
}

/// One atom's state, keyed by tag (the CPU run may permute its rows).
struct Atom {
    std::array<double, 3> unwrapped, v, f;
};

inline std::map<std::int64_t, Atom> atoms(const cpu::Frame& f) {
    std::map<std::int64_t, Atom> out;
    const auto& h = f.box.cell();
    for (int i = 0; i < f.nlocal(); ++i) {
        Atom a;
        a.unwrapped = {f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                       f.span(kAtom_z)[i]};
        for (int d = 0; d < 3; ++d)
            for (int b = 0; b < 3; ++b)
                a.unwrapped[d] +=
                    h(d, b) * static_cast<double>(f.span(kAtom_image)(i, b));
        a.v = {f.span(kAtom_vx)[i], f.span(kAtom_vy)[i], f.span(kAtom_vz)[i]};
        a.f = {f.span(kAtom_fx)[i], f.span(kAtom_fy)[i], f.span(kAtom_fz)[i]};
        out[f.span(kAtom_tag)[i]] = a;
    }
    return out;
}

/// Largest |got − want| of one per-atom vector field, relative to the
/// largest reference magnitude (a vacuous reference field fails).
inline double worst(const std::map<std::int64_t, Atom>& got,
                    const std::map<std::int64_t, Atom>& want,
                    std::array<double, 3> Atom::* field, const char* what) {
    double scale = 0.0, diff = 0.0;
    for (const auto& [tag, w] : want) {
        const Atom& g = got.at(tag);
        for (int d = 0; d < 3; ++d) {
            scale = std::max(scale, std::abs((w.*field)[d]));
            diff = std::max(diff, std::abs((g.*field)[d] - (w.*field)[d]));
        }
    }
    EXPECT_GT(scale, 0.0) << what << " is vacuous";
    return diff / scale;
}

} // namespace nacl_parity
