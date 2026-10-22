/// @file tests/cpu/longrange/periodic_rocksalt.h
/// @brief Shared rock-salt fixtures for the Ewald / PME / PPPM unit tests.
///
/// Pair kernels read raw Cartesian differences between explicit atom and image
/// rows; they apply no minimum image. A periodic test therefore builds its
/// neighbour list from the real owners (CommBrick images + NeighborList), and
/// only a cluster whose every pair is nearer than any image may use the dense
/// owner-only list.
#pragma once

#include <utility>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

#include "../dynamics/periodic_neighbor_fixture.h"

namespace ATV::test {

constexpr double kMadelungNaCl = 1.747564594633182;
constexpr double kRockSaltSkin = 0.3; ///< Verlet skin of the test lists [Å]

// Rock-salt lattice: n³ ions on a simple-cubic grid of spacing d, charges
// alternating (-1)^(i+j+k). Box edge = n·d. Neutral when n is even.
inline cpu::Frame make_rocksalt(double d, int n) {
    const int N = n * n * n;
    cpu::Frame frame(N);
    int idx = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                frame.span(ATV::kAtom_x)[idx] = i * d;
                frame.span(ATV::kAtom_y)[idx] = j * d;
                frame.span(ATV::kAtom_z)[idx] = k * d;
                frame.span(ATV::kAtom_charge)[idx] =
                    ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
                frame.span(ATV::kAtom_mass)[idx] = 1.0;
                frame.span(ATV::kAtom_type)[idx] = 0;
                frame.span(ATV::kAtom_tag)[idx] = idx + 1;
                ++idx;
            }
    frame.box = ATV::cpu::Box::cubic(n * d);
    return frame;
}

// Dense owner-only half list (every j > i). Valid only when no periodic image
// of any pair lies inside the interaction cutoff.
inline void build_dense_halflist(const cpu::Frame& f,
                                 cpu::md::PairList& pairs) {
    const int N = f.nlocal();
    pairs.inum = N;
    pairs.n_rows = N;
    pairs.ilist.resize(N);
    pairs.numneigh.resize(N);
    pairs.firstneigh.resize(N);
    pairs.neighbors.clear();
    int off = 0;
    for (int i = 0; i < N; ++i) {
        pairs.ilist[i] = i;
        pairs.firstneigh[i] = off;
        for (int j = i + 1; j < N; ++j)
            pairs.neighbors.push_back(j);
        pairs.numneigh[i] = N - 1 - i;
        off += N - 1 - i;
    }
    pairs.total_pairs = off;
    pairs.version = 1;
}

inline void zero_force_energy_stress(cpu::Frame& f) {
    for (int i = 0; i < f.n_total(); ++i)
        f.span(ATV::kAtom_fx)[i] = f.span(ATV::kAtom_fy)[i] =
            f.span(ATV::kAtom_fz)[i] = 0;
    f.meta.get(ATV::kEnergy) = 0;
    f.meta.get(ATV::kEnergyPair) = 0;
    for (int s = 0; s < 6; ++s)
        f.meta.get(ATV::kStress)[s] = 0;
}

// Rock-salt with explicit periodic image rows (charges carried by the border
// pass under CoulLong's own ghost declaration) and the real Newton half list.
struct PeriodicRockSalt {
    cpu::Frame frame;
    cpu::CoulLong ghost_probe; ///< supplies `declare_ghosts()` for the fold
    PeriodicNeighborFixture fixture;
    cpu::md::NeighborList<> neighbors;
    cpu::Context& context;

    PeriodicRockSalt(double d, int n, double cutoff)
        : frame(make_rocksalt(d, n)), ghost_probe(1.0, cutoff),
          fixture(frame, cutoff + kRockSaltSkin, ghost_probe.declare_ghosts()),
          neighbors{cutoff, kRockSaltSkin, 1, 0, true},
          context(fixture.context) {
        neighbors.build({context, frame});
        neighbors.launch({context, frame, 0, 0});
    }
    PeriodicRockSalt(const PeriodicRockSalt&) = delete;
    PeriodicRockSalt& operator=(const PeriodicRockSalt&) = delete;
};

// Total electrostatic energy: real-space CoulLong + one reciprocal component,
// on the pair list `ctx` publishes. Ghost forces fold back onto owners when
// the context carries a communicator.
template <typename KSpace>
double electrostatic_energy(cpu::Frame& f, cpu::Context& ctx, double alpha,
                            double cutoff, KSpace&& recip) {
    zero_force_energy_stress(f);
    cpu::BuildContext bctx{ctx, f};
    cpu::LaunchContext lctx{ctx, f, 0, 0};

    cpu::CoulLong real(alpha, cutoff);
    real.build(bctx);
    recip.build(bctx);
    real.launch(lctx);
    recip.launch(lctx);
    if (ctx.md.comm != nullptr)
        ctx.md.comm->reverse_comm(lctx);
    real.destroy({ctx, f});
    recip.destroy({ctx, f});
    return f.meta.get(ATV::kEnergy);
}

} // namespace ATV::test
