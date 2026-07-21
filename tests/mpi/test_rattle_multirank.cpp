/// @file tests/mpi/test_rattle_multirank.cpp
/// @brief Multi-rank RATTLE: rigid water across a domain decomposition.
///
/// The invariant does not care how many ranks there are. Every constrained distance
/// must stay at its target — including for the molecules that straddle a rank
/// boundary, whose atoms are owned by different ranks and are ghosts to each other.
///
/// This exercises the two things multi-rank constraints needed and the engine did not
/// have: a GLOBAL constraint table (the rank owning only an H would otherwise never
/// discover the cluster, because a water's bonds and angle all live on its oxygen),
/// and `comm::HaloRefresh` — a forward halo bindable at an arbitrary phase, because the
/// velocity stage runs at `final_integrate` where no communication has ever happened
/// and ghost velocities do not otherwise exist at all.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/comm/domain.h"
#include "atomiverse/cpu/comm/halo_exchange.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/fix_rattle.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

constexpr double kROH = 0.9572;
constexpr double kThetaDeg = 104.52;
const double kDHH = 2.0 * kROH * std::sin(0.5 * kThetaDeg * M_PI / 180.0);
constexpr double kConstraintTolerance =
    sizeof(StorageReal) == sizeof(float)
        ? 8.0 * cpu::md::FixRattle::default_tolerance()
        : 1.0e-8;

cpu::Frame water_box(int n_side, double spacing) {
    const int n_mol = n_side * n_side * n_side;
    const int n = 3 * n_mol;
    cpu::Frame f(n);
    const double half = 0.5 * kThetaDeg * M_PI / 180.0;
    int idx = 0, mol = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                const double ox = (ix + 0.5) * spacing;
                const double oy = (iy + 0.5) * spacing;
                const double oz = (iz + 0.5) * spacing;
                const double phi = 0.7 * double(mol);
                const int o = idx, h1 = idx + 1, h2 = idx + 2;

                f.span(ATV::kAtom_x)[o] = ox;
                f.span(ATV::kAtom_y)[o] = oy;
                f.span(ATV::kAtom_z)[o] = oz;
                f.span(ATV::kAtom_mass)[o] = 15.9994;
                f.span(ATV::kAtom_type)[o] = 8;
                f.span(ATV::kAtom_x)[h1] =
                    ox + kROH * std::sin(half) * std::cos(phi);
                f.span(ATV::kAtom_y)[h1] =
                    oy + kROH * std::sin(half) * std::sin(phi);
                f.span(ATV::kAtom_z)[h1] = oz + kROH * std::cos(half);
                f.span(ATV::kAtom_mass)[h1] = 1.008;
                f.span(ATV::kAtom_type)[h1] = 1;
                f.span(ATV::kAtom_x)[h2] =
                    ox - kROH * std::sin(half) * std::cos(phi);
                f.span(ATV::kAtom_y)[h2] =
                    oy - kROH * std::sin(half) * std::sin(phi);
                f.span(ATV::kAtom_z)[h2] = oz + kROH * std::cos(half);
                f.span(ATV::kAtom_mass)[h2] = 1.008;
                f.span(ATV::kAtom_type)[h2] = 1;

                idx += 3;
                ++mol;
            }
    f.simbox = ATV::cpu::Box::cubic(n_side * spacing);
    f.rebuild_tag_index(0, n);
    for (int m = 0; m < n_mol; ++m) {
        const int o = 3 * m, h1 = o + 1, h2 = o + 2;
        f.add_bond(f.span(ATV::kAtom_tag)[o], f.span(ATV::kAtom_tag)[h1], 0);
        f.add_bond(f.span(ATV::kAtom_tag)[o], f.span(ATV::kAtom_tag)[h2], 0);
        f.add_angle(f.span(ATV::kAtom_tag)[h1], f.span(ATV::kAtom_tag)[o],
                    f.span(ATV::kAtom_tag)[h2], 0);
    }
    f.build_special_list();
    return f;
}

} // namespace

/// The rigid geometry holds across ranks — for every molecule this rank owns any of.
TEST(RattleMultiRank, GeometryStaysRigidAcrossRanks) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n_side = 4; // 64 waters
    const double spacing = 4.6, cutoff = 6.0, skin = 1.0, dt = 2.0;
    cpu::Frame frame = water_box(n_side, spacing);

    cpu::md::MD sim;
    sim.add_potential(
        cpu::PairLJCut{0.0067, 3.15, cutoff}.set_special_bonds(0, 0, 0));
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});

    cpu::md::FixRattle rattle{dt};
    rattle.constrain_bond(0, kROH);
    rattle.constrain_angle(0, kDHH);
    sim.add_ref(rattle);

    // NOTE: no HaloRefresh is registered here. The driver auto-injects the two the
    // constraint needs (ghost positions after the drift, ghost velocities after the
    // second half-kick) the moment it sees a ComponentRole::Constraint. Making the user
    // remember them would mean that forgetting one line yields constraints that are
    // silently wrong at rank boundaries — no throw, no crash, just a molecule that is
    // not rigid on one side of a cut. This test passing WITHOUT them is the assertion.
    sim.enable_timing(false);
    sim.build();

    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(ATV::kAtom_vx)[i] = 0.003 * std::sin(double(3 * i + 1));
        frame.span(ATV::kAtom_vy)[i] = 0.003 * std::cos(double(5 * i + 2));
        frame.span(ATV::kAtom_vz)[i] = 0.003 * std::sin(double(7 * i + 3));
    }

    sim.run(frame, 100);

    // Check every constraint whose atoms are present on this rank (owner or ghost).
    // Molecules straddling a boundary are precisely the interesting ones.
    //
    // Resolve through the decomposition tag map, not Frame's local tag index: under MPI the
    // frame's own map is not the authority (it still holds the pre-partition identity
    // mapping), and looking tags up there reads the wrong rows entirely — which is how
    // this test first "failed" by 9.5 A while the physics was fine.
    auto dd = sim.release_dd();
    ASSERT_NE(dd, nullptr);
    const auto x = frame.span(kAtom_x);
    const auto y = frame.span(kAtom_y);
    const auto z = frame.span(kAtom_z);
    const auto& tag_index = frame.meta.get(cpu::kTagIndex);
    double worst = 0.0;
    int checked = 0;
    for (int m = 0; m < n_side * n_side * n_side; ++m) {
        const long long to = 3 * m, th1 = to + 1, th2 = to + 2;
        const auto find = [&](long long tag) -> int {
            if (size > 1) {
                const auto it =
                    dd->tag_to_local.find(static_cast<std::int64_t>(tag));
                return (it == dd->tag_to_local.end()) ? -1 : it->second;
            }
            const auto it = tag_index.find(static_cast<TagInt>(tag));
            return (it == tag_index.end()) ? -1 : it->second;
        };
        const int o = find(to), h1 = find(th1), h2 = find(th2);
        if (o < 0 || h1 < 0 || h2 < 0)
            continue;

        const int ii[3] = {o, o, h1};
        const int jj[3] = {h1, h2, h2};
        const double want[3] = {kROH, kROH, kDHH};
        for (int c = 0; c < 3; ++c) {
            double dx = x[ii[c]] - x[jj[c]];
            double dy = y[ii[c]] - y[jj[c]];
            double dz = z[ii[c]] - z[jj[c]];
            frame.simbox.min_image(dx, dy, dz);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            worst = std::max(worst, std::fabs(r - want[c]) / want[c]);
            ++checked;
        }
    }

    EXPECT_GT(checked, 0) << "rank " << rank << " checked no molecules";
    EXPECT_LT(worst, kConstraintTolerance)
        << "on " << size << " ranks, rank " << rank
        << " sees a constrained distance off by " << worst;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
