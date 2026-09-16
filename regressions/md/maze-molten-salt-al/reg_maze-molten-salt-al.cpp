/// @file regressions/md/maze-molten-salt-al/reg_maze-molten-salt-al.cpp
/// @brief E2E product: molten-salt (Fumi–Tosi NaCl) + Al const-pot walls + MaZe.
///
/// Map §9 MW example scenario composition: PairBorn Fumi–Tosi salt, frozen Al
/// walls, ConstPotMaZe charge dynamics, short NVT. Public headers only.
/// See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_maze.h"
#include "atomiverse/cpu/dynamics/fix_freeze.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_born.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

namespace {

constexpr double kLxy = 12.0;  // [Å]
constexpr double kLz = 24.0;   // [Å]
constexpr double kZBot = 3.0;  // Al wall
constexpr double kZTop = 21.0;
constexpr double kEta = 1.5;   // [1/Å] Al-like Gaussian width
constexpr double kPsiBot = 0.0;
constexpr double kPsiTop = 1.0;
constexpr double kCut = 6.0;
constexpr int n_wall_side = 3; // 3×3 per wall
constexpr int n_salt = 8;      // 4 Na + 4 Cl

// Fumi–Tosi-like BMH (same metal eV/Å family as fumi-tosi-nacl regression).
const double kRho = 0.317;
const double kA_NaNa = 25.4435, kSig_NaNa = 2.340, kC_NaNa = 101.1719,
             kD_NaNa = 48.1771;
const double kA_ClCl = 25.4435, kSig_ClCl = 3.170, kC_ClCl = 278.122,
             kD_ClCl = 694.480;
const double kA_NaCl = 20.3548, kSig_NaCl = 2.755, kC_NaCl = 48.1771,
             kD_NaCl = 48.1771;

cpu::Frame build_system() {
    const int n_wall = 2 * n_wall_side * n_wall_side;
    const int N = n_wall + n_salt;
    cpu::Frame f(N);
    int idx = 0;
    for (int wall = 0; wall < 2; ++wall) {
        const double z = (wall == 0) ? kZBot : kZTop;
        const int typ = (wall == 0) ? 2 : 3; // Al walls
        for (int ix = 0; ix < n_wall_side; ++ix)
            for (int iy = 0; iy < n_wall_side; ++iy) {
                f.span(kAtom_x)[idx] = (ix + 0.5) * (kLxy / n_wall_side);
                f.span(kAtom_y)[idx] = (iy + 0.5) * (kLxy / n_wall_side);
                f.span(kAtom_z)[idx] = z;
                f.span(kAtom_type)[idx] = typ;
                f.span(kAtom_charge)[idx] = 0.0;
                f.span(kAtom_mass)[idx] = 26.9815; // Al
                f.span(kAtom_tag)[idx] = idx + 1;
                ++idx;
            }
    }
    // Molten-salt layer: 4 Na + 4 Cl staggered mid-slab.
    for (int i = 0; i < n_salt; ++i) {
        const bool na = (i % 2 == 0);
        f.span(kAtom_x)[idx] = (0.25 + 0.25 * (i % 4)) * kLxy;
        f.span(kAtom_y)[idx] = (0.3 + 0.2 * (i / 4)) * kLxy;
        f.span(kAtom_z)[idx] = 0.5 * (kZBot + kZTop) + ((i < 4) ? -1.5 : 1.5);
        f.span(kAtom_type)[idx] = na ? 0 : 1;
        f.span(kAtom_charge)[idx] = na ? 1.0 : -1.0;
        f.span(kAtom_mass)[idx] = na ? 22.9898 : 35.453;
        f.span(kAtom_tag)[idx] = idx + 1;
        ++idx;
    }
    f.box = cpu::Box::orthorhombic(kLxy, kLxy, kLz);
    return f;
}

double electrode_Q(const cpu::Frame& f, int typ) {
    double q = 0.0;
    for (int i = 0; i < f.nlocal(); ++i)
        if (f.span(kAtom_type)[i] == typ)
            q += f.span(kAtom_charge)[i];
    return q;
}

} // namespace

int main() {
    auto frame = build_system();

    // Snapshot wall positions.
    std::vector<int> wall_idx;
    std::vector<double> wx0, wy0, wz0;
    for (int i = 0; i < frame.nlocal(); ++i) {
        const int t = frame.span(kAtom_type)[i];
        if (t == 2 || t == 3) {
            wall_idx.push_back(i);
            wx0.push_back(frame.span(kAtom_x)[i]);
            wy0.push_back(frame.span(kAtom_y)[i]);
            wz0.push_back(frame.span(kAtom_z)[i]);
        }
    }

    cpu::md::ConstPotMaZe maze(kEta);
    maze.add_electrode(2, kPsiBot);
    maze.add_electrode(3, kPsiTop);
    maze.solver().electroneutrality = true;
    maze.solver().xy_image_range = 30;
    maze.symplectic = true;

    cpu::PairBorn born({{0, 0, kA_NaNa, kRho, kSig_NaNa, kC_NaNa, kD_NaNa},
                        {1, 1, kA_ClCl, kRho, kSig_ClCl, kC_ClCl, kD_ClCl},
                        {0, 1, kA_NaCl, kRho, kSig_NaCl, kC_NaCl, kD_NaCl},
                        // soft Al–ion repulsion (wall exclusion)
                        {0, 2, 10.0, 0.3, 2.0, 0.0, 0.0},
                        {0, 3, 10.0, 0.3, 2.0, 0.0, 0.0},
                        {1, 2, 10.0, 0.3, 2.5, 0.0, 0.0},
                        {1, 3, 10.0, 0.3, 2.5, 0.0, 0.0}},
                       kCut);

    cpu::md::MD sim;
    sim.set_threads(4);
    sim.set_observer(cpu::md::InitVelocity{/*T=*/800.0, /*seed=*/42u}); // molten
    sim.set_integrator(cpu::md::VelocityVerlet{/*dt=*/0.5});
    sim.set_thermostat(cpu::md::NoseHoover{800.0, /*tau=*/50.0, 0.5});
    sim.set_evaluator(std::move(born));
    sim.set_evaluator(cpu::PairCoulCut{kCut});
    sim.set_neighbor(cpu::md::NeighborList<>{kCut, 0.4});
    sim.set_observer(std::move(maze));
    sim.set_observer(cpu::md::FixFreeze{2});
    sim.set_observer(cpu::md::FixFreeze{3});
    sim.enable_timing(false);
    sim.build();

    sim.run(frame, 1);
    double QL = electrode_Q(frame, 2);
    double QR = electrode_Q(frame, 3);
    std::printf("maze-molten step1: QL=%+.6e QR=%+.6e sum=%.3e\n", QL, QR,
                QL + QR);
    if (!(QL < 0.0 && QR > 0.0)) {
        std::printf("FAIL polarity\n");
        return 1;
    }
    if (!(std::fabs(QL + QR) <= 1e-6)) {
        std::printf("FAIL neutrality\n");
        return 1;
    }

    for (int c = 0; c < 4; ++c) {
        sim.run(frame, 40);
        if (!std::isfinite(frame.meta.get(kEnergy))
            || !std::isfinite(frame.meta.get(kKinetic))) {
            std::printf("FAIL non-finite thermo chunk %d\n", c);
            return 1;
        }
        QL = electrode_Q(frame, 2);
        QR = electrode_Q(frame, 3);
        if (!(std::fabs(QL + QR) <= 1e-5)) {
            std::printf("FAIL neutrality drift chunk %d\n", c);
            return 1;
        }
        // g(r) proxy: ions stay separated (no collapse).
        double min_rr = 1e300;
        for (int i = 0; i < frame.nlocal(); ++i) {
            if (frame.span(kAtom_type)[i] > 1)
                continue;
            for (int j = i + 1; j < frame.nlocal(); ++j) {
                if (frame.span(kAtom_type)[j] > 1)
                    continue;
                double dx = frame.span(kAtom_x)[i] - frame.span(kAtom_x)[j];
                double dy = frame.span(kAtom_y)[i] - frame.span(kAtom_y)[j];
                double dz = frame.span(kAtom_z)[i] - frame.span(kAtom_z)[j];
                // crude min-image
                if (dx > 0.5 * kLxy)
                    dx -= kLxy;
                if (dx < -0.5 * kLxy)
                    dx += kLxy;
                if (dy > 0.5 * kLxy)
                    dy -= kLxy;
                if (dy < -0.5 * kLxy)
                    dy += kLxy;
                if (dz > 0.5 * kLz)
                    dz -= kLz;
                if (dz < -0.5 * kLz)
                    dz += kLz;
                min_rr = std::fmin(min_rr, std::sqrt(dx * dx + dy * dy + dz * dz));
            }
        }
        if (!(min_rr > 1.0)) {
            std::printf("FAIL ion collapse min_r=%.3e chunk %d\n", min_rr, c);
            return 1;
        }
    }

    double max_dx = 0.0;
    for (std::size_t k = 0; k < wall_idx.size(); ++k) {
        const int i = wall_idx[k];
        max_dx = std::fmax(max_dx, std::fabs(frame.span(kAtom_x)[i] - wx0[k]));
        max_dx = std::fmax(max_dx, std::fabs(frame.span(kAtom_y)[i] - wy0[k]));
        max_dx = std::fmax(max_dx, std::fabs(frame.span(kAtom_z)[i] - wz0[k]));
    }
    QL = electrode_Q(frame, 2);
    QR = electrode_Q(frame, 3);
    std::printf("after 160 steps: E=%.6e K=%.6e QL=%+.6e QR=%+.6e freeze=%.3e\n",
                frame.meta.get(kEnergy), frame.meta.get(kKinetic), QL, QR,
                max_dx);
    if (!(max_dx <= 1e-10)) {
        std::printf("FAIL freeze\n");
        return 1;
    }
    if (!(QL < 0.0 && QR > 0.0)) {
        std::printf("FAIL end polarity\n");
        return 1;
    }
    std::printf("PASS: maze-molten-salt-al\n");
    return 0;
}
