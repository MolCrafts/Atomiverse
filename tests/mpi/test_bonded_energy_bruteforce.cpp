/// @file tests/mpi/test_bonded_energy_bruteforce.cpp
/// @brief Decisive bonded-set guard: distributed bond/angle/dihedral energy
///        == independent brute force, at 1 and 8 ranks.
///
/// This is the bonded analogue of test_lj_energy_bruteforce. The other
/// multi-rank PE tests check bonded-energy *conservation* / finiteness over a
/// run; a conserving integrator can still be systematically wrong if a rank
/// silently drops a bonded tuple whose partner lives across a domain cut (the
/// exact stale-ghost / O(n_total) class of defect the ResolveView +
/// BondedForceScratch rework addressed). This test removes that blind spot by
/// pinning the *absolute* bond, angle, and dihedral potential energies of a
/// fixed configuration against an O(Σtuples) min-image brute force computed
/// from a pristine copy of the global topology — decomposition-independent, so
/// the same oracle is checked at 1 rank and at 8 ranks.
///
/// The chains are deliberately laid one per axis straddling the central cut of
/// the 2×2×2 (8-rank) Cartesian grid, so at 8 ranks every chain has bond,
/// angle, and dihedral tuples whose partners are *ghosts* owned by a
/// neighbouring rank — i.e. the halo-resolution path the bonded kernels rely
/// on is actually exercised, not bypassed. A small deterministic perturbation
/// pushes every tuple off equilibrium so each energy term is non-degenerate.
///
/// Halo cutoff (6 Å) comfortably exceeds the longest intra-tuple span (a
/// dihedral reaches 2 bonds ≈ 3 Å from its owner), so a correct halo makes
/// every partner resolvable; any disagreement means a bonded tuple's partner
/// was dropped or mis-resolved across the cut.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/comm/halo_exchange.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"

using namespace ATV;

namespace {

constexpr double kEnergyRelativeTolerance = 1.0e-9;

inline constexpr double kKcalPerMolToEv = 1.0 / 23.060547830619026;
inline constexpr double kBondK_kcal = 268.0;
inline constexpr double kBondR0 = 1.53;
inline constexpr double kAngleK_kcal = 60.0;
inline constexpr double kAngleTheta0 = 109.5 * units::pi / 180.0;

inline double bond_K_eV() { return kBondK_kcal * kKcalPerMolToEv; }
inline double angle_K_eV() { return kAngleK_kcal * kKcalPerMolToEv; }
inline std::vector<double> dihedral_Cn_eV() {
    return {
        1.73 * kKcalPerMolToEv,  // C0  (MSU paper, Tschopp 2010)
        -4.49 * kKcalPerMolToEv, // C1
        0.776 * kKcalPerMolToEv, // C2
        6.99 * kKcalPerMolToEv,  // C3
        0.0,                     // C4
    };
}

// Cubic minimum image, multi-wrap safe and independent of Box::min_image's
// internal implementation (the honest-oracle requirement).
void min_image_cubic(double& dx, double& dy, double& dz, double L) {
    while (dx > 0.5 * L)
        dx -= L;
    while (dx < -0.5 * L)
        dx += L;
    while (dy > 0.5 * L)
        dy -= L;
    while (dy < -0.5 * L)
        dy += L;
    while (dz > 0.5 * L)
        dz -= L;
    while (dz < -0.5 * L)
        dz += L;
}

struct BondedEnergies {
    double bond = 0.0;
    double angle = 0.0;
    double dihedral = 0.0;
};

// Build C20 zig-zag chains, one per axis, each centred on the box centre so it
// straddles the central cut of the 8-rank 2×2×2 grid. A small deterministic
// per-atom perturbation drives every tuple off equilibrium (non-zero energy).
cpu::Frame make_axis_crossing_chains(double box_len) {
    const int n_carbons = 20;
    const int n_chains = 3; // one along x, one along y, one along z
    const int n_atoms = n_chains * n_carbons;
    cpu::Frame frame(n_atoms);

    const double half_apex = (units::pi - kAngleTheta0) / 2.0;
    const double along = kBondR0 * std::cos(half_apex); // step along chain axis
    const double zig = kBondR0 * std::sin(half_apex);   // transverse zig-zag
    const double c = box_len / 2.0;
    // Transverse anchor offsets keep the three chains apart (bonded energy is
    // topology-only, but it keeps the geometry tidy / inside the box).
    const double off[3][2] = {{-4.0, +4.0}, {+4.0, -4.0}, {-4.0, -4.0}};

    auto place = [&](int base, int axis) {
        const double start = c - 0.5 * (n_carbons - 1) * along;
        for (int i = 0; i < n_carbons; ++i) {
            const double a = start + i * along;        // along the chain axis
            const double t = (i % 2 == 0) ? 0.0 : zig; // transverse zig-zag
            double p[3];
            p[axis] = a;
            p[(axis + 1) % 3] = c + off[axis][0] + t;
            p[(axis + 2) % 3] = c + off[axis][1];
            // Deterministic off-equilibrium perturbation (sub-Å, smooth).
            const int g = base + i;
            p[0] += 0.17 * std::sin(0.7 * g + 1.0);
            p[1] += 0.13 * std::cos(0.9 * g + 2.0);
            p[2] += 0.11 * std::sin(1.3 * g + 0.5);
            frame.span(ATV::kAtom_x)[g] = p[0];
            frame.span(ATV::kAtom_y)[g] = p[1];
            frame.span(ATV::kAtom_z)[g] = p[2];
            frame.span(ATV::kAtom_mass)[g] = 14.027;
            frame.span(ATV::kAtom_type)[g] = 0;
            frame.span(ATV::kAtom_vx)[g] = 0.0;
            frame.span(ATV::kAtom_vy)[g] = 0.0;
            frame.span(ATV::kAtom_vz)[g] = 0.0;
        }
    };
    for (int chain = 0; chain < n_chains; ++chain)
        place(chain * n_carbons, chain);

    const int n_b_per = n_carbons - 1;
    const int n_a_per = n_carbons - 2;
    const int n_d_per = n_carbons - 3;
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int chain = 0; chain < n_chains; ++chain) {
        const int base = chain * n_carbons;
        for (int k = 0; k < n_b_per; ++k)
            frame.add_bond(base + k, base + k + 1, 0);
        for (int k = 0; k < n_a_per; ++k)
            frame.add_angle(base + k, base + k + 1, base + k + 2, 0);
        for (int k = 0; k < n_d_per; ++k)
            frame.add_dihedral(base + k, base + k + 1, base + k + 2,
                               base + k + 3, 0);
    }
    frame.build_special_list();

    frame.simbox = ATV::cpu::Box::cubic(box_len);
    return frame;
}

// Independent oracle: replay every *stored* bonded tuple on the pristine global
// frame (tag == index there) with cubic min-image. Reads num_bond/bond_atom
// etc. directly, so it matches the kernels' ownership convention by
// construction and the energy formulas verbatim (bond K(r-r0)²; angle
// K(θ-θ0)²; dihedral Σ Cn cosⁿφ with φ from the b1×b2, b2×b3 normals).
BondedEnergies brute_force_bonded(const cpu::Frame& f) {
    const auto x = f.span(kAtom_x);
    const auto y = f.span(kAtom_y);
    const auto z = f.span(kAtom_z);
    const auto num_bond = f.span(kAtom_num_bond);
    const auto bond_atom = f.span(kAtom_bond_atom);
    const auto num_angle = f.span(kAtom_num_angle);
    const auto angle_atom = f.span(kAtom_angle_atom);
    const auto num_dihedral = f.span(kAtom_num_dihedral);
    const auto dihedral_atom = f.span(kAtom_dihedral_atom);
    const double L = f.simbox.cell()(0, 0);
    const int n = f.nlocal();
    const double bk = bond_K_eV();
    const double ak = angle_K_eV();
    const auto cn = dihedral_Cn_eV();
    auto pos = [&](int i, double r[3]) {
        r[0] = x[i];
        r[1] = y[i];
        r[2] = z[i];
    };

    BondedEnergies e;
    for (int i = 0; i < n; ++i) {
        double ri[3];
        pos(i, ri);

        // Bonds owned by i.
        for (int m = 0; m < num_bond[i]; ++m) {
            const int j = static_cast<int>(bond_atom(i, m));
            double rj[3];
            pos(j, rj);
            double dx = ri[0] - rj[0], dy = ri[1] - rj[1], dz = ri[2] - rj[2];
            min_image_cubic(dx, dy, dz, L);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double dr = r - kBondR0;
            e.bond += bk * dr * dr;
        }

        // Angles centred on i.
        for (int m = 0; m < num_angle[i]; ++m) {
            const int pi = static_cast<int>(angle_atom(i, 2 * m + 0));
            const int pk = static_cast<int>(angle_atom(i, 2 * m + 1));
            double rpi[3], rpk[3];
            pos(pi, rpi);
            pos(pk, rpk);
            double jix = rpi[0] - ri[0], jiy = rpi[1] - ri[1],
                   jiz = rpi[2] - ri[2];
            double jkx = rpk[0] - ri[0], jky = rpk[1] - ri[1],
                   jkz = rpk[2] - ri[2];
            min_image_cubic(jix, jiy, jiz, L);
            min_image_cubic(jkx, jky, jkz, L);
            const double rji = std::sqrt(jix * jix + jiy * jiy + jiz * jiz);
            const double rjk = std::sqrt(jkx * jkx + jky * jky + jkz * jkz);
            double cosv = (jix * jkx + jiy * jky + jiz * jkz) / (rji * rjk);
            cosv = std::max(-1.0, std::min(1.0, cosv));
            const double dtheta = std::acos(cosv) - kAngleTheta0;
            e.angle += ak * dtheta * dtheta;
        }

        // Dihedrals owned by i (= the i2 / second atom of i1-i2-i3-i4).
        for (int m = 0; m < num_dihedral[i]; ++m) {
            const int i1 = static_cast<int>(dihedral_atom(i, 3 * m + 0));
            const int i3 = static_cast<int>(dihedral_atom(i, 3 * m + 1));
            const int i4 = static_cast<int>(dihedral_atom(i, 3 * m + 2));
            double r1[3], r3[3], r4[3];
            pos(i1, r1);
            pos(i3, r3);
            pos(i4, r4); // r2 == ri (owner)
            double b1x = ri[0] - r1[0], b1y = ri[1] - r1[1],
                   b1z = ri[2] - r1[2];
            double b2x = r3[0] - ri[0], b2y = r3[1] - ri[1],
                   b2z = r3[2] - ri[2];
            double b3x = r4[0] - r3[0], b3y = r4[1] - r3[1],
                   b3z = r4[2] - r3[2];
            min_image_cubic(b1x, b1y, b1z, L);
            min_image_cubic(b2x, b2y, b2z, L);
            min_image_cubic(b3x, b3y, b3z, L);
            // n1 = b1 × b2, n2 = b2 × b3
            const double n1x = b1y * b2z - b1z * b2y;
            const double n1y = b1z * b2x - b1x * b2z;
            const double n1z = b1x * b2y - b1y * b2x;
            const double n2x = b2y * b3z - b2z * b3y;
            const double n2y = b2z * b3x - b2x * b3z;
            const double n2z = b2x * b3y - b2y * b3x;
            const double n1sq = n1x * n1x + n1y * n1y + n1z * n1z;
            const double n2sq = n2x * n2x + n2y * n2y + n2z * n2z;
            if (n1sq < 1e-20 || n2sq < 1e-20)
                continue;
            double cosp =
                (n1x * n2x + n1y * n2y + n1z * n2z) / std::sqrt(n1sq * n2sq);
            cosp = std::max(-1.0, std::min(1.0, cosp));
            double pw = 1.0, sum = 0.0;
            for (std::size_t k = 0; k < cn.size(); ++k) {
                sum += cn[k] * pw;
                pw *= cosp;
            }
            e.dihedral += sum;
        }
    }
    return e;
}

} // namespace

// ── Absolute bonded energies match brute force at 1 and 8 ranks ─────────────

TEST(BondedEnergyBruteForce, PerTermEnergyMatchesBruteForce) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 1 && world_size != 8) {
        GTEST_SKIP() << "1- or 8-rank specific; got world_size=" << world_size;
    }

    const double box_len = 24.0; // 8-rank slab = 12 Å ≫ halo cutoff 6 Å
    const double cutoff = 6.0;
    auto frame = make_axis_crossing_chains(box_len);

    // Pristine global copy for the oracle, before the pipeline partitions.
    const cpu::Frame ref = frame;
    const BondedEnergies bf = brute_force_bonded(ref);
    ASSERT_GT(std::abs(bf.bond), 1e-9) << "degenerate bond oracle";
    ASSERT_GT(std::abs(bf.angle), 1e-9) << "degenerate angle oracle";
    ASSERT_GT(std::abs(bf.dihedral), 1e-9) << "degenerate dihedral oracle";

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.0}); // dt = 0 → positions frozen
    sim.on(ATV::md::Phase::force)
        .add(cpu::BondHarmonic{{bond_K_eV()}, {kBondR0}})
        .add(cpu::AngleHarmonic{{angle_K_eV()}, {kAngleTheta0}})
        .add(cpu::DihedralMultiHarmonic{{dihedral_Cn_eV()}});
    // Bonded-only sim: no pair evaluator ⇒ halo is not auto-injected, so wire
    // the ghost exchange explicitly (cutoff must span the longest tuple).
    sim.on(ATV::md::Phase::pre_force).add(cpu::comm::HaloSend{cutoff}, -20);
    sim.on(ATV::md::Phase::post_force).add(cpu::comm::HaloRecv{}, -1000);
    sim.build();

    sim.run(frame, 1); // one frozen step → Reduce fills global per-term

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    auto check = [&](const char* term, double dist, double oracle) {
        const double rel = std::abs(dist - oracle) / std::abs(oracle);
        EXPECT_LT(rel, kEnergyRelativeTolerance)
            << "rank " << my_rank << " world=" << world_size << ": distributed "
            << term << " energy " << dist << " disagrees with brute force "
            << oracle << " (rel=" << rel
            << ") — a bonded tuple's cross-cut partner was dropped or "
               "mis-resolved";
    };
    check("bond", frame.meta.get(ATV::kEnergyBond), bf.bond);
    check("angle", frame.meta.get(ATV::kEnergyAngle), bf.angle);
    check("dihedral", frame.meta.get(ATV::kEnergyDihedral), bf.dihedral);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
