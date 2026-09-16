/// @file regressions/md/multihop-widebox-parity/reg_multihop-widebox-parity.cpp
/// @brief END-TO-END: a CHARGED, BONDED molecular lattice on a wide box with
///        TWO decomposed axes reproduces the serial minimum-image direct sum.
///
/// See REFERENCE.md. Public headers + link atomiverse only. oracle=literature:
/// the criterion is the *identity* "a domain decomposition is a partition of
/// the same minimum-image sum", evaluated in-file — no third-party engine is
/// imported, spawned or read at build or run time. perf_exempt: this slug
/// makes no performance claim (rank scaling and wall-clock belong to the
/// bench snapshot in `tools/bench_gaff_lammps/README.md`).
///
/// One binary, two modes:
///  - parent (default): re-launches itself through the configured mpiexec
///    (compile-time pin ATV_MPIEXEC) at np = 1 and at np = 8, then gates the
///    two child reports against each other and against the in-file oracle.
///  - child (`--child <out>`): MPI_Init, run 21 MD steps of a 32-molecule /
///    96-atom bent-triatomic lattice (PairLJCut + PairCoulCut, both with the
///    Amber special-bond weights, + BondHarmonic + AngleHarmonic; halo
///    AUTO-INJECTED by the MD preset), then — from an out-of-library
///    Component at end_of_step — allgather every owned row by tag, recompute
///    the total energy and every per-atom force as a flat O(N^2) min-image
///    direct sum that mirrors the special-bond exclusions, and report the
///    deviations, the per-atom forces, and the published swap-table hops.
///
/// Geometry rationale (why this case exists): 24 x 14 x 14 A box under a
/// forced 4x2x1 decomposition, cutoff 6.0 A with a 0.6 A skin, so the
/// auto-injected shell is max(1.1 * 6.0, 6.0 + 0.6) = 6.6 A and the TWO
/// decomposed axes land in two different branches at once:
///   * x: span 24/4 = 6.0 A < 6.6 A  => max_hops_x = ceil(6.6/6.0) = 2, i.e.
///     the shell CANNOT be filled by one face exchange; round 2 must forward
///     what round 1 received. Before comm-topology-02 this was refused
///     outright. 2 <= dims_x - 1 = 3 keeps it inside the minimum-image bound.
///   * y: span 14/2 = 7.0 A >= 6.6 A => a single hop, but 2 * 6.6 = 13.2 A >
///     7.0 A, so +y and -y are the SAME rank and every atom of the 6.2 A
///     overlap band ships through BOTH faces as two distinct periodic images.
///   * z: undecomposed => zero hops; periodicity is the pair kernel's min
///     image, legal because 6.0 A <= L_z/2 = 7.0 A (as on every axis, which
///     is also what makes the flat min-image oracle below the right one).
/// A 12 A y edge (span 6.0 A) would NOT be legal here: at dims_y = 2 the
/// deepest reachable hop is dims - 1 = 1, so a 6.6 A shell over 6.0 A slabs
/// is refused by `Halo::derive_hops` ("shell <= L - w_self = 6.0 A").
///
/// What the multi-hop shell has to carry here that the LJ-only
/// `comm-topology-multihop` slug does not: ghost CHARGES (PairCoulCut reads
/// `kAtom_charge` on ghost rows) and the BONDED extension (bond/angle tuples
/// whose partners are ghosts across a cut — half the molecules straddle an x
/// or y partition plane by construction), plus the special-bond exclusion
/// lists that must resolve through the same ghost rows.
/// 21 steps at the every=10 halo cadence cross >= 2 FULL rebuilds (borders)
/// with in-step forward refreshes in between; the reverse (post_force)
/// accumulation is gated separately by |sum_i F_i|.

#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

#ifndef ATV_MPIEXEC
#error "ATV_MPIEXEC must be set to the configured MPIEXEC_EXECUTABLE"
#endif

using namespace ATV;

namespace {

// ── Box and forced decomposition ────────────────────────────────────────
constexpr double kBoxX = 24.0; // [A]; 4 slabs of 6.0 A  -> 2 hops
constexpr double kBoxY = 14.0; // [A]; 2 slabs of 7.0 A  -> dual image
constexpr double kBoxZ = 14.0; // [A]; undecomposed
constexpr int kRanksX = 4, kRanksY = 2, kRanksZ = 1;

// ── Molecular lattice: conventional cell + body-centre basis ────────────
constexpr int kCellsX = 4, kCellsY = 2, kCellsZ = 2;
constexpr int kBasis = 2; // corner + body centre
constexpr int kNMol = kCellsX * kCellsY * kCellsZ * kBasis; // 32
constexpr int kAtomsPerMol = 3;                             // A - B - A
constexpr int kNAtoms = kNMol * kAtomsPerMol;               // 96

// Lattice phase: shifts the corner sublattice OFF the partition planes
// (x = 0/6/12/18 A, y = 0/7 A) by a fraction of a bond, so those molecules
// STRADDLE a cut — bond and angle partners land on the far rank — while no
// atom sits exactly on a plane where ownership would be a tie.
constexpr double kPhaseX = 0.13, kPhaseY = 0.07, kPhaseZ = 0.11; // [A]

// ── Force field (generic bent triatomic; not a fitted water model) ──────
constexpr double kEps = 0.0104; // LJ well depth [eV], one type for all sites
constexpr double kSigma = 3.0;  // LJ size [A]
constexpr double kCut = 6.0;    // pair cutoff (LJ and Coulomb) [A]
constexpr double kSkin = 0.6;   // Verlet skin [A]; shell = kCut + kSkin
constexpr int kNeighEvery = 10; // reneighbor / full halo rebuild cadence
constexpr double kQB = +0.70;   // centre charge [e]
constexpr double kQA = -0.35;   // each arm charge [e]; net 0 per molecule
constexpr double kMassB = 15.999, kMassA = 12.011; // [amu]
constexpr double kBondK = 12.0;                    // [eV/A^2]
constexpr double kBondR0 = 1.5;                    // [A]
constexpr double kAngleK = 3.0;                    // [eV/rad^2]
constexpr double kAngleTheta0 = 1.9548;            // 112 deg [rad]
// Amber/GAFF special-bond weights. In a 3-site A-B-A molecule EVERY
// intramolecular pair is 1-2 or 1-3, so the mirrored oracle rule is "skip
// same-molecule pairs"; there is no 1-4 pair for the 0.5 weight to touch.
constexpr double kSpecial12 = 0.0, kSpecial13 = 0.0, kSpecial14 = 0.5;
constexpr double kDt = 0.5; // [fs]
constexpr int kSteps = 21;

// ── Gates (hard-coded; see REFERENCE.md) ────────────────────────────────
constexpr double kEnergyRelTol = 1.0e-9;   // |E_engine - E_oracle| / |E_oracle|
constexpr double kForceTol = 1.0e-9;       // [eV/A], per component
constexpr double kNetForceTol = 1.0e-10;   // [eV/A]
constexpr double kDecompRelTol = 1.0e-10;  // |E(np8) - E(np1)| / |E(np1)|
constexpr double kDecompForceTol = 1.0e-9; // [eV/A], tag-aligned
constexpr int kMinHopsX = 2;               // ceil(6.6 / 6.0) — MULTI-hop
constexpr int kHopsY = 1;                  // dims 2: the deepest legal hop
constexpr int kMinRebuilds = 2;            // full borders passes over 21 steps

/// @brief 3x3 rotation applied to a molecule-local offset.
///
/// Rz(alpha) then Ry(beta), with per-molecule angles, so the 32 molecules
/// are NOT copies of one orientation: every tuple sits at a different angle
/// to the partition planes and no force cancels by lattice symmetry.
void rotate(double alpha, double beta, double& px, double& py, double& pz) {
    const double ca = std::cos(alpha), sa = std::sin(alpha);
    const double x1 = ca * px - sa * py;
    const double y1 = sa * px + ca * py;
    const double z1 = pz;
    const double cb = std::cos(beta), sb = std::sin(beta);
    px = cb * x1 + sb * z1;
    py = y1;
    pz = -sb * x1 + cb * z1;
}

/// @brief The system: 32 bent A-B-A molecules on a body-centred lattice.
///
/// Atom order is molecule-major (tags 3m+1, 3m+2, 3m+3 = A1, B, A2), which
/// is what lets the oracle mirror the special-bond exclusion as
/// "same molecule" without re-deriving the 1-2 / 1-3 lists.
///
/// Bond lengths and apex angles carry a small deterministic per-molecule
/// offset so every bonded term is off equilibrium (a pristine lattice would
/// make the bonded energies and forces vanish and the gates vacuous), and
/// molecule centroids — not the centre atom — sit on the lattice sites, so
/// no atom can land exactly on a partition plane.
///
/// @return Frame with positions [A], charges [e], masses [amu], bonds,
///         angles and the special-bond lists built.
cpu::Frame make_molecular_lattice() {
    cpu::Frame f(kNAtoms);
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    auto vz = f.span(kAtom_vz);
    auto mass = f.span(kAtom_mass);
    auto charge = f.span(kAtom_charge);
    auto type = f.span(kAtom_type);
    auto tag = f.span(kAtom_tag);

    const double cell_x = kBoxX / kCellsX; // 6.0 A = one x subdomain
    const double cell_y = kBoxY / kCellsY; // 7.0 A = one y subdomain
    const double cell_z = kBoxZ / kCellsZ; // 7.0 A

    int m = 0; // molecule index
    for (int ix = 0; ix < kCellsX; ++ix)
        for (int iy = 0; iy < kCellsY; ++iy)
            for (int iz = 0; iz < kCellsZ; ++iz)
                for (int b = 0; b < kBasis; ++b) {
                    const double half = (b == 1) ? 0.5 : 0.0;
                    const double cx = (ix + half) * cell_x + kPhaseX;
                    const double cy = (iy + half) * cell_y + kPhaseY;
                    const double cz = (iz + half) * cell_z + kPhaseZ;

                    const double mm = static_cast<double>(m);
                    const double r = kBondR0 + 0.08 * std::sin(1.7 * mm);
                    const double theta =
                        kAngleTheta0 + 0.12 * std::cos(1.3 * mm);
                    const double alpha = 0.7 * mm + 0.3;
                    const double beta = 0.37 * mm + 1.1;

                    // Molecule-local frame: B at origin, arms symmetric.
                    double px[3] = {r * std::sin(0.5 * theta), 0.0,
                                    -r * std::sin(0.5 * theta)};
                    double py[3] = {r * std::cos(0.5 * theta), 0.0,
                                    r * std::cos(0.5 * theta)};
                    double pz[3] = {0.0, 0.0, 0.0};
                    double gx = 0.0, gy = 0.0, gz = 0.0;
                    for (int a = 0; a < kAtomsPerMol; ++a) {
                        rotate(alpha, beta, px[a], py[a], pz[a]);
                        gx += px[a];
                        gy += py[a];
                        gz += pz[a];
                    }
                    gx /= kAtomsPerMol;
                    gy /= kAtomsPerMol;
                    gz /= kAtomsPerMol;

                    for (int a = 0; a < kAtomsPerMol; ++a) {
                        const int i = m * kAtomsPerMol + a;
                        x[i] = static_cast<StorageReal>(cx + px[a] - gx);
                        y[i] = static_cast<StorageReal>(cy + py[a] - gy);
                        z[i] = static_cast<StorageReal>(cz + pz[a] - gz);
                        vx[i] = vy[i] = vz[i] = 0.0;
                        mass[i] = static_cast<StorageReal>((a == 1) ? kMassB
                                                                    : kMassA);
                        charge[i] =
                            static_cast<StorageReal>((a == 1) ? kQB : kQA);
                        type[i] = 0;
                        tag[i] = i + 1;
                    }
                    ++m;
                }

    f.box = cpu::Box::orthorhombic(kBoxX, kBoxY, kBoxZ);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    // `Frame::add_bond` / `add_angle` take global TAGS (the owner row is
    // looked up through the tag index), not row indices — with tags 1-based
    // the two differ by one, and passing rows would silently shift every
    // tuple onto the neighbouring molecule.
    for (int mol = 0; mol < kNMol; ++mol) {
        const TagInt t_a1 = mol * kAtomsPerMol + 1;
        const TagInt t_b = mol * kAtomsPerMol + 2;
        const TagInt t_a2 = mol * kAtomsPerMol + 3;
        f.add_bond(t_a1, t_b, 0);
        f.add_bond(t_b, t_a2, 0);
        f.add_angle(t_a1, t_b, t_a2, 0);
    }
    f.build_special_list();
    return f;
}

/// @brief Minimum image on an orthorhombic cell, multi-wrap safe.
/// Legal as the oracle's convention because kCut = 6.0 A <= L_d/2 on every
/// axis (12 / 7 / 7 A): each pair has exactly one image inside the cutoff.
void min_image(double& dx, double& dy, double& dz) {
    while (dx > 0.5 * kBoxX)
        dx -= kBoxX;
    while (dx < -0.5 * kBoxX)
        dx += kBoxX;
    while (dy > 0.5 * kBoxY)
        dy -= kBoxY;
    while (dy < -0.5 * kBoxY)
        dy += kBoxY;
    while (dz > 0.5 * kBoxZ)
        dz -= kBoxZ;
    while (dz < -0.5 * kBoxZ)
        dz += kBoxZ;
}

/// @brief Per-term energies of the in-file oracle [eV].
struct OracleEnergy {
    double pair = 0.0;  ///< LJ (shifted) + cut Coulomb
    double bond = 0.0;  ///< sum K (r - r0)^2
    double angle = 0.0; ///< sum K (theta - theta0)^2
    double total() const { return pair + bond + angle; }
};

/// @brief The oracle, evaluated in this file: the flat O(N^2) minimum-image
///        direct sum of the SAME model Hamiltonian, plus its gradient.
///
/// Mirrors each engine kernel term by term:
///  * LJ 12-6 cutoff-shifted exactly as `cpu::PairLJCut` shifts it
///    (u(r) - u(r_c)); force is the unshifted analytic gradient.
///  * Coulomb `E = 14.399645 q_i q_j / r` for r < r_c, NOT shifted, exactly
///    as `cpu::PairCoulCut` computes it (`src/cpu/potentials/pair_coul_cut.cpp`).
///  * Special bonds: with weights (0, 0, 0.5) on a 3-site A-B-A molecule the
///    1-2 and 1-3 pairs — i.e. every same-molecule pair — are dropped from
///    BOTH pair styles; no 1-4 pair exists in this topology.
///  * Bond `K (r - r0)^2`, force -2K(r-r0)/r * d.
///  * Angle `K (theta - theta0)^2` with the LAMMPS harmonic-angle gradient
///    (a_coeff / c_coeff form), so the comparison is term-for-term, not
///    "some other harmonic angle".
///
/// @param x,y,z Assembled positions, index = tag - 1 [Angstrom].
/// @param fx,fy,fz Output forces, resized to N [eV/Angstrom].
/// @return Per-term potential energies [eV].
OracleEnergy brute_force(const std::vector<double>& x,
                         const std::vector<double>& y,
                         const std::vector<double>& z, std::vector<double>& fx,
                         std::vector<double>& fy, std::vector<double>& fz) {
    const std::size_t n = x.size();
    fx.assign(n, 0.0);
    fy.assign(n, 0.0);
    fz.assign(n, 0.0);
    OracleEnergy e;

    const double sigma6 = kSigma * kSigma * kSigma * kSigma * kSigma * kSigma;
    const double rc2 = kCut * kCut;
    const double rc6inv = (1.0 / (rc2 * rc2 * rc2)) * sigma6;
    const double v_shift = 4.0 * kEps * (rc6inv * rc6inv - rc6inv);
    const double coulomb = 14.399645; // eV*A/e^2, ATV::units::coulomb

    // ── Non-bonded: every unordered pair, same-molecule pairs excluded.
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t mol_i = i / kAtomsPerMol;
        const double qi = ((i % kAtomsPerMol) == 1) ? kQB : kQA;
        for (std::size_t j = i + 1; j < n; ++j) {
            if (j / kAtomsPerMol == mol_i)
                continue; // 1-2 / 1-3: special weight 0
            const double qj = ((j % kAtomsPerMol) == 1) ? kQB : kQA;
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rc2)
                continue;
            const double r2inv = 1.0 / r2;
            const double r6inv = r2inv * r2inv * r2inv * sigma6;
            e.pair += 4.0 * kEps * (r6inv * r6inv - r6inv) - v_shift;
            double fpair = 24.0 * kEps * r2inv * (2.0 * r6inv * r6inv - r6inv);

            const double r = std::sqrt(r2);
            const double ecoul = coulomb * qi * qj / r;
            e.pair += ecoul;
            fpair += ecoul * r2inv;

            fx[i] += fpair * dx;
            fy[i] += fpair * dy;
            fz[i] += fpair * dz;
            fx[j] -= fpair * dx;
            fy[j] -= fpair * dy;
            fz[j] -= fpair * dz;
        }
    }

    // ── Bonded: two bonds and one angle per molecule, by construction.
    for (int mol = 0; mol < kNMol; ++mol) {
        const std::size_t a1 = static_cast<std::size_t>(mol * kAtomsPerMol + 0);
        const std::size_t cb = static_cast<std::size_t>(mol * kAtomsPerMol + 1);
        const std::size_t a2 = static_cast<std::size_t>(mol * kAtomsPerMol + 2);

        const std::size_t bonds[2][2] = {{a1, cb}, {cb, a2}};
        for (const auto& b : bonds) {
            const std::size_t i = b[0], j = b[1];
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            min_image(dx, dy, dz);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double dr = r - kBondR0;
            e.bond += kBondK * dr * dr;
            const double fpair = -2.0 * kBondK * dr / r;
            fx[i] += fpair * dx;
            fy[i] += fpair * dy;
            fz[i] += fpair * dz;
            fx[j] -= fpair * dx;
            fy[j] -= fpair * dy;
            fz[j] -= fpair * dz;
        }

        // Angle a1 - cb - a2 (cb is the centre).
        double dx_ji = x[a1] - x[cb], dy_ji = y[a1] - y[cb],
               dz_ji = z[a1] - z[cb];
        double dx_jk = x[a2] - x[cb], dy_jk = y[a2] - y[cb],
               dz_jk = z[a2] - z[cb];
        min_image(dx_ji, dy_ji, dz_ji);
        min_image(dx_jk, dy_jk, dz_jk);
        const double r_ji =
            std::sqrt(dx_ji * dx_ji + dy_ji * dy_ji + dz_ji * dz_ji);
        const double r_jk =
            std::sqrt(dx_jk * dx_jk + dy_jk * dy_jk + dz_jk * dz_jk);
        double cos_theta =
            (dx_ji * dx_jk + dy_ji * dy_jk + dz_ji * dz_jk) / (r_ji * r_jk);
        cos_theta = std::fmax(-1.0, std::fmin(1.0, cos_theta));
        const double theta = std::acos(cos_theta);
        const double dtheta = theta - kAngleTheta0;
        e.angle += kAngleK * dtheta * dtheta;

        const double dV = 2.0 * kAngleK * dtheta;
        const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        if (sin_theta < 1.0e-10)
            continue;
        const double a_coeff = dV / (r_ji * sin_theta);
        const double c_coeff = dV / (r_jk * sin_theta);
        const double fi[3] = {
            a_coeff * (dx_jk / r_jk - cos_theta * dx_ji / r_ji),
            a_coeff * (dy_jk / r_jk - cos_theta * dy_ji / r_ji),
            a_coeff * (dz_jk / r_jk - cos_theta * dz_ji / r_ji)};
        const double fk[3] = {
            c_coeff * (dx_ji / r_ji - cos_theta * dx_jk / r_jk),
            c_coeff * (dy_ji / r_ji - cos_theta * dy_jk / r_jk),
            c_coeff * (dz_ji / r_ji - cos_theta * dz_jk / r_jk)};
        fx[a1] += fi[0];
        fy[a1] += fi[1];
        fz[a1] += fi[2];
        fx[a2] += fk[0];
        fy[a2] += fk[1];
        fz[a2] += fk[2];
        fx[cb] -= fi[0] + fk[0];
        fy[cb] -= fi[1] + fk[1];
        fz[cb] -= fi[2] + fk[2];
    }
    return e;
}

/// @brief Out-of-library end_of_step Component: assemble the global state and
///        confront the decomposed result with the in-file oracle.
///
/// Bound at end_of_step @ +900, i.e. AFTER `mpi::Reduce` (-50) has published
/// the globally summed energies and long after `CommBrick` (post_force
/// @ -1000) folded every ghost force back onto its owner. Every launch
/// overwrites the stored deviations, so the report carries the LAST step;
/// swap-table tallies are kept as extrema over all launches, so a single
/// degraded step cannot hide behind a final step that happens to be right.
class WideBoxChecker : public cpu::Component {
  public:
    std::string name() const override { return "widebox_parity_checker"; }

    void launch(const cpu::LaunchContext& ctx) override {
        const auto& f = ctx.frame;
        auto* dd = ctx.runtime.md.dd;
        const int nloc = dd ? dd->n_local : f.nlocal();
        const int ngh = dd ? dd->n_ghost : 0;

        if (dd && dd->ghost_shell_width > 0.0) {
            const auto need = cpu::comm::Halo::derive_hops(
                *dd, ctx.frame.box, dd->ghost_shell_width);
            const int hx = std::max(need.minus[0], need.plus[0]);
            const int hy = std::max(need.minus[1], need.plus[1]);
            if (hops_x_min_ < 0 || hx < hops_x_min_)
                hops_x_min_ = hx;
            if (hops_y_max_ < 0 || hy > hops_y_max_)
                hops_y_max_ = hy;
            if (round_x_min_ < 0 || hx < round_x_min_)
                round_x_min_ = hx;
        }
        if (ngh > 0 && (nghost_min_ < 0 || ngh < nghost_min_))
            nghost_min_ = ngh;
        if (dd)
            rebuilds_ = static_cast<unsigned long long>(dd->halo_generation);

        // ── Assemble the global configuration from every rank's OWNED rows.
        const auto tag_col = f.span(kAtom_tag);
        const auto x_col = f.span(kAtom_x);
        const auto y_col = f.span(kAtom_y);
        const auto z_col = f.span(kAtom_z);
        const auto fx_col = f.span(kAtom_fx);
        const auto fy_col = f.span(kAtom_fy);
        const auto fz_col = f.span(kAtom_fz);

        int size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        std::vector<int> counts(static_cast<std::size_t>(size), 0);
        MPI_Allgather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      MPI_COMM_WORLD);
        std::vector<int> displs(static_cast<std::size_t>(size), 0);
        std::vector<int> displs6(static_cast<std::size_t>(size), 0);
        std::vector<int> counts6(static_cast<std::size_t>(size), 0);
        int total = 0;
        for (int r = 0; r < size; ++r) {
            displs[static_cast<std::size_t>(r)] = total;
            displs6[static_cast<std::size_t>(r)] = 6 * total;
            counts6[static_cast<std::size_t>(r)] =
                6 * counts[static_cast<std::size_t>(r)];
            total += counts[static_cast<std::size_t>(r)];
        }
        n_gathered_ = total;

        std::vector<long long> my_tags(static_cast<std::size_t>(nloc));
        std::vector<double> my_rows(static_cast<std::size_t>(6 * nloc));
        for (int i = 0; i < nloc; ++i) {
            const std::size_t u = static_cast<std::size_t>(i);
            my_tags[u] = static_cast<long long>(tag_col[i]);
            my_rows[6 * u + 0] = static_cast<double>(x_col[i]);
            my_rows[6 * u + 1] = static_cast<double>(y_col[i]);
            my_rows[6 * u + 2] = static_cast<double>(z_col[i]);
            my_rows[6 * u + 3] = static_cast<double>(fx_col[i]);
            my_rows[6 * u + 4] = static_cast<double>(fy_col[i]);
            my_rows[6 * u + 5] = static_cast<double>(fz_col[i]);
        }
        std::vector<long long> all_tags(static_cast<std::size_t>(total));
        std::vector<double> all_rows(static_cast<std::size_t>(6 * total));
        MPI_Allgatherv(my_tags.data(), nloc, MPI_LONG_LONG, all_tags.data(),
                       counts.data(), displs.data(), MPI_LONG_LONG,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(my_rows.data(), 6 * nloc, MPI_DOUBLE, all_rows.data(),
                       counts6.data(), displs6.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        if (total != kNAtoms) {
            ++assembly_errors_;
            ++launches_;
            return;
        }
        const std::size_t n = static_cast<std::size_t>(kNAtoms);
        std::vector<double> px(n, 0.0), py(n, 0.0), pz(n, 0.0);
        std::vector<int> seen(n, 0);
        force_.assign(3 * n, 0.0);
        for (int i = 0; i < total; ++i) {
            const long long t = all_tags[static_cast<std::size_t>(i)];
            if (t < 1 || t > kNAtoms) {
                ++assembly_errors_;
                ++launches_;
                return;
            }
            const std::size_t u = static_cast<std::size_t>(t - 1);
            ++seen[u];
            const std::size_t s = 6 * static_cast<std::size_t>(i);
            px[u] = all_rows[s + 0];
            py[u] = all_rows[s + 1];
            pz[u] = all_rows[s + 2];
            force_[3 * u + 0] = all_rows[s + 3];
            force_[3 * u + 1] = all_rows[s + 4];
            force_[3 * u + 2] = all_rows[s + 5];
        }
        for (std::size_t u = 0; u < n; ++u)
            if (seen[u] != 1)
                ++assembly_errors_;

        // ── The oracle and the deviations.
        std::vector<double> bx, by, bz;
        const OracleEnergy oe = brute_force(px, py, pz, bx, by, bz);
        e_oracle_ = oe.total();
        e_oracle_pair_ = oe.pair;
        e_oracle_bond_ = oe.bond;
        e_oracle_angle_ = oe.angle;
        e_dd_ = f.meta.get(kEnergy);
        e_dd_pair_ = f.meta.get(kEnergyPair);
        e_dd_bond_ = f.meta.get(kEnergyBond);
        e_dd_angle_ = f.meta.get(kEnergyAngle);

        double max_df = 0.0, max_f = 0.0;
        double sfx = 0.0, sfy = 0.0, sfz = 0.0;
        for (std::size_t u = 0; u < n; ++u) {
            max_df = std::fmax(max_df, std::fabs(force_[3 * u + 0] - bx[u]));
            max_df = std::fmax(max_df, std::fabs(force_[3 * u + 1] - by[u]));
            max_df = std::fmax(max_df, std::fabs(force_[3 * u + 2] - bz[u]));
            sfx += force_[3 * u + 0];
            sfy += force_[3 * u + 1];
            sfz += force_[3 * u + 2];
            max_f = std::fmax(max_f, std::sqrt(bx[u] * bx[u] + by[u] * by[u]
                                               + bz[u] * bz[u]));
        }
        max_dforce_ = max_df;
        net_force_ = std::sqrt(sfx * sfx + sfy * sfy + sfz * sfz);
        max_force_magnitude_ = max_f;
        ++launches_;
    }

    std::vector<double> force_;        ///< engine forces by tag, xyz [eV/A]
    double e_dd_ = 0.0;                ///< reduced engine total energy [eV]
    double e_dd_pair_ = 0.0;           ///< engine LJ + Coulomb share [eV]
    double e_dd_bond_ = 0.0;           ///< engine bond share [eV]
    double e_dd_angle_ = 0.0;          ///< engine angle share [eV]
    double e_oracle_ = 0.0;            ///< in-file direct sum, total [eV]
    double e_oracle_pair_ = 0.0;       ///< in-file direct sum, non-bonded [eV]
    double e_oracle_bond_ = 0.0;       ///< in-file direct sum, bonds [eV]
    double e_oracle_angle_ = 0.0;      ///< in-file direct sum, angles [eV]
    double max_dforce_ = -1.0;         ///< max_i |F_i,dd - F_i,oracle| [eV/A]
    double net_force_ = -1.0;          ///< ||sum_i F_i,dd|| [eV/A]
    double max_force_magnitude_ = 0.0; ///< max_i |F_i,oracle| [eV/A]
    int hops_x_min_ = -1;              ///< min Halo::max_hops(0) over steps
    int hops_y_max_ = -1;              ///< max Halo::max_hops(1) over steps
    int round_x_min_ = -1;             ///< min deepest transmitting x round
    long long nghost_min_ = -1;        ///< min dd->n_ghost over launches
    long long recv_mismatch_ = 0;      ///< launches where sum nrecv != n_ghost
    long long assembly_errors_ = 0;    ///< lost / duplicated owned tags
    unsigned long long rebuilds_ = 0;  ///< dd->halo_generation at the last step
    int n_gathered_ = -1;              ///< owned rows assembled, must be 96
    int launches_ = 0;
};

int run_child(const char* out_path) {
    // Deterministic 4x2x1 split: 6.0 A x-slabs and 7.0 A y-slabs against a
    // 6.6 A shell. Ignored at np = 1 (the override is only honoured when
    // dx*dy*dz equals the world size), which is exactly what makes the np = 1
    // leg a serial control rather than a second decomposed run.
    ::setenv("ATV_FORCE_DIMS", "4,2,1", 1);
    MPI_Init(nullptr, nullptr);

    auto frame = make_molecular_lattice();

    cpu::md::MD sim;
    // `delay 0 check no`: rebuild unconditionally every kNeighEvery steps.
    // Since lammps-comm-brick-03 the halo borders pass is gated on the same
    // nflag as the neighbor rebuild (LAMMPS Verlet), so the ">= 2 full
    // rebuilds" gate below needs the unconditional cadence — the default
    // `check yes` would skip every rebuild on this barely-moving lattice.
    sim.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin, kNeighEvery,
                                             /*delay=*/0, /*check=*/false});
    sim.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut}.set_special_bonds(
        kSpecial12, kSpecial13, kSpecial14));
    sim.set_evaluator(cpu::PairCoulCut{kCut}.set_special_bonds(
        kSpecial12, kSpecial13, kSpecial14));
    sim.set_evaluator(cpu::BondHarmonic{{kBondK}, {kBondR0}});
    sim.set_evaluator(cpu::AngleHarmonic{{kAngleK}, {kAngleTheta0}});
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.enable_timing(false);

    WideBoxChecker checker;
    sim.on(ATV::md::Phase::end_of_step).add_ref(checker, +900);

    sim.build();
    sim.run(frame, kSteps);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rc = 0;
    if (rank == 0) {
        std::FILE* out = std::fopen(out_path, "w");
        if (!out) {
            std::printf("FAIL: child cannot open %s\n", out_path);
            rc = 1;
        } else {
            std::fprintf(out,
                         "%.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e "
                         "%.17e %.17e %.17e %d %d %d %lld %lld %lld %llu %d "
                         "%d\n",
                         checker.e_dd_, checker.e_oracle_, checker.e_dd_pair_,
                         checker.e_oracle_pair_, checker.e_dd_bond_,
                         checker.e_oracle_bond_, checker.e_dd_angle_,
                         checker.e_oracle_angle_, checker.max_dforce_,
                         checker.net_force_, checker.max_force_magnitude_,
                         checker.hops_x_min_, checker.hops_y_max_,
                         checker.round_x_min_, checker.nghost_min_,
                         checker.recv_mismatch_, checker.assembly_errors_,
                         checker.rebuilds_, checker.n_gathered_,
                         checker.launches_);
            for (int i = 0; i < kNAtoms; ++i) {
                const std::size_t u = static_cast<std::size_t>(i);
                const double fxv = (checker.force_.size() >= 3 * u + 3)
                                       ? checker.force_[3 * u + 0]
                                       : 0.0;
                const double fyv = (checker.force_.size() >= 3 * u + 3)
                                       ? checker.force_[3 * u + 1]
                                       : 0.0;
                const double fzv = (checker.force_.size() >= 3 * u + 3)
                                       ? checker.force_[3 * u + 2]
                                       : 0.0;
                std::fprintf(out, "%.17e %.17e %.17e\n", fxv, fyv, fzv);
            }
            std::fclose(out);
        }
    }
    MPI_Finalize();
    return rc;
}

/// @brief One child report; every field is a measured quantity, never a
///        default that could pass a gate by accident.
struct Report {
    double e_dd = 0.0, e_oracle = 0.0;
    double e_dd_pair = 0.0, e_oracle_pair = 0.0;
    double e_dd_bond = 0.0, e_oracle_bond = 0.0;
    double e_dd_angle = 0.0, e_oracle_angle = 0.0;
    double max_dforce = -1.0, net_force = -1.0, max_force = -1.0;
    int hops_x_min = -1, hops_y_max = -1, round_x_min = -1;
    long long nghost_min = -1;
    long long recv_mismatch = -1, assembly_errors = -1;
    unsigned long long rebuilds = 0;
    int n_gathered = -1, launches = -1;
    std::vector<double> force; ///< 3 * kNAtoms, by tag [eV/A]
};

/// @brief Re-launch this binary under mpiexec at `np` ranks and read back the
///        child's report.
/// @param self Absolute path of this executable.
/// @param np Rank count.
/// @param out Parsed report.
/// @return true on a clean child exit with a well-formed report.
bool run_under_mpiexec(const std::string& self, int np, Report& out) {
    const std::string path =
        "/tmp/atv-multihop-widebox-parity-np" + std::to_string(np) + "-"
        + std::to_string(static_cast<long>(::getpid())) + ".txt";
    const std::string cmd = std::string(ATV_MPIEXEC) + " -n "
                            + std::to_string(np) + " \"" + self
                            + "\" --child \"" + path + "\"";
    std::printf("[multihop-widebox-parity] launching: %s\n", cmd.c_str());
    std::fflush(stdout);
    if (std::system(cmd.c_str()) != 0) {
        std::printf("FAIL: np=%d child exited nonzero\n", np);
        return false;
    }
    std::FILE* in = std::fopen(path.c_str(), "r");
    if (!in) {
        std::printf("FAIL: missing np=%d child report %s\n", np, path.c_str());
        return false;
    }
    const int got = std::fscanf(
        in,
        "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %d %d %lld %lld %lld "
        "%llu %d %d",
        &out.e_dd, &out.e_oracle, &out.e_dd_pair, &out.e_oracle_pair,
        &out.e_dd_bond, &out.e_oracle_bond, &out.e_dd_angle,
        &out.e_oracle_angle, &out.max_dforce, &out.net_force, &out.max_force,
        &out.hops_x_min, &out.hops_y_max, &out.round_x_min, &out.nghost_min,
        &out.recv_mismatch, &out.assembly_errors, &out.rebuilds,
        &out.n_gathered, &out.launches);
    int force_read = 0;
    out.force.assign(3 * static_cast<std::size_t>(kNAtoms), 0.0);
    for (int i = 0; i < kNAtoms; ++i) {
        const std::size_t u = static_cast<std::size_t>(i);
        force_read += std::fscanf(in, "%lf %lf %lf", &out.force[3 * u + 0],
                                  &out.force[3 * u + 1], &out.force[3 * u + 2]);
    }
    std::fclose(in);
    std::remove(path.c_str());
    if (got != 20 || force_read != 3 * kNAtoms) {
        std::printf("FAIL: malformed np=%d child report (%d header fields, "
                    "%d force fields)\n",
                    np, got, force_read);
        return false;
    }
    return true;
}

/// @brief Print one gate line and tally it.
/// @return 0 when the gate passes, 1 when it fails.
int gate(const char* label, double got, double tol, bool ok) {
    std::printf("  %-38s got %.3e  want <= %.0e  -> %s\n", label, got, tol,
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--child") == 0)
        return run_child(argv[2]);

    char self_buf[4096];
    const ssize_t n =
        ::readlink("/proc/self/exe", self_buf, sizeof(self_buf) - 1);
    if (n <= 0) {
        std::printf("FAIL: cannot resolve /proc/self/exe\n");
        return 1;
    }
    self_buf[n] = '\0';
    const std::string self(self_buf);

    ::setenv("OMP_NUM_THREADS", "1", 1);
    Report serial, dd;
    if (!run_under_mpiexec(self, 1, serial))
        return 1;
    if (!run_under_mpiexec(self, kRanksX * kRanksY * kRanksZ, dd))
        return 1;

    int failures = 0;
    std::printf("[multihop-widebox-parity] stage 1 gates "
                "(np=1 vs in-file O(N^2) min-image direct sum; "
                "%d atoms / %d molecules, %d steps)\n",
                kNAtoms, kNMol, kSteps);

    // (0) The assembly itself must be sound, or every gate below is vacuous.
    const bool ok_assembly1 = serial.n_gathered == kNAtoms
                              && serial.assembly_errors == 0
                              && serial.launches >= kSteps;
    std::printf("  atoms / errors / launches              got %d / %lld / %d  "
                "want %d / 0 / >=%d -> %s\n",
                serial.n_gathered, serial.assembly_errors, serial.launches,
                kNAtoms, kSteps, ok_assembly1 ? "PASS" : "FAIL");
    if (!ok_assembly1)
        ++failures;

    // (1) np = 1 energy IS the direct sum, term by term. The per-term rows
    //     are what says the special-bond exclusions were mirrored: an
    //     un-excluded 1-2 LJ pair at ~1.5 A would move `pair` by >1e3 eV.
    const double rel_e1 =
        std::fabs(serial.e_dd - serial.e_oracle) / std::fabs(serial.e_oracle);
    failures += gate("|E(np1) - E_oracle| / |E_oracle|", rel_e1, kEnergyRelTol,
                     rel_e1 <= kEnergyRelTol);
    const double rel_pair = std::fabs(serial.e_dd_pair - serial.e_oracle_pair)
                            / std::fabs(serial.e_oracle_pair);
    failures += gate("  pair (LJ + coul) relative", rel_pair, kEnergyRelTol,
                     rel_pair <= kEnergyRelTol);
    const double rel_bond = std::fabs(serial.e_dd_bond - serial.e_oracle_bond)
                            / std::fabs(serial.e_oracle_bond);
    failures += gate("  bond relative", rel_bond, kEnergyRelTol,
                     rel_bond <= kEnergyRelTol);
    const double rel_angle =
        std::fabs(serial.e_dd_angle - serial.e_oracle_angle)
        / std::fabs(serial.e_oracle_angle);
    failures += gate("  angle relative", rel_angle, kEnergyRelTol,
                     rel_angle <= kEnergyRelTol);

    // (2) np = 1 per-atom forces are the direct sum's gradient.
    failures += gate(
        "max_i |F_i(np1) - F_i,oracle| [eV/A]", serial.max_dforce, kForceTol,
        serial.max_dforce >= 0.0 && serial.max_dforce <= kForceTol);

    std::printf("[multihop-widebox-parity] stage 2 gates "
                "(np=%d, ATV_FORCE_DIMS=%d,%d,%d vs np=1)\n",
                kRanksX * kRanksY * kRanksZ, kRanksX, kRanksY, kRanksZ);

    const bool ok_assembly8 = dd.n_gathered == kNAtoms
                              && dd.assembly_errors == 0
                              && dd.launches >= kSteps;
    std::printf("  atoms / errors / launches              got %d / %lld / %d  "
                "want %d / 0 / >=%d -> %s\n",
                dd.n_gathered, dd.assembly_errors, dd.launches, kNAtoms, kSteps,
                ok_assembly8 ? "PASS" : "FAIL");
    if (!ok_assembly8)
        ++failures;

    // (3) Decomposition invariance of the total energy.
    const double rel_np =
        std::fabs(dd.e_dd - serial.e_dd) / std::fabs(serial.e_dd);
    failures += gate("|E(np8) - E(np1)| / |E(np1)|", rel_np, kDecompRelTol,
                     rel_np <= kDecompRelTol);

    // (4) Decomposition invariance of every per-atom force, tag-aligned.
    double max_df_np = 0.0;
    int worst_tag = -1;
    for (int i = 0; i < 3 * kNAtoms; ++i) {
        const double d = std::fabs(dd.force[static_cast<std::size_t>(i)]
                                   - serial.force[static_cast<std::size_t>(i)]);
        if (d > max_df_np) {
            max_df_np = d;
            worst_tag = i / 3 + 1;
        }
    }
    failures += gate("max |F_i(np8) - F_i(np1)| [eV/A]", max_df_np,
                     kDecompForceTol, max_df_np <= kDecompForceTol);
    if (max_df_np > kDecompForceTol)
        std::printf("    worst tag %d\n", worst_tag);

    // (5) np = 8 also stands on its own against the oracle.
    const double rel_e8 =
        std::fabs(dd.e_dd - dd.e_oracle) / std::fabs(dd.e_oracle);
    failures += gate("|E(np8) - E_oracle| / |E_oracle|", rel_e8, kEnergyRelTol,
                     rel_e8 <= kEnergyRelTol);
    failures +=
        gate("max_i |F_i(np8) - F_i,oracle| [eV/A]", dd.max_dforce, kForceTol,
             dd.max_dforce >= 0.0 && dd.max_dforce <= kForceTol);

    // (6) Newton's third law across the cuts: the reverse exchange ran
    //     exactly once at every hop depth (a dropped deepest-round reverse
    //     leaks momentum; a doubled one pumps it).
    failures += gate("||sum_i F_i|| (np8) [eV/A]", dd.net_force, kNetForceTol,
                     dd.net_force >= 0.0 && dd.net_force <= kNetForceTol);

    // (7) THE point of this case: the shell really is multi-hop on x while y
    //     stays at its single legal hop.
    const bool ok_hops = dd.hops_x_min >= kMinHopsX
                         && dd.round_x_min >= kMinHopsX
                         && dd.hops_y_max == kHopsY;
    std::printf("  x max_hops / deepest round / y hops    got %d / %d / %d  "
                "want >=%d / >=%d / %d -> %s\n",
                dd.hops_x_min, dd.round_x_min, dd.hops_y_max, kMinHopsX,
                kMinHopsX, kHopsY, ok_hops ? "PASS" : "FAIL");
    if (!ok_hops)
        ++failures;

    // (8) Slot bookkeeping and rebuild coverage.
    const bool ok_recv = dd.recv_mismatch == 0 && dd.nghost_min > 0;
    std::printf("  n_ghost (min) / recv mismatches        got %lld / %lld  "
                "want > 0 / 0       -> %s\n",
                dd.nghost_min, dd.recv_mismatch, ok_recv ? "PASS" : "FAIL");
    if (!ok_recv)
        ++failures;
    const bool ok_rebuilds =
        dd.rebuilds >= static_cast<unsigned long long>(kMinRebuilds);
    std::printf("  full halo rebuilds                     got %llu  "
                "want >= %d          -> %s\n",
                dd.rebuilds, kMinRebuilds, ok_rebuilds ? "PASS" : "FAIL");
    if (!ok_rebuilds)
        ++failures;

    // (9) Non-vacuous: the lattice really carries force, and every term of
    //     the Hamiltonian is actually populated.
    const bool ok_live = dd.max_force > 1.0e-4
                         && std::fabs(serial.e_oracle_pair) > 1.0e-4
                         && std::fabs(serial.e_oracle_bond) > 1.0e-4
                         && std::fabs(serial.e_oracle_angle) > 1.0e-4;
    std::printf("  max|F| / |E_pair| / |E_bond| / |E_ang| got %.3e / %.3e / "
                "%.3e / %.3e  want > 1e-04 -> %s\n",
                dd.max_force, std::fabs(serial.e_oracle_pair),
                std::fabs(serial.e_oracle_bond),
                std::fabs(serial.e_oracle_angle), ok_live ? "PASS" : "FAIL");
    if (!ok_live)
        ++failures;

    std::printf("[multihop-widebox-parity] E(np=1) = %.16f eV "
                "(pair %.16f, bond %.16f, angle %.16f)\n",
                serial.e_dd, serial.e_dd_pair, serial.e_dd_bond,
                serial.e_dd_angle);
    std::printf("[multihop-widebox-parity] E(np=8) = %.16f eV, "
                "E_oracle(np=1) = %.16f eV\n",
                dd.e_dd, serial.e_oracle);
    if (failures == 0)
        std::printf("PASS\n");
    else
        std::printf("FAIL (%d gate%s)\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
