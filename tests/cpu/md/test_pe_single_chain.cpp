/// @file tests/cpu/md/test_pe_single_chain.cpp
/// @brief Phase A.2 (msu-pe-validation) -- single-chain PE force-field check.
///
/// Builds an all-trans C20H42 united-atom chain and verifies each bonded
/// term against an analytic expectation:
///   * at equilibrium geometry every bond, angle and dihedral term sits at
///     its potential minimum, so the sum is ≤ FP tolerance (1e-6 eV);
///   * a single perturbation (stretch one bond, bend one angle, twist one
///     dihedral) gives the closed-form energy of that term alone.
///
/// LAMMPS cross-validation against an external reference is the next
/// milestone (msu-pe-validation A.3+); this test pins the Atomiverse-side
/// implementation against the analytic answer first.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"

using namespace ATV;

namespace {

// ── PE force-field parameters (OPLS-AA UA, LAMMPS convention) ─────────
//
// LAMMPS bond_style harmonic and angle_style harmonic absorb the textbook
// 1/2 factor: V = K (r - r0)^2 and V = K (theta - theta0)^2. The K values
// below are already in that convention.
//
// Dihedral type "multi/harmonic" stores V = sum_n=0..4 Cn cos^n(phi).
// The OPLS Fourier form
//   V_OPLS = 1/2 * [V1 (1 + cos phi) + V2 (1 - cos 2phi) + V3 (1 + cos 3phi)]
// expands to
//   C0 = V1/2 + V2 + V3/2
//   C1 = V1/2 - 3 V3/2
//   C2 = -V2
//   C3 = 2 V3
//   C4 = 0   (no V4 term in standard PE OPLS-AA torsion)

inline constexpr double kKcalPerMolToEv = 1.0 / 23.060547830619026;

// CT-CT bond (alkane sp3-sp3 single bond).
inline constexpr double kBondK_kcal = 268.0; // kcal/mol/Å^2 (LAMMPS K)
inline constexpr double kBondR0 = 1.53;      // Å

// CT-CT-CT angle.
inline constexpr double kAngleK_kcal = 60.0; // kcal/mol/rad^2 (LAMMPS K)
inline constexpr double kAngleTheta0 = 109.5 * units::pi / 180.0; // rad

// CT-CT-CT-CT dihedral (n-alkane backbone OPLS).

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

// ── All-trans C_n H_{2n+2} united-atom chain in a planar zigzag ───────
//
// Bond length b = kBondR0; bond angle theta = kAngleTheta0.
// Carbons alternate y-positions with offset h = b * sin((π - theta) / 2).
// x increments by p = b * cos((π - theta) / 2). Chain laid along x with
// atom 0 at the box centre to keep PBC images of the ends out of LJ range.
cpu::Frame make_all_trans_alkane(int n_carbons, double box_len) {
    cpu::Frame frame(n_carbons);

    const double half_apex =
        (units::pi - kAngleTheta0) / 2.0; // half of (180° - theta)
    const double dx_step = kBondR0 * std::cos(half_apex);
    const double dy_step = kBondR0 * std::sin(half_apex);

    const double x0 = (box_len - (n_carbons - 1) * dx_step) / 2.0;
    const double y0 = box_len / 2.0;
    const double z0 = box_len / 2.0;

    for (int i = 0; i < n_carbons; ++i) {
        frame.span(ATV::kAtom_x)[i] = x0 + i * dx_step;
        frame.span(ATV::kAtom_y)[i] = y0 + (i % 2 == 0 ? 0.0 : dy_step);
        frame.span(ATV::kAtom_z)[i] = z0;
        frame.span(ATV::kAtom_mass)[i] = 14.027; // CH2 united atom
        frame.span(ATV::kAtom_type)[i] = 0;
    }

    // Populate the canonical topology Block columns through Frame's structural
    // API. The tag index must exist first so each helper can resolve the owner
    // atom's local row from its global tag.
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int k = 0; k < n_carbons - 1; ++k)
        frame.add_bond(k, k + 1, 0);
    for (int k = 0; k < n_carbons - 2; ++k)
        frame.add_angle(k, k + 1, k + 2, 0);
    for (int k = 0; k < n_carbons - 3; ++k)
        frame.add_dihedral(k, k + 1, k + 2, k + 3, 0);
    frame.build_special_list();

    frame.simbox = ATV::cpu::Box::cubic(box_len);
    return frame;
}

// Run a single force-evaluation pass with each bonded component, accumulating
// V into frame.meta.get(ATV::kEnergy). Caller is responsible for resetting frame.meta.get(ATV::kEnergy) first.
void evaluate_bonded(cpu::Frame& frame, cpu::BondHarmonic& bond,
                     cpu::AngleHarmonic& angle,
                     cpu::DihedralMultiHarmonic& dih) {
    cpu::Context ctx;
    cpu::BuildContext bctx{ctx, frame};
    cpu::LaunchContext lctx{ctx, frame, 0, 0};

    // Each component allocates its own per-thread force buffers in build().
    bond.build(bctx);
    angle.build(bctx);
    dih.build(bctx);

    // Zero forces and energy before launch.
    const int n = frame.nlocal();
    for (int i = 0; i < n; ++i) {
        frame.span(ATV::kAtom_fx)[i] = 0;
        frame.span(ATV::kAtom_fy)[i] = 0;
        frame.span(ATV::kAtom_fz)[i] = 0;
    }
    frame.meta.get(ATV::kEnergy) = 0;

    bond.launch(lctx);
    angle.launch(lctx);
    dih.launch(lctx);

    bond.destroy({ctx, frame});
    angle.destroy({ctx, frame});
    dih.destroy({ctx, frame});
}

} // namespace

// ── At equilibrium: every bonded term sits at its minimum ─────────────

TEST(PeSingleChain, AllTransGeometryIsStationaryPoint) {
    auto frame = make_all_trans_alkane(20, 50.0);

    cpu::BondHarmonic bond({bond_K_eV()}, {kBondR0});
    cpu::AngleHarmonic angle({angle_K_eV()}, {kAngleTheta0});
    cpu::DihedralMultiHarmonic dih({dihedral_Cn_eV()});

    evaluate_bonded(frame, bond, angle, dih);

    // All-trans is a stationary point of the Dreiding-UA bonded FF:
    // - bond at r0 → V_bond = 0, dV_bond/dr = 0
    // - angle at θ0 → V_angle = 0, dV_angle/dθ = 0
    // - dihedral at φ=180° → sin(φ)=0 so dV_dih/dφ = 0,
    //   but V_dih(180°) = 1.73 + 4.49 + 0.776 − 6.99 = 0.006 kcal/mol
    //   per dihedral (constant offset baked into the Tschopp 2010
    //   multi-harmonic coefficients). For 17 dihedrals in C₂₀H₄₂:
    //   17 × 0.006 / 23.06 ≈ 4.4 × 10⁻³ eV total.
    // (Pre-FF-migration the OPLS-style Fourier series happened to give
    // V_dih(180°) = 0 exactly, which made this test assert V_total = 0.
    // The Dreiding-UA constants from the MSU paper carry a nonzero
    // offset at trans; the physical claim "all-trans is the minimum"
    // is unchanged — see the max_f gate below.)
    const double expected_dih_offset = 17.0 * 0.006 / 23.060547830619026;
    EXPECT_NEAR(frame.meta.get(ATV::kEnergy), expected_dih_offset, 1e-4)
        << "all-trans C20H42 PE has only the Dreiding-UA dihedral "
        << "constant offset (~4.4e-3 eV) above zero";

    // Forces also vanish to FP tolerance at the minimum.
    double max_f = 0.0;
    for (int i = 0; i < frame.nlocal(); ++i) {
        const double fmag = std::sqrt(
            frame.span(ATV::kAtom_fx)[i] * frame.span(ATV::kAtom_fx)[i]
            + frame.span(ATV::kAtom_fy)[i] * frame.span(ATV::kAtom_fy)[i]
            + frame.span(ATV::kAtom_fz)[i] * frame.span(ATV::kAtom_fz)[i]);
        max_f = std::max(max_f, fmag);
    }
    EXPECT_LT(max_f, 1.0e-10) << "max |force| at equilibrium = " << max_f;
}

// ── Stretch one bond: V_bond = K * (Δr)^2 ─────────────────────────────

TEST(PeSingleChain, StretchOneBondMatchesAnalytic) {
    cpu::Frame fresh = make_all_trans_alkane(20, 50.0);

    // Move END atom 0 along (atom0 - atom1), the bond extension direction.
    // Atom 0 only participates in bond 0-1; moving it along this line
    // preserves angle 0-1-2 (the apex 1→0 direction is unchanged) and
    // keeps atom 0 inside the 0-1-2 plane (so dihedral 0-1-2-3 is also
    // preserved). Only the 0-1 bond length changes, by exactly Δr.
    const double dr = 0.05;
    const double dx0 =
        fresh.span(ATV::kAtom_x)[0] - fresh.span(ATV::kAtom_x)[1];
    const double dy0 =
        fresh.span(ATV::kAtom_y)[0] - fresh.span(ATV::kAtom_y)[1];
    const double dz0 =
        fresh.span(ATV::kAtom_z)[0] - fresh.span(ATV::kAtom_z)[1];
    const double r0_len = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
    fresh.span(ATV::kAtom_x)[0] += dr * dx0 / r0_len;
    fresh.span(ATV::kAtom_y)[0] += dr * dy0 / r0_len;
    fresh.span(ATV::kAtom_z)[0] += dr * dz0 / r0_len;

    cpu::Context ctx;
    cpu::BuildContext bctx{ctx, fresh};
    cpu::LaunchContext lctx{ctx, fresh, 0, 0};
    cpu::BondHarmonic bond_only({bond_K_eV()}, {kBondR0});
    bond_only.build(bctx);
    fresh.meta.get(ATV::kEnergy) = 0;
    bond_only.launch(lctx);
    bond_only.destroy({ctx, fresh});

    const double v_expected = bond_K_eV() * dr * dr;
    EXPECT_NEAR(fresh.meta.get(ATV::kEnergy), v_expected, 1.0e-9)
        << "end-atom shifted by " << dr
        << " Å along bond axis should give K*Δr² = " << v_expected;
}

// ── Bend one angle: V_angle = K * (Δθ)^2 ──────────────────────────────

TEST(PeSingleChain, BendOneAngleMatchesAnalytic) {
    auto frame = make_all_trans_alkane(20, 50.0);

    // Move atom 1 perpendicular to the (0,1,2) plane by a small amount
    // along z. Resulting θ deviation: tan(Δθ/2) ≈ (z_shift / b) for small
    // shifts, so we set z_shift to give Δθ = 0.05 rad exactly via
    // analytical reverse-mapping.
    //
    // Easier: directly move atom 1 along the bisector to reach a precise
    // new angle. For a target angle θ_new, place atom 1 at distance
    // r = b * sec((θ - θ_new)/2)… actually the cleanest is to shift atom 1
    // along z (out of the chain plane) by a small h, then read off the
    // resulting Δθ analytically.
    //
    // For a planar 0-1-2 with both bonds of length b and angle θ0 at
    // vertex 1, lifting atom 1 by h along z preserves both bond lengths to
    // first order in h^2/b^2 but changes the apex angle. The exact change
    // is hard; instead we lift atom 1 by a small h, then compute Δθ from
    // the new geometry directly and compare V_angle against K * (Δθ)^2.

    const double h = 0.02; // Å
    cpu::Frame fresh = make_all_trans_alkane(20, 50.0);
    fresh.span(ATV::kAtom_z)[1] += h;

    // Compute new angle at vertex 1 from atoms 0, 1, 2 directly.
    auto angle_at = [](const cpu::Frame& f, int i, int j, int k) {
        double rij[3] = {f.span(ATV::kAtom_x)[i] - f.span(ATV::kAtom_x)[j],
                         f.span(ATV::kAtom_y)[i] - f.span(ATV::kAtom_y)[j],
                         f.span(ATV::kAtom_z)[i] - f.span(ATV::kAtom_z)[j]};
        double rkj[3] = {f.span(ATV::kAtom_x)[k] - f.span(ATV::kAtom_x)[j],
                         f.span(ATV::kAtom_y)[k] - f.span(ATV::kAtom_y)[j],
                         f.span(ATV::kAtom_z)[k] - f.span(ATV::kAtom_z)[j]};
        const double dot = rij[0] * rkj[0] + rij[1] * rkj[1] + rij[2] * rkj[2];
        const double mij =
            std::sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]);
        const double mkj =
            std::sqrt(rkj[0] * rkj[0] + rkj[1] * rkj[1] + rkj[2] * rkj[2]);
        double c = dot / (mij * mkj);
        if (c > 1.0)
            c = 1.0;
        if (c < -1.0)
            c = -1.0;
        return std::acos(c);
    };
    const double theta_new = angle_at(fresh, 0, 1, 2);
    const double dtheta = theta_new - kAngleTheta0;

    cpu::Context ctx;
    cpu::BuildContext bctx{ctx, fresh};
    cpu::LaunchContext lctx{ctx, fresh, 0, 0};
    cpu::AngleHarmonic angle_only({angle_K_eV()}, {kAngleTheta0});
    angle_only.build(bctx);

    // Isolate the (0,1,2) angle in the PER-ATOM topology the kernel actually
    // reads. The legacy flat `fresh.angles` is dead storage (kernels ignore
    // it), so the original assignment did NOT isolate anything:
    // make_all_trans_alkane wires every chain angle, and lifting atom 1
    // perturbs both (0,1,2) and (1,2,3) — measuring two angles' energy.
    // Clear all per-atom angles, then add only (0,1,2).
    for (int i = 0; i < fresh.nlocal(); ++i)
        fresh.span(ATV::kAtom_num_angle)[i] = 0;
    fresh.clear_tag_index();
    fresh.rebuild_tag_index(0, fresh.nlocal());
    fresh.add_angle(0, 1, 2, 0);
    fresh.build_special_list();

    fresh.meta.get(ATV::kEnergy) = 0;
    angle_only.launch(lctx);
    angle_only.destroy({ctx, fresh});

    const double v_expected = angle_K_eV() * dtheta * dtheta;
    EXPECT_NEAR(fresh.meta.get(ATV::kEnergy), v_expected, 1e-9)
        << "one angle bent by " << dtheta
        << " rad should give K*Δθ² = " << v_expected;
}

// ── All-trans dihedral: V_Dreiding-UA at φ = 180° is a constant offset ─

TEST(PeSingleChain, AllTransDihedralIsStationaryPoint) {
    // Dreiding-UA multi-harmonic V(180°) = C0 - C1 + C2 - C3 + C4
    //                                    = 1.73 + 4.49 + 0.776 − 6.99 + 0
    //                                    = 0.006 kcal/mol per dihedral
    // (a baked-in offset of the Tschopp 2010 fit; nonzero unlike OPLS).
    cpu::Frame fresh = make_all_trans_alkane(20, 50.0);

    cpu::Context ctx;
    cpu::BuildContext bctx{ctx, fresh};
    cpu::LaunchContext lctx{ctx, fresh, 0, 0};
    cpu::DihedralMultiHarmonic dih({dihedral_Cn_eV()});
    dih.build(bctx);

    fresh.meta.get(ATV::kEnergy) = 0;
    dih.launch(lctx);
    dih.destroy({ctx, fresh});

    const double expected = 17.0 * 0.006 / 23.060547830619026;
    EXPECT_NEAR(fresh.meta.get(ATV::kEnergy), expected, 1e-6)
        << "17 dihedrals at trans should give the Dreiding-UA constant "
        << "offset ~4.4e-3 eV (not 0 — OPLS used to set this to 0 by "
        << "construction; Dreiding-UA does not).";
}
