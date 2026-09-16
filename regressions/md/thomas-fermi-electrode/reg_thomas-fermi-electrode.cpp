/// @file regressions/md/thomas-fermi-electrode/reg_thomas-fermi-electrode.cpp
/// @brief E2E: Thomas–Fermi finite metallicity on gold capacitor toy (E7).
///
/// Pinned against the MetalWalls binary (git c8ad175d), `num_pbc 2`,
/// `electrode_charges matrix_inversion`, `charge_neutrality true` — see
/// REFERENCE.md. Public headers + link atomiverse only. Exits non-zero on
/// failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

namespace {

// MetalWalls c8ad175d, total_charges.out of the twin inputs under
// tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/mw_twin/.
// MW charges are already in e (its "atomic unit of charge" is e), so these are
// raw MW output with no conversion; the unit conversion is on the inputs
// (Å→bohr, V→hartree/e, Å⁻¹→bohr⁻¹). See REFERENCE.md.
constexpr double kQTopPerfect = 4.354124119406114e-03; // [e], l_tf = 0
constexpr double kQTopTf2 = 3.707555065364343e-03;     // [e], l_tf = 2 Å
constexpr double kRelTol = 0.01;                       // 1% dual-gate

constexpr double kEta = 1.0 / 0.56; // [1/Å]
constexpr double kA0 = 4.07;        // [Å]
constexpr double kD = 4.0 / (kA0 * kA0 * kA0);
constexpr double kPlane = kA0 / 2.0;
constexpr double kLGap = 20.0;
// `xy_image_range` is ConstPotBO's convergence knob, not a property of the
// system: the truncated square-image sum approaches the exact 2D-periodic
// (MetalWalls num_pbc = 2) elastance only as O(1/R), because each pair misses a
// lattice tail ≈ (π z²/R)/A_cell. With a 20 Å gap over a 4.07 Å cell, R = 40
// is still 6.1% off and R = 1280 is 0.18% off — inside the 1% gate below.
// Convergence table: .../thomas-fermi-electrode/tf_lattice_convergence.py.
constexpr int kImageRange = 1280;

cpu::Frame gold_capacitor() {
    // 3 bot + 3 top electrode sites (types 1, 2), empty vacuum.
    cpu::Frame f(6);
    const double z_surf_bot = 5.0 + 2.0 * kPlane;
    const double z_surf_top = z_surf_bot + kLGap;
    const double z[6] = {
        z_surf_bot - 2.0 * kPlane,
        z_surf_bot - kPlane,
        z_surf_bot,
        z_surf_top,
        z_surf_top + kPlane,
        z_surf_top + 2.0 * kPlane,
    };
    const int typ[6] = {1, 1, 1, 2, 2, 2};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = 196.966553;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0.0;
    }
    // Lz large; electrode solve uses 2D xy images only.
    f.box = cpu::Box::orthorhombic(kA0, kA0, 80.0);
    return f;
}

void run_solve(cpu::md::ConstPotBO& conp, cpu::Frame& f) {
    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{12.0, 0.5});
    sim.set_evaluator(cpu::PairCoulCut{12.0});
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_observer_ref(conp);
    sim.build();
    sim.run(f, 1);
}

int check_case(const char* label, double l_tf, double q_pin) {
    auto f = gold_capacitor();
    cpu::md::ConstPotBO conp(kEta);
    conp.add_electrode(/*type=*/1, /*Psi=*/0.0);
    conp.add_electrode(/*type=*/2, /*Psi=*/1.0);
    conp.electroneutrality = true;
    conp.xy_image_range = kImageRange;
    conp.l_tf = l_tf;
    conp.site_density = kD;
    run_solve(conp, f);

    const double q_bot = conp.total_charge(0);
    const double q_top = conp.total_charge(1);
    const double neut = conp.neutrality_residual();
    const double c = conp.capacitance();
    const double rel_q = std::abs(q_top - q_pin) / std::abs(q_pin);
    const double c_pin = std::abs(q_pin); // ΔΨ = 1 V
    const double rel_c = std::abs(c - c_pin) / std::abs(c_pin);

    std::printf("%s\n", label);
    std::printf("  l_tf    = %.3f Å\n", l_tf);
    std::printf("  Q_bot   = % .12e e\n", q_bot);
    std::printf("  Q_top   = % .12e e  (pin %+ .12e)\n", q_top, q_pin);
    std::printf("  C       = % .12e e/V  (pin % .12e)\n", c, c_pin);
    std::printf("  |sum Q| = % .3e e\n", neut);
    std::printf("  rel Q   = % .4e  (tol %.2f)\n", rel_q, kRelTol);
    std::printf("  rel C   = % .4e  (tol %.2f)\n", rel_c, kRelTol);

    int nfail = 0;
    if (!(neut <= 1e-10)) {
        std::fprintf(stderr, "FAIL %s neutrality |sum Q|=%g\n", label, neut);
        ++nfail;
    }
    if (!(rel_q <= kRelTol)) {
        std::fprintf(stderr, "FAIL %s Q_top rel %g > %g\n", label, rel_q,
                     kRelTol);
        ++nfail;
    }
    if (!(rel_c <= kRelTol)) {
        std::fprintf(stderr, "FAIL %s C rel %g > %g\n", label, rel_c, kRelTol);
        ++nfail;
    }
    if (!(q_top > 0.0 && q_bot < 0.0)) {
        std::fprintf(stderr, "FAIL %s charge signs Q_bot=%g Q_top=%g\n", label,
                     q_bot, q_top);
        ++nfail;
    }
    // Frame write-back: top electrode sites are rows 3..5
    double q_top_sum = 0.0;
    for (int i = 3; i < 6; ++i)
        q_top_sum += static_cast<double>(f.span(kAtom_charge)[i]);
    if (std::abs(q_top_sum - q_top) > 1e-12) {
        std::fprintf(stderr, "FAIL %s kAtom_charge write-back sum=%g vs %g\n",
                     label, q_top_sum, q_top);
        ++nfail;
    }
    return nfail;
}

} // namespace

int main() {
    int nfail = 0;
    nfail += check_case("thomas-fermi-electrode perfect metal (l_tf=0)", 0.0,
                        kQTopPerfect);
    nfail += check_case("thomas-fermi-electrode TF (l_tf=2 Å)", 2.0, kQTopTf2);

    // Dual-gate: finite metallicity must reduce capacitance.
    {
        auto f0 = gold_capacitor();
        cpu::md::ConstPotBO p0(kEta);
        p0.add_electrode(1, 0.0).add_electrode(2, 1.0);
        p0.xy_image_range = kImageRange;
        p0.l_tf = 0.0;
        p0.site_density = kD;
        run_solve(p0, f0);

        auto f2 = gold_capacitor();
        cpu::md::ConstPotBO p2(kEta);
        p2.add_electrode(1, 0.0).add_electrode(2, 1.0);
        p2.xy_image_range = kImageRange;
        p2.l_tf = 2.0;
        p2.site_density = kD;
        run_solve(p2, f2);

        std::printf("dual-gate C(0)=%.6e  C(2)=%.6e\n", p0.capacitance(),
                    p2.capacitance());
        if (!(p2.capacitance() < p0.capacitance())) {
            std::fprintf(stderr, "FAIL C(l_tf=2) not < C(l_tf=0)\n");
            ++nfail;
        }
    }

    if (nfail) {
        std::fprintf(stderr, "%d check(s) failed\n", nfail);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
