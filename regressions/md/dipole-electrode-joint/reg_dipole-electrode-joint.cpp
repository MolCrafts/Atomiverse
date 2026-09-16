/// @file regressions/md/dipole-electrode-joint/reg_dipole-electrode-joint.cpp
/// @brief E2E: joint PIM dipoles ⊕ electrode Q vs the min-image algebra pin.
///
/// The pinned numbers are Atomiverse's own MINIMUM-IMAGE algebra, not
/// MetalWalls output. MetalWalls c8ad175d was run on the exact same 4-site
/// twin (tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/)
/// and lands 9 % away on Q and 16 % away on μ, because MW only implements
/// Ewald electrostatics (num_PBC ∈ {2,3}) while this component truncates at
/// minimum image (xy_image_range = 0). Under a matched 3D lattice sum these
/// same kernels reproduce MetalWalls to 1.2e-8 — i.e. the formulas agree
/// exactly and only the boundary condition differs. Full decomposition,
/// raw MW output and unit conversions: REFERENCE.md.
///
/// See REFERENCE.md. Public headers + link atomiverse only. Exits non-zero on fail.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/dipole_electrode_joint.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

namespace {

cpu::Frame make_frame() {
    cpu::Frame f(4);
    const double z[4] = {2.0, 18.0, 8.0, 12.0};
    const double q0[4] = {0.0, 0.0, 1.0, -1.0};
    const int typ[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = q0[i];
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = (typ[i] < 3) ? 12.0 : 1.0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::orthorhombic(10.0, 10.0, 20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

template <typename Solver>
void solve_once(Solver& s, cpu::Frame& frame) {
    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/8.0, /*skin=*/0.5});
    sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/8.0});
    sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.0});
    sim.set_observer_ref(s);
    sim.build();
    sim.run(frame, 1);
}

} // namespace

int main() {
    // Min-image algebra pin (pin_joint.py; K = units::coulomb = 14.399645).
    // NOT a MetalWalls number — see the file header and REFERENCE.md.
    constexpr double kQBotPin = -7.3285008958883610e-02;
    constexpr double kQTopPin = 7.3285008958883610e-02;
    constexpr double kMuCatZPin = 5.8951916148990660e-02;
    constexpr double kMuAnZPin = 8.7432701391015569e-02;
    constexpr double kQTopAlpha0 = 7.5339627935767609e-02;
    constexpr double kRelTolQ = 1e-6;
    constexpr double kTolMu = 1e-8;

    cpu::Frame frame = make_frame();

    cpu::md::DipoleElectrodeJoint joint(
        /*eta=*/2.0, /*alpha_by_type=*/{0.0, 0.0, 0.0, 1.0, 1.5});
    joint.add_electrode(/*type=*/1, /*Psi=*/0.0);
    joint.add_electrode(/*type=*/2, /*Psi=*/1.0);
    joint.thole_a = 0.39;
    joint.electroneutrality = true;
    joint.xy_image_range = 0;
    joint.joint_tol = 1e-12;
    joint.pol_tol = 1e-12;
    joint.joint_max_iter = 100;
    joint.pol_max_iter = 200;

    solve_once(joint, frame);

    const double q_bot = joint.total_charge(0);
    const double q_top = joint.total_charge(1);
    const double neut = joint.neutrality_residual();
    const auto& mu = joint.last_mu();

    const double rel_bot = std::abs(q_bot - kQBotPin) / std::abs(kQBotPin);
    const double rel_top = std::abs(q_top - kQTopPin) / std::abs(kQTopPin);

    std::printf("dipole-electrode-joint\n");
    std::printf("  Q_bot   = % .16e  (pin % .16e) rel=% .3e\n", q_bot, kQBotPin,
                rel_bot);
    std::printf("  Q_top   = % .16e  (pin % .16e) rel=% .3e\n", q_top, kQTopPin,
                rel_top);
    std::printf("  |sum Q| = % .3e\n", neut);
    std::printf("  mu_cat  = (% .6e, % .6e, % .16e)\n", mu[6], mu[7], mu[8]);
    std::printf("  mu_an   = (% .6e, % .6e, % .16e)\n", mu[9], mu[10], mu[11]);
    std::printf("  outer   = %d  converged=%d\n", joint.last_joint_iters(),
                joint.last_joint_converged() ? 1 : 0);

    int nfail = 0;
    if (!joint.last_joint_converged()) {
        std::fprintf(stderr, "FAIL joint not converged\n");
        ++nfail;
    }
    if (!(neut <= 1e-10)) {
        std::fprintf(stderr, "FAIL neutrality |sum Q|=%g\n", neut);
        ++nfail;
    }
    if (!(rel_bot <= kRelTolQ) || !(rel_top <= kRelTolQ)) {
        std::fprintf(stderr, "FAIL Q rel err bot=%g top=%g tol=%g\n", rel_bot,
                     rel_top, kRelTolQ);
        ++nfail;
    }
    if (!(std::abs(mu[6]) < 1e-10 && std::abs(mu[7]) < 1e-10
          && std::abs(mu[9]) < 1e-10 && std::abs(mu[10]) < 1e-10)) {
        std::fprintf(stderr, "FAIL transverse mu non-zero\n");
        ++nfail;
    }
    if (!(std::abs(mu[8] - kMuCatZPin) <= kTolMu)
        || !(std::abs(mu[11] - kMuAnZPin) <= kTolMu)) {
        std::fprintf(stderr,
                     "FAIL mu_z cat=%.16e want=%.16e an=%.16e want=%.16e\n",
                     mu[8], kMuCatZPin, mu[11], kMuAnZPin);
        ++nfail;
    }
    // Electrolyte permanent charges must remain.
    if (std::abs(static_cast<double>(frame.span(kAtom_charge)[2]) - 1.0) > 1e-14
        || std::abs(static_cast<double>(frame.span(kAtom_charge)[3]) + 1.0)
               > 1e-14) {
        std::fprintf(stderr, "FAIL electrolyte charges mutated\n");
        ++nfail;
    }
    // Frame write-back of electrode Q
    if (std::abs(static_cast<double>(frame.span(kAtom_charge)[0]) - q_bot)
            > 1e-14
        || std::abs(static_cast<double>(frame.span(kAtom_charge)[1]) - q_top)
               > 1e-14) {
        std::fprintf(stderr, "FAIL kAtom_charge electrode write-back\n");
        ++nfail;
    }

    // Dual-gate coupling: α=0 reduction vs ConstPotBO pin.
    {
        cpu::Frame f0 = make_frame();
        cpu::md::DipoleElectrodeJoint j0(2.0, {0.0, 0.0, 0.0, 0.0, 0.0});
        j0.add_electrode(1, 0.0);
        j0.add_electrode(2, 1.0);
        j0.electroneutrality = true;
        j0.xy_image_range = 0;
        j0.joint_tol = 1e-12;
        solve_once(j0, f0);
        const double q0 = j0.total_charge(1);
        const double rel0 = std::abs(q0 - kQTopAlpha0) / std::abs(kQTopAlpha0);
        std::printf("  alpha0 Q_top = % .16e  rel=% .3e\n", q0, rel0);
        if (!(rel0 <= kRelTolQ)) {
            std::fprintf(stderr, "FAIL alpha0 Q_top rel=%g\n", rel0);
            ++nfail;
        }
        const double couple =
            std::abs(q_top - q0) / std::max(std::abs(q0), 1e-30);
        std::printf("  coupling |ΔQ|/|Q0| = % .6e\n", couple);
        if (!(couple >= 1e-6)) {
            std::fprintf(stderr, "FAIL no coupling shift\n");
            ++nfail;
        }

        // ConstPotBO parity at α=0
        cpu::Frame fbo = make_frame();
        cpu::md::ConstPotBO bo(2.0);
        bo.add_electrode(1, 0.0);
        bo.add_electrode(2, 1.0);
        bo.electroneutrality = true;
        bo.xy_image_range = 0;
        solve_once(bo, fbo);
        if (std::abs(bo.total_charge(1) - q0) > 1e-10) {
            std::fprintf(stderr, "FAIL alpha0 vs ConstPotBO |Δ|=%g\n",
                         std::abs(bo.total_charge(1) - q0));
            ++nfail;
        }
    }

    // MD binding path (one step).
    {
        cpu::Frame f2 = make_frame();
        cpu::md::DipoleElectrodeJoint j2(2.0, {0.0, 0.0, 0.0, 1.0, 1.5});
        j2.add_electrode(1, 0.0);
        j2.add_electrode(2, 1.0);
        j2.thole_a = 0.39;
        j2.electroneutrality = true;
        j2.xy_image_range = 0;
        j2.joint_tol = 1e-12;
        j2.pol_tol = 1e-12;
        j2.rebuild_matrix_every_step = true;

        cpu::md::MD sim;
        sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/8.0, /*skin=*/0.5});
        sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/8.0});
        sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
        sim.set_observer_ref(j2);
        sim.build();
        sim.run(f2, 1);

        const double q_md = j2.total_charge(1);
        const double rel_md = std::abs(q_md - kQTopPin) / std::abs(kQTopPin);
        std::printf("  MD path Q_top = % .16e  rel=% .3e  |sumQ|=% .3e\n", q_md,
                    rel_md, j2.neutrality_residual());
        if (!(rel_md <= kRelTolQ) || !(j2.neutrality_residual() <= 1e-10)
            || !j2.last_joint_converged()) {
            std::fprintf(stderr, "FAIL MD-bound joint rel=%g neut=%g\n", rel_md,
                         j2.neutrality_residual());
            ++nfail;
        }
    }

    if (nfail) {
        std::fprintf(stderr, "%d check(s) failed\n", nfail);
        return 1;
    }
    std::printf(
        "PASS: dipole-electrode-joint vs min-image algebra pin "
        "(MetalWalls c8ad175d twin: not_expressible, see REFERENCE.md)\n");
    return 0;
}
