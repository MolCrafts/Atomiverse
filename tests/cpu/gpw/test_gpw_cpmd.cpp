/// @file tests/cpu/gpw/test_gpw_cpmd.cpp
/// @brief Unit tests: GPWEvaluator CPMD build() guards + the multi-k
///        fixed-density H_k/S_k export the CPMD path consumes.
///
/// The Cpmd / CpmdCplx propagator classes themselves live in
/// cpu/electronic/ and are pinned by tests/cpu/electronic/test_cpmd.cpp.
/// No multi-step MD / full water SCF trajectory here (that is the e2e
/// regressions/dynamics/cpmd-water and cpmd-rattle-water pins). build()
/// validation, the tiny inline H2 payload, and one direct `launch` probe on
/// the smallest possible system (H2, no driver, no loop) for the electronic
/// energy publication contract.
///
/// **Timestep:** the nuclear dt is owned by the MD integrator and pushed in
/// through `cpu::TimestepConsumer::set_timestep` (esmm-cpmd-02) — the
/// evaluator has no electronic-dt member of its own to set. Guards that are
/// not *about* the dt therefore call `set_timestep(kDtFs)` first, so each one
/// fails on its own gate rather than on the missing-provider gate that
/// precedes most of them. The push walk itself (`cpu::md::MD`
/// provider/consumer rules) is pinned by
/// tests/cpu/dynamics/test_timestep_protocol.cpp.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/electronic/car_parrinello.h"
#include "atomiverse/cpu/electronic/xl_bomd_force.h"
#include "atomiverse/cpu/electronic/second_gen_cp.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"

// ── GPWEvaluator cpmd guards ────────────────────────────────────────────

namespace {

using ATV::cpu::electronic::CarParrinello;
using ATV::cpu::electronic::XlBomdForce;
using ATV::cpu::electronic::SecondGenCp;
using ATV::cpu::electronic::XlBomdLinearized;
using ATV::cpu::electronic::ElectronicThermostat;

/// A valid nuclear timestep [fs], standing in for the MD integrator's push.
constexpr double kDtFs = 0.1;

bool message_mentions(const std::exception& e, const std::string& needle) {
    return std::string(e.what()).find(needle) != std::string::npos;
}

ATV::cpu::Frame water_frame() {
    ATV::cpu::Frame f(3);
    const int z[3] = {8, 1, 1};
    const double x[3] = {5.0000, 5.7586, 4.2414};
    const double y[3] = {5.0000, 5.0000, 5.0000};
    const double zc[3] = {5.0000, 5.5043, 5.5043};
    for (int i = 0; i < 3; ++i) {
        f.span(ATV::kAtom_type)[i] = z[i];
        f.span(ATV::kAtom_x)[i] = x[i];
        f.span(ATV::kAtom_y)[i] = y[i];
        f.span(ATV::kAtom_z)[i] = zc[i];
    }
    f.box = ATV::cpu::Box::cubic(10.0);
    return f;
}

ATV::cpu::gpw::GPWEvaluator water_eval() {
    const ATV::cpu::gpw::Kind O{
        molrs::Element::O,
        ATV::cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        ATV::cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const ATV::cpu::gpw::Kind H{
        molrs::Element::H,
        ATV::cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
        ATV::cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};
    return ATV::cpu::gpw::GPWEvaluator(
        {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
        /*mix*/ 0.3, ATV::cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
        /*compute_forces*/ true);
}

} // namespace

TEST(GpwEvaluatorCpmd, BuildAcceptsCpmdWithUks) {
    // Γ-UKS CPMD is product (dual α/β propagators); O2-like occupations.
    auto frame = water_frame();
    auto base = water_eval();
    base.n_alpha = 5;
    base.n_beta = 3;
    ATV::cpu::electronic::CarParrinello cp{std::move(base), 400.0};
    cp.set_timestep(kDtFs);
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(cp.build(bc));
}

TEST(GpwEvaluatorCpmd, BuildRejectsUksCpmdWithZeroAlpha) {
    auto frame = water_frame();
    auto base = water_eval();
    base.n_alpha = 0;
    base.n_beta = 3;
    CarParrinello eval{std::move(base), 400.0};
    eval.set_timestep(kDtFs);
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_THROW(eval.build(bc), std::invalid_argument);
}

TEST(GpwEvaluatorCpmd, BuildRejectsCpmdWithOt) {
    auto frame = water_frame();
    auto base = water_eval();
    base.use_ot = true;
    CarParrinello eval{std::move(base), 400.0};
    eval.set_timestep(kDtFs);
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_THROW(eval.build(bc), std::invalid_argument);
}

TEST(GpwEvaluatorCpmd, CpmdAndXlAreDistinctTypes) {
    // Mutual exclusion is type-level: CarParrinello vs XlBomdForce, not flags.
    auto base1 = water_eval();
    auto base2 = water_eval();
    CarParrinello cp{std::move(base1), 400.0};
    XlBomdForce xl{std::move(base2), 5};
    EXPECT_NE(cp.name(), xl.name());
}

TEST(GpwEvaluatorCpmd, CtorRejectsNonPositiveMu) {
    EXPECT_THROW((CarParrinello{water_eval(), 0.0}), std::invalid_argument);
    EXPECT_THROW((CarParrinello{water_eval(), -1.0}), std::invalid_argument);
}

TEST(GpwEvaluatorCpmd, CtorForcesComputeForces) {
    auto base = water_eval();
    base.compute_forces = false;
    CarParrinello eval{std::move(base), 400.0};
    EXPECT_TRUE(eval.method().compute_forces);
}

// ── Nuclear timestep protocol (esmm-cpmd-02) ────────────────────────────
//
// dt has exactly one owner (the MD integrator). The evaluator is a
// `cpu::TimestepConsumer`: it validates what arrives, and refuses to build
// CPMD when nothing ever arrived.

TEST(GpwEvaluatorCpmd, SetTimestepRejectsNonPositive) {
    auto base = water_eval();
    auto eval = std::move(base);
    EXPECT_THROW(eval.set_timestep(0.0), std::invalid_argument);
    EXPECT_THROW(eval.set_timestep(-0.1), std::invalid_argument);
}

TEST(GpwEvaluatorCpmd, SetTimestepRejectsNonFinite) {
    auto base = water_eval();
    auto eval = std::move(base);
    EXPECT_THROW(eval.set_timestep(std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
    EXPECT_THROW(eval.set_timestep(std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
}

/// ac-003: a cpmd evaluator that no `TimestepProvider` ever pushed to has no
/// nuclear clock to ride — building it outside an MD loop is meaningless, and
/// the message must name the missing provider path so the fix is readable.
TEST(GpwEvaluatorCpmd, BuildWithoutTimestepThrows) {
    auto frame = water_frame();
    // CarParrinello without set_timestep: no nuclear clock.
    CarParrinello eval{water_eval(), 400.0};
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    try {
        eval.build(bc);
        FAIL() << "expected build() to refuse cpmd with no pushed timestep";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "CPMD outside MD")) << e.what();
        EXPECT_TRUE(message_mentions(e, "TimestepProvider")) << e.what();
    }
}

/// ac-004: every CPMD branch of `launch` returns before the stress block, so a
/// stress request used to be dropped in silence. It must be refused instead.
TEST(GpwEvaluatorCpmd, BuildRefusesCpmdWithStress) {
    auto frame = water_frame();
    auto base = water_eval();
    base.compute_stress = true;
    CarParrinello eval{std::move(base), 400.0};
    eval.set_timestep(kDtFs);
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    try {
        eval.build(bc);
        FAIL() << "expected build() to refuse cpmd + compute_stress";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "variable-cell")) << e.what();
    }
}

// ── Nosé–Hoover electronic thermostat guards ────────────────────────────

TEST(GpwEvaluatorCpmd, BareGpwHasNoPublicThermostatFlag) {
    // Thermostat only exists via CarParrinello(ElectronicThermostat{...}).
    // Bare GPWEvaluator cannot enable electronic thermostat.
    auto frame = water_frame();
    auto eval = water_eval();
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorCpmd, CtorRejectsBadThermostatParams) {
    EXPECT_THROW(
        (CarParrinello{water_eval(), 400.0, ElectronicThermostat{0.0, 50.0}}),
        std::invalid_argument);
    EXPECT_THROW(
        (CarParrinello{water_eval(), 400.0, ElectronicThermostat{1e-5, -5.0}}),
        std::invalid_argument);
}

TEST(GpwEvaluatorCpmd, BuildAcceptsCpmdWithThermostat) {
    auto frame = water_frame();
    auto base = water_eval();
    CarParrinello eval{std::move(base), 400.0, ElectronicThermostat{1e-5, 50.0}};
    eval.set_timestep(kDtFs);
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

// ── Multi-k Bloch CPMD evaluator guards ─────────────────────────────────

TEST(GpwEvaluatorCpmd, BuildAcceptsMultiKRksCpmd) {
    auto frame = water_frame();
    auto base = water_eval();
    CarParrinello eval{std::move(base), 400.0};
    eval.set_timestep(kDtFs);
    eval.method().k_mesh[0] = 1;
    eval.method().k_mesh[1] = 1;
    eval.method().k_mesh[2] = 2;
    eval.method().k_time_reversal = false;
    eval.method().nonperiodic = false;
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorCpmd, BuildAcceptsMultiKUksCpmd) {
    // Multi-k UKS CPMD is product (dual-spin Bloch C_k^σ, one CpmdCplx per
    // (k, spin)); the old build() refusal is deleted (cpmd-uks-kpoints).
    auto frame = water_frame();
    auto base = water_eval();
    base.n_alpha = 5;
    base.n_beta = 3;
    CarParrinello eval{std::move(base), 400.0};
    eval.set_timestep(kDtFs);
    eval.method().k_mesh[2] = 2;
    eval.method().k_time_reversal = false;
    eval.method().nonperiodic = false;
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorCpmd, BuildRejectsMultiKUksCpmdWithZeroAlpha) {
    // The alpha channel stays primary on the k path too.
    auto frame = water_frame();
    auto base = water_eval();
    base.n_alpha = 0;
    base.n_beta = 3;
    CarParrinello eval{std::move(base), 400.0};
    eval.set_timestep(kDtFs);
    eval.method().k_mesh[2] = 2;
    eval.method().nonperiodic = false;
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_THROW(eval.build(bc), std::invalid_argument);
}

TEST(GpwEvaluatorCpmd, BuildAcceptsMultiKCpmdThermostat) {
    // Multi-k CPMD + electronic NH is product (cpmd-thermostat-kpoints):
    // one thermostatted CpmdCplx per (k, sigma) channel — the old
    // "NVE electrons only on the k path" refusal is deleted.
    auto frame = water_frame();
    auto base = water_eval();
    CarParrinello eval{std::move(base), 400.0, ElectronicThermostat{1e-5, 50.0}};
    eval.set_timestep(kDtFs);
    eval.method().k_mesh[2] = 2;
    eval.method().k_time_reversal = false;
    eval.method().nonperiodic = false;
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorCpmd, BuildAcceptsMultiKUksCpmdThermostat) {
    // The dual-spin k path thermostats every (k, sigma) channel too.
    auto frame = water_frame();
    auto base = water_eval();
    base.n_alpha = 5;
    base.n_beta = 3;
    CarParrinello eval{std::move(base), 400.0, ElectronicThermostat{1e-5, 50.0}};
    eval.set_timestep(kDtFs);
    eval.method().k_mesh[2] = 2;
    eval.method().k_time_reversal = false;
    eval.method().nonperiodic = false;
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

// ── Electronic energy publication (esmm-cpmd-03 ac-001) ─────────────────
//
// Producer side of `cpu::md::CpmdConserved` (assembler pinned in
// tests/cpu/dynamics/test_cpmd_conserved.cpp): every CPMD `launch` must leave
// BOTH open-schema electronic keys on the frame, holding exactly what the
// `last_electronic_*_ev()` readbacks report. Bit-for-bit is the contract, not
// a tolerance — meta and readback are two views of one stored double, so any
// difference at all means a recomputation crept in between them and the
// conserved energy the user reads is not the one the propagator advanced.

namespace {


/// @brief H₂ (r = 0.74 Å) centred in a 6 Å box — the cheapest frame that
///        reaches `GPWEvaluator::launch` at all.
///
/// The CPMD seed needs a *converged* cold SCF with occupied MOs, so a real
/// basis/pseudopotential pair is unavoidable; H₂ keeps it to 2 atoms / 10 AOs
/// / 1 occupied orbital instead of water's 3 atoms / 29 AOs. This is a
/// single-`launch` component probe, not an MD trajectory: no driver, no loop,
/// no step budget.
ATV::cpu::Frame h2_frame() {
    ATV::cpu::Frame f(2);
    const double dx = 0.37; // half the 0.74 Å bond
    for (int i = 0; i < 2; ++i) {
        f.span(ATV::kAtom_type)[i] = 1;
        f.span(ATV::kAtom_x)[i] = 3.0 + (i == 0 ? -dx : dx);
        f.span(ATV::kAtom_y)[i] = 3.0;
        f.span(ATV::kAtom_z)[i] = 3.0;
    }
    f.box = ATV::cpu::Box::cubic(6.0);
    return f;
}

/// @brief Γ-RKS CPMD evaluator for `h2_frame()` (60 Ry, non-periodic).
///
/// Cutoff and tolerance are the cheapest settings that still converge the cold
/// seed — nothing here pins an energy, so grid accuracy only has to be good
/// enough for the SCF to reach `tol` inside `max_iter` (an unconverged seed
/// throws, so "it ran" already means "it converged"). ~0.2 s for the whole
/// test, which keeps this file inside the seconds-scale unit budget.
CarParrinello h2_cpmd_eval() {
    const ATV::cpu::gpw::Kind H{
        molrs::Element::H,
        ATV::cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
        ATV::cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};
    ATV::cpu::gpw::GPWEvaluator e(
        {H}, /*cutoff_ry*/ 60.0, /*max_iter*/ 60, /*tol*/ 1e-5,
        /*mix*/ 0.3, ATV::cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
        /*compute_forces*/ true);
    CarParrinello cp{std::move(e), 2000.0}; // as in regressions/dynamics/cpmd-water
    cp.set_timestep(kDtFs);
    return cp;
}

} // namespace

/// ac-001: `kEnergyElectronKinetic` / `kEnergyElectronThermostat` appear on the
/// frame at the first CPMD launch and equal the readbacks exactly.
///
/// Both branches that a CPMD run walks are sampled by one and the same
/// assertion, because it is one contract: the cold **seed** launch (electrons
/// have not moved — both terms are exactly 0, and the keys must still be
/// published, since an absent key makes `CpmdConserved` throw rather than
/// silently drop T_e) and the first fixed-density **production** launch, where
/// T_e leaves zero and the equality stops being vacuous.
TEST(GpwEvaluatorCpmd, PublishesElectronicEnergyMetaKeys) {
    auto frame = h2_frame();
    auto eval = h2_cpmd_eval();
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    ASSERT_NO_THROW(eval.build(bc));

    // Open-schema keys: `Frame(int)` does not pre-insert them, and nothing has
    // published them yet — so "present afterwards" below is a real observation.
    ASSERT_FALSE(frame.meta.contains(ATV::kEnergyElectronKinetic.key));
    ASSERT_FALSE(frame.meta.contains(ATV::kEnergyElectronThermostat.key));

    auto expect_published = [&frame, &eval]() {
        ASSERT_TRUE(frame.meta.contains(ATV::kEnergyElectronKinetic.key));
        ASSERT_TRUE(frame.meta.contains(ATV::kEnergyElectronThermostat.key));
        const double te = frame.meta.get(ATV::kEnergyElectronKinetic);
        const double etherm = frame.meta.get(ATV::kEnergyElectronThermostat);
        EXPECT_TRUE(std::isfinite(te)) << te;
        EXPECT_TRUE(std::isfinite(etherm)) << etherm;
        EXPECT_EQ(te, eval.last_electronic_ke_ev());
        EXPECT_EQ(etherm, eval.last_electronic_thermostat_ev());
    };

    ATV::cpu::LaunchContext lc{rt, frame, ATV::slot_of(ATV::md::Phase::force),
                               /*step=*/0};

    eval.launch(lc); // cold SCF + CPMD seed: electrons still at rest
    {
        SCOPED_TRACE("seed launch");
        expect_published();
        EXPECT_EQ(frame.meta.get(ATV::kEnergyElectronKinetic), 0.0);
        EXPECT_EQ(frame.meta.get(ATV::kEnergyElectronThermostat), 0.0);
    }

    // First production launch: fixed-density KS eval + one electronic
    // velocity-Verlet step, so T_e is no longer zero (electrons NVE — no
    // thermostat requested, hence E_therm stays 0).
    eval.launch(lc);
    {
        SCOPED_TRACE("first production launch");
        ASSERT_EQ(eval.last_result().iters, 0)
            << "production CP launch must be a fixed-density eval, not an SCF";
        expect_published();
        EXPECT_GT(frame.meta.get(ATV::kEnergyElectronKinetic), 0.0)
            << "T_e must leave zero once the electrons are propagated, "
               "otherwise the equality above is vacuous";
    }
}

// Multi-k fixed-density payload (tiny H2 GpwDft).
namespace {


ATV::cpu::gpw::GridBasisFunction s_basis_cp(double alpha, const double c[3]) {
    ATV::cpu::gpw::GridPrimitive p;
    p.alpha = alpha;
    p.center[0] = c[0];
    p.center[1] = c[1];
    p.center[2] = c[2];
    p.l[0] = p.l[1] = p.l[2] = 0;
    p.coeff = std::pow(2.0 * alpha / M_PI, 0.75);
    return ATV::cpu::gpw::GridBasisFunction{p};
}

ATV::cpu::gpw::GpwAtom hydrogen_cp(const double pos[3]) {
    ATV::cpu::gpw::GpwAtom a;
    a.z_val = 1.0;
    a.z = 1;
    a.pos[0] = pos[0];
    a.pos[1] = pos[1];
    a.pos[2] = pos[2];
    a.basis = {s_basis_cp(0.5, pos)};
    a.rloc = 0.2;
    a.cloc = {-4.17890044, 0.72446331};
    return a;
}

std::vector<ATV::cpu::gpw::GpwAtom> small_h2_cp(double box = 8.0) {
    const double rA[3] = {box / 2 - 0.7, box / 2, box / 2};
    const double rB[3] = {box / 2 + 0.7, box / 2, box / 2};
    return {hydrogen_cp(rA), hydrogen_cp(rB)};
}

ATV::cpu::gpw::GpwDft small_cfg_cp(double box = 8.0) {
    ATV::cpu::gpw::GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = box;
    cfg.cutoff_ry = 120.0;
    cfg.n_occ = 1;
    cfg.max_iter = 80;
    cfg.tol = 1e-7;
    cfg.mix = 0.5;
    return cfg;
}

} // namespace

TEST(GpwSolverFixedDensityK, MultiKFixedDensityExportsHkSk) {
    const auto atoms = small_h2_cp(/*box=*/8.0);
    ATV::cpu::gpw::GpwDft cold = small_cfg_cp(/*box=*/8.0);
    cold.k_mesh[0] = 1;
    cold.k_mesh[1] = 1;
    cold.k_mesh[2] = 2;
    cold.k_time_reversal = false;
    cold.nonperiodic = false;
    cold.build(atoms);
    const auto ref = cold.run(atoms);
    ASSERT_TRUE(ref.converged);
    ASSERT_GE(ref.n_kpts, 2);
    ASSERT_EQ(static_cast<int>(ref.P_k.size()), ref.n_kpts);

    ATV::cpu::gpw::GpwDft fix = small_cfg_cp(/*box=*/8.0);
    fix.k_mesh[0] = 1;
    fix.k_mesh[1] = 1;
    fix.k_mesh[2] = 2;
    fix.k_time_reversal = false;
    fix.nonperiodic = false;
    fix.max_iter = 0;
    fix.build(atoms);
    const auto& r = fix.run(atoms, nullptr, nullptr, nullptr, &ref.P_k);
    EXPECT_TRUE(r.converged);
    EXPECT_EQ(r.iters, 0);
    EXPECT_TRUE(std::isfinite(r.e_total));
    EXPECT_EQ(static_cast<int>(r.H_ks_k.size()), r.n_kpts);
    EXPECT_EQ(static_cast<int>(r.S_k.size()), r.n_kpts);
    EXPECT_EQ(static_cast<int>(r.P_k.size()), r.n_kpts);
    // Energy near the cold SCF (same density).
    EXPECT_NEAR(r.e_total, ref.e_total, 1e-6);
}

TEST(GpwSolverFixedDensityK, MultiKFixedDensityWithoutGuessThrows) {
    const auto atoms = small_h2_cp();
    ATV::cpu::gpw::GpwDft fix = small_cfg_cp();
    fix.k_mesh[2] = 2;
    fix.k_time_reversal = false;
    fix.nonperiodic = false;
    fix.max_iter = 0;
    fix.build(atoms);
    EXPECT_THROW(fix.run(atoms), std::invalid_argument);
}

// ── Multi-k UKS fixed-density (dual-spin Bloch CPMD payload) ────────────

namespace {

/// The `small_cfg_cp` fixture on the collinear UKS path with closed-shell
/// fillings (n_alpha = n_beta = 1): both spin channels non-empty, and the
/// RKS total density is exactly Pα + Pβ, so the spin machinery is exercised
/// without changing the physics of the tiny H2 fixture.
ATV::cpu::gpw::GpwDft small_cfg_cp_uks(double box = 8.0) {
    ATV::cpu::gpw::GpwDft cfg = small_cfg_cp(box);
    cfg.n_occ = 0;
    cfg.n_alpha = 1;
    cfg.n_beta = 1;
    return cfg;
}

} // namespace

TEST(GpwSolverFixedDensityKUks, DualSpinFixedDensityExportsHkSkBothSpins) {
    const auto atoms = small_h2_cp(/*box=*/8.0);
    ATV::cpu::gpw::GpwDft cold = small_cfg_cp_uks(/*box=*/8.0);
    cold.k_mesh[0] = 1;
    cold.k_mesh[1] = 1;
    cold.k_mesh[2] = 2;
    cold.k_time_reversal = false;
    cold.nonperiodic = false;
    cold.build(atoms);
    const auto ref = cold.run(atoms);
    ASSERT_TRUE(ref.converged);
    ASSERT_GE(ref.n_kpts, 2);
    ASSERT_EQ(static_cast<int>(ref.P_k.size()), ref.n_kpts);
    ASSERT_EQ(static_cast<int>(ref.P_k_beta.size()), ref.n_kpts);

    ATV::cpu::gpw::GpwDft fix = small_cfg_cp_uks(/*box=*/8.0);
    fix.k_mesh[0] = 1;
    fix.k_mesh[1] = 1;
    fix.k_mesh[2] = 2;
    fix.k_time_reversal = false;
    fix.nonperiodic = false;
    fix.max_iter = 0;
    fix.build(atoms);
    const auto& r = fix.run(atoms, /*guess_C=*/nullptr, /*guess_P=*/nullptr,
                            /*guess_P_beta=*/nullptr, &ref.P_k,
                            &ref.P_k_beta);
    EXPECT_TRUE(r.converged);
    EXPECT_EQ(r.iters, 0);
    EXPECT_TRUE(std::isfinite(r.e_total));
    EXPECT_EQ(static_cast<int>(r.H_ks_k.size()), r.n_kpts);
    EXPECT_EQ(static_cast<int>(r.H_ks_k_beta.size()), r.n_kpts);
    EXPECT_EQ(static_cast<int>(r.S_k.size()), r.n_kpts);
    EXPECT_EQ(static_cast<int>(r.P_k.size()), r.n_kpts);
    EXPECT_EQ(static_cast<int>(r.P_k_beta.size()), r.n_kpts);
    // Energy near the cold SCF (same dual-spin density).
    EXPECT_NEAR(r.e_total, ref.e_total, 1e-6);
}

TEST(GpwSolverFixedDensityKUks, DualSpinFixedDensityWithoutBetaGuessThrows) {
    const auto atoms = small_h2_cp(/*box=*/8.0);
    ATV::cpu::gpw::GpwDft cold = small_cfg_cp_uks(/*box=*/8.0);
    cold.k_mesh[2] = 2;
    cold.k_time_reversal = false;
    cold.nonperiodic = false;
    cold.build(atoms);
    const auto ref = cold.run(atoms);
    ASSERT_TRUE(ref.converged);

    ATV::cpu::gpw::GpwDft fix = small_cfg_cp_uks(/*box=*/8.0);
    fix.k_mesh[2] = 2;
    fix.k_time_reversal = false;
    fix.nonperiodic = false;
    fix.max_iter = 0;
    fix.build(atoms);
    // α-only guess: the UKS k path refuses a partial spin guess.
    EXPECT_THROW(fix.run(atoms, /*guess_C=*/nullptr, /*guess_P=*/nullptr,
                         /*guess_P_beta=*/nullptr, &ref.P_k),
                 std::invalid_argument);
}

TEST(GpwSolverFixedDensityKUks, GammaLimitMatchesGammaUksFixedDensity) {
    // A3: one explicit Γ point on the k path must reproduce the shipped
    // Γ dual-spin fixed-density eval (energy and per-spin KS matrices) —
    // the Γ-limit identity anchoring multi-k UKS CPMD to the cpmd-uks
    // product surface.
    const auto atoms = small_h2_cp(/*box=*/8.0);

    // Γ-UKS cold SCF supplies the dual densities.
    ATV::cpu::gpw::GpwDft cold = small_cfg_cp_uks(/*box=*/8.0);
    cold.nonperiodic = false;
    cold.build(atoms);
    const auto ref = cold.run(atoms);
    ASSERT_TRUE(ref.converged);
    ASSERT_GT(ref.P.size(), 0u);
    ASSERT_GT(ref.P_beta.size(), 0u);

    // Γ dual-spin fixed-density eval (the cpmd-uks product path).
    ATV::cpu::gpw::GpwDft fix_g = small_cfg_cp_uks(/*box=*/8.0);
    fix_g.nonperiodic = false;
    fix_g.max_iter = 0;
    fix_g.build(atoms);
    const auto rg = fix_g.run(atoms, /*guess_C=*/nullptr, &ref.P, &ref.P_beta);
    ASSERT_TRUE(rg.converged);
    ASSERT_EQ(rg.iters, 0);
    ASSERT_GT(rg.H_ks.size(), 0u);
    ASSERT_GT(rg.H_ks_beta.size(), 0u);

    // Same densities, complexified, on the k path at one explicit Γ point.
    std::vector<ATV::cpu::linalg::CplxMatrix> Pa_k(1), Pb_k(1);
    const auto nbs = ref.P.shape(0);
    Pa_k[0] = ATV::cpu::linalg::CplxMatrix(
        xt::zeros<std::complex<double>>({nbs, nbs}));
    Pb_k[0] = ATV::cpu::linalg::CplxMatrix(
        xt::zeros<std::complex<double>>({nbs, nbs}));
    for (std::size_t i = 0; i < nbs; ++i)
        for (std::size_t j = 0; j < nbs; ++j) {
            Pa_k[0](i, j) = ref.P(i, j);
            Pb_k[0](i, j) = ref.P_beta(i, j);
        }
    ATV::cpu::gpw::GpwDft fix_k = small_cfg_cp_uks(/*box=*/8.0);
    fix_k.nonperiodic = false;
    fix_k.k_points = {{0.0, 0.0, 0.0}};
    fix_k.k_point_weights = {1.0};
    fix_k.max_iter = 0;
    fix_k.build(atoms);
    const auto rk = fix_k.run(atoms, /*guess_C=*/nullptr, /*guess_P=*/nullptr,
                              /*guess_P_beta=*/nullptr, &Pa_k, &Pb_k);
    ASSERT_TRUE(rk.converged);
    ASSERT_EQ(rk.iters, 0);
    ASSERT_EQ(rk.n_kpts, 1);
    ASSERT_EQ(static_cast<int>(rk.H_ks_k.size()), 1);
    ASSERT_EQ(static_cast<int>(rk.H_ks_k_beta.size()), 1);

    EXPECT_NEAR(rk.e_total, rg.e_total, 1e-7);
    for (std::size_t i = 0; i < nbs; ++i)
        for (std::size_t j = 0; j < nbs; ++j) {
            EXPECT_NEAR(rk.H_ks_k[0](i, j).real(), rg.H_ks(i, j), 1e-7)
                << "alpha i=" << i << " j=" << j;
            EXPECT_NEAR(rk.H_ks_k[0](i, j).imag(), 0.0, 1e-9)
                << "alpha i=" << i << " j=" << j;
            EXPECT_NEAR(rk.H_ks_k_beta[0](i, j).real(), rg.H_ks_beta(i, j),
                        1e-7)
                << "beta i=" << i << " j=" << j;
            EXPECT_NEAR(rk.H_ks_k_beta[0](i, j).imag(), 0.0, 1e-9)
                << "beta i=" << i << " j=" << j;
        }
}

// ── Standalone overlap parity (esmm-cpmd-06-single-eval, ac-001) ────────
//
// Every GPW CPMD production step is about to build S standalone
// (`GpwDft::overlap` / `overlap_k`), align the propagated coefficients to
// it, and run exactly ONE fixed-density KS evaluation — instead of the
// old discard-and-rerun double eval. That deletion is only safe because S
// is geometry-only: the standalone overlap must be bit-identical to the
// `.S` / `.S_k` a fixed-density `run()` produces at the same geometry
// (same image counts, same summation order — extraction, not
// re-implementation). Reuses the tiny inline H2 `GpwDft` fixture above
// (`small_h2_cp` / `small_cfg_cp`, shared with `GpwSolverFixedDensityK`).

TEST(GpwSolverOverlap, StandaloneOverlapMatchesFixedDensityRun) {
    const auto atoms = small_h2_cp(/*box=*/8.0);

    // Converged cold SCF supplies the density guess the fixed-density
    // (max_iter == 0) path requires (see MultiKFixedDensityWithoutGuessThrows).
    ATV::cpu::gpw::GpwDft cold = small_cfg_cp(/*box=*/8.0);
    cold.build(atoms);
    const auto ref = cold.run(atoms);
    ASSERT_TRUE(ref.converged);

    ATV::cpu::gpw::GpwDft fix = small_cfg_cp(/*box=*/8.0);
    fix.max_iter = 0;
    fix.build(atoms);
    const auto& r = fix.run(atoms, /*guess_C=*/nullptr, &ref.P);
    ASSERT_TRUE(r.converged);
    ASSERT_EQ(r.iters, 0);
    ASSERT_GT(r.S.size(), 0u);

    // Standalone overlap at the SAME geometry, on the SAME solver instance.
    const auto s = fix.overlap(atoms);
    ASSERT_EQ(s.shape(0), r.S.shape(0));
    ASSERT_EQ(s.shape(1), r.S.shape(1));
    for (std::size_t i = 0; i < s.shape(0); ++i)
        for (std::size_t j = 0; j < s.shape(1); ++j)
            EXPECT_EQ(s(i, j), r.S(i, j)) << "i=" << i << " j=" << j;
}

TEST(GpwSolverOverlap, StandaloneOverlapKMatchesFixedDensityRun) {
    const auto atoms = small_h2_cp(/*box=*/8.0);

    ATV::cpu::gpw::GpwDft cold = small_cfg_cp(/*box=*/8.0);
    cold.k_mesh[0] = 1;
    cold.k_mesh[1] = 1;
    cold.k_mesh[2] = 2;
    cold.k_time_reversal = false;
    cold.nonperiodic = false;
    cold.build(atoms);
    const auto ref = cold.run(atoms);
    ASSERT_TRUE(ref.converged);
    ASSERT_GE(ref.n_kpts, 2);

    ATV::cpu::gpw::GpwDft fix = small_cfg_cp(/*box=*/8.0);
    fix.k_mesh[0] = 1;
    fix.k_mesh[1] = 1;
    fix.k_mesh[2] = 2;
    fix.k_time_reversal = false;
    fix.nonperiodic = false;
    fix.max_iter = 0;
    fix.build(atoms);
    const auto& r = fix.run(atoms, /*guess_C=*/nullptr, /*guess_P=*/nullptr,
                             /*guess_P_beta=*/nullptr, &ref.P_k);
    ASSERT_TRUE(r.converged);
    ASSERT_EQ(r.iters, 0);
    ASSERT_EQ(static_cast<int>(r.S_k.size()), r.n_kpts);

    // Standalone multi-k overlap at the SAME geometry, on the SAME solver.
    const auto s_k = fix.overlap_k(atoms);
    ASSERT_EQ(static_cast<int>(s_k.size()), r.n_kpts);
    for (int ik = 0; ik < r.n_kpts; ++ik) {
        const auto& s_ref = r.S_k[static_cast<std::size_t>(ik)];
        const auto& s_new = s_k[static_cast<std::size_t>(ik)];
        ASSERT_EQ(s_new.shape(0), s_ref.shape(0));
        ASSERT_EQ(s_new.shape(1), s_ref.shape(1));
        for (std::size_t i = 0; i < s_ref.shape(0); ++i)
            for (std::size_t j = 0; j < s_ref.shape(1); ++j) {
                EXPECT_EQ(s_new(i, j).real(), s_ref(i, j).real())
                    << "k=" << ik << " i=" << i << " j=" << j;
                EXPECT_EQ(s_new(i, j).imag(), s_ref(i, j).imag())
                    << "k=" << ik << " i=" << i << " j=" << j;
            }
    }
}
