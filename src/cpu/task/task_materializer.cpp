/// @file cpu/task/task_materializer.cpp
/// @brief Typed TaskSpec registration replay onto MD/Evaluate/Minimize.

#include "atomiverse/cpu/task/task_materializer.h"

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#ifdef ATV_HAS_MD
#include "atomiverse/cpu/dynamics/barostat.h"
#include "atomiverse/cpu/dynamics/fix_csvr.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/parrinello_rahman_npt.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#endif
#ifdef ATV_HAS_KSPACE
#include "atomiverse/cpu/longrange/pme.h"
#endif
#ifdef ATV_HAS_MINIMIZER
#include "atomiverse/cpu/minimize/cg.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#endif
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#ifdef ATV_HAS_PW
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#endif
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_fene.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/improper_harmonic.h"
#include "atomiverse/cpu/potentials/pair_born.h"
#include "atomiverse/cpu/potentials/pair_buck.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_dpd.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_morse.h"
#include "atomiverse/cpu/potentials/pair_wca.h"
#include "atomiverse/cpu/potentials/pair_yukawa.h"
#include "atomiverse/cpu/record/thermo.h"
#ifdef ATV_HAS_IO
#include "atomiverse/cpu/record/xyz_recorder.h"
#endif
#ifdef ATV_HAS_SEMIEMPIRICAL
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#endif
#ifdef ATV_HAS_MINIMIZER
#include "atomiverse/cpu/task/internal/minimize_builders.h"
#endif
#include "atomiverse/cpu/task/internal/potential_builders.h"

namespace ATV::cpu::task {
namespace {

// Only the MD replay names it.
#if defined(ATV_HAS_MD) && !defined(ATV_HAS_KSPACE)
/// @brief Reject a `KSpacePmeParams` entry in a build without the PME solver.
///
/// PME is smooth Particle-Mesh Ewald, the reciprocal-space half of the
/// Ewald-split Coulomb sum; its solver needs FFTW3 and is therefore behind
/// the `ATV_BUILD_KSPACE` build flag, while the wire mirror
/// `KSpacePmeParams` is compiled unconditionally. A payload naming PME
/// therefore still *decodes* on a build without the solver and has to be
/// refused here, at the point where the component would be constructed.
///
/// Both replay paths — explicit binding and categorical setter — hit the same
/// wall when the reciprocal solver was compiled out; one thrower is what keeps
/// the two diagnostics from drifting apart.
///
/// @throws std::runtime_error Always, naming the flag to rebuild with.
[[noreturn]] void throw_kspace_not_built() {
    throw std::runtime_error("KSpacePME requires ATV_BUILD_KSPACE=ON (PME "
                             "reciprocal solver not compiled in)");
}
#endif

// Only the MD replay names it.
#if defined(ATV_HAS_MD) && !defined(ATV_HAS_IO)
/// @brief Reject an `XyzRecorderParams` entry in a build without the writer.
///
/// `cpu::XyzRecorder` writes through molrs and is therefore behind the
/// `ATV_BUILD_IO` build flag, while the wire mirror `XyzRecorderParams` and
/// its component ID are compiled unconditionally. A payload naming the
/// recorder therefore still *decodes* on an IO-less build and has to be
/// refused here, at the point where the component would be constructed —
/// running a trajectory-less MD instead would be a silent no-op.
///
/// Both replay paths — explicit binding and categorical setter — hit the same
/// wall; one thrower is what keeps the two diagnostics from drifting apart.
///
/// @throws std::runtime_error Always, naming the flag to rebuild with.
[[noreturn]] void throw_io_not_built() {
    throw std::runtime_error("XyzRecorder requires ATV_BUILD_IO=ON "
                             "(trajectory writer not compiled in)");
}
#endif

// Only the MD replay names it.
#if defined(ATV_HAS_MD) && !defined(ATV_HAS_PW)
/// @brief Reject a `PwEvaluatorParams` entry in a build without the solver.
///
/// `cpu::pw::PW` is plane-wave density-functional theory — the electronic
/// structure is expanded in plane waves, which needs FFTW3, so the whole
/// domain sits behind the `ATV_BUILD_PW` build flag while the wire mirror
/// `PwEvaluatorParams` and its component ID are compiled unconditionally. A
/// payload naming PW therefore still *decodes* on a PW-less build and has to
/// be refused here, at the point where the component would be constructed:
/// the alternative is an MD run whose only force source silently vanished.
///
/// Both replay paths — explicit binding and categorical setter — hit the same
/// wall; one thrower is what keeps the two diagnostics from drifting apart.
///
/// @throws std::runtime_error Always, naming the flag to rebuild with.
[[noreturn]] void throw_pw_not_built() {
    throw std::runtime_error("PWEvaluator requires ATV_BUILD_PW=ON "
                             "(plane-wave solver not compiled in)");
}
#endif

#ifndef ATV_HAS_SEMIEMPIRICAL
/// @brief Reject an `AM1EvaluatorParams` entry, or a `SinglePointTask`, in a
///        build without the semiempirical domain.
///
/// `cpu::semiempirical::AM1` sits behind the `ATV_BUILD_SEMIEMPIRICAL` build
/// flag, while the wire mirror `AM1EvaluatorParams` is compiled
/// unconditionally. A payload naming AM1 therefore still *decodes* on an
/// SE-less build and has to be refused here, at the point where the
/// component would be constructed. Every AM1 home — explicit binding,
/// categorical setter, single point — hits this one thrower.
///
/// @throws std::runtime_error Always, naming the flag to rebuild with.
[[noreturn]] void throw_semiempirical_not_built() {
    throw std::runtime_error("AM1Evaluator requires ATV_BUILD_SEMIEMPIRICAL=ON "
                             "(semiempirical domain not compiled in)");
}
#endif

#ifndef ATV_HAS_MD
/// @brief Reject an `MDTask` in a build without the MD domain.
///
/// `TaskSpec` and its wire codec are compiled unconditionally, so an MD
/// payload still *decodes* on an MD-less build and has to be refused where
/// the driver would be constructed.
///
/// @throws std::runtime_error Always, naming the flag to rebuild with.
[[noreturn]] void throw_md_not_built() {
    throw std::runtime_error("MDTask requires ATV_BUILD_MD=ON "
                             "(molecular dynamics not compiled in)");
}
#endif

#ifndef ATV_HAS_EVALUATE
/// @brief Reject a single-point task when the Evaluate domain is disabled.
/// @throws std::runtime_error Always, naming the required build flag.
[[noreturn]] void throw_evaluate_not_built() {
    throw std::runtime_error("SinglePointTask requires ATV_BUILD_EVALUATE=ON "
                             "(single-point evaluation not compiled in)");
}
#endif

#ifndef ATV_HAS_MINIMIZER
/// @brief Reject a `MinimizeTask` in a build without the minimizer domain.
///
/// Same shape as `throw_md_not_built`: the payload decodes, the driver and
/// its optimizers were compiled out.
///
/// @throws std::runtime_error Always, naming the flag to rebuild with.
[[noreturn]] void throw_minimizer_not_built() {
    throw std::runtime_error("MinimizeTask requires ATV_BUILD_MINIMIZER=ON "
                             "(geometry optimization not compiled in)");
}
#endif

#ifdef ATV_HAS_MD
/// @brief Build a `FixLangevin` with its temperature-ramp span installed.
///
/// A *temperature ramp* moves the thermostat setpoint linearly from
/// `T_start` to `T_stop` over a *run span* — the first and last MD step
/// index of the run, which is what LAMMPS reads from
/// `update->beginstep`/`endstep`. The component cannot know that span from
/// its constructor arguments, so whoever owns the step budget installs it:
/// this materializer on the launcher path, `MdHandle::run` on the Python
/// path, and the caller itself in hand-written C++.
///
/// The span is installed unconditionally: `T_stop < 0` folds to `T_start` in
/// the constructor, so a span on a component that was never asked to ramp
/// interpolates between two equal temperatures and is inert.
///
/// @param p       Typed Langevin params (temperatures [K], damp/dt [fs]).
/// @param n_steps Step budget of the owning MDTask — ramp span `[0, n_steps]`.
/// @return Thermostat with its ramp span wired, ready for either
///         registration path (explicit binding or categorical setter).
cpu::md::FixLangevin ramped_fix_langevin(const FixLangevinParams& p,
                                         std::int64_t n_steps) {
    cpu::md::FixLangevin lang{p.T_start, p.T_stop, p.damp, p.dt, p.seed};
    lang.set_run_span(0, n_steps);
    return lang;
}

/// @brief Build a `NoseHooverNPT` with its temperature-ramp span installed.
///
/// Same unconditional span as `ramped_fix_langevin`, for the same reason:
/// `T_stop < 0` folds to `T_target` in the constructor, so the span is inert
/// unless a ramp was actually requested. The constructor argument order is
/// the ten-argument one mirrored by `NoseHooverNptParams`; passing the
/// fields positionally is what keeps the two in step.
///
/// @param p       Typed Martyna-Tobias-Klein NPT params (temperatures [K],
///                pressure [eV/Å³], coupling times and timestep [fs]).
/// @param n_steps Step budget of the owning MDTask — ramp span `[0, n_steps]`.
/// @return Integrator with its ramp span wired, ready for either
///         registration path (explicit binding or categorical setter).
cpu::md::NoseHooverNPT ramped_nose_hoover_npt(const NoseHooverNptParams& p,
                                              std::int64_t n_steps) {
    cpu::md::NoseHooverNPT nh{p.T_target,    p.tau_T, p.P_target, p.tau_P,
                              p.dt,          p.drag,  p.mtchain,  p.mpchain,
                              p.couple_axis, p.T_stop};
    nh.set_run_span(0, n_steps);
    return nh;
}

/// @brief Build an `AM1Evaluator` that computes nuclear forces.
///
/// The same `AM1EvaluatorParams` payload serves a `SinglePointTask`, where the
/// caller reads back energy and Mulliken charges, and an `MDTask`, where the
/// evaluator is a *force* source — Born-Oppenheimer molecular dynamics, one
/// SCF solve per step with the nuclei moving on the resulting energy surface.
/// Forces are the point of the MD home, so the flag is spelled out here
/// rather than inherited from the component default: were that default to
/// flip, an MD run would keep integrating a frame the SCF never pushed on —
/// silently, since a force-free evaluator still publishes a plausible
/// energy.
///
/// Both replay paths (explicit binding and categorical setter) go through
/// this one builder, so neither can drift into registering a gradient-less
/// AM1 behind the other's back.
///
/// @param p Typed AM1 params (charge [e], multiplicity, SCF iteration cap,
///          density residual tolerance, energy tolerance [eV]).
/// @return Evaluator with gradients enabled, ready for either registration
///         path.
#ifdef ATV_HAS_SEMIEMPIRICAL
cpu::semiempirical::AM1Evaluator gradient_am1(const AM1EvaluatorParams& p) {
    cpu::semiempirical::AM1Evaluator evaluator{
        p.charge, p.multiplicity, p.max_iter, p.density_tol, p.energy_tol};
    evaluator.with_gradient(true);
    return evaluator;
}
#endif

#ifdef ATV_HAS_PW
/// @brief Build a `PWEvaluator` that computes nuclear forces.
///
/// Plane-wave Born-Oppenheimer dynamics: the electronic structure is
/// re-converged at every geometry the nuclei reach, and the ionic forces are
/// analytic (local Hellmann-Feynman + Ewald + Kleinman-Bylander) at the
/// converged density rather than differenced from energies. A force-producing
/// step costs two self-consistent-field solves, not one — the energy solve
/// (`PW::run`), then the force path's own (`pw_forces_analytic`), which is
/// what publishes the forces.
///
/// Gradients are spelled out here rather than inherited from the component
/// default (`false`) for the same reason as `gradient_am1`: a force-free
/// evaluator still publishes a plausible energy, so the MD would integrate a
/// frame nothing ever pushed on.
///
/// The fifteen members are assigned in the params' DECLARATION order — the
/// order the codec writes and a hand-built twin spells — so a dropped or
/// swapped field is a visible diff rather than a silent C++ default. The two
/// enum mirrors are cast back from their wire integers here, the one place
/// that knows both the flag-gated enums and the flagless wire struct.
///
/// @param p Typed plane-wave params (cutoff [Ry], k-mesh, SCF tolerance and
///          cap, mixing, functional, spin, smearing [Ry]).
/// @return Evaluator owning a freshly configured solver, gradients enabled,
///         ready for either registration path.
cpu::pw::PWEvaluator gradient_pw(const PwEvaluatorParams& p) {
    cpu::pw::PW pw;
    pw.upf_path = p.upf_path;
    pw.gth_name = p.gth_name;
    pw.ecut_ry = p.ecut_ry;
    pw.k_mesh = p.k_mesh;
    pw.tol = p.tol;
    pw.max_iter = p.max_iter;
    pw.mix_beta = p.mix_beta;
    pw.pulay_depth = p.pulay_depth;
    pw.xc = static_cast<cpu::scf::XCType>(p.xc);
    pw.spin_polarized = p.spin_polarized;
    pw.multiplicity = p.multiplicity;
    pw.smearing = static_cast<cpu::pw::Smearing>(p.smearing);
    pw.degauss = p.degauss;
    pw.mp_order = p.mp_order;
    pw.extra_bands = p.extra_bands;
    // `PW` is move-only (it owns FFTW plans and SCF workspaces), so the
    // solver moves into the evaluator.
    return cpu::pw::PWEvaluator{std::move(pw)}.with_gradient(true);
}
#endif

/// @brief Replay one typed MD entry onto the driver.
///
/// Two registration paths, chosen by whether the entry carries a binding:
/// with one, the component goes to `on(phase).add(component, order)`;
/// without one, to the categorical setter for its kind (`set_integrator`,
/// `set_thermostat`, `set_barostat`, …). Both paths construct the same
/// component from the same fields — the span-wiring helpers above exist so
/// that a ramping component cannot reach only one of the two with its run
/// span installed.
///
/// Nothing is cross-validated here: this function replays one entry and
/// knows nothing about the others. The rule that `CoulLong` and
/// `KSpacePME` are the two halves of one Ewald split — exactly one of each,
/// at the same splitting parameter α — is checked once by `md_->build()`
/// in the `TaskMaterializer` constructor, so a payload naming only one half
/// registers cleanly and then fails at build with a named diagnostic.
///
/// @param driver  Target MD driver.
/// @param entry   Typed params + optional explicit phase binding.
/// @param n_steps Step budget of the owning MDTask — the ramp span
///                `[0, n_steps]` handed to the temperature-ramping
///                components (LAMMPS `update->beginstep`/`endstep`).
void register_md_entry(md::MD& driver, const MDComponentEntry& entry,
                       std::int64_t n_steps) {
    std::visit(
        [&](const auto& p) {
            using P = std::decay_t<decltype(p)>;
            if (entry.binding) {
                auto bind = [&](auto component) {
                    driver.on(entry.binding->phase)
                        .add(std::move(component), entry.binding->order);
                };
                if constexpr (std::is_same_v<P, PairLJCutParams>)
                    bind(cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
                else if constexpr (std::is_same_v<P, NeighborListParams>)
                    bind(cpu::md::NeighborList<>{p.cutoff, p.skin});
                else if constexpr (std::is_same_v<P, BondHarmonicParams>)
                    bind(cpu::BondHarmonic{p.K, p.r0});
                else if constexpr (std::is_same_v<P, AngleHarmonicParams>)
                    bind(cpu::AngleHarmonic{p.K, p.theta0});
                else if constexpr (std::is_same_v<P,
                                                  DihedralMultiHarmonicParams>)
                    bind(cpu::DihedralMultiHarmonic{p.Cn});
                else if constexpr (std::is_same_v<P, PairMorseParams>)
                    bind(cpu::PairMorse{p.D, p.alpha, p.r0, p.cutoff});
                else if constexpr (std::is_same_v<P, PairBuckParams>)
                    bind(cpu::PairBuck{p.A, p.rho, p.C, p.cutoff});
                else if constexpr (std::is_same_v<P, PairBornParams>)
                    // Parallel vectors transposed back into the coefficient
                    // rows the C++ constructor takes; the multi-type
                    // constructor covers the single-type one as a one-row
                    // table.
                    bind(cpu::PairBorn{internal::born_table(p), p.cutoff});
                else if constexpr (std::is_same_v<P, PairYukawaParams>)
                    bind(cpu::PairYukawa{p.A, p.kappa, p.cutoff});
                else if constexpr (std::is_same_v<P, PairWcaParams>)
                    // No cutoff argument: WCA derives it as 2^(1/6)·sigma.
                    bind(cpu::PairWCA{p.epsilon, p.sigma});
                else if constexpr (std::is_same_v<P, PairDpdParams>)
                    // The scalar-amplitude constructor; the multi-component
                    // `a_matrix` one has no wire mirror.
                    bind(cpu::PairDpd{p.a, p.gamma, p.temperature, p.cutoff,
                                      p.dt, p.seed});
                else if constexpr (std::is_same_v<P, BondFeneParams>)
                    bind(cpu::BondFene{p.K, p.R0});
                else if constexpr (std::is_same_v<P, DihedralCharmmParams>)
                    bind(cpu::DihedralCharmm{p.K, p.n, p.delta});
                else if constexpr (std::is_same_v<P, ImproperHarmonicParams>)
                    bind(cpu::ImproperHarmonic{p.K, p.chi0});
                else if constexpr (std::is_same_v<P, PairCoulCutParams>)
                    bind(cpu::PairCoulCut{p.cutoff});
                else if constexpr (std::is_same_v<P, CoulLongParams>)
                    bind(cpu::CoulLong{p.alpha, p.cutoff});
                else if constexpr (std::is_same_v<P, KSpacePmeParams>) {
#ifdef ATV_HAS_KSPACE
                    // Move-only (it owns FFTW plans): the prvalue goes
                    // straight into the sink, no named local to move from.
                    bind(cpu::KSpacePME{p.alpha, p.grid_spacing, p.order});
#else
                    throw_kspace_not_built();
#endif
                } else if constexpr (std::is_same_v<P, VelocityVerletParams>)
                    bind(cpu::md::VelocityVerlet{p.dt});
                else if constexpr (std::is_same_v<P, InitVelocityParams>)
                    bind(cpu::md::InitVelocity{p.temperature, p.seed});
                else if constexpr (std::is_same_v<P, NoseHooverParams>)
                    bind(cpu::md::NoseHoover{p.T_start, p.tau_T, p.dt});
                else if constexpr (std::is_same_v<P, FixLangevinParams>)
                    bind(ramped_fix_langevin(p, n_steps));
                else if constexpr (std::is_same_v<P, FixCsvrParams>)
                    bind(cpu::md::FixCSVR{p.T_start, p.tau, p.dt, p.seed});
                else if constexpr (std::is_same_v<P, NoseHooverNptParams>)
                    bind(ramped_nose_hoover_npt(p, n_steps));
                else if constexpr (std::is_same_v<P, ParrinelloRahmanNptParams>)
                    // No temperature ramp on PR-NPT — no span to install.
                    bind(cpu::md::ParrinelloRahmanNPT{p.T_target, p.P_target,
                                                      p.tau_T, p.tau_P, p.dt});
                else if constexpr (std::is_same_v<P, BerendsenBarostatParams>)
                    bind(cpu::md::Barostat{p.P_target, p.tau_P, p.dt,
                                           p.bulk_modulus});
                else if constexpr (std::is_same_v<P, AM1EvaluatorParams>) {
#ifdef ATV_HAS_SEMIEMPIRICAL
                    bind(gradient_am1(p));
#else
                    throw_semiempirical_not_built();
#endif
                } else if constexpr (std::is_same_v<P, PwEvaluatorParams>) {
#ifdef ATV_HAS_PW
                    bind(gradient_pw(p));
#else
                    throw_pw_not_built();
#endif
                } else if constexpr (std::is_same_v<P, ThermoParams>)
                    bind(cpu::Thermo{p.freq});
                else if constexpr (std::is_same_v<P, XyzRecorderParams>) {
#ifdef ATV_HAS_IO
                    bind(cpu::XyzRecorder{p.filename, p.freq});
#else
                    throw_io_not_built();
#endif
                } else
                    static_assert(detail::unsupported_component_v<P>);
                return;
            }
            // Categorical path — no public Driver::add
            if constexpr (std::is_same_v<P, PairLJCutParams>)
                driver.set_evaluator(
                    cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
            else if constexpr (std::is_same_v<P, NeighborListParams>)
                driver.set_neighbor(cpu::md::NeighborList<>{p.cutoff, p.skin});
            else if constexpr (std::is_same_v<P, BondHarmonicParams>)
                driver.set_evaluator(cpu::BondHarmonic{p.K, p.r0});
            else if constexpr (std::is_same_v<P, AngleHarmonicParams>)
                driver.set_evaluator(cpu::AngleHarmonic{p.K, p.theta0});
            else if constexpr (std::is_same_v<P, DihedralMultiHarmonicParams>)
                driver.set_evaluator(cpu::DihedralMultiHarmonic{p.Cn});
            else if constexpr (std::is_same_v<P, PairMorseParams>)
                driver.set_evaluator(
                    cpu::PairMorse{p.D, p.alpha, p.r0, p.cutoff});
            else if constexpr (std::is_same_v<P, PairBuckParams>)
                driver.set_evaluator(cpu::PairBuck{p.A, p.rho, p.C, p.cutoff});
            else if constexpr (std::is_same_v<P, PairBornParams>)
                driver.set_evaluator(
                    cpu::PairBorn{internal::born_table(p), p.cutoff});
            else if constexpr (std::is_same_v<P, PairYukawaParams>)
                driver.set_evaluator(cpu::PairYukawa{p.A, p.kappa, p.cutoff});
            else if constexpr (std::is_same_v<P, PairWcaParams>)
                driver.set_evaluator(cpu::PairWCA{p.epsilon, p.sigma});
            else if constexpr (std::is_same_v<P, PairDpdParams>)
                driver.set_evaluator(cpu::PairDpd{p.a, p.gamma, p.temperature,
                                                  p.cutoff, p.dt, p.seed});
            else if constexpr (std::is_same_v<P, BondFeneParams>)
                driver.set_evaluator(cpu::BondFene{p.K, p.R0});
            else if constexpr (std::is_same_v<P, DihedralCharmmParams>)
                driver.set_evaluator(cpu::DihedralCharmm{p.K, p.n, p.delta});
            else if constexpr (std::is_same_v<P, ImproperHarmonicParams>)
                driver.set_evaluator(cpu::ImproperHarmonic{p.K, p.chi0});
            else if constexpr (std::is_same_v<P, PairCoulCutParams>)
                driver.set_evaluator(cpu::PairCoulCut{p.cutoff});
            else if constexpr (std::is_same_v<P, CoulLongParams>)
                driver.set_evaluator(cpu::CoulLong{p.alpha, p.cutoff});
            else if constexpr (std::is_same_v<P, KSpacePmeParams>) {
#ifdef ATV_HAS_KSPACE
                // Move-only (it owns FFTW plans): the prvalue goes straight
                // into the owning setter, no named local to move from.
                driver.set_evaluator(
                    cpu::KSpacePME{p.alpha, p.grid_spacing, p.order});
#else
                throw_kspace_not_built();
#endif
            } else if constexpr (std::is_same_v<P, VelocityVerletParams>)
                driver.set_integrator(cpu::md::VelocityVerlet{p.dt});
            else if constexpr (std::is_same_v<P, InitVelocityParams>)
                driver.set_observer(
                    cpu::md::InitVelocity{p.temperature, p.seed});
            else if constexpr (std::is_same_v<P, NoseHooverParams>)
                driver.set_thermostat(
                    cpu::md::NoseHoover{p.T_start, p.tau_T, p.dt});
            else if constexpr (std::is_same_v<P, FixLangevinParams>)
                driver.set_thermostat(ramped_fix_langevin(p, n_steps));
            else if constexpr (std::is_same_v<P, FixCsvrParams>)
                driver.set_thermostat(
                    cpu::md::FixCSVR{p.T_start, p.tau, p.dt, p.seed});
            else if constexpr (std::is_same_v<P, NoseHooverNptParams>)
                driver.set_integrator(ramped_nose_hoover_npt(p, n_steps));
            else if constexpr (std::is_same_v<P, ParrinelloRahmanNptParams>)
                // No temperature ramp on PR-NPT — no span to install.
                driver.set_integrator(cpu::md::ParrinelloRahmanNPT{
                    p.T_target, p.P_target, p.tau_T, p.tau_P, p.dt});
            else if constexpr (std::is_same_v<P, BerendsenBarostatParams>)
                driver.set_barostat(cpu::md::Barostat{p.P_target, p.tau_P, p.dt,
                                                      p.bulk_modulus});
            else if constexpr (std::is_same_v<P, AM1EvaluatorParams>) {
#ifdef ATV_HAS_SEMIEMPIRICAL
                driver.set_evaluator(gradient_am1(p));
#else
                throw_semiempirical_not_built();
#endif
            } else if constexpr (std::is_same_v<P, PwEvaluatorParams>) {
#ifdef ATV_HAS_PW
                driver.set_evaluator(gradient_pw(p));
#else
                throw_pw_not_built();
#endif
            } else if constexpr (std::is_same_v<P, ThermoParams>)
                driver.set_thermo(p.freq);
            else if constexpr (std::is_same_v<P, XyzRecorderParams>) {
#ifdef ATV_HAS_IO
                driver.set_record(cpu::XyzRecorder{p.filename, p.freq});
#else
                throw_io_not_built();
#endif
            } else
                static_assert(detail::unsupported_component_v<P>);
        },
        entry.params);
}

#endif // ATV_HAS_MD

#ifdef ATV_HAS_MINIMIZER
/// @brief Replay one typed Minimize entry onto the driver.
///
/// Same two paths as `register_md_entry`, minus the ramp wiring: no
/// minimizer component carries a temperature setpoint, so none needs a span.
///
/// @param driver Target minimizer driver.
/// @param entry  Typed params + optional explicit phase binding.
void register_minimize_entry(minimizer::Minimize& driver,
                             const MinimizeComponentEntry& entry) {
    std::visit(
        [&](const auto& p) {
            using P = std::decay_t<decltype(p)>;
            if (entry.binding) {
                auto bind = [&](auto component) {
                    driver.on(entry.binding->phase)
                        .add(std::move(component), entry.binding->order);
                };
                if constexpr (std::is_same_v<P, PairLJCutParams>)
                    bind(cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
                else if constexpr (std::is_same_v<P, NeighborListParams>)
                    bind(cpu::md::NeighborList<>{p.cutoff, p.skin});
                else if constexpr (std::is_same_v<P, LBFGSParams>)
                    bind(cpu::minimizer::LBFGS{p.memory, p.alpha0, p.max_ls});
                else if constexpr (std::is_same_v<P, CGParams>)
                    bind(cpu::minimizer::CG{p.alpha0});
                else if constexpr (std::is_same_v<P, FIREParams>)
                    bind(internal::tuned_fire(p));
                else if constexpr (std::is_same_v<P, IsotropicCellRelaxParams>)
                    bind(internal::tuned_isotropic_cell_relax(p));
                else if constexpr (std::is_same_v<P,
                                                  AnisotropicCellRelaxParams>)
                    bind(internal::tuned_anisotropic_cell_relax(p));
                else if constexpr (std::is_same_v<P, TriclinicCellRelaxParams>)
                    bind(internal::tuned_triclinic_cell_relax(p));
                else if constexpr (std::is_same_v<P, JointIsotropicRelaxParams>)
                    bind(internal::tuned_joint_isotropic_relax(p));
                else
                    static_assert(detail::unsupported_component_v<P>);
                return;
            }
            if constexpr (std::is_same_v<P, PairLJCutParams>)
                driver.set_evaluator(
                    cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
            else if constexpr (std::is_same_v<P, NeighborListParams>)
                driver.set_neighbor(cpu::md::NeighborList<>{p.cutoff, p.skin});
            else if constexpr (std::is_same_v<P, LBFGSParams>)
                driver.set_optimizer(
                    cpu::minimizer::LBFGS{p.memory, p.alpha0, p.max_ls});
            else if constexpr (std::is_same_v<P, CGParams>)
                driver.set_optimizer(cpu::minimizer::CG{p.alpha0});
            else if constexpr (std::is_same_v<P, FIREParams>)
                driver.set_optimizer(internal::tuned_fire(p));
            else if constexpr (std::is_same_v<P, IsotropicCellRelaxParams>)
                driver.set_optimizer(internal::tuned_isotropic_cell_relax(p));
            else if constexpr (std::is_same_v<P, AnisotropicCellRelaxParams>)
                driver.set_optimizer(internal::tuned_anisotropic_cell_relax(p));
            else if constexpr (std::is_same_v<P, TriclinicCellRelaxParams>)
                driver.set_optimizer(internal::tuned_triclinic_cell_relax(p));
            else if constexpr (std::is_same_v<P, JointIsotropicRelaxParams>)
                driver.set_optimizer(internal::tuned_joint_isotropic_relax(p));
            else
                static_assert(detail::unsupported_component_v<P>);
        },
        entry.params);
}
#endif // ATV_HAS_MINIMIZER

} // namespace

TaskMaterializer::TaskMaterializer(const TaskSpec& spec) : kind_(spec.kind()) {
    if (kind_ == TaskKind::MD) {
#ifdef ATV_HAS_MD
        const auto& task = spec.md();
        budget_ = static_cast<int>(task.n_steps);
        md_.emplace();
        for (const auto& entry : task.entries)
            register_md_entry(*md_, entry, task.n_steps);
        md_->build();
        return;
#else
        throw_md_not_built();
#endif
    }

    if (kind_ == TaskKind::SinglePoint) {
#ifndef ATV_HAS_EVALUATE
        throw_evaluate_not_built();
#elif defined(ATV_HAS_SEMIEMPIRICAL)
        const auto& p = spec.single_point().evaluator;
        eval_.emplace();
        eval_->set_serial_on_replica();
        eval_->on(ATV::evaluate::Phase::evaluate)
            .add(semiempirical::AM1Evaluator{p.charge, p.multiplicity,
                                             p.max_iter, p.density_tol,
                                             p.energy_tol});
        eval_->build();
        return;
#else
        throw_semiempirical_not_built();
#endif
    }

#ifdef ATV_HAS_MINIMIZER
    const auto& task = spec.minimize();
    budget_ = static_cast<int>(task.max_iter);
    minimize_.emplace();
    internal::add_minimize_criterion(*minimize_, task.convergence);
    for (const auto& entry : task.entries)
        register_minimize_entry(*minimize_, entry);
    minimize_->build();
#else
    throw_minimizer_not_built();
#endif
}

// The constructor refuses a task kind whose domain is compiled out, so the
// guarded branches below are the only ones a live materializer can reach.

TaskResult TaskMaterializer::run(cpu::Frame& frame) {
    completed_dd_.reset();
#ifdef ATV_HAS_MD
    if (kind_ == TaskKind::MD) {
        const int iterations = md_->run(frame, budget_);
        return TaskResult{iterations, md_->converged(), md_->stop_reason()};
    }
#endif
#ifdef ATV_HAS_EVALUATE
    if (kind_ == TaskKind::SinglePoint) {
        eval_->run(frame);
        return TaskResult{1, false, {}};
    }
#endif
#ifdef ATV_HAS_MINIMIZER
    auto completed = minimize_->run_detached(frame, budget_);
    const int iterations = completed.iterations;
    completed_dd_ = std::move(completed.decomposition);
    return TaskResult{iterations, minimize_->converged(),
                      minimize_->stop_reason()};
#else
    throw_minimizer_not_built();
#endif
}

std::unique_ptr<md::mpi::DomainDecomposition> TaskMaterializer::release_dd() {
#ifdef ATV_HAS_MD
    if (kind_ == TaskKind::MD)
        return md_->release_dd();
#endif
    if (kind_ == TaskKind::Minimize)
        return std::move(completed_dd_);
    return nullptr;
}

const Plan<cpu::Component>& TaskMaterializer::plan() const {
#ifdef ATV_HAS_MD
    if (kind_ == TaskKind::MD)
        return md_->plan();
#endif
#ifdef ATV_HAS_EVALUATE
    if (kind_ == TaskKind::SinglePoint)
        return eval_->plan();
#endif
#ifdef ATV_HAS_MINIMIZER
    return minimize_->plan();
#else
    throw_minimizer_not_built();
#endif
}

} // namespace ATV::cpu::task
