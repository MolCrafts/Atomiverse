/// @file src/cpu/semiempirical/am1_solver.cpp
#include "atomiverse/cpu/semiempirical/am1_solver.h"

#include <stdexcept>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

AM1::AM1(int charge_, int multiplicity_, int max_iter_, double density_tol_,
         double energy_tol_)
    : charge(charge_), multiplicity(multiplicity_), max_iter(max_iter_),
      density_tol(density_tol_), energy_tol(energy_tol_) {}

AM1::~AM1() = default;

AM1::AM1(AM1&& other)
    : NddoMethod(std::move(other)), charge(other.charge),
      multiplicity(other.multiplicity), max_iter(other.max_iter),
      density_tol(other.density_tol), energy_tol(other.energy_tol),
      engine_(std::move(other.engine_)) {}

AM1::AM1(const AM1& other)
    : NddoMethod(), charge(other.charge), multiplicity(other.multiplicity),
      max_iter(other.max_iter), density_tol(other.density_tol),
      energy_tol(other.energy_tol) {}

void AM1::sync_engine_() {
    NDDOConfig c;
    c.method = ATV::semiempirical::SemiEmpiricalMethod::AM1;
    c.charge = charge;
    c.multiplicity = multiplicity;
    c.max_iter = max_iter;
    c.density_tol = density_tol;
    c.energy_tol = energy_tol;
    engine_ = std::make_unique<NDDOSolver>(c);
}

void AM1::build(const cpu::Frame& reference) {
    sync_engine_();
    engine_->build(reference);
}

AM1Result& AM1::run(const cpu::Frame& frame) {
    if (!engine_ || !engine_->built())
        build(frame);
    engine_->set_scf_controls(max_iter, density_tol, energy_tol);
    auto& result = drive_(*engine_, frame);
    engine_->commit_density();
    return result;
}

AM1Result& AM1::run(cpu::Frame& frame) {
    publish(frame, /*forces_requested=*/false, /*accumulate=*/false);
    return engine_->result();
}

AM1Result& AM1::forces(cpu::Frame& frame) {
    publish(frame, /*forces_requested=*/true, /*accumulate=*/false);
    return engine_->result();
}

const AM1Result& AM1::last_result() const {
    if (!engine_) {
        static AM1Result empty;
        return empty;
    }
    return engine_->last_result();
}

bool AM1::built() const noexcept { return engine_ && engine_->built(); }

bool AM1::has_density() const noexcept {
    return engine_ && engine_->has_density();
}

void AM1::prepare(const cpu::BuildContext& qm_ctx) {
    // Repeated parent passes must not erase the last accepted guess.
    // scf_prepare validates/rebuilds changed atom identities at evaluation.
    if (!built())
        build(qm_ctx.frame);
}

void AM1::publish(cpu::Frame& frame, bool forces_requested, bool accumulate) {
    if (!engine_ || !engine_->built())
        build(frame);
    engine_->set_scf_controls(max_iter, density_tol, energy_tol);
    publish_(*engine_, frame, forces_requested, accumulate);
}

void AM1::evaluate(const cpu::LaunchContext& qm_ctx) {
    // Private QM frame: overwrite energy/forces.
    publish_in_context_(qm_ctx, /*accumulate=*/false);
}

void AM1::reset_history() {
    // Drop persisted density by rebuilding the engine on the next run.
    engine_.reset();
}

const NDDOSystem& AM1::system() const {
    if (!engine_)
        throw std::runtime_error("AM1::system before build");
    return engine_->system();
}

const NDDOScfState& AM1::state() const {
    if (!engine_)
        throw std::runtime_error("AM1::state before build");
    return engine_->state();
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
