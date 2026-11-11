/// @file src/cpu/semiempirical/pm3_solver.cpp
#include "atomiverse/cpu/semiempirical/pm3_solver.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_solver.h"
#include <utility>

#include "atomiverse/core/frame_schema.h"
namespace ATV {
namespace cpu {
namespace semiempirical {
PM3::PM3(int charge_, int multiplicity_, int max_iter_, double density_tol_,
         double energy_tol_)
    : charge(charge_), multiplicity(multiplicity_), max_iter(max_iter_),
      density_tol(density_tol_), energy_tol(energy_tol_) {}

PM3::~PM3() = default;

PM3::PM3(PM3&& other)
    : NddoMethod(std::move(other)), charge(other.charge),
      multiplicity(other.multiplicity), max_iter(other.max_iter),
      density_tol(other.density_tol), energy_tol(other.energy_tol),
      engine_(std::move(other.engine_)) {}

void PM3::sync_engine_() {
    discard_evaluation();
    NDDOConfig c;
    c.method = ATV::semiempirical::SemiEmpiricalMethod::PM3;
    c.charge = charge;
    c.multiplicity = multiplicity;
    c.max_iter = max_iter;
    c.density_tol = density_tol;
    c.energy_tol = energy_tol;
    engine_ = std::make_unique<NDDOSolver>(c);
}
void PM3::build(const cpu::Frame& reference) {
    sync_engine_();
    engine_->build(reference);
}
PM3Result& PM3::run(const cpu::Frame& frame) {
    if (!engine_ || !engine_->built())
        build(frame);
    engine_->set_scf_controls(max_iter, density_tol, energy_tol);
    auto& result = drive_(*engine_, frame);
    engine_->commit_density();
    return result;
}
PM3Result& PM3::run(cpu::Frame& frame) {
    publish(frame, /*forces_requested=*/false, /*accumulate=*/false);
    return engine_->result();
}

PM3Result& PM3::forces(cpu::Frame& frame) {
    publish(frame, /*forces_requested=*/true, /*accumulate=*/false);
    return engine_->result();
}

void PM3::prepare(const cpu::BuildContext& qm_ctx) {
    // Repeated parent passes must not erase the last accepted guess.
    // scf_prepare validates/rebuilds changed atom identities at evaluation.
    if (!built())
        build(qm_ctx.frame);
}

void PM3::publish(cpu::Frame& frame, bool forces_requested, bool accumulate) {
    if (!engine_ || !engine_->built())
        build(frame);
    engine_->set_scf_controls(max_iter, density_tol, energy_tol);
    publish_(*engine_, frame, forces_requested, accumulate);
}

void PM3::evaluate(const cpu::LaunchContext& qm_ctx) {
    publish_in_context_(qm_ctx, /*accumulate=*/false);
}

void PM3::reset_history() {
    discard_evaluation();
    if (engine_)
        engine_->reset_history();
}
const PM3Result& PM3::last_result() const {
    if (!engine_) {
        static const PM3Result empty = [] {
            PM3Result result;
            result.method = ATV::semiempirical::SemiEmpiricalMethod::PM3;
            return result;
        }();
        return empty;
    }
    return engine_->last_result();
}
bool PM3::built() const noexcept { return engine_ && engine_->built(); }
bool PM3::has_density() const noexcept {
    return engine_ && engine_->has_density();
}
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
