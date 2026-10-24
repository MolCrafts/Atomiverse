/// @file src/cpu/semiempirical/mndo_solver.cpp
#include "atomiverse/cpu/semiempirical/mndo_solver.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_solver.h"
#include <utility>

#include "atomiverse/core/frame_schema.h"
namespace ATV {
namespace cpu {
namespace semiempirical {
MNDO::MNDO(int charge_, int multiplicity_, int max_iter_, double density_tol_,
           double energy_tol_)
    : charge(charge_), multiplicity(multiplicity_), max_iter(max_iter_),
      density_tol(density_tol_), energy_tol(energy_tol_) {}

MNDO::~MNDO() = default;

MNDO::MNDO(MNDO&& other)
    : NddoMethod(std::move(other)), charge(other.charge),
      multiplicity(other.multiplicity), max_iter(other.max_iter),
      density_tol(other.density_tol), energy_tol(other.energy_tol),
      engine_(std::move(other.engine_)) {}

void MNDO::sync_engine_() {
    discard_evaluation();
    NDDOConfig c;
    c.method = ATV::semiempirical::SemiEmpiricalMethod::MNDO;
    c.charge = charge;
    c.multiplicity = multiplicity;
    c.max_iter = max_iter;
    c.density_tol = density_tol;
    c.energy_tol = energy_tol;
    engine_ = std::make_unique<NDDOSolver>(c);
}
void MNDO::build(const cpu::Frame& reference) {
    sync_engine_();
    engine_->build(reference);
}
MNDOResult& MNDO::run(const cpu::Frame& frame) {
    if (!engine_ || !engine_->built())
        build(frame);
    engine_->set_scf_controls(max_iter, density_tol, energy_tol);
    auto& result = drive_(*engine_, frame);
    engine_->commit_density();
    return result;
}
MNDOResult& MNDO::run(cpu::Frame& frame) {
    publish(frame, /*forces_requested=*/false, /*accumulate=*/false);
    return engine_->result();
}

MNDOResult& MNDO::forces(cpu::Frame& frame) {
    publish(frame, /*forces_requested=*/true, /*accumulate=*/false);
    return engine_->result();
}

void MNDO::prepare(const cpu::BuildContext& qm_ctx) {
    // Repeated parent passes must not erase the last accepted guess.
    // scf_prepare validates/rebuilds changed atom identities at evaluation.
    if (!built())
        build(qm_ctx.frame);
}

void MNDO::publish(cpu::Frame& frame, bool forces_requested, bool accumulate) {
    if (!engine_ || !engine_->built())
        build(frame);
    engine_->set_scf_controls(max_iter, density_tol, energy_tol);
    publish_(*engine_, frame, forces_requested, accumulate);
}

void MNDO::evaluate(const cpu::LaunchContext& qm_ctx) {
    publish_in_context_(qm_ctx, /*accumulate=*/false);
}

void MNDO::reset_history() {
    discard_evaluation();
    if (engine_)
        engine_->reset_history();
}
const MNDOResult& MNDO::last_result() const {
    if (!engine_) {
        static const MNDOResult empty = [] {
            MNDOResult result;
            result.method = ATV::semiempirical::SemiEmpiricalMethod::MNDO;
            return result;
        }();
        return empty;
    }
    return engine_->last_result();
}
bool MNDO::built() const noexcept { return engine_ && engine_->built(); }
bool MNDO::has_density() const noexcept {
    return engine_ && engine_->has_density();
}
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
