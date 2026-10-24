/// @file src/cpu/semiempirical/rm1_solver.cpp
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include <stdexcept>
#include <utility>

#include "atomiverse/core/frame_schema.h"
namespace ATV {
namespace cpu {
namespace semiempirical {
RM1::RM1(int charge_, int multiplicity_, int max_iter_, double density_tol_,
         double energy_tol_)
    : charge(charge_), multiplicity(multiplicity_), max_iter(max_iter_),
      density_tol(density_tol_), energy_tol(energy_tol_) {}

RM1::~RM1() = default;

RM1::RM1(RM1&& other)
    : NddoMethod(std::move(other)), charge(other.charge),
      multiplicity(other.multiplicity), max_iter(other.max_iter),
      density_tol(other.density_tol), energy_tol(other.energy_tol),
      engine_(std::move(other.engine_)) {}

void RM1::sync_engine_() {
    NDDOConfig c;
    c.method = ATV::semiempirical::SemiEmpiricalMethod::RM1;
    c.charge = charge;
    c.multiplicity = multiplicity;
    c.max_iter = max_iter;
    c.density_tol = density_tol;
    c.energy_tol = energy_tol;
    engine_ = std::make_unique<NDDOSolver>(c);
}
void RM1::build(const cpu::Frame& reference) {
    sync_engine_();
    engine_->build(reference);
}
RM1Result& RM1::run(const cpu::Frame& frame) {
    if (!engine_ || !engine_->built())
        build(frame);
    engine_->set_scf_controls(max_iter, density_tol, energy_tol);
    auto& result = drive_(*engine_, frame);
    engine_->commit_density();
    return result;
}
RM1Result& RM1::run(cpu::Frame& frame) {
    publish(frame, /*forces_requested=*/false, /*accumulate=*/false);
    return engine_->result();
}

RM1Result& RM1::forces(cpu::Frame& frame) {
    publish(frame, /*forces_requested=*/true, /*accumulate=*/false);
    return engine_->result();
}

void RM1::prepare(const cpu::BuildContext& qm_ctx) {
    // Repeated parent passes must not erase the last accepted guess.
    // scf_prepare validates/rebuilds changed atom identities at evaluation.
    if (!built())
        build(qm_ctx.frame);
}

void RM1::publish(cpu::Frame& frame, bool forces_requested, bool accumulate) {
    if (!engine_ || !engine_->built())
        build(frame);
    engine_->set_scf_controls(max_iter, density_tol, energy_tol);
    publish_(*engine_, frame, forces_requested, accumulate);
}

void RM1::evaluate(const cpu::LaunchContext& qm_ctx) {
    publish_in_context_(qm_ctx, /*accumulate=*/false);
}

void RM1::reset_history() { engine_.reset(); }
const RM1Result& RM1::last_result() const {
    if (!engine_) {
        static RM1Result empty;
        return empty;
    }
    return engine_->last_result();
}
bool RM1::built() const noexcept { return engine_ && engine_->built(); }
bool RM1::has_density() const noexcept {
    return engine_ && engine_->has_density();
}
const NDDOSystem& RM1::system() const {
    if (!engine_)
        throw std::runtime_error("RM1::system before build");
    return engine_->system();
}
const NDDOScfState& RM1::state() const {
    if (!engine_)
        throw std::runtime_error("RM1::state before build");
    return engine_->state();
}
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
