/// @file cuda/driver/driver.cu
/// @brief CUDA Backend implementation + Driver explicit instantiations.

#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/precision.h"
#include <cmath>
#include <cstdint>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/driver_template_impl.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/log_callback.h"
#include "atomiverse/cuda/device/timer.cuh"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/driver/backend_traits.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

// ── Backend ────────────────────────────────────────────────────────

void Backend::init() {
    cuda::register_log_callback();
    cudaStream_t main, defer;
    ATV_CUDA_CHECK(cudaStreamCreate(&main));
    context_.stream = main;
    // Retain each acquired stream immediately so failure cleanup can release
    // the first stream if creation of the second one fails.
    ATV_CUDA_CHECK(cudaStreamCreate(&defer));
    context_.defer_stream = defer;
    context_.exec.backend = this;
    context_.exec.evaluation_transaction = &evaluation_;
    release_fault_ring_();
    ATV_CUDA_CHECK(cudaMallocHost(&fault_ring_, kFaultRing * sizeof(unsigned)));
    if (!device_step_)
        ATV_CUDA_CHECK(cudaMalloc(&device_step_, sizeof(long long)));
    context_.device_step = device_step_;
    device_step_next_ = -1;
    for (int i = 0; i < kFaultRing; ++i)
        ATV_CUDA_CHECK(
            cudaEventCreateWithFlags(&fault_ready_[i], cudaEventDisableTiming));
}

void Backend::release_fault_ring_() {
    for (int i = 0; i < kFaultRing; ++i) {
        if (fault_ready_[i])
            cudaEventDestroy(fault_ready_[i]);
        fault_ready_[i] = nullptr;
        fault_pending_[i] = false;
    }
    if (fault_ring_)
        cudaFreeHost(fault_ring_);
    fault_ring_ = nullptr;
    fault_slot_ = 0;
    fault_observed_ = false;
}

// A new frame or an explicit input/retry boundary starts with a clean fault
// record; drop copies of the old one (the caller has synchronized).
void Backend::forget_fault_() {
    for (bool& pending : fault_pending_)
        pending = false;
    fault_observed_ = false;
}

// Fold in every earlier copy whose event has completed, and every
// kFaultPollEvery steps append the fault word to the pinned ring. A copy on
// the compute stream orders the next kernel behind a DMA, so it is not paid
// every step; the device state froze at the faulting step, so observing it
// late loses nothing. cudaEventQuery never blocks; a slot still in flight is
// simply re-armed (faults are sticky, so a later copy carries the verdict).
void Backend::observe_fault_(BigInt step_idx) {
    if (!fault_ring_ || frame_.force_fault == nullptr)
        return;
    for (int i = 0; i < kFaultRing; ++i) {
        if (!fault_pending_[i])
            continue;
        const cudaError_t state = cudaEventQuery(fault_ready_[i]);
        if (state == cudaErrorNotReady)
            continue;
        ATV_CUDA_CHECK(state);
        fault_pending_[i] = false;
        fault_observed_ |= fault_ring_[i] != 0;
    }
    if (step_idx % kFaultPollEvery != 0)
        return;
    const auto stream = static_cast<cudaStream_t>(context_.stream);
    const int slot = fault_slot_;
    fault_slot_ = (fault_slot_ + 1) % kFaultRing;
    ATV_CUDA_CHECK(cudaMemcpyAsync(&fault_ring_[slot],
                                   &frame_.force_fault->kinds, sizeof(unsigned),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaEventRecord(fault_ready_[slot], stream));
    fault_pending_[slot] = true;
}

namespace {
// Exactly the input columns consumed by cpu::to_cuda; arbitrary host-only
// custom fields do not change the single-GPU execution identity.
bool device_input(std::string_view block, const cpu::ColumnRef& column) {
    if (block != "atoms")
        return false;
    switch (static_cast<AtomFieldId>(column.wire_id)) {
    case AtomFieldId::x:
    case AtomFieldId::y:
    case AtomFieldId::z:
    case AtomFieldId::vx:
    case AtomFieldId::vy:
    case AtomFieldId::vz:
    case AtomFieldId::mass:
    case AtomFieldId::charge:
    case AtomFieldId::type:
    case AtomFieldId::tag:
    case AtomFieldId::image:
    case AtomFieldId::num_bond:
    case AtomFieldId::bond_atom:
    case AtomFieldId::bond_type:
    case AtomFieldId::num_angle:
    case AtomFieldId::angle_atom:
    case AtomFieldId::angle_type:
    case AtomFieldId::num_dihedral:
    case AtomFieldId::dihedral_atom:
    case AtomFieldId::dihedral_type:
        return true;
    default:
        return false;
    }
}

__global__ void set_step_k(long long* step, long long value) { *step = value; }

__global__ void advance_step_k(long long* step) { ++*step; }

__global__ void publish_cell(const geom_t* cell, Box seed, Box* cache) {
    const Box previous = *cache;
    seed.resident = nullptr;
    seed.assign_cell(cell);
    bool changed = false;
    for (int k = 0; k < 9; ++k)
        changed |= !(seed.cell[k] == previous.cell[k]);
    seed.revision = previous.revision + (changed ? 1 : 0);
    *cache = seed;
}
} // namespace

void Backend::publish_box_() {
    publish_cell<<<1, 1, 0, static_cast<cudaStream_t>(context_.stream)>>>(
        frame_.box, context_.host_box,
        const_cast<Box*>(context_.host_box.resident));
    ATV_CUDA_CHECK(cudaGetLastError());
}

void Backend::materialize(const cpu::Frame& frame) {
    forces_ready_ = false;
    forget_fault_();
    forget_graphs_();
    frame_ = cpu::to_cuda(frame);
    context_.md.n_dof_global = ATV::md::degrees_of_freedom(frame.nlocal());

    auto& box = context_.host_box;
    geom_t cell[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cell[3 * i + j] = frame.box.cell()(i, j);
    box.assign_cell(cell);
    Box* resident = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&resident, sizeof(Box)));
    // Zeroed cache: the first publish is a cell change (revision 1).
    ATV_CUDA_CHECK(cudaMemset(resident, 0, sizeof(Box)));
    box.resident = resident;
    publish_box_();
    input_snapshot_.capture(frame, device_input);
    inputs_dirty_ = identity_dirty_ = false;
}

void Backend::load(const Plan<Component>& plan) {
    std::vector<std::pair<int, Component*>> bindings;
    if (plan.kind == PhaseKind::MD || plan.kind == PhaseKind::Minimizer)
        for (const auto& slot : plan.slots)
            for (const auto& op : slot.ops)
                bindings.emplace_back(slot.phase, op.component);
    if (bindings != force_bindings_)
        forces_ready_ = false;
    force_bindings_ = std::move(bindings);
    plan_ = &plan;
    forget_graphs_();
    // MD evaluates energy/virial only on steps a consumer asked for (LAMMPS
    // Integrate::ev_setup). Other plans keep their inherited requests.
    demands_.clear();
    gate_evaluation_ = plan.kind == PhaseKind::MD;
    virial_demanded_ = false;
    report_step_ = -1;
    if (gate_evaluation_)
        for (auto* component : plan.components()) {
            const auto d = component->demand();
            if (d.every > 0 && (d.energy || d.virial))
                demands_.push_back(d);
            virial_demanded_ |= d.virial && d.every > 0;
        }
}

// LAMMPS Integrate::ev_set: a quantity is evaluated on a step when some
// consumer's cadence falls on it; the reported final step always evaluates
// the energy, and the virial when anyone uses it.
void Backend::apply_demand_(LaunchContext& ctx, BigInt step) const {
    if (!gate_evaluation_)
        return;
    bool energy = step == report_step_;
    bool virial = energy && virial_demanded_;
    for (const auto& d : demands_)
        if (step % d.every == d.phase) {
            energy |= d.energy;
            virial |= d.virial;
        }
    ctx.energy_requested &= energy;
    ctx.stress_requested &= virial;
}

void Backend::build_components() {
    forget_graphs_(); // captured segments bake component buffers
    build_plan(*plan_);
    ATV_CUDA_CHECK(
        cudaStreamSynchronize(static_cast<cudaStream_t>(context_.stream)));
    if (timer_)
        timer_->impl()->flush();
}

void Backend::build_plan(const Plan<Component>& plan) {
    evaluation_.reserve(plan.components().size());
    BuildContext build{context_, frame_};
    DestroyContext destroy{context_, frame_};
    lifecycle_.build(
        plan, [&](Component& c) { c.build(build); },
        [&](Component& c) { c.destroy(destroy); });
}

void Backend::destroy_plan(const Plan<Component>& plan) {
    forget_graphs_();
    DestroyContext ctx{context_, frame_};
    lifecycle_.destroy(plan, [&](Component& c) { c.destroy(ctx); });
}

void Backend::prepare_run(BigInt step_idx) {
    if (!plan_
        || (plan_->kind != PhaseKind::MD && plan_->kind != PhaseKind::Minimizer)
        || forces_ready_)
        return;
    struct DispatchGuard {
        bool& active;
        bool saved;
        ~DispatchGuard() { active = saved; }
    } guard{dispatching_, dispatching_};
    dispatching_ = true;
    plan_->prepare_forces(
        [](int) {},
        [&](const auto& slot, const auto& op, bool force_slot) {
            LaunchContext ctx{context_, frame_, slot.phase, step_idx};
            ctx.initializing_forces = true;
            ctx.forces_requested &= slot.forces_requested;
            // The initial evaluation is what run(frame, 0) reports, and what
            // a step-0 barostat reads: energy always, virial when demanded.
            ctx.stress_requested &= slot.stress_requested
                                    && (!gate_evaluation_ || virial_demanded_);
            if (force_slot)
                op.component->launch(ctx);
            else
                op.component->initialize_forces(ctx);
        });
    forces_ready_ = true;
}

void Backend::execute_phases(BigInt step_idx, int first_phase, int end_phase) {
    if (!plan_)
        throw std::invalid_argument("CUDA Backend has no loaded plan");
    execute_phases(*plan_, step_idx, first_phase, end_phase);
}

void Backend::execute_phases(const Plan<Component>& plan, BigInt step_idx,
                             int first_phase, int end_phase) {
    if (first_phase < 0 || end_phase < first_phase
        || end_phase > static_cast<int>(plan.slots.size()))
        throw std::invalid_argument(
            "CUDA Backend phase range is outside the loaded plan");
    const bool timed = timer_ && !dispatching_;
    struct DispatchGuard {
        bool& active;
        bool saved;
        ~DispatchGuard() { active = saved; }
    } guard{dispatching_, dispatching_};
    dispatching_ = true;
    for (int phase = first_phase; phase < end_phase; ++phase) {
        auto& slot = plan.slots[static_cast<std::size_t>(phase)];
        if (slot.ops.empty())
            continue;
        for (auto& op : slot.ops) {
            if (op.frequency > 0 && step_idx % op.frequency != 0
                && !(step_idx == report_step_
                     && op.component->samples_final_step()))
                continue;
            auto stream = static_cast<cudaStream_t>(
                (op.scheduling == Scheduling::deferrable)
                    ? context_.defer_stream
                    : context_.stream);
            LaunchContext ctx{context_, frame_, slot.phase, step_idx};
            ctx.forces_requested &= slot.forces_requested;
            ctx.stress_requested &= slot.stress_requested;
            apply_demand_(ctx, step_idx);
            if (timed)
                timer_->impl()->begin(stream);
            try {
                op.component->launch(ctx);
            } catch (...) {
                if (timed) {
                    try {
                        timer_->impl()->end(stream, slot.phase,
                                            op.component->name());
                    } catch (...) {
                        // Profiling failure must not replace the component error.
                    }
                    try {
                        timer_->impl()->flush();
                    } catch (...) {
                        // Consume the pending batch even if recording end failed.
                    }
                }
                throw;
            }
            if (op.component->changes_box(slot.phase))
                publish_box_();
            if (timed)
                timer_->impl()->end(stream, slot.phase, op.component->name());
        }
        // Per-phase syncs intentionally removed
        // (cuda-md-review-cleanup-03-launch-purity ac-005):
        //   - Components within a phase already serialize via CUDA's
        //     in-stream FIFO ordering. The previous unconditional
        //     `cudaStreamSynchronize(stream) + cudaStreamSynchronize(defer)`
        //     drained both streams 6× per step (per-phase × main + defer),
        //     ~80 µs/step pure host wait on Hopper for no semantic gain.
        //   - Cross-stream dependencies (defer_stream reads main_stream's
        //     buffers) are explicitly ordered by AsyncBridge::launch via
        //     cudaEventRecord(main) + cudaStreamWaitEvent(defer) per
        //     deferred op, so cross-stream reads see consistent values
        //     without per-phase sync.
        //   - End-of-run sync happens in Backend::sync() / retrieve(). No
        //     component waits on the device inside a step (iron law 10);
        //     host-consumed values arrive through async copies.
        if (timed)
            timer_->impl()->flush();
    }
}

void Backend::step(BigInt step_idx) {
    EvaluationTransaction::Scope evaluation(evaluation_);
    prepare_run(step_idx);
    sync_device_step_(step_idx);
    // MD steps replay captured segments; a profiled step stays eager so the
    // timer brackets every component.
    if (gate_evaluation_ && !timer_) {
        execute_step_graphed_(step_idx);
    } else {
        execute_phases(step_idx, 0, static_cast<int>(plan_->slots.size()));
        advance_device_step_(static_cast<cudaStream_t>(context_.stream));
    }
    if (!evaluation_.empty()) {
        sync();
        throw_if_force_fault(frame_);
    }
    observe_fault_(step_idx);
    if (timer_)
        timer_->step_done();
    evaluation.accept();
}

// The device step copy follows the host step; only a jump (new run, retry,
// continuation from another index) costs a launch.
void Backend::sync_device_step_(BigInt step_idx) {
    if (!device_step_ || device_step_next_ == step_idx)
        return;
    set_step_k<<<1, 1, 0, static_cast<cudaStream_t>(context_.stream)>>>(
        device_step_, static_cast<long long>(step_idx));
    ATV_CUDA_CHECK(cudaGetLastError());
    device_step_next_ = step_idx;
}

void Backend::advance_device_step_(cudaStream_t stream) {
    if (!device_step_)
        return;
    advance_step_k<<<1, 1, 0, stream>>>(device_step_);
    ATV_CUDA_CHECK(cudaGetLastError());
    ++device_step_next_;
}

void Backend::forget_graphs_() noexcept { step_graphs_.clear(); }

// One MD step as replayed segments: consecutive graph-capturable main-stream
// components form a segment captured once per signature (components, phase,
// evaluation flags) and replayed from a GraphRing; any other component is
// launched directly between segments. The device step advance closes the
// last segment. Launch order and stream order are those of execute_phases.
void Backend::execute_step_graphed_(BigInt step_idx) {
    const auto& plan = *plan_;
    const auto stream = static_cast<cudaStream_t>(context_.stream);
    struct Entry {
        Component* component; // null: the device step advance
        int phase;
        bool forces, energy, stress, box;
    };
    std::vector<Entry> segment;
    auto flush = [&] {
        if (segment.empty())
            return;
        std::vector<std::uintptr_t> key;
        key.reserve(segment.size() * 3);
        for (const auto& e : segment) {
            key.push_back(reinterpret_cast<std::uintptr_t>(e.component));
            key.push_back(e.component ? static_cast<std::uintptr_t>(
                                            e.component->graph_epoch())
                                      : 0);
            key.push_back(static_cast<std::uintptr_t>(e.phase) << 4
                          | (e.forces ? 1u : 0u) | (e.energy ? 2u : 0u)
                          | (e.stress ? 4u : 0u) | (e.box ? 8u : 0u));
        }
        auto found = step_graphs_.find(key);
        if (found == step_graphs_.end()) {
            // One capture per ring slot: a spliced conditional node (the
            // neighbor rebuild) allows a single instantiation per graph.
            auto capture = [&]() -> cudaGraph_t {
                cudaGraph_t graph = nullptr;
                ATV_CUDA_CHECK(cudaStreamBeginCapture(
                    stream, cudaStreamCaptureModeThreadLocal));
                try {
                    for (const auto& e : segment) {
                        if (!e.component) {
                            advance_step_k<<<1, 1, 0, stream>>>(device_step_);
                            continue;
                        }
                        LaunchContext ctx{context_, frame_, e.phase, step_idx};
                        ctx.forces_requested = e.forces;
                        ctx.energy_requested = e.energy;
                        ctx.stress_requested = e.stress;
                        e.component->launch(ctx);
                        if (e.box)
                            publish_box_();
                    }
                } catch (...) {
                    cudaStreamEndCapture(stream, &graph);
                    if (graph)
                        cudaGraphDestroy(graph);
                    throw;
                }
                ATV_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
                return graph;
            };
            GraphRing ring;
            ring.build(capture);
            found = step_graphs_.emplace(std::move(key), std::move(ring)).first;
        }
        ATV_CUDA_CHECK(found->second.launch(stream));
        segment.clear();
    };
    struct DispatchGuard {
        bool& active;
        bool saved;
        ~DispatchGuard() { active = saved; }
    } guard{dispatching_, dispatching_};
    dispatching_ = true;
    for (const auto& slot : plan.slots) {
        for (const auto& op : slot.ops) {
            if (op.frequency > 0 && step_idx % op.frequency != 0
                && !(step_idx == report_step_
                     && op.component->samples_final_step()))
                continue;
            LaunchContext ctx{context_, frame_, slot.phase, step_idx};
            ctx.forces_requested &= slot.forces_requested;
            ctx.stress_requested &= slot.stress_requested;
            apply_demand_(ctx, step_idx);
            const bool box = op.component->changes_box(slot.phase);
            if (op.scheduling != Scheduling::deferrable
                && op.component->graph_capturable()) {
                segment.push_back({op.component, slot.phase,
                                   ctx.forces_requested, ctx.energy_requested,
                                   ctx.stress_requested, box});
                continue;
            }
            flush();
            op.component->launch(ctx);
            if (box)
                publish_box_();
        }
    }
    segment.push_back({nullptr, 0, false, false, false, false});
    flush();
    ++device_step_next_;
}

void Backend::run(int n_steps) {
    for (int s = 0; s < n_steps; ++s)
        step(s);
}

void Backend::destroy_components() {
    forget_graphs_();
    DestroyContext ctx{context_, frame_};
    lifecycle_.destroy_all([&](Component& c) { c.destroy(ctx); });
}

void Backend::retrieve(cpu::Frame& frame, bool copy_forces) {
    sync();
    cuda::to_cpu(frame_, frame, copy_forces);
    input_snapshot_.capture(frame, device_input);
    inputs_dirty_ = identity_dirty_ = false;
}

InputChange Backend::validate_inputs(const cpu::Frame& frame) const {
    auto changes = input_snapshot_.changes(frame, device_input);
    if (identity_dirty_ || frame.nlocal() != frame_.n_atoms())
        changes |= InputChange::Identity;
    if (changes == InputChange::None && !inputs_dirty_)
        return changes;
    // Preflight the complete plan before touching device state or any cache.
    // A retry restores numeric inputs even if the caller returned to the last
    // checkpoint; that restoration is not a newly edited charge/identity.
    const auto invalidation =
        inputs_dirty_ ? changes | InputChange::Geometry | InputChange::Cell
                      : changes;
    for (auto* component : plan_->components())
        if (!component->supports_input_changes(invalidation))
            throw std::invalid_argument(
                "CUDA component '" + component->name()
                + "' does not support these input changes; its input-derived "
                  "caches require a new run object");
    return changes;
}

void Backend::refresh_inputs(const cpu::Frame& frame) {
    const auto changes = validate_inputs(frame);
    if (changes == InputChange::None && !inputs_dirty_)
        return;
    const auto invalidation =
        inputs_dirty_ ? changes | InputChange::Geometry | InputChange::Cell
                      : changes;
    sync();
    const bool restore_all = inputs_dirty_;
    mark_inputs_dirty();
    if (has_input_change(changes, InputChange::Identity)) {
        // Build replacement storage before releasing the current allocation.
        // The Frame object stays at the same address for borrowed method plans.
        auto replacement = cpu::to_cuda(frame);
        cuda::destroy(frame_);
        frame_ = replacement;
        identity_dirty_ = true;
    } else {
        auto upload = changes;
        if (restore_all)
            upload |= InputChange::Geometry | InputChange::Velocity
                      | InputChange::Charge | InputChange::Cell;
        cuda::update_inputs(frame, frame_, upload);
    }
    auto& box = context_.host_box;
    geom_t cell[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cell[3 * i + j] = frame.box.cell()(i, j);
    box.assign_cell(cell);
    publish_box_();
    context_.md.n_dof_global = ATV::md::degrees_of_freedom(frame.nlocal());
    // A reported scientific force fault belongs to the discarded evaluation.
    // Reset it only at an explicit input/retry boundary, never during a step.
    const ForceFault clean_fault{};
    ATV_CUDA_CHECK(cudaMemcpy(frame_.force_fault, &clean_fault,
                              sizeof(clean_fault), cudaMemcpyHostToDevice));
    forget_fault_();
    forget_graphs_();
    for (auto* component : plan_->components())
        component->invalidate_inputs(invalidation, {context_, frame_});
}

void Backend::sync() {
    ATV_CUDA_CHECK(
        cudaStreamSynchronize(static_cast<cudaStream_t>(context_.stream)));
    ATV_CUDA_CHECK(cudaStreamSynchronize(
        static_cast<cudaStream_t>(context_.defer_stream)));
}

void Backend::shutdown() {
    std::exception_ptr failure;
    auto cleanup = [&](auto&& operation) {
        try {
            operation();
        } catch (...) {
            if (!failure)
                failure = std::current_exception();
        }
    };
    if (!lifecycle_.empty()) {
        // Drain both streams even when one reports an asynchronous error.
        cleanup([&] {
            ATV_CUDA_CHECK(cudaStreamSynchronize(
                static_cast<cudaStream_t>(context_.stream)));
        });
        cleanup([&] {
            ATV_CUDA_CHECK(cudaStreamSynchronize(
                static_cast<cudaStream_t>(context_.defer_stream)));
        });
        DestroyContext ctx{context_, frame_};
        lifecycle_.destroy_all([&](Component& component) {
            cleanup([&] { component.destroy(ctx); });
        });
    }
    if (context_.host_box.resident)
        cleanup([&] {
            ATV_CUDA_CHECK(
                cudaFree(const_cast<Box*>(context_.host_box.resident)));
        });
    cuda::destroy(frame_);
    frame_ = {};
    auto main = static_cast<cudaStream_t>(context_.stream);
    auto defer = static_cast<cudaStream_t>(context_.defer_stream);
    if (main)
        cleanup([&] { ATV_CUDA_CHECK(cudaStreamDestroy(main)); });
    if (defer)
        cleanup([&] { ATV_CUDA_CHECK(cudaStreamDestroy(defer)); });
    context_ = {};
    input_snapshot_.clear();
    inputs_dirty_ = identity_dirty_ = false;
    forces_ready_ = false;
    plan_ = nullptr;
    release_fault_ring_();
    forget_graphs_();
    if (device_step_)
        cleanup([&] { ATV_CUDA_CHECK(cudaFree(device_step_)); });
    device_step_ = nullptr;
    device_step_next_ = -1;
    cuda::unregister_log_callback();
    // Cleanup attempts every release; a poisoned CUDA context can still require
    // process restart. Reporting an error does not leave a live host-side ledger.
    if (failure)
        std::rethrow_exception(failure);
}

} // namespace cuda

// ── Explicit instantiations (CUDA) ───────────────────────────────

template class Driver<md::Phase, cuda::device_tag>;
template class Driver<minimizer::Phase, cuda::device_tag>;
template class Driver<evaluate::Phase, cuda::device_tag>;
template class Driver<scf::Phase, cuda::device_tag>;
#ifdef ATV_HAS_PW
template class Driver<pw::Phase, cuda::device_tag>;
#endif
#ifdef ATV_HAS_GPW
template class Driver<gpw::Phase, cuda::device_tag>;
#endif
#ifdef ATV_HAS_SEMIEMPIRICAL
template class Driver<nddo::Phase, cuda::device_tag>;
#endif

} // namespace ATV
