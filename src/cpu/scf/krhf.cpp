/// @file cpu/scf/krhf.cpp
/// @brief cpu::scf::KRHF — Driver<scf::Phase> over the FFTDF kernel.
///
/// PySCF 2.14.0 `pyscf/pbc/scf/khf.py` (`KRHF`, `KSCF.get_fock`,
/// `energy_elec`, `eig`, `make_rdm1`) and `pyscf/scf/hf.py` `kernel`.
/// DIIS stays off (`get_fock` diis=None), so the mix stage is the identity
/// and the only residual is |ΔE|.

#include "atomiverse/cpu/scf/krhf.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/scf/internal/kpoint_hf.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ATV {
namespace cpu {
namespace scf {
namespace {

using internal::KrhfCycleOwner;

KrhfCycleOwner* require_cycle(KrhfCycleOwner* cycle) {
    if (!cycle || !*cycle)
        throw std::runtime_error("cpu::scf::KRHF: SCF cycle is not open");
    return cycle;
}

class FockStage : public cpu::Component {
  public:
    explicit FockStage(KrhfCycleOwner* cycle) : cycle_(cycle) {}

    std::string name() const override { return "krhf_fock"; }

    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::scf::Phase::fock))
            return;
        // KSCF.get_fock with diis=None: F = H + J − ½K, then energy_elec.
        internal::krhf_cycle_fock(**require_cycle(cycle_));
    }

  private:
    KrhfCycleOwner* cycle_ = nullptr;
};

class MixStage : public cpu::Component {
  public:
    explicit MixStage(KrhfCycleOwner*) {}

    std::string name() const override { return "krhf_mix"; }

    void launch(const cpu::LaunchContext& ctx) override {
        // Identity. khf.KSCF.get_fock is called with diis=None.
        if (!ctx.at(ATV::scf::Phase::mix))
            return;
    }
};

class DiagonalizeStage : public cpu::Component {
  public:
    explicit DiagonalizeStage(KrhfCycleOwner* cycle) : cycle_(cycle) {}

    std::string name() const override { return "krhf_diagonalize"; }

    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::scf::Phase::diagonalize))
            return;
        internal::krhf_cycle_diagonalize(**require_cycle(cycle_));
    }

  private:
    KrhfCycleOwner* cycle_ = nullptr;
};

class DensityStage : public cpu::Component {
  public:
    explicit DensityStage(KrhfCycleOwner* cycle) : cycle_(cycle) {}

    std::string name() const override { return "krhf_density"; }

    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::scf::Phase::density))
            return;
        internal::krhf_cycle_density(**require_cycle(cycle_));
    }

  private:
    KrhfCycleOwner* cycle_ = nullptr;
};

class CheckStage : public cpu::Component {
  public:
    explicit CheckStage(KrhfCycleOwner* cycle) : cycle_(cycle) {}

    std::string name() const override { return "krhf_check"; }

    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::scf::Phase::check))
            return;
        if (!ctx.runtime.loop.metrics)
            throw std::runtime_error(
                "cpu::scf::KRHF: check stage has no ConvergenceMetrics");
        auto& cycle = **require_cycle(cycle_);
        const double ha = internal::krhf_cycle_energy(cycle);
        const double dha = internal::krhf_cycle_delta(cycle);
        ctx.runtime.loop.metrics->energy = ha * units::kHartreeInEv;
        ctx.runtime.loop.metrics->energy_change = dha * units::kHartreeInEv;
    }

  private:
    KrhfCycleOwner* cycle_ = nullptr;
};

/// @brief |ΔE| only. ScfConverged also demands a DIIS residual this kernel
///        does not build.
class EnergyDelta : public cpu::Criterion {
  public:
    explicit EnergyDelta(double tol_ha)
        : tol_ev_(tol_ha * units::kHartreeInEv) {}

    std::string name() const override { return "krhf_energy_delta"; }

    bool contributes_to_convergence() const override { return true; }

    void launch(const cpu::LaunchContext&) override {}

    ATV::Verdict verdict(const cpu::LoopState& s) override {
        if (s.metrics.energy_change < tol_ev_)
            return ATV::Verdict::Converged;
        return ATV::Verdict::Continue;
    }

  private:
    double tol_ev_ = 0.0;
};

} // namespace

struct KRHF::Loop {
    internal::KrhfCycleOwner cycle{nullptr, &internal::krhf_cycle_delete};
    FockStage fock;
    MixStage mix;
    DiagonalizeStage diag;
    DensityStage density;
    CheckStage check;
    EnergyDelta criterion;

    explicit Loop(double tol_ha)
        : fock(&cycle), mix(&cycle), diag(&cycle), density(&cycle),
          check(&cycle), criterion(tol_ha) {}
};

KRHF::KRHF(BasisSet basis_in, std::array<int, 3> k_mesh_in, int charge_in,
           double tol_in, int max_iter_in)
    : basis(basis_in), k_mesh(k_mesh_in), charge(charge_in), tol(tol_in),
      max_iter(max_iter_in), loop_(std::make_unique<Loop>(tol)) {
    this->on(ATV::scf::Phase::fock).add_ref(loop_->fock);
    this->on(ATV::scf::Phase::mix).add_ref(loop_->mix);
    this->on(ATV::scf::Phase::diagonalize).add_ref(loop_->diag);
    this->on(ATV::scf::Phase::density).add_ref(loop_->density);
    this->on(ATV::scf::Phase::check).add_ref(loop_->check);
    this->add_criterion_ref(loop_->criterion);
    this->set_serial_on_replica();
    this->build();
}

KRHF::~KRHF() {
    if (loop_)
        loop_->cycle.reset();
    // Drop built stages while the heap loop is still alive.
    this->shutdown();
}

KRHF::KRHF(KRHF&& other) noexcept
    : cpu::Driver<ATV::scf::Phase>(std::move(other)), HF(std::move(other)),
      basis(other.basis), k_mesh(other.k_mesh), charge(other.charge),
      tol(other.tol), max_iter(other.max_iter), loop_(std::move(other.loop_)),
      energy_ha_(other.energy_ha_), iterations_(other.iterations_) {}

void KRHF::build_on(cpu::Runtime&) {
    if (this->plan().slots.empty())
        this->build();
}

void KRHF::destroy_on(cpu::Runtime&) {
    if (loop_)
        loop_->cycle.reset();
}

int KRHF::drive(cpu::Runtime* parent, cpu::Context& ctx, cpu::Frame& frame,
                int budget, Write write) {
    if (!(tol > 0.0) || max_iter < 1)
        throw std::runtime_error(
            "KRHF: tol must be positive and max_iter >= 1");
    const int mesh[3] = {k_mesh[0], k_mesh[1], k_mesh[2]};
    loop_->cycle = internal::krhf_cycle_open(frame, basis, mesh, charge);
    struct ReleaseCycle {
        Loop* loop = nullptr;
        ~ReleaseCycle() {
            if (loop)
                loop->cycle.reset();
        }
    } release{loop_.get()};

    const cpu::LoopServices saved = ctx.loop;
    ctx.loop.status = &this->status_;
    ctx.loop.metrics = &this->metrics_;
    struct RestoreLoop {
        cpu::Context& ctx;
        cpu::LoopServices saved;
        ~RestoreLoop() { ctx.loop = saved; }
    } restore{ctx, saved};
    this->status_.reset();
    this->metrics_ = ConvergenceMetrics{};

    const int cap = budget > 0 ? budget : max_iter;
    int executed = 0;
    bool ok = false;
    for (int it = 0; it < cap; ++it) {
        this->mutable_outer_status().reset();
        if (parent)
            parent->step_plan(this->plan(), it);
        else
            this->step_once(it);
        ++executed;
        if (this->poll_criteria_(it, frame, ctx) != ATV::Verdict::Continue) {
            ok = this->converged();
            break;
        }
    }
    if (!ok) {
        const double last = internal::krhf_cycle_energy(*loop_->cycle);
        throw std::runtime_error(
            "KRHF: SCF did not converge in " + std::to_string(cap)
            + " iterations (last |energy| = " + std::to_string(last) + " Ha)");
    }
    energy_ha_ = internal::krhf_cycle_energy(*loop_->cycle);
    iterations_ = executed;
    if (write != Write::None) {
        const double e_ev = energy_ha_ * units::kHartreeInEv;
        if (write == Write::Accumulate) {
            frame.meta.get(kEnergy) += e_ev;
            frame.meta.get(kEnergyElectronic) += e_ev;
        } else {
            frame.meta.get(kEnergy) = e_ev;
            frame.meta.get(kEnergyElectronic) = e_ev;
        }
    }
    return executed;
}

void KRHF::run(const Frame& frame) {
    Frame scratch(frame);
    this->setup_pass(scratch);
    try {
        drive(nullptr, this->backend_->context(), scratch, max_iter,
              Write::None);
        this->teardown_pass();
    } catch (...) {
        this->teardown_pass();
        throw;
    }
}

void KRHF::run(Frame& frame) {
    this->setup_pass(frame);
    try {
        drive(nullptr, this->backend_->context(), frame, max_iter,
              Write::Overwrite);
        this->teardown_pass();
    } catch (...) {
        this->teardown_pass();
        throw;
    }
}

int KRHF::run_nested_erased(cpu::Context& ctx, cpu::Frame& frame, int budget) {
    if (!ctx.exec.runtime)
        throw std::runtime_error("cpu::scf::KRHF: nested run has no Runtime");
    cpu::Runtime& rt = *ctx.exec.runtime;
    rt.build_plan(this->plan());
    struct ReleasePlan {
        cpu::Runtime& rt;
        const Plan<cpu::Component>& plan;
        ~ReleasePlan() { rt.destroy_plan(plan); }
    } release{rt, this->plan()};
    return drive(&rt, ctx, frame, budget, Write::Accumulate);
}

GradientResult KRHF::forces(Frame& /*frame*/) {
    throw std::runtime_error(
        "KRHF::forces: analytic k-point nuclear forces are not implemented "
        "(missing kernel krhf_nuclear_forces; PySCF entry is "
        "pyscf.pbc.grad.krhf.Gradients, outside this product)");
}

} // namespace scf
} // namespace cpu
} // namespace ATV
