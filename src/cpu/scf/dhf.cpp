/// @file cpu/scf/dhf.cpp
/// @brief cpu::scf::DHF — Driver<scf::Phase> over the Dirac–Coulomb kernel.
///
/// PySCF 2.14.0 `pyscf/scf/dhf.py` (`DHF.get_hcore`, `get_ovlp`,
/// `get_jk_coulomb`, `get_occ`, `energy_elec`, `kernel`) and
/// `pyscf/scf/hf.py` (`kernel`, `get_fock`, `eig`, `make_rdm1`,
/// `energy_elec`, `energy_tot`). CDIIS is the mix stage. The residual is
/// |ΔE|; two successive steps inside `tol` is the convergence requirement.
/// The method does not assign convergence.

#include "atomiverse/cpu/scf/dhf.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/scf/internal/dhf.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ATV {
namespace cpu {
namespace scf {
namespace {

using Cycle = internal::DhfCycle;
using CyclePtr = std::unique_ptr<Cycle>;

CyclePtr* require_cycle(CyclePtr* cycle) {
    if (!cycle || !*cycle)
        throw std::runtime_error("cpu::scf::DHF: SCF cycle is not open");
    return cycle;
}

/// One plan op for every SCF phase. Mix is CDIIS (`hf.get_fock`).
class Stage : public cpu::Component {
  public:
    enum class Kind { Fock, Mix, Diag, Density, Check };

    Stage(CyclePtr* cycle, Kind kind) : cycle_(cycle), kind_(kind) {}

    std::string name() const override {
        switch (kind_) {
        case Kind::Fock:
            return "dhf_fock";
        case Kind::Mix:
            return "dhf_mix";
        case Kind::Diag:
            return "dhf_diagonalize";
        case Kind::Density:
            return "dhf_density";
        case Kind::Check:
            return "dhf_check";
        }
        throw std::logic_error("cpu::scf::DHF: unknown stage");
    }

    void launch(const cpu::LaunchContext& ctx) override {
        // Bound to one phase, so the runtime only calls this on that phase.
        auto& cycle = **require_cycle(cycle_);
        if (kind_ == Kind::Fock) {
            // get_jk_coulomb, then dhf.energy_elec + hf.energy_tot.
            cycle.fock();
            return;
        }
        if (kind_ == Kind::Mix) {
            cycle.mix();
            return;
        }
        if (kind_ == Kind::Diag) {
            cycle.diagonalize();
            return;
        }
        if (kind_ == Kind::Density) {
            cycle.density();
            return;
        }
        if (!ctx.runtime.loop.metrics)
            throw std::runtime_error(
                "cpu::scf::DHF: check stage has no ConvergenceMetrics");
        const double ha = cycle.energy;
        const double dha = cycle.delta();
        ctx.runtime.loop.metrics->energy = ha * units::kHartreeInEv;
        ctx.runtime.loop.metrics->energy_change = dha * units::kHartreeInEv;
    }

  private:
    CyclePtr* cycle_ = nullptr;
    Kind kind_ = Kind::Fock;
};

/// @brief Two successive |ΔE| steps. Matches the previous Dirac gate:
///        one close step is not enough. Does not write loop status.
class EnergyDelta : public cpu::Criterion {
  public:
    explicit EnergyDelta(double tol_ha)
        : tol_ev_(tol_ha * units::kHartreeInEv) {}

    std::string name() const override { return "dhf_energy_delta"; }

    bool contributes_to_convergence() const override { return true; }

    void launch(const cpu::LaunchContext&) override {}

    /// @brief Forget history so the next solve starts on its first energy.
    void reset_history() {
        seen_ = false;
        was_close_ = false;
    }

    ATV::Verdict verdict(const cpu::LoopState& s) override {
        const bool close = seen_ && s.metrics.energy_change < tol_ev_;
        const bool done = close && was_close_;
        was_close_ = close;
        seen_ = true;
        if (done)
            return ATV::Verdict::Converged;
        return ATV::Verdict::Continue;
    }

  private:
    double tol_ev_ = 0.0;
    bool seen_ = false;
    bool was_close_ = false;
};

void require_controls(double tol, int max_iter) {
    if (!(tol > 0.0) || max_iter < 1)
        throw std::invalid_argument(
            "DHF::run: tol must be positive and max_iter >= 1");
}

} // namespace

struct DHF::Loop {
    /// Opens one SCF cycle and drops it on every exit, including a throw.
    struct Hold {
        Loop* loop = nullptr;

        Hold(Loop& loop_in, const Frame& frame, BasisSet basis, int charge)
            : loop(&loop_in) {
            loop->criterion.reset_history();
            loop->cycle = std::make_unique<Cycle>(frame, basis, charge);
        }

        ~Hold() {
            if (loop)
                loop->cycle.reset();
        }

        Hold(const Hold&) = delete;
        Hold& operator=(const Hold&) = delete;
    };

    CyclePtr cycle;
    Stage fock;
    Stage mix;
    Stage diag;
    Stage density;
    Stage check;
    EnergyDelta criterion;

    explicit Loop(double tol_ha)
        : fock(&cycle, Stage::Kind::Fock), mix(&cycle, Stage::Kind::Mix),
          diag(&cycle, Stage::Kind::Diag),
          density(&cycle, Stage::Kind::Density),
          check(&cycle, Stage::Kind::Check), criterion(tol_ha) {}
};

DHF::DHF(BasisSet basis_in, int charge_in, double tol_in, int max_iter_in)
    : basis(basis_in), charge(charge_in), tol(tol_in), max_iter(max_iter_in),
      loop_(std::make_unique<Loop>(tol)) {
    this->on(ATV::scf::Phase::fock).add_ref(loop_->fock);
    this->on(ATV::scf::Phase::mix).add_ref(loop_->mix);
    this->on(ATV::scf::Phase::diagonalize).add_ref(loop_->diag);
    this->on(ATV::scf::Phase::density).add_ref(loop_->density);
    this->on(ATV::scf::Phase::check).add_ref(loop_->check);
    this->add_criterion_ref(loop_->criterion);
    this->set_serial_on_replica();
    this->build();
}

DHF::~DHF() {
    if (loop_)
        loop_->cycle.reset();
    // Drop built stages while the heap loop is still alive.
    this->shutdown();
}

DHF::DHF(DHF&& other) noexcept
    : cpu::Driver<ATV::scf::Phase>(std::move(other)), HF(std::move(other)),
      basis(other.basis), charge(other.charge), tol(other.tol),
      max_iter(other.max_iter), loop_(std::move(other.loop_)),
      energy_ha_(other.energy_ha_), iterations_(other.iterations_) {}

void DHF::destroy_on(cpu::Runtime& rt) {
    if (loop_)
        loop_->cycle.reset();
    cpu::Driver<ATV::scf::Phase>::destroy_on(rt);
}

void DHF::commit(Frame& frame, int executed, int cap, Write write) {
    const double last = loop_->cycle->energy;
    if (!this->converged()) {
        throw std::runtime_error(
            "DHF: SCF did not converge in " + std::to_string(cap)
            + " iterations (last |energy| = " + std::to_string(last) + " Ha)");
    }
    energy_ha_ = last;
    iterations_ = executed;
    if (write == Write::None)
        return;
    const double e_ev = energy_ha_ * units::kHartreeInEv;
    if (write == Write::Accumulate) {
        frame.meta.get(kEnergy) += e_ev;
        frame.meta.get(kEnergyElectronic) += e_ev;
    } else {
        frame.meta.get(kEnergy) = e_ev;
        frame.meta.get(kEnergyElectronic) = e_ev;
    }
}

void DHF::solve(Frame& frame, Write write) {
    require_controls(tol, max_iter);
    this->setup_pass(frame);
    try {
        Loop::Hold hold(*loop_, frame, basis, charge);
        int executed = 0;
        for (int it = 0; it < max_iter; ++it) {
            this->mutable_outer_status().reset();
            this->step_once(it);
            ++executed;
            if (this->poll_criteria_(it) != ATV::Verdict::Continue)
                break;
        }
        commit(frame, executed, max_iter, write);
        this->teardown_pass();
    } catch (...) {
        this->teardown_pass();
        throw;
    }
}

void DHF::run(const Frame& frame) {
    Frame scratch(frame);
    solve(scratch, Write::None);
}

void DHF::run(Frame& frame) { solve(frame, Write::Overwrite); }

int DHF::run_nested_erased(cpu::Context& ctx, cpu::Frame& frame, int budget) {
    if (!ctx.exec.runtime)
        throw std::runtime_error("cpu::scf::DHF: nested run has no Runtime");
    require_controls(tol, max_iter);
    Loop::Hold hold(*loop_, frame, basis, charge);
    const int cap = budget > 0 ? budget : max_iter;
    const int executed =
        cpu::Driver<ATV::scf::Phase>::run_nested(ctx, frame, cap);
    commit(frame, executed, cap, Write::Accumulate);
    return executed;
}

GradientResult DHF::forces(Frame& /*frame*/) {
    throw std::runtime_error(
        "DHF::forces: analytic Dirac–Hartree–Fock nuclear forces are not "
        "implemented (missing kernel dhf_nuclear_forces; PySCF entry is "
        "pyscf.grad.dhf.Gradients, outside this product)");
}

} // namespace scf
} // namespace cpu
} // namespace ATV
