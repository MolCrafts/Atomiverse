/// @file cpu/scf/kuhf.cpp
/// @brief cpu::scf::KUHF — Driver<scf::Phase> over the FFTDF kernel.
///
/// PySCF 2.14.0 `pyscf/pbc/scf/kuhf.py` (`KUHF.nelec`, `get_veff`,
/// `get_occ`, `energy_elec`, `eig`), `pyscf/pbc/scf/khf.py` `KSCF.eig`,
/// and `pyscf/scf/hf.py` `kernel`. DIIS stays off (`get_fock` diis=None),
/// so the mix stage is the identity and the only residual is |ΔE|.

#include "atomiverse/cpu/scf/kuhf.h"

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

using internal::KuhfCycleOwner;

KuhfCycleOwner* require_cycle(KuhfCycleOwner* cycle) {
    if (!cycle || !*cycle)
        throw std::runtime_error("cpu::scf::KUHF: SCF cycle is not open");
    return cycle;
}

/// One plan op for every SCF phase. Mix is the identity (`diis=None`).
class Stage : public cpu::Component {
  public:
    enum class Kind { Fock, Mix, Diag, Density, Check };

    Stage(KuhfCycleOwner* cycle, Kind kind) : cycle_(cycle), kind_(kind) {}

    std::string name() const override {
        switch (kind_) {
        case Kind::Fock:
            return "kuhf_fock";
        case Kind::Mix:
            return "kuhf_mix";
        case Kind::Diag:
            return "kuhf_diagonalize";
        case Kind::Density:
            return "kuhf_density";
        case Kind::Check:
            return "kuhf_check";
        }
        throw std::logic_error("cpu::scf::KUHF: unknown stage");
    }

    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(phase_of(kind_)) || kind_ == Kind::Mix)
            return;
        auto& cycle = **require_cycle(cycle_);
        if (kind_ == Kind::Fock) {
            // KUHF.get_veff: J = Ja+Jb, vhf = J−K, then energy_elec.
            internal::kuhf_cycle_fock(cycle);
            return;
        }
        if (kind_ == Kind::Diag) {
            internal::kuhf_cycle_diagonalize(cycle);
            return;
        }
        if (kind_ == Kind::Density) {
            internal::kuhf_cycle_density(cycle);
            return;
        }
        if (!ctx.runtime.loop.metrics)
            throw std::runtime_error(
                "cpu::scf::KUHF: check stage has no ConvergenceMetrics");
        const double ha = internal::kuhf_cycle_energy(cycle);
        const double dha = internal::kuhf_cycle_delta(cycle);
        ctx.runtime.loop.metrics->energy = ha * units::kHartreeInEv;
        ctx.runtime.loop.metrics->energy_change = dha * units::kHartreeInEv;
    }

  private:
    static ATV::scf::Phase phase_of(Kind kind) {
        switch (kind) {
        case Kind::Fock:
            return ATV::scf::Phase::fock;
        case Kind::Mix:
            return ATV::scf::Phase::mix;
        case Kind::Diag:
            return ATV::scf::Phase::diagonalize;
        case Kind::Density:
            return ATV::scf::Phase::density;
        case Kind::Check:
            return ATV::scf::Phase::check;
        }
        throw std::logic_error("cpu::scf::KUHF: unknown stage");
    }

    KuhfCycleOwner* cycle_ = nullptr;
    Kind kind_ = Kind::Fock;
};

/// @brief |ΔE| only. ScfConverged also demands a DIIS residual this kernel
///        does not build.
class EnergyDelta : public cpu::Criterion {
  public:
    explicit EnergyDelta(double tol_ha)
        : tol_ev_(tol_ha * units::kHartreeInEv) {}

    std::string name() const override { return "kuhf_energy_delta"; }

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

void require_controls(double tol, int max_iter) {
    if (!(tol > 0.0) || max_iter < 1)
        throw std::runtime_error(
            "KUHF: tol must be positive and max_iter >= 1");
}

} // namespace

struct KUHF::Loop {
    internal::KuhfCycleOwner cycle{nullptr, &internal::kuhf_cycle_delete};
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

KUHF::KUHF(BasisSet basis_in, std::array<int, 3> k_mesh_in, int n_alpha_in,
           int n_beta_in, int charge_in, double tol_in, int max_iter_in)
    : basis(basis_in), k_mesh(k_mesh_in), n_alpha(n_alpha_in),
      n_beta(n_beta_in), charge(charge_in), tol(tol_in), max_iter(max_iter_in),
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

KUHF::~KUHF() {
    if (loop_)
        loop_->cycle.reset();
    // Drop built stages while the heap loop is still alive.
    this->shutdown();
}

KUHF::KUHF(KUHF&& other) noexcept
    : cpu::Driver<ATV::scf::Phase>(std::move(other)), HF(std::move(other)),
      basis(other.basis), k_mesh(other.k_mesh), n_alpha(other.n_alpha),
      n_beta(other.n_beta), charge(other.charge), tol(other.tol),
      max_iter(other.max_iter), loop_(std::move(other.loop_)),
      energy_ha_(other.energy_ha_), iterations_(other.iterations_) {}

void KUHF::destroy_on(cpu::Runtime& rt) {
    if (loop_)
        loop_->cycle.reset();
    cpu::Driver<ATV::scf::Phase>::destroy_on(rt);
}

void KUHF::commit(Frame& frame, int executed, int cap, Write write) {
    if (!this->converged()) {
        const double last = internal::kuhf_cycle_energy(*loop_->cycle);
        throw std::runtime_error(
            "KUHF: SCF did not converge in " + std::to_string(cap)
            + " iterations (last |energy| = " + std::to_string(last) + " Ha)");
    }
    energy_ha_ = internal::kuhf_cycle_energy(*loop_->cycle);
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

void KUHF::solve(Frame& frame, Write write) {
    require_controls(tol, max_iter);
    this->setup_pass(frame);
    try {
        const int mesh[3] = {k_mesh[0], k_mesh[1], k_mesh[2]};
        loop_->cycle = internal::kuhf_cycle_open(frame, basis, mesh, n_alpha,
                                                 n_beta, charge);
        struct Release {
            Loop* loop = nullptr;
            ~Release() {
                if (loop)
                    loop->cycle.reset();
            }
        } release{loop_.get()};
        this->status_.reset();
        this->metrics_ = ConvergenceMetrics{};
        int executed = 0;
        for (int it = 0; it < max_iter; ++it) {
            this->mutable_outer_status().reset();
            this->step_once(it);
            ++executed;
            if (this->poll_criteria_(it) != ATV::Verdict::Continue)
                break;
        }
        commit(frame, executed, max_iter, write);
        loop_->cycle.reset();
        this->teardown_pass();
    } catch (...) {
        this->teardown_pass();
        throw;
    }
}

void KUHF::run(const Frame& frame) {
    Frame scratch(frame);
    solve(scratch, Write::None);
}

void KUHF::run(Frame& frame) { solve(frame, Write::Overwrite); }

int KUHF::run_nested_erased(cpu::Context& ctx, cpu::Frame& frame, int budget) {
    if (!ctx.exec.runtime)
        throw std::runtime_error("cpu::scf::KUHF: nested run has no Runtime");
    require_controls(tol, max_iter);
    const int mesh[3] = {k_mesh[0], k_mesh[1], k_mesh[2]};
    loop_->cycle =
        internal::kuhf_cycle_open(frame, basis, mesh, n_alpha, n_beta, charge);
    struct Release {
        Loop* loop = nullptr;
        ~Release() {
            if (loop)
                loop->cycle.reset();
        }
    } release{loop_.get()};
    const int cap = budget > 0 ? budget : max_iter;
    const int executed =
        cpu::Driver<ATV::scf::Phase>::run_nested(ctx, frame, cap);
    commit(frame, executed, cap, Write::Accumulate);
    return executed;
}

GradientResult KUHF::forces(Frame& /*frame*/) {
    throw std::runtime_error(
        "KUHF::forces: analytic k-point nuclear forces are not implemented "
        "(missing kernel kuhf_nuclear_forces; PySCF entry is "
        "pyscf.pbc.grad.kuhf.Gradients, outside this product)");
}

} // namespace scf
} // namespace cpu
} // namespace ATV
