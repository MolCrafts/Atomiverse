/// @file src/cpu/path/remd.cpp
#include "atomiverse/cpu/path/remd.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/velocity_verlet.h"

namespace ATV {
namespace cpu {
namespace path {

Remd::Remd(RemdConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.temperatures.size() < 2)
        throw std::invalid_argument("Remd: need ≥ 2 temperatures");
    for (std::size_t i = 1; i < cfg_.temperatures.size(); ++i) {
        if (!(cfg_.temperatures[i] > cfg_.temperatures[i - 1]))
            throw std::invalid_argument(
                "Remd: temperatures must be strictly increasing");
        if (!(cfg_.temperatures[i] > 0.0) || !(cfg_.temperatures[i - 1] > 0.0))
            throw std::invalid_argument("Remd: temperatures must be positive");
    }
    if (cfg_.steps_per_exchange < 1)
        throw std::invalid_argument("Remd: steps_per_exchange must be ≥ 1");
    expansion_ = std::make_unique<IdentityReplication>(cfg_.temperatures);
    exchange_rng_.seed(cfg_.seed);
}

void Remd::set_program_builder(ProgramBuilder builder, void* user) {
    builder_ = builder;
    builder_user_ = user;
}

void Remd::ensure_built_(cpu::Frame& frame) {
    if (built_)
        return;
    if (!builder_)
        throw std::runtime_error("Remd: set_program_builder before run");

    band_ = std::make_unique<ReplicaSet>(frame, *expansion_);
    const int n = band_->size();
    programs_.clear();
    programs_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        auto md = std::make_unique<cpu::md::MD>();
        md->set_integrator(cpu::md::VelocityVerlet{cfg_.dt});
        // Independent InitVelocity stream per replica (seed ^ (i+1)).
        md->add(cpu::md::InitVelocity{
            cfg_.temperatures[static_cast<std::size_t>(i)],
            static_cast<unsigned>(cfg_.seed ^ (i + 1u))});
        builder_(*md, cfg_.temperatures[static_cast<std::size_t>(i)],
                 builder_user_);
        md->enable_timing(false);
        md->build();
        programs_.push_back(std::move(md));
    }
    built_ = true;
}

void Remd::propagate_() {
    const int n = band_->size();
    for (int i = 0; i < n; ++i)
        programs_[static_cast<std::size_t>(i)]->run(band_->frame(i),
                                                    cfg_.steps_per_exchange);
}

void Remd::exchange_round_(bool even_edges) {
    const int n = band_->size();
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int i = even_edges ? 0 : 1; i + 1 < n; i += 2) {
        ++attempts_;
        auto& fi = band_->frame(i);
        auto& fj = band_->frame(i + 1);
        const double Ui = fi.meta.get(kEnergy);
        const double Uj = fj.meta.get(kEnergy);
        const double Ti = cfg_.temperatures[static_cast<std::size_t>(i)];
        const double Tj = cfg_.temperatures[static_cast<std::size_t>(i + 1)];
        const double beta_i = 1.0 / (units::kB_ev_per_k * Ti);
        const double beta_j = 1.0 / (units::kB_ev_per_k * Tj);
        const double dbeta = beta_i - beta_j;
        const double dU = Ui - Uj;
        const double arg = dbeta * dU;
        const double p = (arg >= 0.0) ? 1.0 : std::exp(arg);
        if (u01(exchange_rng_) < p) {
            ++accepts_;
            // Swap configurations; then rescale velocities so each slot keeps
            // its temperature: after swap, fi holds j's old v → *√(Ti/Tj).
            const double si = std::sqrt(Ti / Tj);
            const double sj = std::sqrt(Tj / Ti);
            const int nloc = fi.nlocal();
            for (int a = 0; a < nloc; ++a) {
                std::swap(fi.span(kAtom_x)[a], fj.span(kAtom_x)[a]);
                std::swap(fi.span(kAtom_y)[a], fj.span(kAtom_y)[a]);
                std::swap(fi.span(kAtom_z)[a], fj.span(kAtom_z)[a]);
                std::swap(fi.span(kAtom_vx)[a], fj.span(kAtom_vx)[a]);
                std::swap(fi.span(kAtom_vy)[a], fj.span(kAtom_vy)[a]);
                std::swap(fi.span(kAtom_vz)[a], fj.span(kAtom_vz)[a]);
                fi.span(kAtom_vx)[a] *= si;
                fi.span(kAtom_vy)[a] *= si;
                fi.span(kAtom_vz)[a] *= si;
                fj.span(kAtom_vx)[a] *= sj;
                fj.span(kAtom_vy)[a] *= sj;
                fj.span(kAtom_vz)[a] *= sj;
            }
            std::swap(fi.meta.get(kEnergy), fj.meta.get(kEnergy));
        }
    }
}

void Remd::run(cpu::Frame& frame, int n_exchanges) {
    if (n_exchanges < 1)
        throw std::invalid_argument("Remd::run: n_exchanges must be ≥ 1");
    ensure_built_(frame);
    for (int r = 0; r < n_exchanges; ++r) {
        propagate_();
        exchange_round_(/*even_edges=*/(r % 2 == 0));
    }
    band_->collapse_into(frame, *expansion_);
}

} // namespace path
} // namespace cpu
} // namespace ATV
