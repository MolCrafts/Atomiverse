/// @file src/cpu/path/remd.cpp
#include "atomiverse/cpu/path/remd.h"

#include <cmath>
#include <stdexcept>

#include <unordered_map>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/fix_csvr.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"

namespace ATV {
namespace cpu {
namespace path {

Remd::Remd(std::vector<double> temperatures_, double dt_,
           int steps_per_exchange_, std::uint64_t seed_)
    : temperatures(std::move(temperatures_)), dt(dt_),
      steps_per_exchange(steps_per_exchange_), seed(seed_) {
    if (temperatures.size() < 2)
        throw std::invalid_argument("Remd: need ≥ 2 temperatures");
    for (std::size_t i = 1; i < temperatures.size(); ++i) {
        if (!(temperatures[i] > temperatures[i - 1]))
            throw std::invalid_argument(
                "Remd: temperatures must be strictly increasing");
        if (!(temperatures[i] > 0.0) || !(temperatures[i - 1] > 0.0))
            throw std::invalid_argument("Remd: temperatures must be positive");
    }
    if (steps_per_exchange < 1)
        throw std::invalid_argument("Remd: steps_per_exchange must be ≥ 1");
    if (!(thermostat_tau > 0.0))
        throw std::invalid_argument("Remd: thermostat_tau must be > 0 fs");
    expansion_ = std::make_unique<IdentityReplication>(temperatures);
    exchange_rng_.seed(seed);
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
    if (!(thermostat_tau > 0.0))
        throw std::invalid_argument("Remd: thermostat_tau must be > 0 fs");
    for (int i = 0; i < n; ++i) {
        const double t_i = temperatures[static_cast<std::size_t>(i)];
        auto md = std::make_unique<cpu::md::MD>();
        md->set_integrator(cpu::md::VelocityVerlet{dt});
        // Independent InitVelocity stream per replica (seed ^ (i+1)).
        md->set_observer(
            cpu::md::InitVelocity{t_i, static_cast<unsigned>(seed ^ (i + 1u))});
        // The exchange test is the Metropolis rule for swapping between two
        // CANONICAL ensembles. Without a thermostat each replica is an NVE
        // trajectory carrying whatever its velocity draw gave it, β_i is a
        // label rather than a distribution, and the accept ratio measures
        // nothing.
        md->set_thermostat(cpu::md::FixCSVR{
            t_i, thermostat_tau, dt,
            static_cast<std::uint32_t>((seed * 2654435761u) ^ (i + 1u))});
        builder_(*md, t_i, builder_user_);
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
                                                    steps_per_exchange);
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
        const double Ti = temperatures[static_cast<std::size_t>(i)];
        const double Tj = temperatures[static_cast<std::size_t>(i + 1)];
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
            if (fj.nlocal() != nloc)
                throw std::runtime_error(
                    "Remd: replicas hold different atom counts — an exchange "
                    "swaps one configuration for another, so the two must be "
                    "the same system");
            // Pair the atoms by TAG, not by row. Each replica runs its own MD
            // program with its own neighbour list, and `atom_modify sort`
            // spatially reorders owned rows at its own cadence in each — so
            // two replicas that started identical do not stay row-aligned. A
            // row-wise swap would then hand replica i atom 7's coordinates
            // under atom 3's tag.
            const auto tag_i = fi.span(kAtom_tag);
            const auto tag_j = fj.span(kAtom_tag);
            std::unordered_map<TagInt, int> row_of_j;
            row_of_j.reserve(static_cast<std::size_t>(nloc));
            for (int b = 0; b < nloc; ++b)
                row_of_j.emplace(tag_j[b], b);
            for (int a = 0; a < nloc; ++a) {
                const auto it = row_of_j.find(tag_i[a]);
                if (it == row_of_j.end())
                    throw std::runtime_error(
                        "Remd: replicas do not carry the same tags — an "
                        "exchange is a swap between two copies of ONE system");
                const int b = it->second;
                std::swap(fi.span(kAtom_x)[a], fj.span(kAtom_x)[b]);
                std::swap(fi.span(kAtom_y)[a], fj.span(kAtom_y)[b]);
                std::swap(fi.span(kAtom_z)[a], fj.span(kAtom_z)[b]);
                std::swap(fi.span(kAtom_vx)[a], fj.span(kAtom_vx)[b]);
                std::swap(fi.span(kAtom_vy)[a], fj.span(kAtom_vy)[b]);
                std::swap(fi.span(kAtom_vz)[a], fj.span(kAtom_vz)[b]);
                fi.span(kAtom_vx)[a] *= si;
                fi.span(kAtom_vy)[a] *= si;
                fi.span(kAtom_vz)[a] *= si;
                fj.span(kAtom_vx)[b] *= sj;
                fj.span(kAtom_vy)[b] *= sj;
                fj.span(kAtom_vz)[b] *= sj;
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
