/// @file src/cpu/path/pimd.cpp
#include "atomiverse/cpu/path/pimd.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"

#include <random>

namespace ATV {
namespace cpu {
namespace path {

namespace {

/// @brief External isotropic harmonic trap F = −k r (about origin).
class HarmonicTrap : public Component {
  public:
    explicit HarmonicTrap(double k_ev_per_a2) : k_(k_ev_per_a2) {
        if (!(k_ev_per_a2 >= 0.0))
            throw std::invalid_argument("HarmonicTrap: k must be ≥ 0");
    }
    std::string name() const override { return "harmonic_trap"; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::force, +10)};
    }
    void launch(const LaunchContext& ctx) override {
        if (!(k_ > 0.0))
            return;
        auto& f = ctx.frame;
        const int n = f.nlocal();
        const auto x = f.span(kAtom_x);
        const auto y = f.span(kAtom_y);
        const auto z = f.span(kAtom_z);
        auto fx = f.span(kAtom_fx);
        auto fy = f.span(kAtom_fy);
        auto fz = f.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            fx[i] += -k_ * x[i];
            fy[i] += -k_ * y[i];
            fz[i] += -k_ * z[i];
        }
    }

  private:
    double k_;
};

} // namespace

/// @brief Draw thermal velocities onto a bead Frame (no COM wipe for N=1).
static void seed_bead_velocity_(cpu::Frame& f, double temperature_K,
                                unsigned seed) {
    const int n = f.nlocal();
    if (n <= 0)
        return;
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    auto vz = f.span(kAtom_vz);
    const auto mass = f.span(kAtom_mass);
    const double kT = units::kB_ev_per_k * temperature_K;
    for (int i = 0; i < n; ++i) {
        const double m = static_cast<double>(mass[i]);
        const double sig =
            std::sqrt(kT / (units::mass_v2_to_energy * m)); // [Å/fs]
        vx[i] = sig * g(rng);
        vy[i] = sig * g(rng);
        vz[i] = sig * g(rng);
    }
    if (n >= 2) {
        // Remove COM velocity (multi-atom hosts).
        double px = 0, py = 0, pz = 0, M = 0;
        for (int i = 0; i < n; ++i) {
            const double m = static_cast<double>(mass[i]);
            px += m * vx[i];
            py += m * vy[i];
            pz += m * vz[i];
            M += m;
        }
        const double invM = 1.0 / M;
        for (int i = 0; i < n; ++i) {
            vx[i] -= px * invM;
            vy[i] -= py * invM;
            vz[i] -= pz * invM;
        }
    }
}

double Pimd::compute_k_spring_(double temperature_K, int n_beads,
                               double mass_amu) {
    // ℏ = h / 2π [eV·fs]
    const double hbar = units::h_planck_ev_fs / (2.0 * units::pi);
    const double kT = units::kB_ev_per_k * temperature_K;
    const double beta = 1.0 / kT;
    // k = m_eff * P / (β² ℏ²) with m_eff = mass_v2_to_energy * m [eV·fs²/Å²]
    const double m_eff = units::mass_v2_to_energy * mass_amu;
    const double beta_hbar = beta * hbar;
    return m_eff * static_cast<double>(n_beads) / (beta_hbar * beta_hbar);
}

Pimd::Pimd(int n_beads_, double temperature_, double dt_, int steps_per_block_,
           std::uint64_t seed_, double k_external_)
    : n_beads(n_beads_), temperature(temperature_), dt(dt_),
      steps_per_block(steps_per_block_), seed(seed_), k_external(k_external_) {
    if (n_beads < 2)
        throw std::invalid_argument("Pimd: n_beads must be ≥ 2");
    if (!(temperature > 0.0))
        throw std::invalid_argument("Pimd: temperature must be > 0 K");
    if (!(dt > 0.0))
        throw std::invalid_argument("Pimd: dt must be > 0 fs");
    if (steps_per_block < 1)
        throw std::invalid_argument("Pimd: steps_per_block must be ≥ 1");
    if (!(k_external >= 0.0))
        throw std::invalid_argument("Pimd: k_external must be ≥ 0");

    std::vector<double> bead_ids(static_cast<std::size_t>(n_beads));
    for (int i = 0; i < n_beads; ++i)
        bead_ids[static_cast<std::size_t>(i)] = static_cast<double>(i);
    expansion_ = std::make_unique<IdentityReplication>(std::move(bead_ids));
}

void Pimd::set_program_builder(ProgramBuilder builder, void* user) {
    builder_ = builder;
    builder_user_ = user;
}

void Pimd::ensure_built_(cpu::Frame& frame) {
    if (built_)
        return;

    // Spring mass from first atom (uniform-mass skeleton).
    double mass = 1.0;
    if (frame.nlocal() > 0)
        mass = static_cast<double>(frame.span(kAtom_mass)[0]);
    if (!(mass > 0.0))
        throw std::invalid_argument("Pimd: atom mass must be > 0 amu");
    mass_amu_ = mass;
    k_spring_ = compute_k_spring_(temperature, n_beads, mass);

    band_ = std::make_unique<ReplicaSet>(frame, *expansion_);
    const int P = band_->size();
    programs_.clear();
    springs_.clear();
    programs_.reserve(static_cast<std::size_t>(P));
    springs_.reserve(static_cast<std::size_t>(P));

    for (int s = 0; s < P; ++s) {
        auto spring =
            std::make_unique<RingPolymerSpring>(band_.get(), s, k_spring_);
        springs_.push_back(std::move(spring));
    }
    for (int s = 0; s < P; ++s) {
        auto md = std::make_unique<cpu::md::MD>();
        md->set_integrator(cpu::md::VelocityVerlet{dt});
        if (k_external > 0.0)
            md->on(ATV::md::Phase::force).add(HarmonicTrap{k_external});
        // Spring after optional trap / user forces (borrow springs_[s]).
        md->on(ATV::md::Phase::force)
            .add_ref(*springs_[static_cast<std::size_t>(s)]);
        if (builder_)
            builder_(*md, s, builder_user_);
        md->enable_timing(false);
        md->build();
        // Seed Maxwell–Boltzmann velocities on the replica Frame.
        // InitVelocity always subtracts COM velocity; for nlocal==1 that
        // zeros v entirely — seed without COM removal in that case.
        seed_bead_velocity_(band_->frame(s), temperature,
                            static_cast<unsigned>(seed ^ (s + 1u)));
        programs_.push_back(std::move(md));
    }
    built_ = true;
}

void Pimd::propagate_() {
    const int P = band_->size();
    for (int s = 0; s < P; ++s)
        programs_[static_cast<std::size_t>(s)]->run(band_->frame(s),
                                                    steps_per_block);
}

void Pimd::collapse_centroid_(cpu::Frame& out) {
    const int P = band_->size();
    const int n = out.nlocal();
    auto ox = out.span(kAtom_x);
    auto oy = out.span(kAtom_y);
    auto oz = out.span(kAtom_z);
    for (int i = 0; i < n; ++i) {
        ox[i] = 0;
        oy[i] = 0;
        oz[i] = 0;
    }
    for (int s = 0; s < P; ++s) {
        const auto& f = band_->frame(s);
        const auto x = f.span(kAtom_x);
        const auto y = f.span(kAtom_y);
        const auto z = f.span(kAtom_z);
        for (int i = 0; i < n; ++i) {
            ox[i] += x[i];
            oy[i] += y[i];
            oz[i] += z[i];
        }
    }
    const double invP = 1.0 / static_cast<double>(P);
    for (int i = 0; i < n; ++i) {
        ox[i] *= invP;
        oy[i] *= invP;
        oz[i] *= invP;
    }

    // Diagnostic: mean squared bead–centroid displacement.
    double acc = 0;
    long long count = 0;
    for (int s = 0; s < P; ++s) {
        const auto& f = band_->frame(s);
        const auto x = f.span(kAtom_x);
        const auto y = f.span(kAtom_y);
        const auto z = f.span(kAtom_z);
        for (int i = 0; i < n; ++i) {
            const double dx = x[i] - ox[i];
            const double dy = y[i] - oy[i];
            const double dz = z[i] - oz[i];
            acc += dx * dx + dy * dy + dz * dz;
            ++count;
        }
    }
    mean_spread_ = (count > 0) ? acc / static_cast<double>(count) : 0.0;
    // R_g² = (1/P) Σ_b |r_b − r_c|² for the (single) free particle case n=1;
    // for multi-atom hosts average over atoms as well.
    rg2_ = mean_spread_;
}

double Pimd::free_particle_rg2_analytic() const {
    // 3D free ring polymer (single particle):
    // ⟨R_g²⟩ = (ℏ² β² / (4 m_eff)) (1 − 1/P²)
    // with m_eff = mass_v2_to_energy * m [eV fs² / Å²], ℏ in eV·fs.
    // Ref: Tuckerman, Statistical Mechanics; Chandler & Wolynes (1981).
    const double hbar = units::h_planck_ev_fs / (2.0 * units::pi);
    const double kT = units::kB_ev_per_k * temperature;
    const double beta = 1.0 / kT;
    const double m_eff = units::mass_v2_to_energy * mass_amu_;
    const double P = static_cast<double>(n_beads);
    const double hb = hbar * beta;
    return (hb * hb / (4.0 * m_eff)) * (1.0 - 1.0 / (P * P));
}

void Pimd::run(cpu::Frame& frame, int n_blocks) {
    if (n_blocks < 1)
        throw std::invalid_argument("Pimd::run: n_blocks must be ≥ 1");
    ensure_built_(frame);
    for (int b = 0; b < n_blocks; ++b)
        propagate_();
    collapse_centroid_(frame);
}

} // namespace path
} // namespace cpu
} // namespace ATV
