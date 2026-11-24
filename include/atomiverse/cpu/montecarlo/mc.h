/// @file cpu/montecarlo/mc.h
/// @brief CPU equilibrium Monte Carlo driver and shared sampling lifecycle.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/montecarlo/transaction.h"

namespace ATV::cpu::mc {

/// Runs equilibrium Monte Carlo trials using a mover and an outcome policy.
class MC : public Driver<ATV::sampling::Phase> {
  public:
    explicit MC(std::uint64_t seed = 0);
    MC(AccReal temperature_K, std::uint64_t seed);
    ~MC() override;

    MC& set_nvt(AccReal temperature_K);
    MC& set_npt(AccReal temperature_K, AccReal pressure);
    MC& set_gcmc(AccReal temperature_K, AccReal mu, AccReal mass);

    template <typename P>
    std::decay_t<P>& set_outcome(P&& policy) {
        auto& installed =
            Driver<ATV::sampling::Phase>::set_outcome(std::forward<P>(policy));
        bootstrapped_ = false;
        return installed;
    }

    /// Advance up to budget trials or kinetic events. Consecutive calls on
    /// the same Frame continue the run. An absorbing state stops early.
    BigInt run(cpu::Frame& frame, BigInt budget);
    void build_on(cpu::Runtime& runtime) override;
    int run_nested_erased(cpu::Context& ctx, cpu::Frame& frame,
                          int budget) override;

    /// Install one proposal generator. Use MoveSelector to mix moves.
    template <typename M>
        requires(!std::is_lvalue_reference_v<M>)
    MC& set_mover(M&& mover) {
        this->set_singular_default_owned_(std::forward<M>(mover), mover_set_,
                                          "set_mover");
        return *this;
    }

    template <typename M>
    MC& set_mover_ref(M& mover) {
        this->set_singular_default_ref_(mover, mover_set_, "set_mover_ref");
        return *this;
    }

    template <typename M>
    MC& set_mover(M&) = delete;

    bool changed_connectivity() const noexcept { return changed_connectivity_; }

    virtual AccReal current_energy() const;
    AccReal current_volume() const noexcept { return current_volume_; }
    long long attempts() const;
    long long accepts() const;
    long long attempts(Transaction::Verb verb) const;
    long long accepts(Transaction::Verb verb) const;
    double acceptance_ratio() const;
    double acceptance_ratio(Transaction::Verb verb) const;

  protected:
    /// Construct the shared loop without equilibrium stage components.
    struct EmptyPlan {};
    MC(std::uint64_t seed, EmptyPlan);

    virtual void prepare_sampling_(cpu::Runtime& runtime, cpu::Frame& frame);
    /// Return false to finish without consuming an iteration or observing it.
    virtual bool advance_step_(cpu::Runtime& runtime, cpu::Frame& frame,
                               BigInt step);
    std::mt19937_64 rng_;
    Transaction proposal_;

    bool changed_connectivity_ = false;

    virtual void require_trial_statistics_() const {}

    void install_defaults() override;
    void validate_components() const override;
    bool phase_wants_forces(ATV::sampling::Phase phase) const override;

  private:
    BigInt advance_(cpu::Runtime& runtime, cpu::Frame& frame, BigInt budget);
    void score_(cpu::Runtime& runtime, BigInt step);

    std::optional<std::string> mover_set_;
    bool bootstrapped_ = false;
    AccReal current_energy_ = 0;
    AccReal current_volume_ = 0;
    static constexpr std::size_t kNVerbs = 5;
    std::array<long long, kNVerbs> attempts_{};
    std::array<long long, kNVerbs> accepts_{};
};

} // namespace ATV::cpu::mc
