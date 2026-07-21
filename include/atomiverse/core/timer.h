/// @file core/timer.h
/// @brief Systematic per-component timing registry. CUDA-free, header-only.
///
/// One `TimerRegistry` records the wall time of every component launch (keyed
/// by component name + phase + a category rollup) plus optional RAII
/// sub-scopes (`ScopedTimer`) for drilling into regions inside a component.
/// It is opt-in (`set_enabled`) so the engine pays nothing when off, and it is
/// timing-source-agnostic (`record` takes a plain elapsed double) so a future
/// CUDA backend can feed it event-elapsed times into the same data model.
///
/// Aggregation (per-rank min/avg/max, load balance) and rendering live in the
/// MD layer (`cpu/md/timing_report.h`) which knows the category names; this
/// header stays domain-agnostic. Output is routed through the log module's
/// report path (`core/report.h`). @see core/report.h
#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ATV {

/// @brief Identity of a timed unit. `subkey` is empty for a top-level
/// component launch; non-empty for a `ScopedTimer` sub-region.
struct TimerKey {
    std::string component;
    int phase = -1;
    int category = -1;
    std::string subkey;
};

/// @brief Accumulated stats for one key.
struct TimerStat {
    double total_s = 0.0;
    std::uint64_t calls = 0;
    double min_s = std::numeric_limits<double>::infinity();
    double max_s = 0.0;

    void add(double s) {
        total_s += s;
        ++calls;
        if (s < min_s)
            min_s = s;
        if (s > max_s)
            max_s = s;
    }
    double avg_s() const {
        return calls ? total_s / static_cast<double>(calls) : 0.0;
    }
};

/// @brief Per-component timing registry, owned by a Backend and reached by
/// components via a `Context` pointer.
class TimerRegistry {
  public:
    bool enabled() const noexcept { return enabled_; }
    void set_enabled(bool on) noexcept { enabled_ = on; }

    /// @brief Register a top-level op once (at build time); returns a dense
    /// handle for the hot path. Safe to call when disabled.
    int register_op(TimerKey key) {
        const int h = static_cast<int>(top_.size());
        top_keys_.push_back(std::move(key));
        top_.emplace_back();
        return h;
    }

    /// @brief Record a top-level op's elapsed time (hot path).
    void record(int handle, double secs) noexcept {
        if (handle >= 0 && handle < static_cast<int>(top_.size()))
            top_[static_cast<std::size_t>(handle)].add(secs);
    }

    /// @brief Record a scoped sub-region (rarer path; keyed by subkey string).
    void record_scoped(std::string_view subkey, int phase, int category,
                       double secs) {
        auto& st = scoped_[std::string(subkey)];
        if (st.calls == 0) {
            scoped_meta_[std::string(subkey)] = TimerKey{
                /*component=*/"", phase, category, std::string(subkey)};
        }
        st.add(secs);
    }

    void set_loop(double secs, int nlocal) noexcept {
        loop_seconds_ += secs;
        nlocal_ = nlocal;
    }
    void tick_step() noexcept { ++steps_; }

    /// @brief Clear stats but keep registered handles/keys (the plan is stable
    /// across a stage). Used to reset between stages or after warmup.
    void reset() {
        for (auto& s : top_)
            s = TimerStat{};
        scoped_.clear();
        scoped_meta_.clear();
        loop_seconds_ = 0.0;
        steps_ = 0;
        nlocal_ = -1;
    }

    /// @brief Roll up top-level stats into per-category buckets, indexed by
    /// category id. Sized to (max category id + 1).
    std::vector<TimerStat> rollup_by_category() const {
        int ncat = 0;
        for (const auto& k : top_keys_)
            ncat = (k.category + 1 > ncat) ? k.category + 1 : ncat;
        std::vector<TimerStat> out(static_cast<std::size_t>(ncat));
        for (std::size_t i = 0; i < top_.size(); ++i) {
            const int c = top_keys_[i].category;
            if (c < 0)
                continue;
            auto& dst = out[static_cast<std::size_t>(c)];
            dst.total_s += top_[i].total_s;
            dst.calls += top_[i].calls;
            if (top_[i].min_s < dst.min_s)
                dst.min_s = top_[i].min_s;
            if (top_[i].max_s > dst.max_s)
                dst.max_s = top_[i].max_s;
        }
        return out;
    }

    const std::vector<TimerStat>& per_component() const { return top_; }
    const std::vector<TimerKey>& keys() const { return top_keys_; }
    const std::unordered_map<std::string, TimerStat>& scoped() const {
        return scoped_;
    }
    double loop_seconds() const noexcept { return loop_seconds_; }
    std::uint64_t steps() const noexcept { return steps_; }
    int nlocal() const noexcept { return nlocal_; }

  private:
    bool enabled_ = false;
    std::vector<TimerKey> top_keys_;
    std::vector<TimerStat> top_;
    std::unordered_map<std::string, TimerStat> scoped_;
    std::unordered_map<std::string, TimerKey> scoped_meta_;
    double loop_seconds_ = 0.0;
    std::uint64_t steps_ = 0;
    int nlocal_ = -1;
};

/// @brief RAII timer for a sub-region inside a component. Records into the
/// registry's scoped map on destruction. No-op when the registry is null or
/// disabled (so the macro is free when timing is off).
class ScopedTimer {
  public:
    ScopedTimer(TimerRegistry* reg, std::string_view subkey, int phase,
                int category)
        : reg_((reg && reg->enabled()) ? reg : nullptr), subkey_(subkey),
          phase_(phase), category_(category),
          t0_(reg_ ? std::chrono::steady_clock::now()
                   : std::chrono::steady_clock::time_point{}) {}

    ~ScopedTimer() {
        if (!reg_)
            return;
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0_)
                             .count();
        reg_->record_scoped(subkey_, phase_, category_, s);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

  private:
    TimerRegistry* reg_;
    std::string_view subkey_;
    int phase_;
    int category_;
    std::chrono::steady_clock::time_point t0_;
};

} // namespace ATV

/// @brief Profile a scope into the registry reachable from a launch context.
/// `ctx.runtime.timer` is the `TimerRegistry*` published by the Backend.
#define ATV_PROFILE_SCOPE(ctx, name, phase, category)                          \
    ::ATV::ScopedTimer ATV_PROFILE_SCOPE_NAME_(_atv_scope_, __LINE__)(         \
        (ctx).runtime.timer, (name), (phase), (category))
#define ATV_PROFILE_SCOPE_CAT_(a, b) a##b
#define ATV_PROFILE_SCOPE_NAME_(a, b) ATV_PROFILE_SCOPE_CAT_(a, b)
