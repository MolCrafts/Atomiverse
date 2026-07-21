/// @file cpu/task/task_spec.h
/// @brief Closed, versioned, typed sum of launcher-dispatchable CPU tasks.
///
/// TaskSpec is an in-memory/wire boundary, not a universal component IR. Each
/// task alternative owns its own phase enum, component allow-list, and budget:
/// MD cannot carry a minimizer phase and Minimize cannot carry an MD phase.
/// Historic wire layouts are rejected; there is no compatibility decoder.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "atomiverse/facade/stage.h"

namespace ATV::cpu::task {

/// @brief Which concrete CPU driver pipeline a TaskSpec materializes.
enum class TaskKind : std::uint8_t {
    MD = 0,
    SinglePoint = 1,
    Minimize = 2,
};

inline const char* task_kind_name(TaskKind kind) noexcept {
    switch (kind) {
    case TaskKind::MD:
        return "MD";
    case TaskKind::SinglePoint:
        return "SinglePoint";
    case TaskKind::Minimize:
        return "Minimize";
    }
    return "unknown";
}

// ── Constructor parameter mirrors ─────────────────────────────────────────

struct PairLJCutParams {
    double epsilon;
    double sigma;
    double cutoff;
};
inline bool operator==(const PairLJCutParams& a, const PairLJCutParams& b) {
    return a.epsilon == b.epsilon && a.sigma == b.sigma && a.cutoff == b.cutoff;
}

struct NeighborListParams {
    double cutoff;
    double skin;
};
inline bool operator==(const NeighborListParams& a,
                       const NeighborListParams& b) {
    return a.cutoff == b.cutoff && a.skin == b.skin;
}

struct BondHarmonicParams {
    std::vector<double> K;
    std::vector<double> r0;
};
inline bool operator==(const BondHarmonicParams& a,
                       const BondHarmonicParams& b) {
    return a.K == b.K && a.r0 == b.r0;
}

struct AngleHarmonicParams {
    std::vector<double> K;
    std::vector<double> theta0;
};
inline bool operator==(const AngleHarmonicParams& a,
                       const AngleHarmonicParams& b) {
    return a.K == b.K && a.theta0 == b.theta0;
}

struct DihedralMultiHarmonicParams {
    std::vector<std::vector<double>> Cn;
};
inline bool operator==(const DihedralMultiHarmonicParams& a,
                       const DihedralMultiHarmonicParams& b) {
    return a.Cn == b.Cn;
}

struct VelocityVerletParams {
    double dt;
};
inline bool operator==(const VelocityVerletParams& a,
                       const VelocityVerletParams& b) {
    return a.dt == b.dt;
}

struct InitVelocityParams {
    double temperature;
    unsigned seed;
};
inline bool operator==(const InitVelocityParams& a,
                       const InitVelocityParams& b) {
    return a.temperature == b.temperature && a.seed == b.seed;
}

struct NoseHooverParams {
    double T_start;
    double tau_T;
    double dt;
};
inline bool operator==(const NoseHooverParams& a, const NoseHooverParams& b) {
    return a.T_start == b.T_start && a.tau_T == b.tau_T && a.dt == b.dt;
}

struct ThermoParams {
    int freq;
};
inline bool operator==(const ThermoParams& a, const ThermoParams& b) {
    return a.freq == b.freq;
}

struct AM1EvaluatorParams {
    int charge = 0;
    int multiplicity = 1;
    int max_iter = 200;
    double density_tol = 1e-8;
    double energy_tol = 1e-10;
};
inline bool operator==(const AM1EvaluatorParams& a,
                       const AM1EvaluatorParams& b) {
    return a.charge == b.charge && a.multiplicity == b.multiplicity
           && a.max_iter == b.max_iter && a.density_tol == b.density_tol
           && a.energy_tol == b.energy_tol;
}

/// @brief Mirrors the `cpu::minimizer::LBFGS` constructor exactly.
struct LBFGSParams {
    int memory = 5;
    double alpha0 = 0.01;
    int max_ls = 10;
};
inline bool operator==(const LBFGSParams& a, const LBFGSParams& b) {
    return a.memory == b.memory && a.alpha0 == b.alpha0 && a.max_ls == b.max_ls;
}

/// @brief Mirrors all five `cpu::ForceConverged` constructor tolerances.
struct ForceConvergedParams {
    double ftol = 0.05;
    double rms_ftol = 0.03;
    double max_dtol = 0.002;
    double rms_dtol = 0.001;
    double etol = 1e-5;
};
inline bool operator==(const ForceConvergedParams& a,
                       const ForceConvergedParams& b) {
    return a.ftol == b.ftol && a.rms_ftol == b.rms_ftol
           && a.max_dtol == b.max_dtol && a.rms_dtol == b.rms_dtol
           && a.etol == b.etol;
}

/// @brief Stable wire IDs, independent of variant ordering.
enum class ComponentId : std::uint16_t {
    PairLJCut = 0,
    NeighborList = 1,
    BondHarmonic = 2,
    AngleHarmonic = 3,
    DihedralMultiHarmonic = 4,
    VelocityVerlet = 5,
    InitVelocity = 6,
    NoseHoover = 7,
    Thermo = 8,
    AM1Evaluator = 9,
    LBFGS = 10,
};

inline const char* component_name(ComponentId id) noexcept {
    switch (id) {
    case ComponentId::PairLJCut:
        return "PairLJCut";
    case ComponentId::NeighborList:
        return "NeighborList";
    case ComponentId::BondHarmonic:
        return "BondHarmonic";
    case ComponentId::AngleHarmonic:
        return "AngleHarmonic";
    case ComponentId::DihedralMultiHarmonic:
        return "DihedralMultiHarmonic";
    case ComponentId::VelocityVerlet:
        return "VelocityVerlet";
    case ComponentId::InitVelocity:
        return "InitVelocity";
    case ComponentId::NoseHoover:
        return "NoseHoover";
    case ComponentId::Thermo:
        return "Thermo";
    case ComponentId::AM1Evaluator:
        return "AM1Evaluator";
    case ComponentId::LBFGS:
        return "LBFGS";
    }
    return "unknown";
}

using MDComponentParams =
    std::variant<PairLJCutParams, NeighborListParams, BondHarmonicParams,
                 AngleHarmonicParams, DihedralMultiHarmonicParams,
                 VelocityVerletParams, InitVelocityParams, NoseHooverParams,
                 ThermoParams>;

using MinimizeComponentParams =
    std::variant<PairLJCutParams, NeighborListParams, LBFGSParams>;

namespace detail {
template <typename>
inline constexpr bool unsupported_component_v = false;
} // namespace detail

inline ComponentId component_id(const MDComponentParams& params) noexcept {
    return std::visit(
        [](const auto& p) -> ComponentId {
            using P = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<P, PairLJCutParams>)
                return ComponentId::PairLJCut;
            else if constexpr (std::is_same_v<P, NeighborListParams>)
                return ComponentId::NeighborList;
            else if constexpr (std::is_same_v<P, BondHarmonicParams>)
                return ComponentId::BondHarmonic;
            else if constexpr (std::is_same_v<P, AngleHarmonicParams>)
                return ComponentId::AngleHarmonic;
            else if constexpr (std::is_same_v<P, DihedralMultiHarmonicParams>)
                return ComponentId::DihedralMultiHarmonic;
            else if constexpr (std::is_same_v<P, VelocityVerletParams>)
                return ComponentId::VelocityVerlet;
            else if constexpr (std::is_same_v<P, InitVelocityParams>)
                return ComponentId::InitVelocity;
            else if constexpr (std::is_same_v<P, NoseHooverParams>)
                return ComponentId::NoseHoover;
            else if constexpr (std::is_same_v<P, ThermoParams>)
                return ComponentId::Thermo;
            else
                static_assert(detail::unsupported_component_v<P>);
        },
        params);
}

inline ComponentId
component_id(const MinimizeComponentParams& params) noexcept {
    return std::visit(
        [](const auto& p) -> ComponentId {
            using P = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<P, PairLJCutParams>)
                return ComponentId::PairLJCut;
            else if constexpr (std::is_same_v<P, NeighborListParams>)
                return ComponentId::NeighborList;
            else if constexpr (std::is_same_v<P, LBFGSParams>)
                return ComponentId::LBFGS;
            else
                static_assert(detail::unsupported_component_v<P>);
        },
        params);
}

struct MDBinding {
    ATV::md::Phase phase;
    int order;
};
inline bool operator==(const MDBinding& a, const MDBinding& b) {
    return a.phase == b.phase && a.order == b.order;
}

struct MinimizeBinding {
    ATV::minimizer::Phase phase;
    int order;
};
inline bool operator==(const MinimizeBinding& a, const MinimizeBinding& b) {
    return a.phase == b.phase && a.order == b.order;
}

struct MDComponentEntry {
    MDComponentParams params;
    std::optional<MDBinding> binding;
};
inline bool operator==(const MDComponentEntry& a, const MDComponentEntry& b) {
    return a.params == b.params && a.binding == b.binding;
}

struct MinimizeComponentEntry {
    MinimizeComponentParams params;
    std::optional<MinimizeBinding> binding;
};
inline bool operator==(const MinimizeComponentEntry& a,
                       const MinimizeComponentEntry& b) {
    return a.params == b.params && a.binding == b.binding;
}

struct MDTask {
    std::vector<MDComponentEntry> entries;
    std::int64_t n_steps;
};
inline bool operator==(const MDTask& a, const MDTask& b) {
    return a.entries == b.entries && a.n_steps == b.n_steps;
}

struct SinglePointTask {
    AM1EvaluatorParams evaluator;
};
inline bool operator==(const SinglePointTask& a, const SinglePointTask& b) {
    return a.evaluator == b.evaluator;
}

struct MinimizeTask {
    std::vector<MinimizeComponentEntry> entries;
    std::int64_t max_iter;
    ForceConvergedParams convergence;
};
inline bool operator==(const MinimizeTask& a, const MinimizeTask& b) {
    return a.entries == b.entries && a.max_iter == b.max_iter
           && a.convergence == b.convergence;
}

using TaskPayload = std::variant<MDTask, SinglePointTask, MinimizeTask>;

/// @brief Control-plane result; simulation state remains exclusively in Frame.
struct TaskResult {
    std::int64_t iterations = 0;
    bool converged = false;
    std::string stop_reason;
};
inline bool operator==(const TaskResult& a, const TaskResult& b) {
    return a.iterations == b.iterations && a.converged == b.converged
           && a.stop_reason == b.stop_reason;
}

class TaskSpec {
  public:
    /// v3 is a hard cut to task-specific typed payloads. No v2 decoder exists.
    static constexpr std::uint32_t kWireVersion = 3;

    explicit TaskSpec(MDTask task);
    explicit TaskSpec(SinglePointTask task);
    explicit TaskSpec(MinimizeTask task);

    TaskKind kind() const noexcept;
    const TaskPayload& payload() const noexcept { return payload_; }
    const MDTask& md() const { return std::get<MDTask>(payload_); }
    const SinglePointTask& single_point() const {
        return std::get<SinglePointTask>(payload_);
    }
    const MinimizeTask& minimize() const {
        return std::get<MinimizeTask>(payload_);
    }

  private:
    TaskPayload payload_;
};

} // namespace ATV::cpu::task
