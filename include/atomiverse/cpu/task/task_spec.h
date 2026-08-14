/// @file cpu/task/task_spec.h
/// @brief Closed, versioned, typed sum of launcher-dispatchable CPU tasks.
///
/// TaskSpec is an in-memory/wire boundary, not a universal component IR. Each
/// task alternative owns its own phase enum, component allow-list, and budget:
/// MD cannot carry a minimizer phase and Minimize cannot carry an MD phase.
/// Historic wire layouts are rejected; there is no compatibility decoder.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "atomiverse/core/stage.h"

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

/// @brief Mirrors `cpu::md::FixLangevin(T_start, T_stop, damp, dt, seed)`.
struct FixLangevinParams {
    double T_start;
    double T_stop;
    double damp;
    double dt;
    std::uint32_t seed;
};
inline bool operator==(const FixLangevinParams& a, const FixLangevinParams& b) {
    return a.T_start == b.T_start && a.T_stop == b.T_stop && a.damp == b.damp
           && a.dt == b.dt && a.seed == b.seed;
}

/// @brief Mirrors `cpu::md::FixCSVR(T_start, tau, dt, seed)`.
struct FixCsvrParams {
    double T_start;
    double tau;
    double dt;
    std::uint32_t seed;
};
inline bool operator==(const FixCsvrParams& a, const FixCsvrParams& b) {
    return a.T_start == b.T_start && a.tau == b.tau && a.dt == b.dt
           && a.seed == b.seed;
}

/// @brief Wire mirror of the ten-argument `cpu::md::NoseHooverNPT`
///        constructor, argument for argument.
///
/// `NoseHooverNPT` is the Martyna-Tobias-Klein (MTK) integrator for the NPT
/// ensemble — constant particle number N, pressure P and temperature T — so
/// it propagates the simulation cell as well as the atoms. Field order here
/// is exactly the constructor order (`T_target, tau_T, P_target, tau_P, dt,
/// drag, mtchain, mpchain, couple_axis, T_stop`), and `task_codec` writes
/// the fields in that same order — so this declaration doubles as the byte
/// layout. Changing either side is a `kWireVersion` bump.
///
/// Not mirrored — a launcher-dispatched NPT run gets the C++ defaults for
/// `NoseHooverNPT::set_couple_xy` (lockstep x/y cell edges) and
/// `set_surface_tension` (constant-surface-tension target). The ramp span
/// `set_run_span` is absent for a different reason: it is derived, not
/// chosen, and `TaskMaterializer` installs `[0, n_steps]` from the owning
/// `MDTask`.
struct NoseHooverNptParams {
    /// Thermostat setpoint, and the start value of the optional ramp [K].
    double T_target;
    /// Thermostat coupling time [fs] (LAMMPS `Tdamp`) — the timescale on
    /// which the thermostat pulls the kinetic temperature to `T_target`.
    double tau_T;
    /// Barostat setpoint [eV/Å³], LAMMPS sign (positive ⇔ compressive).
    /// 1 bar ≈ 6.2415e-7 eV/Å³.
    double P_target;
    /// Barostat coupling time [fs] (LAMMPS `Pdamp`).
    double tau_P;
    /// Timestep [fs].
    double dt;
    /// Empirical extra damping of the Nose-Hoover chain momenta (LAMMPS
    /// `drag`); dimensionless, 0 disables it.
    double drag;
    /// Length of the atom thermostat's Nose-Hoover chain — a chain of
    /// thermostat variables, each one thermostatting the previous
    /// [count; LAMMPS `mtchain`].
    int mtchain;
    /// Length of the barostat's own Nose-Hoover chain
    /// [count; LAMMPS `mpchain`].
    int mpchain;
    /// Per-axis barostat enable `{x, y, z}` (LAMMPS `p_flag`). A `false`
    /// axis keeps its cell edge rigid, i.e. that direction stays NVT.
    std::array<bool, 3> couple_axis;
    /// End temperature of the linear setpoint ramp [K]. A negative value
    /// folds to `T_target` in the constructor, i.e. no ramp.
    double T_stop;
};
inline bool operator==(const NoseHooverNptParams& a,
                       const NoseHooverNptParams& b) {
    return a.T_target == b.T_target && a.tau_T == b.tau_T
           && a.P_target == b.P_target && a.tau_P == b.tau_P && a.dt == b.dt
           && a.drag == b.drag && a.mtchain == b.mtchain
           && a.mpchain == b.mpchain && a.couple_axis == b.couple_axis
           && a.T_stop == b.T_stop;
}

/// @brief Wire mirror of the five leading `cpu::md::ParrinelloRahmanNPT`
///        constructor arguments.
///
/// `ParrinelloRahmanNPT` is the same MTK NPT scheme as `NoseHooverNPT` but
/// with a full cell matrix: all six independent strain-rate components
/// (xx, yy, zz, xy, xz, yz) evolve, so the cell can shear, not just breathe.
///
/// Watch the argument order — both setpoints first, then both coupling times
/// (`T_target, P_target, tau_T, tau_P, dt`) — where `NoseHooverNptParams`
/// pairs each setpoint with its own coupling time. The codec writes the
/// fields in this declaration order.
///
/// Not mirrored (C++ defaults apply on the launcher path): the optional
/// sixth constructor argument `W_cell` (barostat mass [eV·fs²]) and the
/// setters `set_drag` / `set_mtchain` / `set_mpchain`.
struct ParrinelloRahmanNptParams {
    /// Thermostat setpoint [K].
    double T_target;
    /// Target hydrostatic pressure [eV/Å³], LAMMPS sign (positive ⇔
    /// compressive). 1 bar ≈ 6.2415e-7 eV/Å³.
    double P_target;
    /// Thermostat coupling time [fs] (LAMMPS `Tdamp`).
    double tau_T;
    /// Barostat coupling time [fs] (LAMMPS `Pdamp`).
    double tau_P;
    /// Timestep [fs].
    double dt;
};
inline bool operator==(const ParrinelloRahmanNptParams& a,
                       const ParrinelloRahmanNptParams& b) {
    return a.T_target == b.T_target && a.P_target == b.P_target
           && a.tau_T == b.tau_T && a.tau_P == b.tau_P && a.dt == b.dt;
}

/// @brief Wire mirror of `cpu::md::Barostat(P_target, tau_P, dt,
///        bulk_modulus)` — the Berendsen barostat.
///
/// A barostat drives the cell volume towards a target pressure; the
/// Berendsen variant does it by rescaling the box and every atom position by
/// one scalar factor per step. Unlike the two NPT integrators above, it
/// rides *alongside* an integrator and a thermostat instead of replacing
/// them, so it is materialized through `MD::set_barostat`.
///
/// No field is defaulted here: every wire payload states `bulk_modulus`
/// explicitly. The C++ default of 1.0 is supplied by whoever builds this
/// struct (e.g. the Python binding), not by the wire layer.
struct BerendsenBarostatParams {
    /// Target pressure [eV/Å³], LAMMPS sign (positive ⇔ compressive).
    /// 1 bar ≈ 6.2415e-7 eV/Å³.
    double P_target;
    /// Barostat coupling time [fs] (LAMMPS `Pdamp`).
    double tau_P;
    /// Timestep [fs].
    double dt;
    /// Tunable modulus β [eV/Å³] (LAMMPS `modulus`). It divides the
    /// per-step rescale, so a larger β means a gentler box change and
    /// slower pressure relaxation. The C++ default is 1.0 — deliberately
    /// not LAMMPS's 10.0; the reasoning is in `cpu/dynamics/barostat.h`.
    double bulk_modulus;
};
inline bool operator==(const BerendsenBarostatParams& a,
                       const BerendsenBarostatParams& b) {
    return a.P_target == b.P_target && a.tau_P == b.tau_P && a.dt == b.dt
           && a.bulk_modulus == b.bulk_modulus;
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
///
/// The value, not the position, is what crosses the wire, so this list is
/// append-only: an ID is never reused or renumbered. `task_codec` decodes it
/// with an exhaustive switch and throws on an unknown value.
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
    FixLangevin = 11,
    FixCSVR = 12,
    NoseHooverNPT = 13,
    ParrinelloRahmanNPT = 14,
    BerendsenBarostat = 15,
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
    case ComponentId::FixLangevin:
        return "FixLangevin";
    case ComponentId::FixCSVR:
        return "FixCSVR";
    case ComponentId::NoseHooverNPT:
        return "NoseHooverNPT";
    case ComponentId::ParrinelloRahmanNPT:
        return "ParrinelloRahmanNPT";
    case ComponentId::BerendsenBarostat:
        return "BerendsenBarostat";
    }
    return "unknown";
}

using MDComponentParams =
    std::variant<PairLJCutParams, NeighborListParams, BondHarmonicParams,
                 AngleHarmonicParams, DihedralMultiHarmonicParams,
                 VelocityVerletParams, InitVelocityParams, NoseHooverParams,
                 ThermoParams, FixLangevinParams, FixCsvrParams,
                 NoseHooverNptParams, ParrinelloRahmanNptParams,
                 BerendsenBarostatParams>;

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
            else if constexpr (std::is_same_v<P, FixLangevinParams>)
                return ComponentId::FixLangevin;
            else if constexpr (std::is_same_v<P, FixCsvrParams>)
                return ComponentId::FixCSVR;
            else if constexpr (std::is_same_v<P, NoseHooverNptParams>)
                return ComponentId::NoseHooverNPT;
            else if constexpr (std::is_same_v<P, ParrinelloRahmanNptParams>)
                return ComponentId::ParrinelloRahmanNPT;
            else if constexpr (std::is_same_v<P, BerendsenBarostatParams>)
                return ComponentId::BerendsenBarostat;
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

/// @brief One launcher-dispatchable task: a `TaskKind` tag plus the typed
///        payload that materializes it.
class TaskSpec {
  public:
    /// @brief Version tag written at the head of every encoded TaskSpec;
    ///        `task_codec` rejects a payload whose tag differs.
    ///
    /// v3 was a hard cut to task-specific typed payloads; v4 is a hard cut to
    /// the grown `NoseHooverNptParams` byte layout (the chain-length,
    /// per-axis and ramp fields). Adding the `ParrinelloRahmanNPT` and
    /// `BerendsenBarostat` component IDs did not by itself force a bump — a
    /// new ID only ever appears in payloads that use it — but both land
    /// under v4. No v2 or v3 decoder exists: an older payload is an error,
    /// never a best-effort read.
    static constexpr std::uint32_t kWireVersion = 4;

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
