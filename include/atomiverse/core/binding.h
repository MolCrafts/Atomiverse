/// @file core/binding.h
/// @brief DefaultBinding -- shared between CPU and CUDA component bases. CUDA-free header.
#pragma once

namespace ATV {

/// @brief Workflow discriminant for the phase index carried by a DefaultBinding.
///
/// Every driver category owns its own `Phase` enum (facade/stage.h) and the
/// numeric values overlap (all start at 0), so a raw `int phase` cannot be
/// told apart across workflows. `PhaseKind` records which workflow's enum a
/// binding was built from, letting `Driver::add()` reject a component whose
/// default bindings target a different workflow at registration time.
enum class PhaseKind { MD, Minimizer, Evaluate, Vib, Sampling, Path, SCF };

/// @brief Human-readable workflow name for a PhaseKind.
/// @param k Workflow discriminant.
/// @return Static string: "MD", "Minimizer", "Evaluate", "Vib", "Sampling",
///         "Path" or "SCF".
constexpr const char* phase_kind_name(PhaseKind k) {
    switch (k) {
    case PhaseKind::MD:
        return "MD";
    case PhaseKind::Minimizer:
        return "Minimizer";
    case PhaseKind::Evaluate:
        return "Evaluate";
    case PhaseKind::Vib:
        return "Vib";
    case PhaseKind::Sampling:
        return "Sampling";
    case PhaseKind::Path:
        return "Path";
    case PhaseKind::SCF:
        return "SCF";
    }
    return "unknown";
}

/// @brief Default stage binding declared by a component.
///
/// Construct via `ATV::make_binding(phase, order)` (facade/stage.h) — it
/// derives `kind` from the enum type so the discriminant can never disagree
/// with the phase index. `kind` is deliberately the first field and carries
/// no default member initializer: any legacy two-field `{phase, order}`
/// aggregate initializer fails to compile, which is the migration
/// completeness gate.
struct DefaultBinding {
    PhaseKind kind; ///< Workflow whose Phase enum `phase` indexes into.
    int phase;      ///< Target phase index.
    int order;      ///< Ordering within phase (lower = earlier).
};

} // namespace ATV
