/// @file facade/component_role.h
/// @brief Role — bitset of RoleFlags on every Component.
///
/// Loop coupling is **one** axis, named for the soft path:
///   - **`Deferred` set** → optional; may run off the hot path / async
///   - **`Deferred` unset** → main loop depends on this component this step
///
/// There is no named “role preset” (kRole*). Compose flags where needed:
///   Observer → FrameImmutable | Deferred
///   Recorder → FrameImmutable | Deferred | ExternalIO
///   default (0) → required, may mutate frame
#pragma once

#include <cstdint>

namespace ATV {

enum class RoleFlag : uint32_t {
    None = 0,
    /// Does not modify x / v / f / box / topology.
    FrameImmutable = 1u << 0,
    /// Optional / may be deferred or async. **Unset** ⇒ main-loop required.
    Deferred = 1u << 1,
    /// Writes outside the process (files, …).
    ExternalIO = 1u << 2,
};

using Role = uint32_t;

constexpr Role role_bit(RoleFlag f) noexcept { return static_cast<Role>(f); }

constexpr Role operator|(RoleFlag a, RoleFlag b) noexcept {
    return role_bit(a) | role_bit(b);
}
constexpr Role operator|(Role a, RoleFlag b) noexcept {
    return a | role_bit(b);
}
constexpr Role operator|(RoleFlag a, Role b) noexcept {
    return role_bit(a) | b;
}
constexpr Role& operator|=(Role& a, RoleFlag b) noexcept {
    a = a | b;
    return a;
}

constexpr bool has_role(Role set, RoleFlag flag) noexcept {
    return (set & role_bit(flag)) == role_bit(flag);
}

constexpr bool role_empty(Role r) noexcept { return r == 0; }

/// @brief True when the main loop depends on this component this step.
constexpr bool loop_required(Role set) noexcept {
    return !has_role(set, RoleFlag::Deferred);
}

} // namespace ATV
