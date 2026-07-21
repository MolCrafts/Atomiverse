/// @file cpu/component/halo_participant.h
/// @brief HaloParticipant — opt-in protocol for MPI ghost-exchange
///        participation. CUDA-free header.
///
/// Split out of `Component` (arch-refactor-02-halo): the halo contract
/// concerns exactly three parties — the `comm::HaloSend` / `comm::HaloRecv`
/// endpoints and any component that wants extra per-atom fields shipped with
/// the per-face messages. Every other component keeps a clean vtable.
///
/// Follows the `PairEvaluator` opt-in precedent: a component that
/// participates inherits this protocol *in addition to* `Component` and
/// overrides `Component::halo_participant()` to return `this`; the
/// driver's infrastructure scan probes that single typed accessor — no
/// `dynamic_cast`, no `name()` matching.
///
/// The field contract is declarative (LAMMPS `comm_forward`/`comm_reverse`
/// semantics, engine-side packing): a participant declares per-atom arrays
/// via `CommField{data, width}` and the halo engine packs them into its
/// existing per-face messages — one message per face regardless of how many
/// participants declare fields. Forward fields are owner→ghost refreshed
/// every halo step; reverse fields are ghost→owner accumulated exactly once
/// (Newton's-third-law partial sums), and their ghost rows are zeroed at
/// each forward pass alongside the intrinsic force columns.
///
/// Reference: Thompson et al., Comput. Phys. Commun. 271, 108171 (2022),
/// DOI: 10.1016/j.cpc.2021.108171 (LAMMPS forward/reverse ghost comm).
#pragma once

#include <vector>

#include "atomiverse/core/precision.h"

namespace ATV {
namespace cpu {

/// @brief Opt-in protocol for components that take part in the MPI halo.
class HaloParticipant {
  public:
    virtual ~HaloParticipant() = default;

    /// @brief Intrinsic exchange endpoint role.
    ///
    /// `send` — the forward endpoint (`comm::HaloSend`); `recv` — the
    /// reverse endpoint (`comm::HaloRecv`); `none` — a pure field
    /// participant that only declares extra per-atom payload.
    enum class Endpoint { none, send, recv };

    /// @brief This participant's endpoint role.
    /// @return `Endpoint::none` unless overridden by HaloSend / HaloRecv.
    virtual Endpoint halo_endpoint() const { return Endpoint::none; }

    /// @brief One declared per-atom communication field.
    ///
    /// `data` must point at a per-atom array with capacity for owned +
    /// ghost rows, and the pointer must stay **stable for the whole run**
    /// — the endpoints flatten descriptors once at first launch and cache
    /// them. Size to the worst-case `frame.capacity()` in `build()`
    /// (frame-capacity growth mid-run would dangle a cached pointer; a
    /// participant that cannot pre-size must not declare fields). Packed
    /// contiguously as `data[i * width + k]`, `k in [0, width)`.
    struct CommField {
        StorageReal* data = nullptr; ///< Per-atom array (owned+ghost capacity).
        int width = 1;               ///< Components per atom [dimensionless].
    };

    /// @brief Extra per-atom fields refreshed owner→ghost on every forward
    ///        halo pass. Default: none.
    /// @return Field descriptors; pointers must stay valid for the run.
    virtual std::vector<CommField> comm_forward_fields() const { return {}; }

    /// @brief Extra per-atom fields accumulated ghost→owner on the reverse
    ///        pass (exactly-once partial sums). Default: none.
    /// @return Field descriptors; pointers must stay valid for the run.
    virtual std::vector<CommField> comm_reverse_fields() const { return {}; }

    /// @brief Endpoint-only hook: the driver injects the list of field
    ///        participants at build time (after `install_defaults()`,
    ///        before component `build()`), so endpoints query field
    ///        descriptors lazily at each launch. Default no-op.
    /// @param participants Non-owning; valid for the driver's lifetime.
    virtual void
    bind_participants(const std::vector<const HaloParticipant*>& participants) {
        (void)participants;
    }
};

} // namespace cpu
} // namespace ATV
