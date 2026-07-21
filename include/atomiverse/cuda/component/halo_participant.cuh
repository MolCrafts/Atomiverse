/// @file cuda/component/halo_participant.cuh
/// @brief cuda::HaloParticipant — opt-in protocol for multi-GPU ghost
///        exchange participation. Deliberately CUDA-free header (pointer
///        declarations only), same discipline as component.cuh.
///
/// Member-for-member mirror of `cpu::HaloParticipant`
/// (cpu/component/halo_participant.h) — same names, same semantics — with
/// the field pointer in the CUDA per-atom storage tier (`real_t*`, device
/// memory). The multi-GPU halo gathers declared fields into its combined
/// P2P staging buffer; there is no in-tree CUDA field declarant yet, so the
/// consumption wiring lands with the first declarant (see
/// arch-refactor-02-halo Out of scope).
#pragma once

#include <vector>

#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

/// @brief Opt-in protocol for components that take part in the multi-GPU
///        halo. See `cpu::HaloParticipant` for the full contract.
class HaloParticipant {
  public:
    virtual ~HaloParticipant() = default;

    /// @brief Intrinsic exchange endpoint role (`none` = field-only).
    enum class Endpoint { none, send, recv };

    /// @brief This participant's endpoint role.
    /// @return `Endpoint::none` unless overridden by an exchange endpoint.
    virtual Endpoint halo_endpoint() const { return Endpoint::none; }

    /// @brief One declared per-atom DEVICE communication field.
    ///
    /// `data` must point at a device array with owned+ghost capacity,
    /// packed contiguously as `data[i * width + k]`.
    struct CommField {
        real_t* data =
            nullptr;   ///< Per-atom device array (owned+ghost capacity).
        int width = 1; ///< Components per atom [dimensionless].
    };

    /// @brief Extra per-atom fields refreshed owner→ghost each halo step.
    /// @return Field descriptors; pointers must stay valid for the run.
    virtual std::vector<CommField> comm_forward_fields() const { return {}; }

    /// @brief Extra per-atom fields accumulated ghost→owner (exactly once).
    /// @return Field descriptors; pointers must stay valid for the run.
    virtual std::vector<CommField> comm_reverse_fields() const { return {}; }

    /// @brief Endpoint-only hook: driver injects field participants at
    ///        build time. Default no-op.
    /// @param participants Non-owning; valid for the driver's lifetime.
    virtual void
    bind_participants(const std::vector<const HaloParticipant*>& participants) {
        (void)participants;
    }
};

} // namespace cuda
} // namespace ATV
