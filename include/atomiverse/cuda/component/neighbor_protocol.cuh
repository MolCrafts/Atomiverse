/// @file cuda/component/neighbor_protocol.cuh
/// @brief cuda::NeighborConsumer / cuda::NeighborProvider — opt-in
///        protocols for the neighbor-list contract. Deliberately
///        CUDA-free header, same discipline as component.cuh.
///
/// Member-for-member mirror of the CPU protocols
/// (cpu/component/neighbor_protocol.h) with the cutoff in `double`
/// (matching `cuda::Pair::pair_cutoff()`). Closes the historical CUDA
/// gap: the CUDA base never exposed a cutoff hint, so `cuda::Pair` now
/// reports its cutoff through the protocol like its CPU sibling.
#pragma once

namespace ATV {
namespace cuda {

/// @brief Consumer side of the neighbor-list contract. See
///        `cpu::NeighborConsumer` for the full semantics.
class NeighborConsumer {
  public:
    virtual ~NeighborConsumer() = default;

    /// @brief Interaction cutoff the neighbor list must cover [Å].
    /// @return Cutoff radius [Å]; `0` = no cutoff declared.
    virtual double neighbor_cutoff() const = 0;
};

/// @brief Supply side of the neighbor-list contract (pure tag; the
///        publication channel stays `Context::pair_list`).
class NeighborProvider {
  public:
    virtual ~NeighborProvider() = default;
};

} // namespace cuda
} // namespace ATV
