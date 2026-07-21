/// @file torch/neighbor/neighbor_list.cuh
/// @brief NeighborList for the torch backend (CUDA path).
///
/// Wraps ATV::NeighborList. build() allocates once; update() rebuilds the
/// pair list from current positions.
///
/// Per the CLAUDE.md Frame purity rule the neighbor pair list is no longer
/// part of `torch::Frame`: `torch::NeighborList` OWNS it. The pair Tensors
/// live in the nested `PairList` struct below and are written by
/// build()/update(); downstream consumers (e.g. `torch::lj`) read them from
/// the `NeighborList` instance, not from the `Frame`.
///
/// @see ATV::NeighborList for the CUDA backend component
/// @see torch::Frame for the Tensor-based system state
/// @see torch::NeighborList::PairList for the relocated pair list
#pragma once

#ifdef ATV_HAS_TORCH

#include <memory>

#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/torch/frame/frame.cuh"

namespace ATV {

class NeighborList;

namespace torch {

/// @brief Neighbor list builder for torch backend (CUDA tensors).
///
/// Wraps ATV::NeighborList. The underlying component is created once in
/// build() and reused across update() calls. The resulting pair list is
/// owned by this object (see `PairList`), not by the `torch::Frame`.
///
/// Usage:
/// @code{.cpp}
/// torch::NeighborList nl(cutoff);
/// nl.build(frame);    // allocate once
/// nl.update(frame);   // rebuild pair list (call after positions change)
/// // nl.pairs().pair_i etc. are now populated Tensors
/// @endcode
class NeighborList {
  public:
    /// @brief Neighbor pair list owned by `torch::NeighborList`.
    ///
    /// All Tensors are CUDA-resident, contiguous, self-contained (cloned)
    /// and carry no autograd grad. Empty (undefined Tensors, n_pairs == 0)
    /// until the first build()/update() call. Mirrors the field set of
    /// `cuda::PairList` in the torch device representation.
    struct PairList {
        Tensor pair_i, pair_j; ///< Pair atom indices [n_pairs], kIntDtype.
        Tensor pair_dx, pair_dy,
            pair_dz;      ///< Pair displacements (j - i) [n_pairs, Angstrom].
        Tensor pair_dist; ///< Pair distances [n_pairs, Angstrom].
        int n_pairs_ = 0; ///< Host-cached pair count.

        /// @brief Number of pairs in the pair list.
        /// @return Pair count.
        int n_pairs() const { return n_pairs_; }
    };

    /// @brief Construct with cutoff and optional skin width.
    /// @param cutoff Interaction cutoff [Angstrom].
    /// @param skin   Verlet skin width [Angstrom]. Negative = auto (0.3 * cutoff).
    explicit NeighborList(double cutoff, double skin = -1.0);
    ~NeighborList();

    NeighborList(NeighborList&&) noexcept;
    NeighborList& operator=(NeighborList&&) noexcept;

    /// @brief Allocate internal resources from frame dimensions.
    /// @param frame torch::Frame with positions, box, and metadata set.
    void build(Frame& frame);

    /// @brief Rebuild pair list from current positions.
    ///
    /// Writes the result into this object's own `PairList` (see `pairs()`).
    ///
    /// @param frame torch::Frame with up-to-date positions.
    void update(Frame& frame);

    /// @brief Access the owned pair list (populated after build()/update()).
    /// @return Const reference to the pair Tensors and pair count.
    const PairList& pairs() const { return pairs_; }

    /// @brief Number of pairs in the owned pair list.
    /// @return Pair count (0 before the first update()).
    int n_pairs() const { return pairs_.n_pairs_; }

  private:
    double cutoff_;
    double skin_;
    std::unique_ptr<ATV::NeighborList> nl_;
    cuda::Context ctx_;
    cuda::Frame cframe_;
    PairList pairs_; ///< Pair list owned by this NeighborList.
    bool built_ = false;
};

/// @brief Convenience: construct + build + update in one call.
///
/// @param frame   torch::Frame with positions and box.
/// @param cutoff  Interaction cutoff [Angstrom].
/// @param skin    Verlet skin width [Angstrom]. Negative = auto (0.3 * cutoff).
/// @return A built NeighborList whose `pairs()` are populated for `frame`.
NeighborList neighbor_list(Frame& frame, double cutoff, double skin = -1.0);

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH
