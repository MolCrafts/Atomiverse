/// @file src/cpu/communication/swap_list.cpp
/// @brief `SwapList` implementation: multi-hop derivation, swap expansion,
/// band scan, ghost-slot prefix sum. Purely local — no MPI here by
/// construction, so the per-rank hop counts are what this file produces and
/// the `MPI_Allreduce(MAX)` that makes them rank-uniform under a non-uniform
/// partition belongs to the transport caller (`HaloSend`).

#include "atomiverse/cpu/communication/swap_list.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace comm {

namespace {

/// Sentinel for the open side of a slab band [Angstrom]: larger than any
/// physical box by many orders of magnitude, so the comparison against it is
/// always true without a branch on `sign`.
constexpr double kBigSlab = 1.0e30;

/// Relative slack when a non-uniform slab chain is compared against the shell
/// width. The chain accumulates `(cuts[p+1] - cuts[p]) * L` term by term, so a
/// chain that covers the shell *exactly* can land a few ulps short; without
/// the slack that would silently cost one extra round. The uniform path needs
/// none — its closed form is a single division.
constexpr double kChainSlack = 1.0e-12;

/// @brief One send direction's slab chain on a non-uniform axis.
struct ChainWalk {
    /// Partners visited before the cumulative width covered the shell.
    int hops = 0;
    /// Whether it ever did, within the partners available.
    bool covered = false;
};

/// @brief Walk one send direction of a non-uniform axis, accumulating the
/// *partner* slab widths until they cover the shell.
///
/// @param cuts Fractional cut planes of the axis, `dims + 1` entries.
/// @param dims Rank count on the axis.
/// @param coords This rank's coordinate on the axis.
/// @param step `-1` toward decreasing coords, `+1` toward increasing.
/// @param axis_length Box edge along the axis [Angstrom].
/// @param shell Ghost shell width to cover [Angstrom].
/// @param reach Partners available in this direction (the whole ring minus
///        this rank when periodic, the ranks before the boundary otherwise).
/// @return Hops walked and whether the shell ended up covered.
ChainWalk walk_chain(const std::vector<double>& cuts, int dims, int coords,
                     int step, double axis_length, double shell, int reach) {
    const double target = shell - kChainSlack * shell;
    ChainWalk walk;
    double covered_width = 0.0;
    while (covered_width < target && walk.hops < reach) {
        ++walk.hops;
        const int partner = ((coords + step * walk.hops) % dims + dims) % dims;
        covered_width += (cuts[static_cast<std::size_t>(partner) + 1]
                          - cuts[static_cast<std::size_t>(partner)])
                         * axis_length;
    }
    walk.covered = (covered_width >= target);
    return walk;
}

/// @brief The minimum-image refusal shared by both derivation paths.
///
/// A periodic axis can grow its ghost shell over every slab *except* this
/// rank's own; past that an atom would see its own periodic image.
///
/// @param d Axis index.
/// @param dd Decomposition the refusal is reported against.
/// @param axis_length Box edge along `d` [Angstrom].
/// @param shell Requested ghost shell width [Angstrom].
/// @param hops Hops the derivation asked for (`> dims[d] - 1`).
/// @return Never; always throws.
[[noreturn]] void refuse_minimum_image(int d,
                                       const md::mpi::DomainDecomposition& dd,
                                       double axis_length, double shell,
                                       int hops) {
    const double self_width = dd.subhi[d] - dd.sublo[d];
    throw std::runtime_error(
        "SwapList: the requested ghost shell (" + std::to_string(shell)
        + " A) violates the minimum-image convention on periodic axis "
        + std::to_string(d) + ": covering it would take " + std::to_string(hops)
        + " hops of the " + std::to_string(dd.dims[d] - 1)
        + " available, i.e. the shell would wrap past every other slab and "
          "reach this rank's own periodic image. The bound is shell <= L - "
          "w_self = "
        + std::to_string(axis_length) + " - " + std::to_string(self_width)
        + " = " + std::to_string(axis_length - self_width)
        + " A. Enlarge the box or shorten the cutoff. (The shell is "
          "max(pair cutoff, bonded extension); a many-body style such as EAM "
          "asks for a ghost-of-ghost shell of 2*cutoff.)");
}

/// @brief Range-check an axis index for the per-axis accessors.
/// @param what Accessor name, for the message.
/// @param d Axis index that must lie in `[0, 3)`.
void check_axis(const char* what, int d) {
    if (d < 0 || d >= 3) {
        throw std::runtime_error(std::string("SwapList::") + what + ": axis "
                                 + std::to_string(d) + " is outside [0, 3).");
    }
}

} // namespace

SwapList::HopNeed SwapList::derive_hops(const md::mpi::DomainDecomposition& dd,
                                        const Box& box, double shell_width) {
    if (!(shell_width >= 0.0)) {
        throw std::runtime_error(
            "SwapList::derive_hops: the ghost shell must be a non-negative "
            "width in Angstrom, got "
            + std::to_string(shell_width)
            + ". It is max(pair cutoff, bonded extension) and is derived by "
              "HaloSend; a negative or NaN value means the caller never "
              "computed it.");
    }

    HopNeed hops;
    for (int d = 0; d < 3; ++d) {
        // Undecomposed axis: the Cartesian partner is this rank itself, no
        // ghost is ever produced, and periodicity is the pair kernel's
        // min-image. Zero hops, and exempt from every width check — a shell
        // wider than such an axis is legal (EAM-style 2*cutoff on a thin
        // axis).
        if (dd.dims[d] <= 1)
            continue;

        const double axis_length = box.length(d);
        if (!(axis_length > 0.0)) {
            throw std::runtime_error(
                "SwapList::derive_hops: box edge along axis "
                + std::to_string(d) + " is " + std::to_string(axis_length)
                + " A; the slab widths the hop count is measured against are "
                  "undefined. Set the box before deriving the swap table.");
        }

        const std::vector<double>& cuts = dd.cuts[static_cast<std::size_t>(d)];
        if (!cuts.empty() && static_cast<int>(cuts.size()) != dd.dims[d] + 1) {
            throw std::runtime_error(
                "SwapList::derive_hops: the non-uniform cut table of axis "
                + std::to_string(d) + " has " + std::to_string(cuts.size())
                + " entries but the axis carries " + std::to_string(dd.dims[d])
                + " ranks, so it must have " + std::to_string(dd.dims[d] + 1)
                + " (fractions from 0 to 1). Leave it empty for a uniform "
                  "partition.");
        }

        // Partners available in each direction. A periodic axis offers the
        // whole ring minus this rank; a non-periodic one ends at the global
        // boundary, where the chain simply stops — there is no partner left
        // and no periodic image to collide with, so no refusal either.
        const bool periodic = dd.periods[d] != 0;
        const int reach_minus = periodic ? dd.dims[d] - 1 : dd.coords[d];
        const int reach_plus =
            periodic ? dd.dims[d] - 1 : dd.dims[d] - 1 - dd.coords[d];

        if (cuts.empty()) {
            // Uniform brick: every partner slab is L_d / dims_d wide, so the
            // chain has a closed form and both directions agree. `ceil` keeps
            // a shell exactly one slab wide at a single hop.
            const int needed = static_cast<int>(std::ceil(
                shell_width * static_cast<double>(dd.dims[d]) / axis_length));
            if (periodic && needed > dd.dims[d] - 1)
                refuse_minimum_image(d, dd, axis_length, shell_width, needed);
            hops.minus[d] = std::min(needed, reach_minus);
            hops.plus[d] = std::min(needed, reach_plus);
            continue;
        }

        // Non-uniform (RCB / Rebalance): it is the *partner* width chain in
        // each send direction that decides how many rounds the shell takes,
        // not this rank's own width.
        const ChainWalk minus =
            walk_chain(cuts, dd.dims[d], dd.coords[d], -1, axis_length,
                       shell_width, reach_minus);
        const ChainWalk plus = walk_chain(cuts, dd.dims[d], dd.coords[d], +1,
                                          axis_length, shell_width, reach_plus);
        if (periodic && !minus.covered)
            refuse_minimum_image(d, dd, axis_length, shell_width,
                                 minus.hops + 1);
        if (periodic && !plus.covered)
            refuse_minimum_image(d, dd, axis_length, shell_width,
                                 plus.hops + 1);
        hops.minus[d] = minus.hops;
        hops.plus[d] = plus.hops;
    }
    return hops;
}

void SwapList::setup(const md::mpi::DomainDecomposition& dd, const Box& box,
                     double halo_cutoff, int my_rank) {
    if (!(halo_cutoff >= 0.0)) {
        throw std::runtime_error(
            "SwapList::setup: halo cutoff must be a non-negative width in "
            "Angstrom, got "
            + std::to_string(halo_cutoff)
            + ". It is max(pair cutoff, bonded extension) and is derived by "
              "HaloSend; a negative or NaN value means the caller never "
              "computed it.");
    }

    // Derive every hop count first: a refusal must leave the previous table
    // intact rather than half-rewritten.
    const HopNeed hops = derive_hops(dd, box, halo_cutoff);

    int rounds[3];
    int pairs[3];
    for (int d = 0; d < 3; ++d) {
        if (dd.dims[d] > 1) {
            const double sub = dd.subhi[d] - dd.sublo[d];
            if (!(sub > 0.0)) {
                throw std::runtime_error(
                    "SwapList::setup: this rank's subdomain along axis "
                    + std::to_string(d) + " has non-positive width ("
                    + std::to_string(sub)
                    + " A), so the ghost shell has nothing to grow from. Call "
                      "DomainDecomposition::compute_sublo_subhi(box) with the "
                      "current box before setting up the swap table.");
            }
        }

        // This rank's own round count on the axis is the deeper of its two
        // send directions; making it uniform across ranks (the spec's
        // MPI_Allreduce(MAX) under a non-uniform partition) is the transport
        // caller's job — SwapList performs no MPI.
        rounds[d] = std::max(hops.minus[d], hops.plus[d]);
        // An axis needing no hop at all (undecomposed, or a zero-width shell)
        // still keeps one `sendself` placeholder pair, so every axis owns a
        // contiguous, non-empty run of swaps.
        pairs[d] = std::max(rounds[d], 1);
    }

    for (int d = 0; d < 3; ++d) {
        hops_[d] = rounds[d];
        maxneed_[d] = pairs[d];
    }
    shell_width_ = halo_cutoff;

    // `nswap = 2 * sum_d maxneed_d`, the axis runs staying contiguous and in
    // round order. `resize` keeps the surviving swaps' send-list heap
    // capacity across the rebuild.
    const int nswap_total = 2 * (maxneed_[0] + maxneed_[1] + maxneed_[2]);
    swaps_.resize(static_cast<std::size_t>(nswap_total));

    std::size_t iswap = 0;
    for (int d = 0; d < 3; ++d) {
        for (int round = 1; round <= maxneed_[d]; ++round) {
            for (int face_side = 0; face_side < 2; ++face_side) { // lo, hi
                Swap& swap = swaps_[iswap++];
                const bool hi = (face_side == 1);
                const int neighbor = dd.face_neighbors[static_cast<std::size_t>(
                    2 * d + face_side)];

                swap.dim = d;
                swap.sign = hi ? +1 : -1;
                swap.round = round;
                // Every round of the chain goes through the SAME face
                // neighbour: a deeper shell is filled by forwarding what the
                // previous round received one rank further, never by a direct
                // message to a rank two hops away.
                swap.sendproc = neighbor;
                swap.recvproc = neighbor;
                swap.sendself = (neighbor == my_rank); // == (dims[d] <= 1)

                swap.pbc[0] = 0;
                swap.pbc[1] = 0;
                swap.pbc[2] = 0;
                if (!swap.sendself) {
                    // The image shift is a property of the *link*, not of the
                    // round: it applies exactly when this rank's message to
                    // that neighbour crosses the global periodic boundary,
                    // whether it carries owners or forwarded ghosts.
                    if (hi && dd.coords[d] == dd.dims[d] - 1)
                        swap.pbc[d] = -1; // partner wraps to the low side
                    else if (!hi && dd.coords[d] == 0)
                        swap.pbc[d] = +1; // partner sits at the far side
                }

                if (round > 1) {
                    // Forwarding round: the rows it ships already sit outside
                    // this rank's subdomain, so no band of this rank's own
                    // coordinates describes them. The open band takes whatever
                    // the caller's scan window offers.
                    swap.slablo = -kBigSlab;
                    swap.slabhi = +kBigSlab;
                } else if (hi) {
                    swap.slablo = dd.subhi[d] - halo_cutoff;
                    swap.slabhi = +kBigSlab;
                } else {
                    swap.slablo = -kBigSlab;
                    swap.slabhi = dd.sublo[d] + halo_cutoff;
                }

                swap.sendlist.clear();
                swap.nrecv = 0;
                swap.firstrecv = 0;
            }
        }
    }
}

void SwapList::rebuild_sendlist(int iswap, const Frame& frame,
                                int sources_end) {
    if (iswap < 0 || static_cast<std::size_t>(iswap) >= swaps_.size()) {
        throw std::runtime_error(
            "SwapList::rebuild_sendlist: swap index " + std::to_string(iswap)
            + " is outside the table of " + std::to_string(swaps_.size())
            + " swaps. Call setup() first and iterate [0, size()).");
    }
    if (sources_end < 0 || sources_end > frame.capacity()) {
        throw std::runtime_error(
            "SwapList::rebuild_sendlist: scan bound "
            + std::to_string(sources_end) + " is outside the frame's "
            + std::to_string(frame.capacity())
            + " allocated rows. Pass n_local for the first axis stage and "
              "n_local + ghosts received so far for later ones.");
    }

    Swap& swap = swaps_[static_cast<std::size_t>(iswap)];
    const StorageReal* __restrict q = (swap.dim == 0)   ? frame.span(kAtom_x)
                                      : (swap.dim == 1) ? frame.span(kAtom_y)
                                                        : frame.span(kAtom_z);

    // Strict on both edges: an atom exactly on the band edge is not shipped.
    const double lo = swap.slablo;
    const double hi = swap.slabhi;
    swap.sendlist.clear();
    for (int i = 0; i < sources_end; ++i) {
        const double qi = static_cast<double>(q[i]);
        if (lo < qi && qi < hi)
            swap.sendlist.push_back(i);
    }
}

int SwapList::commit_recv(int iswap, int nrecv, int next_slot) {
    if (iswap < 0 || static_cast<std::size_t>(iswap) >= swaps_.size()) {
        throw std::runtime_error(
            "SwapList::commit_recv: swap index " + std::to_string(iswap)
            + " is outside the table of " + std::to_string(swaps_.size())
            + " swaps. Call setup() first and iterate [0, size()).");
    }
    if (nrecv < 0 || next_slot < 0) {
        throw std::runtime_error(
            "SwapList::commit_recv: negative ghost count ("
            + std::to_string(nrecv) + ") or slot (" + std::to_string(next_slot)
            + "). Drive the prefix sum in swap order starting from n_local.");
    }

    Swap& swap = swaps_[static_cast<std::size_t>(iswap)];
    swap.nrecv = nrecv;
    swap.firstrecv = next_slot;
    return next_slot + nrecv;
}

int SwapList::total_recv() const {
    int total = 0;
    for (const Swap& swap : swaps_)
        total += swap.nrecv;
    return total;
}

int SwapList::max_hops(int d) const {
    check_axis("max_hops", d);
    return hops_[d];
}

int SwapList::nswap(int d) const {
    check_axis("nswap", d);
    return 2 * hops_[d];
}

int SwapList::maxneed(int d) const {
    check_axis("maxneed", d);
    return maxneed_[d];
}

void SwapList::invalidate() {
    for (Swap& swap : swaps_) {
        swap.sendlist.clear();
        swap.nrecv = 0;
        swap.firstrecv = 0;
    }
}

} // namespace comm
} // namespace cpu
} // namespace ATV
