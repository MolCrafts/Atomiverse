/// @file src/cpu/communication/swap_list.cpp
/// @brief `SwapList` implementation: swap expansion, band scan, ghost-slot
/// prefix sum. Purely local — no MPI here by construction.

#include "atomiverse/cpu/communication/swap_list.h"

#include <cmath>
#include <stdexcept>
#include <string>

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

} // namespace

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
    int need[3];
    for (int d = 0; d < 3; ++d) {
        // Undecomposed axis: the Cartesian partner is this rank itself, the
        // swap is skipped, no ghost is ever produced, and periodicity is the
        // pair kernel's min-image. A shell wider than the whole axis is legal
        // here and is refused only on decomposed axes below — mirroring the
        // `if (dims[d] <= 1) continue` short circuit of the outgoing guard.
        if (dd.dims[d] <= 1) {
            need[d] = 1;
            continue;
        }

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

        // ceil(r_halo / sub) == ceil(r_halo * dims_d / L_d) for the uniform
        // brick, and stays exact under a non-uniform partition. `> 1` is
        // algebraically `r_halo > sub`, the outgoing guard's test, so a shell
        // exactly as wide as the subdomain remains a single hop.
        need[d] = static_cast<int>(std::ceil(halo_cutoff / sub));
        if (need[d] < 1)
            need[d] = 1; // a zero-width shell is still one (empty) swap pair

        if (need[d] > 1) {
            throw std::runtime_error(
                "SwapList::setup: the requested ghost shell ("
                + std::to_string(halo_cutoff)
                + " A) is wider than this rank's subdomain along axis "
                + std::to_string(d) + " (" + std::to_string(sub) + " A of a "
                + std::to_string(box.length(d)) + " A box split over "
                + std::to_string(dd.dims[d])
                + " ranks), so filling it would "
                  "need atoms "
                + std::to_string(need[d])
                + " hops away. The halo exchanges with immediate neighbour "
                  "ranks only, so a deeper shell cannot be filled and the "
                  "ghost region would be silently truncated. Use a larger "
                  "box, fewer ranks along that axis, or a smaller cutoff. (A "
                  "many-body style such as EAM asks for a ghost-of-ghost "
                  "shell of 2*cutoff.)");
        }
    }

    for (int d = 0; d < 3; ++d)
        maxneed_[d] = need[d];

    // Every axis derives `need == 1` here, so `nswap = 2 * sum_d maxneed_d`
    // is exactly the low/high pair per axis. `resize` keeps the surviving
    // swaps' send-list heap capacity across the rebuild.
    const int nswap = 2 * (maxneed_[0] + maxneed_[1] + maxneed_[2]);
    swaps_.resize(static_cast<std::size_t>(nswap));

    std::size_t iswap = 0;
    for (int d = 0; d < 3; ++d) {
        for (int face_side = 0; face_side < 2; ++face_side) { // 0 = lo, 1 = hi
            Swap& swap = swaps_[iswap++];
            const bool hi = (face_side == 1);
            const int neighbor =
                dd.face_neighbors[static_cast<std::size_t>(2 * d + face_side)];

            swap.dim = d;
            swap.sign = hi ? +1 : -1;
            swap.sendproc = neighbor;
            swap.recvproc = neighbor;
            swap.sendself = (neighbor == my_rank); // == (dims[d] <= 1)

            swap.pbc[0] = 0;
            swap.pbc[1] = 0;
            swap.pbc[2] = 0;
            if (!swap.sendself) {
                if (hi && dd.coords[d] == dd.dims[d] - 1)
                    swap.pbc[d] = -1; // partner wraps to the low side
                else if (!hi && dd.coords[d] == 0)
                    swap.pbc[d] = +1; // partner sits at the far side
            }

            if (hi) {
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

int SwapList::maxneed(int d) const {
    if (d < 0 || d >= 3) {
        throw std::runtime_error("SwapList::maxneed: axis " + std::to_string(d)
                                 + " is outside [0, 3).");
    }
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
