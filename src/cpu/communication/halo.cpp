/// @file src/cpu/communication/halo.cpp
/// @brief `Halo` implementation: multi-hop derivation, swap expansion,
/// band scan, ghost-slot prefix sum. Purely local — no MPI here by
/// construction. Rank-uniform round counts come from
/// `derive_uniform_rounds` (a local sweep over each axis's coordinates);
/// the transport caller (`CommBrick`) then expands them via the five-argument
/// `setup`.
///
/// Algorithm transplanted from LAMMPS `src/comm_brick.cpp` sendlist
/// (checkout `970d4ba`). Triclinic fractional slabs and periodic self copies
/// follow src/comm_brick.cpp at a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b.
/// Copyright (2003) Sandia Corporation. GPL-2.0-only.

#include "atomiverse/cpu/communication/halo.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
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

/// Sentinel for the open side of a fractional slab band: larger than any
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
/// @param reach Maximum hops including complete periodic rings.
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

/// @brief Reject a shell width that is not a usable length, for either
/// `setup` entry point.
/// @param halo_cutoff Ghost shell width [Angstrom] the caller passed.
void check_shell_width(double halo_cutoff) {
    if (!(halo_cutoff >= 0.0) || !std::isfinite(halo_cutoff)) {
        throw std::runtime_error(
            "Halo::setup: halo cutoff must be a non-negative width in "
            "Angstrom, got "
            + std::to_string(halo_cutoff)
            + ". It is max(pair cutoff, bonded extension) and is derived by "
              "CommBrick; a negative or NaN value means the caller never "
              "computed it.");
    }
}

/// @brief Range-check an axis index for the per-axis accessors.
/// @param what Accessor name, for the message.
/// @param d Axis index that must lie in `[0, 3)`.
void check_axis(const char* what, int d) {
    if (d < 0 || d >= 3) {
        throw std::runtime_error(std::string("Halo::") + what + ": axis "
                                 + std::to_string(d) + " is outside [0, 3).");
    }
}

} // namespace

Halo::HopNeed Halo::derive_hops(const md::mpi::DomainDecomposition& dd,
                                const Box& box, double shell_width) {
    return derive_hops(dd, box, shell_width, dd.coords);
}

Halo::HopNeed Halo::derive_hops(const md::mpi::DomainDecomposition& dd,
                                const Box& box, double shell_width,
                                const int (&coords)[3]) {
    if (!(shell_width >= 0.0) || !std::isfinite(shell_width)) {
        throw std::runtime_error(
            "Halo::derive_hops: the ghost shell must be a non-negative "
            "width in Angstrom, got "
            + std::to_string(shell_width)
            + ". It is max(pair cutoff, bonded extension) and is derived by "
              "CommBrick; a negative or NaN value means the caller never "
              "computed it.");
    }

    HopNeed hops;
    for (int d = 0; d < 3; ++d) {
        const double axis_length = box.face_height(d);
        if (!(axis_length > 0.0)) {
            throw std::runtime_error(
                "Halo::derive_hops: box edge along axis " + std::to_string(d)
                + " is " + std::to_string(axis_length)
                + " A; the slab widths the hop count is measured against are "
                  "undefined. Set the box before deriving the swap table.");
        }

        const std::vector<double>& cuts = dd.cuts[static_cast<std::size_t>(d)];
        if (!cuts.empty() && static_cast<int>(cuts.size()) != dd.dims[d] + 1) {
            throw std::runtime_error(
                "Halo::derive_hops: the non-uniform cut table of axis "
                + std::to_string(d) + " has " + std::to_string(cuts.size())
                + " entries but the axis carries " + std::to_string(dd.dims[d])
                + " ranks, so it must have " + std::to_string(dd.dims[d] + 1)
                + " (fractions from 0 to 1). Leave it empty for a uniform "
                  "partition.");
        }

        if (dd.dims[d] <= 0 || coords[d] < 0 || coords[d] >= dd.dims[d])
            throw std::invalid_argument(
                "Halo: invalid process grid coordinate");
        if (!cuts.empty()) {
            if (cuts.front() != 0.0 || cuts.back() != 1.0)
                throw std::invalid_argument(
                    "Halo: cuts must span fractional [0,1]");
            for (std::size_t i = 1; i < cuts.size(); ++i)
                if (!std::isfinite(cuts[i]) || !(cuts[i] > cuts[i - 1]))
                    throw std::invalid_argument(
                        "Halo: cuts must be finite and increasing");
        }

        // LAMMPS periodic brick can traverse an entire ring and reach its own
        // images. A full ring covers one face height even for nonuniform cuts.
        const double layers =
            shell_width * static_cast<double>(dd.dims[d]) / axis_length;
        const double periodic_reach =
            (std::floor(shell_width / axis_length) + 1.0) * dd.dims[d];
        if (!std::isfinite(layers) || !std::isfinite(periodic_reach)
            || std::max(layers, periodic_reach)
                   >= static_cast<double>(std::numeric_limits<int>::max() / 6))
            throw std::length_error(
                "Halo: periodic shell exceeds addressable swap count");
        const int reach = static_cast<int>(periodic_reach);

        if (cuts.empty()) {
            // Uniform brick: every partner slab is L_d / dims_d wide, so the
            // chain has a closed form and both directions agree. Include the
            // next slab at an exact boundary, as LAMMPS maxneed does.
            const int needed = static_cast<int>(layers) + 1;
            hops.minus[d] = std::min(needed, reach);
            hops.plus[d] = std::min(needed, reach);
            continue;
        }

        // Non-uniform (RCB / Rebalance): it is the *partner* width chain in
        // each send direction that decides how many rounds the shell takes,
        // not this rank's own width.
        const ChainWalk minus = walk_chain(cuts, dd.dims[d], coords[d], -1,
                                           axis_length, shell_width, reach);
        const ChainWalk plus = walk_chain(cuts, dd.dims[d], coords[d], +1,
                                          axis_length, shell_width, reach);
        hops.minus[d] = minus.hops;
        hops.plus[d] = plus.hops;
    }
    return hops;
}

std::array<int, 3>
Halo::derive_uniform_rounds(const md::mpi::DomainDecomposition& dd,
                            const Box& box, double shell_width) {
    std::array<int, 3> rounds{0, 0, 0};
    // Coords override instead of copying the decomposition per scan point:
    // a DomainDecomposition carries the tag→local hash and the ghost-tag
    // log, so the old `scan = dd` deep-copied thousands of entries per
    // coordinate — measurable at every borders pass on bonded systems.
    const HopNeed local = derive_hops(dd, box, shell_width);
    int coords[3] = {dd.coords[0], dd.coords[1], dd.coords[2]};
    for (int d = 0; d < 3; ++d) {
        if (dd.cuts[d].empty()) {
            rounds[d] = local.minus[d];
            continue;
        }
        int axis_max = 0;
        const int saved = coords[d];
        for (int c = 0; c < dd.dims[d]; ++c) {
            coords[d] = c;
            const HopNeed hops = derive_hops(dd, box, shell_width, coords);
            axis_max =
                std::max(axis_max, std::max(hops.minus[d], hops.plus[d]));
        }
        coords[d] = saved;
        rounds[static_cast<std::size_t>(d)] = axis_max;
    }
    return rounds;
}

void Halo::setup(const md::mpi::DomainDecomposition& dd, const Box& box,
                 double halo_cutoff, int my_rank) {
    check_shell_width(halo_cutoff);

    // Derive every hop count first: a refusal must leave the previous table
    // intact rather than half-rewritten.
    const HopNeed hops = derive_hops(dd, box, halo_cutoff);

    int rounds[3];
    for (int d = 0; d < 3; ++d) {
        // This rank's own round count on the axis is the deeper of its two
        // send directions.
        rounds[d] = std::max(hops.minus[d], hops.plus[d]);
    }
    expand(dd, box, halo_cutoff, my_rank, rounds);
}

void Halo::setup(const md::mpi::DomainDecomposition& dd, const Box& box,
                 double halo_cutoff, int my_rank, const int rounds[3]) {
    check_shell_width(halo_cutoff);

    // Validate geometry and retain at least this rank's locally required reach.
    const HopNeed hops = derive_hops(dd, box, halo_cutoff);

    int reconciled[3];
    for (int d = 0; d < 3; ++d) {
        const int local = std::max(hops.minus[d], hops.plus[d]);
        const int imposed = (rounds[d] > 0) ? rounds[d] : 0;
        reconciled[d] = std::max(local, imposed);
    }
    expand(dd, box, halo_cutoff, my_rank, reconciled);
}

void Halo::setup_uniform(const md::mpi::DomainDecomposition& dd, const Box& box,
                         double halo_cutoff, int my_rank) {
    const std::array<int, 3> rounds =
        derive_uniform_rounds(dd, box, halo_cutoff);
    setup(dd, box, halo_cutoff, my_rank, rounds.data());
}

void Halo::expand(const md::mpi::DomainDecomposition& dd, const Box& box,
                  double halo_cutoff, int my_rank, const int rounds[3]) {
    int pairs[3];
    for (int d = 0; d < 3; ++d) {
        if (dd.dims[d] > 1) {
            const double sub = dd.subhi[d] - dd.sublo[d];
            if (!(sub > 0.0)) {
                throw std::runtime_error(
                    "Halo::setup: this rank's subdomain along axis "
                    + std::to_string(d) + " has non-positive width ("
                    + std::to_string(sub)
                    + " A), so the ghost shell has nothing to grow from. Call "
                      "DomainDecomposition::compute_sublo_subhi(box) with the "
                      "current box before setting up the swap table.");
            }
        }

        // An axis needing no hop at all (undecomposed, or a zero-width shell)
        // still keeps one `sendself` placeholder pair, so every axis owns a
        // contiguous, non-empty run of swaps.
        pairs[d] = std::max(rounds[d], 1);
    }

    // Imposed round counts also enter here; validate before mutating an
    // existing table or performing signed-int arithmetic on its size.
    const std::int64_t nswap_wide =
        2 * (static_cast<std::int64_t>(pairs[0]) + pairs[1] + pairs[2]);
    if (nswap_wide > std::numeric_limits<int>::max())
        throw std::length_error(
            "Halo::setup: imposed rounds exceed addressable swap count");
    const int nswap_total = static_cast<int>(nswap_wide);
    for (int d = 0; d < 3; ++d) {
        hops_[d] = rounds[d];
        maxneed_[d] = pairs[d];
    }
    shell_width_ = halo_cutoff;

    // `nswap = 2 * sum_d maxneed_d`, the axis runs staying contiguous and in
    // round order. `resize` keeps the surviving swaps' send-list heap
    // capacity across the rebuild.
    swaps_.resize(static_cast<std::size_t>(nswap_total));

    std::size_t iswap = 0;
    for (int d = 0; d < 3; ++d) {
        const double slab_cutoff = halo_cutoff / box.face_height(d);
        for (int round = 1; round <= maxneed_[d]; ++round) {
            for (int face_side = 0; face_side < 2; ++face_side) { // lo, hi
                Swap& swap = swaps_[iswap++];
                const bool hi = (face_side == 1);
                const int send_nbr = dd.face_neighbors[static_cast<std::size_t>(
                    2 * d + face_side)];
                const int recv_nbr = dd.face_neighbors[static_cast<std::size_t>(
                    2 * d + 1 - face_side)];

                swap.dim = d;
                swap.sign = hi ? +1 : -1;
                swap.round = round;
                // Every round of the chain goes through the SAME face
                // neighbours: a deeper shell is filled by forwarding what the
                // previous round received one rank further, never by a direct
                // message to a rank two hops away.
                //
                // LAMMPS `CommBrick::setup`: send across THIS face, hear back
                // across the opposite one (`procneigh[dim][0]` /
                // `procneigh[dim][1]`). Both ranks run the same rank-uniform
                // table in the same order, so swap `s` is answered by the
                // partner's swap `s` — one message per pair per swap, tag 0,
                // no round-as-a-unit rule and no mirrored index.
                swap.sendproc = send_nbr;
                swap.recvproc = recv_nbr;
                swap.sendself = (send_nbr == my_rank);

                swap.pbc[0] = 0;
                swap.pbc[1] = 0;
                swap.pbc[2] = 0;
                {
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
                    // Forwarding round — LAMMPS `CommBrick::setup` slab
                    // bounds for `ineed >= 2` (`src/comm_brick.cpp`).
                    //
                    // Depth bound: the low-going swap hands rows to the
                    // neighbour whose own high edge is this rank's
                    // `sublo`, so that neighbour's shell reaches exactly
                    // `sublo + halo_cutoff`; anything past it is outside
                    // every shell the chain will ever serve, because the
                    // next hop clips at a strictly lower bound again. An
                    // open band ships the whole forwarded slab instead of
                    // the part that is still in reach, and the surplus
                    // compounds once per hop: at the bench geometry
                    // (7.875 A subdomain, 11 A shell) it inflates the
                    // per-axis ghost span 29.9 -> 39.4 A, i.e. 1.76x the
                    // ghost rows over three axes — paid again in halo
                    // bytes, neighbour build and reverse comm.
                    //
                    // Mid-plane bound: LAMMPS, exactly. One window holds
                    // the previous round's receipts from BOTH faces, and
                    // the mid-plane is what sorts them — a ghost that
                    // arrived across one face sits past the opposite edge
                    // of this subdomain, so it fails its own face's band
                    // and passes the other's. That is what keeps a row from
                    // bouncing straight back to where it came from.
                    const double mid = 0.5 * (dd.sublo[d] + dd.subhi[d]);
                    if (hi) {
                        swap.slablo = dd.subhi[d] - slab_cutoff;
                        swap.slabhi = mid;
                    } else {
                        swap.slablo = mid;
                        swap.slabhi = dd.sublo[d] + slab_cutoff;
                    }
                } else if (hi) {
                    swap.slablo = dd.subhi[d] - slab_cutoff;
                    swap.slabhi = +kBigSlab;
                } else {
                    swap.slablo = -kBigSlab;
                    swap.slabhi = dd.sublo[d] + slab_cutoff;
                }

                swap.sendlist.clear();
                swap.nrecv = 0;
                swap.firstrecv = 0;
            }
        }
    }
}

void Halo::rebuild_sendlist(int iswap, const Frame& frame, int sources_end) {
    rebuild_sendlist(iswap, frame, /*first=*/0, /*last=*/sources_end);
}

void Halo::rebuild_sendlist(int iswap, const Frame& frame, int first,
                            int last) {
    if (iswap < 0 || static_cast<std::size_t>(iswap) >= swaps_.size()) {
        throw std::runtime_error(
            "Halo::rebuild_sendlist: swap index " + std::to_string(iswap)
            + " is outside the table of " + std::to_string(swaps_.size())
            + " swaps. Call setup() first and iterate [0, size()).");
    }
    if (first < 0 || last < first || last > frame.capacity()) {
        throw std::runtime_error(
            "Halo::rebuild_sendlist: scan window [" + std::to_string(first)
            + ", " + std::to_string(last)
            + ") is not an ascending range inside " + "the frame's "
            + std::to_string(frame.capacity())
            + " allocated rows. Pass [0, n_local) for the first round of an "
              "axis, [0, n_local + ghosts from earlier axes) for a later axis, "
              "and the previous round's received slot range for a forwarding "
              "round.");
    }

    Swap& swap = swaps_[static_cast<std::size_t>(iswap)];
    const StorageReal* __restrict q = (swap.dim == 0)   ? frame.span(kAtom_x)
                                      : (swap.dim == 1) ? frame.span(kAtom_y)
                                                        : frame.span(kAtom_z);

    // Inclusive on both edges, as LAMMPS `CommBrick::borders`
    // (`x[i][dim] >= lo && x[i][dim] <= hi`). A row sitting exactly on a
    // band edge is shipped: an extra ghost is free, a missing one is a lost
    // pair, and a coordinate that arrived across a periodic boundary can
    // land on the edge by round-off alone.
    const double lo = swap.slablo;
    const double hi = swap.slabhi;
    swap.sendlist.clear();
    const bool ortho = frame.box.is_orthorhombic();
    const StorageReal* x = frame.span(kAtom_x);
    const StorageReal* y = frame.span(kAtom_y);
    const StorageReal* z = frame.span(kAtom_z);
    for (int i = first; i < last; ++i) {
        const double qi =
            ortho ? static_cast<double>(q[i]) / frame.box.length(swap.dim)
                  : frame.box.to_fractional({x[i], y[i], z[i]})[swap.dim];
        if (lo <= qi && qi <= hi)
            swap.sendlist.push_back(i);
    }
}

int Halo::commit_recv(int iswap, int nrecv, int next_slot) {
    if (iswap < 0 || static_cast<std::size_t>(iswap) >= swaps_.size()) {
        throw std::runtime_error(
            "Halo::commit_recv: swap index " + std::to_string(iswap)
            + " is outside the table of " + std::to_string(swaps_.size())
            + " swaps. Call setup() first and iterate [0, size()).");
    }
    if (nrecv < 0 || next_slot < 0) {
        throw std::runtime_error(
            "Halo::commit_recv: negative ghost count (" + std::to_string(nrecv)
            + ") or slot (" + std::to_string(next_slot)
            + "). Drive the prefix sum in swap order starting from n_local.");
    }

    const std::int64_t end = static_cast<std::int64_t>(next_slot) + nrecv;
    if (end > std::numeric_limits<int>::max())
        throw std::overflow_error(
            "Halo::commit_recv: ghost slot range exceeds INT_MAX");
    Swap& swap = swaps_[static_cast<std::size_t>(iswap)];
    swap.nrecv = nrecv;
    swap.firstrecv = next_slot;
    return static_cast<int>(end);
}

int Halo::total_recv() const {
    std::int64_t total = 0;
    for (const Swap& swap : swaps_)
        total += swap.nrecv;
    if (total > std::numeric_limits<int>::max())
        throw std::overflow_error(
            "Halo::total_recv: ghost count exceeds INT_MAX");
    return static_cast<int>(total);
}

void Swap::pbc_shift(const Box& box, double out[3]) const {
    const Vec3 shift = box.to_cartesian({static_cast<double>(pbc[0]),
                                         static_cast<double>(pbc[1]),
                                         static_cast<double>(pbc[2])});
    for (int d = 0; d < 3; ++d)
        out[d] = shift[d];
}

int Halo::stage_end(int begin) const {
    const int n = static_cast<int>(swaps_.size());
    const int dim = swaps_[static_cast<std::size_t>(begin)].dim;
    int end = begin;
    while (end < n && swaps_[static_cast<std::size_t>(end)].dim == dim)
        ++end;
    return end;
}

int Halo::round_end(int begin) const {
    const int n = static_cast<int>(swaps_.size());
    const Swap& first = swaps_[static_cast<std::size_t>(begin)];
    int end = begin;
    while (end < n && swaps_[static_cast<std::size_t>(end)].dim == first.dim
           && swaps_[static_cast<std::size_t>(end)].round == first.round)
        ++end;
    return end;
}

int Halo::max_hops(int d) const {
    check_axis("max_hops", d);
    return hops_[d];
}

int Halo::nswap(int d) const {
    check_axis("nswap", d);
    return 2 * hops_[d];
}

int Halo::maxneed(int d) const {
    check_axis("maxneed", d);
    return maxneed_[d];
}

void Halo::invalidate() {
    for (Swap& swap : swaps_) {
        swap.sendlist.clear();
        swap.nrecv = 0;
        swap.firstrecv = 0;
    }
}

} // namespace comm
} // namespace cpu
} // namespace ATV
