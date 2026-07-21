/// @file cpu/md/atom_sort.cpp
/// @brief Spatial reordering of owned atoms (LAMMPS atom_modify sort).
#include "atomiverse/cpu/md/atom_sort.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {
namespace md {

bool sort_owned_atoms(Frame& f, int n_local, double binsize) {
    if (n_local < 2 || !(binsize > 0.0))
        return false;

    const double Lx = f.simbox.length(0);
    const double Ly = f.simbox.length(1);
    const double Lz = f.simbox.length(2);
    if (!(Lx > 0.0) || !(Ly > 0.0) || !(Lz > 0.0))
        return false;

    // Number of bins per axis (at least 1). Match LAMMPS: box / binsize.
    auto nbins = [&](double L) -> int {
        int n = static_cast<int>(std::floor(L / binsize));
        return n < 1 ? 1 : n;
    };
    const int nx = nbins(Lx);
    const int ny = nbins(Ly);
    const int nz = nbins(Lz);
    const int nbin = nx * ny * nz;

    const double inv_dx = static_cast<double>(nx) / Lx;
    const double inv_dy = static_cast<double>(ny) / Ly;
    const double inv_dz = static_cast<double>(nz) / Lz;

    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);

    // bin counts → prefix → scatter order[new_pos] = old_index
    std::vector<int> counts(static_cast<std::size_t>(nbin), 0);
    auto bin_of = [&](int i) -> int {
        // Fold into [0, L) then bin (orthorhombic).
        double px = x[i], py = y[i], pz = z[i];
        px -= Lx * std::floor(px / Lx);
        py -= Ly * std::floor(py / Ly);
        pz -= Lz * std::floor(pz / Lz);
        int ix = static_cast<int>(px * inv_dx);
        int iy = static_cast<int>(py * inv_dy);
        int iz = static_cast<int>(pz * inv_dz);
        if (ix < 0)
            ix = 0;
        if (iy < 0)
            iy = 0;
        if (iz < 0)
            iz = 0;
        if (ix >= nx)
            ix = nx - 1;
        if (iy >= ny)
            iy = ny - 1;
        if (iz >= nz)
            iz = nz - 1;
        return iz + nz * (iy + ny * ix);
    };

    for (int i = 0; i < n_local; ++i)
        ++counts[static_cast<std::size_t>(bin_of(i))];

    std::vector<int> start(static_cast<std::size_t>(nbin), 0);
    int sum = 0;
    for (int b = 0; b < nbin; ++b) {
        start[static_cast<std::size_t>(b)] = sum;
        sum += counts[static_cast<std::size_t>(b)];
    }
    // order[new_index] = old_index
    std::vector<int> order(static_cast<std::size_t>(n_local));
    std::vector<int> cursor = start;
    for (int i = 0; i < n_local; ++i) {
        const int b = bin_of(i);
        const int pos = cursor[static_cast<std::size_t>(b)]++;
        order[static_cast<std::size_t>(pos)] = i;
    }

    // Already sorted?
    bool changed = false;
    for (int i = 0; i < n_local; ++i) {
        if (order[static_cast<std::size_t>(i)] != i) {
            changed = true;
            break;
        }
    }
    if (!changed)
        return false;

    // Apply permutation via cycle-following + Frame::copy_row (migrate cols).
    // Need one free slot: use capacity-1 if free, else do out-of-place with
    // a temp row at n_local if capacity allows, else double-buffer index map.
    //
    // Safest portable approach: inverse map + temp storage at a scratch index.
    // If capacity == n_local (tight), use std::vector swaps via two-pass:
    // follow cycles with a single scratch row at the end of capacity when
    // capacity > n_local; otherwise allocate a temporary Frame row via
    // grow(n_local+1) is too heavy — use element-wise cycle with heap scratch
    // for the largest migrate column set via copy_row to a reserved ghost slot.
    //
    // Policy: if capacity > n_local, use slot n_local as scratch (ghosts may
    // occupy it under MPI — only use when n_ghost==0 OR scratch beyond
    // n_local+n_ghost). When ghosts present, use capacity-1 if free, else
    // rebuild-style out-of-place permute into a temp order of copy_row via
    // an index vector and a boolean visited with in-place cycle + one
    // heap-backed "tmp atom" using a temporary column buffer.
    //
    // Simpler: out-of-place using a second index array and copy_row into
    // temporary storage at the high end of capacity after growing by 1 if needed.

    const int cap = f.capacity();
    int scratch = -1;
    if (cap > f.n_total())
        scratch = f.n_total(); // unused capacity past ghosts
    else if (cap > n_local && f.nghost() == 0)
        scratch = n_local;
    else {
        // Grow by 1 for a scratch row (rare — usually capacity has headroom).
        f.grow(cap + 1);
        scratch = f.n_total(); // old n_total is still owners+ghosts; new cap has room
        // After grow, capacity increased; use last slot.
        scratch = f.capacity() - 1;
    }

    // Apply a[i] ← a[order[i]] in-place by following cycles (with scratch row).
    std::vector<char> visited(static_cast<std::size_t>(n_local), 0);
    for (int i = 0; i < n_local; ++i) {
        if (visited[static_cast<std::size_t>(i)])
            continue;
        const int p0 = order[static_cast<std::size_t>(i)];
        if (p0 == i) {
            visited[static_cast<std::size_t>(i)] = 1;
            continue;
        }
        // temp = a[i]
        f.copy_row(scratch, i);
        int j = i;
        int k = p0;
        while (k != i) {
            // a[j] = a[k]
            f.copy_row(j, k);
            visited[static_cast<std::size_t>(j)] = 1;
            j = k;
            k = order[static_cast<std::size_t>(j)];
        }
        // a[j] = temp
        f.copy_row(j, scratch);
        visited[static_cast<std::size_t>(j)] = 1;
    }
    return true;
}

} // namespace md
} // namespace cpu
} // namespace ATV
