/// @file cpu/comm/gather.h
/// @brief `gather_to_root` -- collect distributed `cpu::Frame` owners to one rank.
///
/// Used by recorders (XYZ, thermo) and benchmarks that need a single-rank view
/// of the simulation. Each rank ships its local owner range; the root reorders
/// the result by global `tag` so the output matches the original (single-rank)
/// atom ordering.
///
/// Single-rank mode: a fast-path direct copy from the owned rows
/// into the result; no MPI calls.
///
/// Stops short of a parallel HDF5 / per-rank Zarr writer.
///
/// @see .claude/specs/parallel-md-engine.md  § A.9
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/precision.h"

namespace ATV {

namespace cpu {

class Frame;

namespace md {
namespace mpi {

struct DomainDecomposition;

struct GatheredColumn {
    std::string key;
    ColDType dtype = ColDType::Storage;
    int cols = 1;
    ColGroup group = ColGroup::Core;
    int entries_per_item = 0;
    unsigned flags = kColNone;
    std::vector<std::byte> bytes;
};

/// @brief Gathered (root-only) view of all atoms from a distributed frame.
///
/// On the root rank: every per-atom field has size `n_total`, sorted by `tag`
/// in ascending order so callers see a deterministic atom order regardless of
/// the underlying decomposition. On non-root ranks: every per-atom field is
/// empty (size 0); only `n_total` and `box` are populated for ergonomics.
struct GatheredFrame {
    /// @brief Total atom count across all ranks (populated on every rank).
    int n_total = 0;
    /// @brief Root rank that holds the data (defaults to 0).
    int root_rank = 0;

    /// @brief x positions of every atom, ordered by global tag [Angstrom].
    std::vector<StorageReal> x;
    /// @brief y positions of every atom, ordered by global tag [Angstrom].
    std::vector<StorageReal> y;
    /// @brief z positions of every atom, ordered by global tag [Angstrom].
    std::vector<StorageReal> z;
    /// @brief x velocities of every atom, ordered by global tag [Angstrom/fs].
    std::vector<StorageReal> vx;
    /// @brief y velocities of every atom, ordered by global tag [Angstrom/fs].
    std::vector<StorageReal> vy;
    /// @brief z velocities of every atom, ordered by global tag [Angstrom/fs].
    std::vector<StorageReal> vz;
    /// @brief x forces of every atom, ordered by global tag [eV/Angstrom].
    std::vector<AccReal> fx;
    /// @brief y forces of every atom, ordered by global tag [eV/Angstrom].
    std::vector<AccReal> fy;
    /// @brief z forces of every atom, ordered by global tag [eV/Angstrom].
    std::vector<AccReal> fz;
    /// @brief Atomic masses, ordered by global tag [amu].
    std::vector<StorageReal> mass;
    /// @brief Per-atom charges, ordered by global tag [e]. Result data for
    /// charge-writing evaluators (Capability::Charge), not just a static
    /// input -- the task-protocol result path returns it to rank 0.
    std::vector<StorageReal> charge;
    /// @brief Atom types, ordered by global tag.
    std::vector<int> type;
    /// @brief Global atom tags. Ascending: `tag[i] == i` after a clean run.
    std::vector<TagInt> tag;
    /// @brief PBC crossing counts in flat row-major `[n_total * 3]` layout.
    std::vector<std::int8_t> image;
    /// Runtime-added exchanged columns, native and row-major. The schema is
    /// taken directly from the exchange PackPlan; no gather-only field list.
    std::vector<GatheredColumn> runtime_columns;
    /// @brief Simulation cell (replicated to every rank from the local frame).
    Mat3 box{};
};

/// @brief Gather every rank's owner range into a single `GatheredFrame` on `root`.
///
/// Single-rank: returns a direct copy of the owned `atoms` Block rows; `tag`
/// already increases monotonically by construction so no resort is performed.
///
/// Multi-rank: uses `MPI_Gather` for the per-rank owner counts and a sequence
/// of `MPI_Gatherv` calls for each per-atom field. The root then sorts the
/// gathered atoms by `tag` so subsequent callers see a deterministic ordering
/// (i.e. the I/O output is independent of how atoms were partitioned across
/// ranks).
///
/// Non-MPI build: identical to single-rank mode.
///
/// @param frame Host-resident frame; only the first `dd.n_local` rows are read.
/// @param dd Domain decomposition state (for `n_local` and the active comm).
/// @param root Rank index that receives the gathered data. Defaults to 0.
/// @return Gathered frame -- populated on `root`, empty buffers on others.
GatheredFrame gather_to_root(const cpu::Frame& frame,
                             const DomainDecomposition& dd, int root = 0);

/// @brief Materialize a `GatheredFrame` into a host `cpu::Frame` (root-side
/// I/O convenience for recorders that consume `Frame`-shaped geometry, e.g.
/// `cpu::XYZ::write`, and the task-protocol result path). Copies positions,
/// velocities, forces, mass, charge, type, tag, and image in the gathered
/// (global-tag) order and sets the box. Bonded topology does NOT travel
/// through the gather (a per-run static input by the write_back contract).
/// Called on a non-root rank (empty buffers) it returns a 0-atom Frame.
cpu::Frame to_frame(const GatheredFrame& gathered);

} // namespace mpi
} // namespace md
} // namespace cpu
} // namespace ATV
