/// @file cpu/comm/domain.h
/// @brief `DomainDecomposition` — multi-rank spatial ownership / halo state.
///
/// **Not used for serial physics.** Without multi-rank MPI there is no
/// spatial decomposition: components take atom counts from `Frame` via
/// `cpu::owned_count` / `cpu::force_count` (see `owned_range.h`). This
/// type is for multi-rank ownership slabs, ghost ranges, and halo
/// bookkeeping only. Single-rank may still allocate a trivial
/// `dims={1,1,1}` stub for infrastructure, but force kernels must not
/// require it — prefer the owned_range helpers.
///
/// Owned by `cpu::Runtime`; non-owning pointer on `cpu::Context::dd`.
///
/// @see .claude/specs/parallel-md-engine.md  § A.1
#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "atomiverse/core/precision.h" // TagInt

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {

struct Mat3;

namespace cpu {

class Frame;
class Box;

namespace md {
namespace mpi {

/// @brief MD spatial domain decomposition state for a single rank.
///
/// Holds (a) MPI Cartesian topology, (b) this rank's owned spatial slab,
/// (c) authoritative atom counts (`n_local`, `n_ghost`, `capacity`),
/// (d) a generation counter that consumers (neighbor lists, halo lists)
/// rebuild against, and (e) the tag→local map.
///
/// It does **not** hold the halo's derived indices: the per-swap send lists
/// and ghost-slot layout live on `comm::Halo`, private to `comm::CommBrick`
/// and published as `ctx.md.comm`.
///
/// `Frame::capacity()` is the allocated row capacity in multi-rank mode;
/// `n_local` is the owned-atom count. Force kernels and integrators iterate
/// `n_local` (via `owned_count()` / `owned_range.h`), not capacity,
/// so ghost rows in `[n_local, n_local + n_ghost)` are never double-counted.
struct DomainDecomposition {
#ifdef ATV_HAS_MPI
    /// @brief Cartesian sub-communicator created by `MPI_Cart_create`.
    MPI_Comm cart_comm = MPI_COMM_NULL;
#endif

    /// @brief Cartesian process grid dimensions. Single-rank: {1,1,1}.
    int dims[3] = {1, 1, 1};

    /// @brief This rank's coords in the Cartesian grid. Single-rank: {0,0,0}.
    int coords[3] = {0, 0, 0};

    /// @brief Rank owning grid cell (cx, cy, cz): x-slowest row-major brick
    /// (periodic wrap NOT applied — caller wraps). The single source of
    /// truth for the rank ↔ coords convention; `dest_rank_for` and the
    /// benchmark-side ownership pre-filters use it. (A LAMMPS-twolevel
    /// node-aware mapping was evaluated here and measured indistinguishable
    /// from row-major on this fabric at 1024 ranks — latency-bound, not
    /// inter-node-bandwidth-bound — so the simple mapping stays.)
    int rank_of(int cx, int cy, int cz) const noexcept {
        return cx * (dims[1] * dims[2]) + cy * dims[2] + cz;
    }

    /// @brief Inverse of `rank_of`: grid coords of `rank`.
    void coords_of(int rank, int out[3]) const noexcept {
        out[0] = rank / (dims[1] * dims[2]);
        out[1] = (rank / dims[2]) % dims[1];
        out[2] = rank % dims[2];
    }

    /// @brief Periodic flags per dimension (1=periodic, 0=non-periodic).
    int periods[3] = {1, 1, 1};

    /// @brief Face neighbor ranks: order is -x, +x, -y, +y, -z, +z.
    /// Single-rank: all zeros (this rank is its own neighbor along periodic axes).
    std::array<int, 6> face_neighbors{};

    /// @brief This rank's owned spatial slab, low corner [Angstrom].
    double sublo[3]{};

    /// @brief This rank's owned spatial slab, high corner [Angstrom].
    double subhi[3]{};

    /// @brief Number of locally owned atoms (authoritative count).
    int n_local = 0;

    /// @brief Number of ghost copies received from neighbors.
    int n_ghost = 0;

    /// @brief Allocated atom-array capacity. `n_local + n_ghost <= capacity`.
    int capacity = 0;

    /// @brief Generation counter, bumped on migration. Neighbor-list / bonded /
    /// halo-list consumers rebuild when they observe a bump.
    std::uint64_t neighbor_generation = 0;

    /// @brief Generation counter, bumped at every `CommBrick` *full
    /// rebuild* (not on position-only refreshes). Distinct from
    /// `neighbor_generation` because a full halo rebuild can reshuffle
    /// ghost slot indices (atoms entering / leaving the cutoff zone, or
    /// just re-binning across axes when a swap's received count changes),
    /// which invalidates any cached structure that stores raw frame
    /// indices for ghost atoms — most importantly the `NeighborList`
    /// pair-list CSR, whose `j` entries hold raw slot indices. Without
    /// this signal the NL keeps pointing at slots whose *atoms* have
    /// been replaced, producing a per-rebuild PE / virial discontinuity
    /// that injects 1-3 eV per halo cycle into the MTK extended
    /// Hamiltonian and accumulates to a step-1000 catastrophic ε̇
    /// runaway under MSU PE stage-2 NPT. (`Comm::exchange` keys off
    /// `neighbor_generation` only — bumping that counter from the halo
    /// path interacts disastrously with its first_launch handling on a
    /// Zarr-loaded state; see
    /// `.claude/notes/msu_stage2_conserved_drift_2026-05-27.md`.)
    std::uint64_t halo_generation = 0;

    /// @brief Depth [Angstrom] of the ghost shell the halo actually builds:
    /// how far beyond `sublo` / `subhi` a legitimate ghost may sit.
    ///
    /// Written by `CommBrick` (`max(pair cutoff, bonded extension)`) at every
    /// full rebuild. `0.0` means no halo has been built yet — the first
    /// `NeighborList::build()` precedes the first `CommBrick::build()`, and
    /// single-rank runs never write it at all — in which case consumers use
    /// their own reach and behave exactly as before this field existed.
    ///
    /// Consumed by `CellList::compute_grid`, which pads every decomposed
    /// axis by `max(cutoff + skin, ghost_shell_width)`. Under a multi-hop
    /// halo (subdomain thinner than the shell) or a bonded extension wider
    /// than `cutoff + skin`, ghosts land deeper than the Verlet reach; a
    /// grid padded only by `cutoff + skin` lets `bin_atoms` clamp them into
    /// the edge cell — silently, with no missing atom and no error, only
    /// wrong cell membership and therefore wrong pairs.
    double ghost_shell_width = 0.0;

    // ── atom_modify sort (LAMMPS: exchange → sort → borders) ──────────
    // NeighborList publishes these; CommBrick applies sort on full rebuild
    // *before* rebuilding ghost slots so the swap table's reverse send
    // lists stay valid.
    bool atom_sort_enabled = true;
    int atom_sort_nfreq = 1000;
    double atom_sort_binsize = 0.0;   ///< ≤0 → 0.5 * atom_sort_neigh_cut
    double atom_sort_neigh_cut = 0.0; ///< neighbor cutoff+skin for auto binsize
    int atom_sort_step = 0;
    int atom_sort_next = 1000; ///< first sort after nfreq steps

    /// @brief Map from global atom tag to local row index.
    ///
    /// Includes both owned atoms (slots `[0, n_local)`) and current-step
    /// ghost atoms (slots `[n_local, n_local + n_ghost)`), so bonded force
    /// kernels can resolve a tuple's atom-tags to local rows regardless of
    /// whether the atom is an owner or a halo ghost.
    ///
    /// Owner entries are rebuilt on every `partition_atoms()` /
    /// `Comm::exchange`. Ghost entries are appended by `CommBrick`
    /// and erased back out at the start of the next halo via the
    /// companion `ghost_tags` log.
    std::unordered_map<TagInt, int> tag_to_local;

    /// @brief Tags of the ghost atoms that `CommBrick` last
    /// inserted into `tag_to_local`, in slot order. Used to roll back
    /// ghost entries before the next halo write so a different ghost set
    /// does not leave stale tag→slot bindings behind.
    std::vector<TagInt> ghost_tags;

    /// @brief Per-axis cut planes as **fractions** of the box length, in `[0, 1]`.
    ///
    /// `cuts[d]` has either size `0` (uniform partition along axis `d`, default)
    /// or size `dims[d] + 1` with `cuts[d].front() == 0.0` and `cuts[d].back() == 1.0`.
    /// When non-empty, `compute_sublo_subhi()` reads from these instead of the
    /// uniform `coord[d] / dims[d]` formula, and `Comm::exchange dest_rank_for`
    /// binary-searches the cuts to decide each owner's destination rank.
    ///
    /// Stored as fractions so the partition survives NPT box rescaling without
    /// needing to be recomputed every step. Populated by `Rebalance` or any
    /// future load-balance component; left empty in the default uniform-brick
    /// decomposition.
    ///
    /// All ranks must agree on these values for migration to be consistent.
    /// `Rebalance` derives them from a global atom histogram via
    /// `MPI_Allreduce`, which guarantees consistency by construction.
    std::array<std::vector<double>, 3> cuts{};

    /// @brief Bind to a frame: populate `n_local`, `capacity`, `tag_to_local`.
    ///
    /// Single-rank initialization. Sets `n_local = capacity = frame.nlocal()`,
    /// `n_ghost = 0`, and rebuilds `tag_to_local` from `kAtom_tag`.
    /// Does not touch MPI topology fields.
    ///
    /// @param frame Host-side frame whose atoms this DD describes.
    void bind(const Frame& frame);

#ifdef ATV_HAS_MPI
    /// @brief Release the Cartesian sub-communicator if still alive.
    ///
    /// Called by `Backend::shutdown()` before `MPI_Finalize` so OpenMPI
    /// does not segfault trying to clean up a user comm that outlived the
    /// runtime. Safe to call when `cart_comm == MPI_COMM_NULL`.
    /// @param mpi_is_finalizing If true, the MPI runtime is shutting
    ///        down and we skip the `MPI_Comm_free` (avoid double-free
    ///        against an already-dead runtime).
    void release_comm(bool mpi_is_finalizing = false);
#endif

    /// @brief Rank that owns position `(x, y, z)` under this decomposition.
    ///
    /// The single owner-lookup of the engine: `Comm::exchange` homes atoms
    /// with it and the benchmark-side pre-filters agree with it by
    /// construction. Coordinates are wrapped into the box first, then
    /// resolved per axis — against `cuts[d]` when the axis carries a
    /// non-uniform partition, against equal-width slabs otherwise.
    ///
    /// Flat coordinates deliberately: this used to exist as three separate
    /// file-local copies (two in the communication directory, one here).
    /// @param x Position along x [Angstrom].
    /// @param y Position along y [Angstrom].
    /// @param z Position along z [Angstrom].
    /// @param box Current simulation box.
    /// @return Owning rank in the Cartesian communicator.
    int owner_of(double x, double y, double z, const ATV::cpu::Box& box) const;

    /// @brief Compute and store this rank's spatial slab from the box.
    ///
    /// Orthorhombic: `sublo[d] = (L_d / dims[d]) * coords[d]`,
    /// `subhi[d] = sublo[d] + L_d / dims[d]`. Idempotent. Single-rank
    /// (`dims == {1,1,1}`): the slab is the full box. Not MPI-gated: the
    /// slab record is useful for any spatial component, e.g. the neighbor
    /// list and future observer shell analysis.
    /// @param box Simulation box (orthorhombic).
    void compute_sublo_subhi(const ATV::cpu::Box& box);

#ifdef ATV_HAS_MPI
    /// @brief Build the 3D Cartesian topology for `world` and discover the
    /// 6 face neighbors.
    ///
    /// `MPI_Dims_create` chooses a balanced `dims` factorization of the
    /// world size; `MPI_Cart_create` produces `cart_comm` with periodic
    /// boundaries on all axes; `MPI_Cart_coords` populates `coords`;
    /// `MPI_Cart_shift` populates `face_neighbors` in the order
    /// `{-x, +x, -y, +y, -z, +z}`.
    ///
    /// Single-rank world (`world_size == 1`): the call is a no-op and
    /// `dims` stays `{1,1,1}`.
    ///
    /// Calling this twice is allowed; the existing `cart_comm` (if any)
    /// is freed before the new one is created.
    /// @param world Parent communicator (typically `MPI_COMM_WORLD`).
    /// @param lx Box edge along x [Angstrom]; pass 0 when the box is not
    ///        yet known (falls back to a near-cubic factorization).
    /// @param ly Box edge along y [Angstrom].
    /// @param lz Box edge along z [Angstrom].
    void init_topology(MPI_Comm world, double lx = 0.0, double ly = 0.0,
                       double lz = 0.0);

    /// @brief Pick the processor grid `dims_out[3]` (product == world_size)
    /// that minimizes the per-rank halo surface for a box of edge lengths
    /// `lx, ly, lz` — LAMMPS ProcMap-style: more cuts along longer axes.
    /// With `MPI_Dims_create`-style blind factorization an anisotropic box
    /// (e.g. the PE melt at 1:1:4) gets its finest cuts on the SHORTEST
    /// axis, producing sub-cutoff-thin slabs whose ghost shells dominate
    /// the step at high rank counts (measured: PE strong-1024 Comm 63% of
    /// the loop, negative 512→1024 scaling).
    ///
    /// Honors the `ATV_FORCE_DIMS=Nx,Ny,Nz` env override (shared by every
    /// caller so builder-side pre-filtering and the DD always agree).
    /// Non-positive lengths fall back to a cube (near-cubic factorization,
    /// equivalent to `MPI_Dims_create`). Pure function of its arguments —
    /// every rank computes the identical grid.
    /// @param world_size Total rank count P.
    /// @param lx Box edge along x [Angstrom] (0 = unknown).
    /// @param ly Box edge along y [Angstrom].
    /// @param lz Box edge along z [Angstrom].
    /// @param dims_out Chosen factorization {dx, dy, dz}, dx*dy*dz == P.
    static void choose_dims(int world_size, double lx, double ly, double lz,
                            int dims_out[3]);

    /// @brief Partition atoms across ranks by spatial position.
    ///
    /// Each rank inspects all owned rows in the `atoms` Block, picks the
    /// subset whose (PBC-wrapped) positions fall in this rank's slab,
    /// compacts those atoms to the front of every per-atom array, and
    /// updates `n_local`, `n_ghost = 0`, `capacity = atoms.size()`. Stale
    /// rows in `[n_local, capacity)` are reused later by `CommBrick`
    /// as ghost slots.
    ///
    /// **Initial-partition contract:** every rank starts with the same
    /// `frame` and claims its spatial subset. Owners that later drift across
    /// a slab boundary are re-homed by `ctx.md.comm->exchange()`, so long
    /// runs where atoms move between slabs are handled correctly (see the
    /// MSU-PE multi-rank NPT validation).
    ///
    /// Single-rank: behaves like `bind(frame)`.
    /// @param frame Frame whose atoms this rank's slab claims a subset of.
    void partition_atoms(ATV::cpu::Frame& frame);
#endif
};

} // namespace mpi
} // namespace md
} // namespace cpu
} // namespace ATV
