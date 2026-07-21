/// @file src/cpu/comm/halo_exchange.cpp
/// @brief `HaloSend` / `HaloRecv` real implementation.
///
/// Orthorhombic, LAMMPS-style multi-stage forward comm: 3 sequential axis
/// stages (`x`, then `y`, then `z`), within each stage the ±-direction send
/// pair is dispatched concurrently. The key invariant is that each axis
/// stage's send list scans `[0, n_local + n_ghost_accumulated_so_far)` —
/// so ghosts received from an earlier axis are eligible to be **forwarded**
/// to a later-axis neighbor. After three stages every rank has the
/// owned-atom plus 1-, 2- and 3-axis-shifted ghost copies it needs, which
/// reaches all 26 cartesian neighbors via cascade (6 face + 12 edge + 8
/// corner). The previous single-round 6-face exchange dropped every
/// edge/corner ghost which silently dropped bonded tuples whose partner
/// lived diagonally — see `.claude/notes/cpu_mpi_halo_edge_corner_bug_2026-05-27.md`.
///
/// Ghost slots are appended in face order `{-x, +x, -y, +y, -z, +z}` so
/// `Reverse` can mirror the layout via `dd->recv_counts[]`; `Reverse`
/// processes the axes in reverse order so the cascade plays backwards.

#include "atomiverse/cpu/comm/halo_exchange.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/comm/atom_pack.h"
#include "atomiverse/cpu/comm/domain.h"
#include "atomiverse/cpu/comm/schema_mpi.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/atom_sort.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace comm {

// ── Per-instance buffer storage (PIMPL) ──────────────────────────────
//
// `FaceBufs` holds only native byte arrays and compiled plans, so the type is
// usable regardless of whether MPI is linked in; the send-/recv-force
// arrays simply stay empty on the !MPI path. Keeping the full definition
// here (rather than under `#ifdef ATV_HAS_MPI`) lets the non-MPI stubs
// and the destructors in both classes see a complete type.

namespace detail {

struct FaceBufs {
    std::array<std::vector<std::byte>, 6> send_pos;
    std::array<std::vector<std::byte>, 6> recv_pos;
    std::array<std::vector<std::byte>, 6> send_force;
    std::array<std::vector<std::byte>, 6> recv_force;
    PackPlan border_plan;
    PackPlan comm_plan;
    PackPlan reverse_plan;
};

} // namespace detail

namespace {
/// @brief Flatten the declared fields of every bound participant into
/// `out`, skipping null/degenerate descriptors. One-time (first launch).
void append_declared_fields(
    const std::vector<const HaloParticipant*>& participants, bool forward,
    std::vector<HaloParticipant::CommField>& out) {
    for (const auto* p : participants) {
        const auto fs =
            forward ? p->comm_forward_fields() : p->comm_reverse_fields();
        for (const auto& f : fs)
            if (f.data && f.width > 0)
                out.push_back(f);
    }
}
} // namespace

// ── Special members (defined here so the PIMPL dtor sees complete FaceBufs) ──

HaloSend::HaloSend() = default;
HaloSend::HaloSend(double cutoff, int every)
    : cutoff_(cutoff), every_(every < 1 ? 1 : every) {}
HaloSend::~HaloSend() = default;
HaloSend::HaloSend(HaloSend&&) noexcept = default;
HaloSend& HaloSend::operator=(HaloSend&&) noexcept = default;

HaloRecv::HaloRecv() = default;
HaloRecv::~HaloRecv() = default;
HaloRecv::HaloRecv(HaloRecv&&) noexcept = default;
HaloRecv& HaloRecv::operator=(HaloRecv&&) noexcept = default;

std::size_t HaloSend::send_buf_capacity_for_test() const {
    return bufs_ ? bufs_->send_pos[0].capacity() : 0;
}

namespace {
// Resident bytes of a FaceBufs (4 arrays × 6 per-face byte vectors).
std::size_t facebufs_bytes(const detail::FaceBufs* b) {
    if (!b)
        return 0;
    // size() (live elements) ≈ resident; reserved slack is lazily mapped.
    std::size_t n = 0;
    for (int f = 0; f < 6; ++f)
        n += (b->send_pos[f].size() + b->recv_pos[f].size()
              + b->send_force[f].size() + b->recv_force[f].size())
             * sizeof(std::byte);
    return n;
}
} // namespace

std::size_t HaloSend::memory_bytes() const {
    return facebufs_bytes(bufs_.get());
}
std::size_t HaloRecv::memory_bytes() const {
    return facebufs_bytes(bufs_.get());
}

// ── Topology-scan helpers (available with or without MPI) ────────────

namespace {

/// @brief Try to resolve a global atom tag to a local row. Returns -1 if
/// the atom is not present on this rank (either as owner or ghost). When
/// `dd == nullptr` (pre-partition / single-rank path) the tag is assumed
/// to equal the row index, matching the default `tag[i] = i` layout set
/// up by `Frame::Atoms`'s constructor.
/// @brief Max intra-tuple span across every tuple whose atoms all resolve
/// (locally) on this rank. Each rank's local max is post-combined with an
/// `MPI_Allreduce(MAX)` in `compute_bonded_extension_mpi` so the global
/// bonded span is visible to every caller even when every cross-rank
/// tuple is partial on every rank.
double compute_bonded_extension_local(const Frame& frame,
                                      const md::mpi::DomainDecomposition* dd) {
    // Reads per-atom bonded topology directly from the canonical atoms Block.
    // Single-rank: `dd` may be null; we treat all atoms as locally
    // resolvable. Multi-rank: `dd->tag_to_local` (still maintained by
    // the old halo / migration path until full switchover) is the
    // resolver for the rank-local Frame rows.
    double max_sq = 0.0;

    // Hoisted ONCE at the launch boundary; `pair_d2` below only ever reads
    // through these (topology arrays num_bond/bond_atom/... stay direct
    // member access -- they are not migrated per the frame-block-02 scope).
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);
    const TagInt* __restrict sp_tag = frame.span(kAtom_tag);
    const int* __restrict num_bond = frame.span(kAtom_num_bond).data();
    const auto bond_atom = frame.span(kAtom_bond_atom);
    const int* __restrict num_angle = frame.span(kAtom_num_angle).data();
    const auto angle_atom = frame.span(kAtom_angle_atom);
    const int* __restrict num_dihedral = frame.span(kAtom_num_dihedral).data();
    const auto dihedral_atom = frame.span(kAtom_dihedral_atom);

    auto pair_d2 = [&](int i, int j) {
        double dx = sp_x[i] - sp_x[j];
        double dy = sp_y[i] - sp_y[j];
        double dz = sp_z[i] - sp_z[j];
        frame.simbox.min_image(dx, dy, dz);
        return dx * dx + dy * dy + dz * dz;
    };

    auto resolve_via_dd = [&](TagInt tag) -> int {
        if (dd == nullptr)
            return static_cast<int>(tag);
        auto it = dd->tag_to_local.find(tag);
        return (it == dd->tag_to_local.end()) ? -1 : it->second;
    };

    // Iterate the *compacted* owner range (`dd->n_local` in multi-rank;
    // `atom.nlocal` single-rank). Beyond that range, rows hold stale
    // pre-partition topology that would resolve to current rows whose
    // positions belong to entirely different atoms — giving spurious
    // 50+ Å bond distances and an astronomically wrong bonded extent.
    const int nlocal_iter = (dd != nullptr) ? dd->n_local : frame.nlocal();
    for (int i_owner = 0; i_owner < nlocal_iter; ++i_owner) {
        const TagInt ti = sp_tag[i_owner];
        // Bonds: owner is the smaller-tag atom; partner stored in
        // bond_atom[i][m].
        const int nb = num_bond[i_owner];
        for (int m = 0; m < nb; ++m) {
            const int i = resolve_via_dd(ti);
            const int j = resolve_via_dd(bond_atom(i_owner, m));
            if (i < 0 || j < 0)
                continue;
            max_sq = std::max(max_sq, pair_d2(i, j));
        }
        // Angles: center atom owns; non-center partners are
        // (angle_atom(i, 2m), angle_atom(i, 2m+1)).
        const int na = num_angle[i_owner];
        for (int m = 0; m < na; ++m) {
            const int j = resolve_via_dd(ti); // center
            const int i = resolve_via_dd(angle_atom(i_owner, 2 * m + 0));
            const int k = resolve_via_dd(angle_atom(i_owner, 2 * m + 1));
            if (i < 0 || j < 0 || k < 0)
                continue;
            max_sq = std::max(max_sq, pair_d2(i, j));
            max_sq = std::max(max_sq, pair_d2(j, k));
            max_sq = std::max(max_sq, pair_d2(i, k));
        }
        // Dihedrals: atom-j owns; non-j partners are (i, k, l) at offsets
        // 3m+0/1/2.
        const int nd = num_dihedral[i_owner];
        for (int m = 0; m < nd; ++m) {
            const int j = resolve_via_dd(ti);
            const int i = resolve_via_dd(dihedral_atom(i_owner, 3 * m + 0));
            const int k = resolve_via_dd(dihedral_atom(i_owner, 3 * m + 1));
            const int l = resolve_via_dd(dihedral_atom(i_owner, 3 * m + 2));
            if (i < 0 || j < 0 || k < 0 || l < 0)
                continue;
            max_sq = std::max(max_sq, pair_d2(i, j));
            max_sq = std::max(max_sq, pair_d2(i, k));
            max_sq = std::max(max_sq, pair_d2(i, l));
            max_sq = std::max(max_sq, pair_d2(j, k));
            max_sq = std::max(max_sq, pair_d2(j, l));
            max_sq = std::max(max_sq, pair_d2(k, l));
        }
    }
    return (max_sq > 0.0) ? std::sqrt(max_sq) : 0.0;
}

} // namespace

double HaloSend::compute_bonded_extension(const cpu::Frame& frame) {
    // The no-DD variant assumes canonical row-index tags, as initialized by
    // Frame. Rank-local multi-rank scans use the DD tag map instead.
    return compute_bonded_extension_local(frame, /*dd=*/nullptr);
}

#ifdef ATV_HAS_MPI

namespace {

using detail::FaceBufs;

/// @brief Helper: opposite face index (toggle low bit of {-x, +x, -y, +y, -z, +z}).
inline int opposite_face(int f) { return f ^ 1; }

/// @brief Map face index to {dim, sign}. dim = f / 2, sign = (f & 1) ? +1 : -1.
inline int face_dim(int f) { return f >> 1; }
inline int face_sign(int f) { return (f & 1) ? +1 : -1; }

/// @brief PBC shift along `dim` for atoms going out face `f` from this rank.
///
/// If this rank sits on the periodic edge in the +dim direction
/// (`coords[dim] == dims[dim]-1`) and we are sending out the +dim face,
/// the receiving rank is at the opposite edge of the global box;
/// pre-subtract `L_dim` from the packed coord so the receiver sees the
/// atom in its own coordinate frame. Symmetric for the -dim edge.
inline double pbc_shift_for_face(const md::mpi::DomainDecomposition& dd, int f,
                                 double L_dim) {
    const int dim = face_dim(f);
    const int sign = face_sign(f);
    if (sign > 0 && dd.coords[dim] == dd.dims[dim] - 1)
        return -L_dim;
    if (sign < 0 && dd.coords[dim] == 0)
        return +L_dim;
    return 0.0;
}

/// @brief Build the ±dim send pair, scanning frame indices `[0, sources_end)`.
///
/// Multi-stage forward comm scans owned atoms in the first axis stage, then
/// owned + axis-0 ghosts in the second stage, etc. The eligibility test is
/// the same proximity check used before (within `cutoff` of the slab
/// boundary along the send direction); the only change is the scan upper
/// bound, which is the caller's responsibility.
///
/// `dd.send_lists[2*dim]` (= face `−dim`) and `dd.send_lists[2*dim+1]` (=
/// face `+dim`) are cleared and refilled. Capacity is preserved.
void rebuild_send_lists_for_axis(const Frame& frame,
                                 md::mpi::DomainDecomposition& dd, int dim,
                                 int sources_end, double cutoff) {
    const int f_lo = 2 * dim;
    const int f_hi = 2 * dim + 1;
    dd.send_lists[static_cast<std::size_t>(f_lo)].clear();
    dd.send_lists[static_cast<std::size_t>(f_hi)].clear();

    const StorageReal* __restrict q = (dim == 0)   ? frame.span(kAtom_x)
                                      : (dim == 1) ? frame.span(kAtom_y)
                                                   : frame.span(kAtom_z);
    const double lo = dd.sublo[dim];
    const double hi = dd.subhi[dim];

    for (int i = 0; i < sources_end; ++i) {
        if (q[i] - lo < cutoff)
            dd.send_lists[static_cast<std::size_t>(f_lo)].push_back(i);
        if (hi - q[i] < cutoff)
            dd.send_lists[static_cast<std::size_t>(f_hi)].push_back(i);
    }
}

using CommField = HaloParticipant::CommField;

void hash_wire_value(std::uint64_t& hash, const void* data, std::size_t bytes) {
    constexpr std::uint64_t prime = 1099511628211ull;
    const auto* input = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= input[i];
        hash *= prime;
    }
}

template <typename T>
void hash_wire_value(std::uint64_t& hash, const T& value) {
    hash_wire_value(hash, &value, sizeof(value));
}

std::uint64_t halo_payload_fingerprint(const PackPlan& plan,
                                       const std::vector<CommField>& extras,
                                       std::uint64_t discriminator = 0) {
    std::uint64_t hash = 1469598103934665603ull;
    hash_wire_value(hash, plan.version);
    hash_wire_value(hash, plan.fingerprint);
    hash_wire_value(hash, discriminator);
    const std::uint64_t count = extras.size();
    hash_wire_value(hash, count);
    const std::uint64_t element_size = sizeof(StorageReal);
    hash_wire_value(hash, element_size);
    for (const CommField& field : extras) {
        if (field.width <= 0)
            throw std::invalid_argument(
                "halo participant field width must be positive");
        const std::int64_t width = field.width;
        hash_wire_value(hash, width);
    }
    return hash;
}

void require_collective_halo_schema(const PackPlan& plan,
                                    const std::vector<CommField>& extras,
                                    MPI_Comm communicator,
                                    std::string_view operation,
                                    std::uint64_t discriminator = 0) {
    require_collective_fingerprint(
        plan.version, halo_payload_fingerprint(plan, extras, discriminator),
        communicator, operation);
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* operation) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error(std::string(operation) + ": size overflow");
    return a * b;
}

std::size_t extra_row_bytes(const std::vector<CommField>& fields) {
    std::size_t result = 0;
    for (const CommField& field : fields) {
        const std::size_t bytes =
            checked_mul(static_cast<std::size_t>(field.width),
                        sizeof(StorageReal), "halo extra field");
        if (bytes > std::numeric_limits<std::size_t>::max() - result)
            throw std::overflow_error("halo extra-field row size overflow");
        result += bytes;
    }
    return result;
}

void pack_extra_fields(const std::vector<CommField>& fields,
                       const std::vector<int>& rows,
                       std::vector<std::byte>& buffer, std::size_t offset) {
    for (int row : rows) {
        for (const CommField& field : fields) {
            const std::size_t bytes =
                static_cast<std::size_t>(field.width) * sizeof(StorageReal);
            const auto* source = reinterpret_cast<const std::byte*>(
                field.data + static_cast<std::size_t>(row) * field.width);
            if (bytes != 0)
                std::memcpy(buffer.data() + offset, source, bytes);
            offset += bytes;
        }
    }
}

void unpack_extra_fields(const std::vector<CommField>& fields, int first,
                         int count, const std::vector<std::byte>& buffer,
                         std::size_t offset) {
    for (int row = first; row < first + count; ++row) {
        for (const CommField& field : fields) {
            const std::size_t bytes =
                static_cast<std::size_t>(field.width) * sizeof(StorageReal);
            auto* destination = reinterpret_cast<std::byte*>(
                field.data + static_cast<std::size_t>(row) * field.width);
            if (bytes != 0)
                std::memcpy(destination, buffer.data() + offset, bytes);
            offset += bytes;
        }
    }
}

std::size_t forward_wire_bytes(const PackPlan& plan, int count) {
    return plan.kind == PackKind::Border ? border_bytes(plan, count)
                                         : comm_bytes(plan, count);
}

std::size_t plan_row_bytes(const PackPlan& plan) {
    std::size_t bytes = 0;
    for (const ColumnRef& field : plan.fields) {
        if (field.row_bytes() > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("halo plan row size overflow");
        bytes += field.row_bytes();
    }
    return bytes;
}

void pack_forward_for_face(const Frame& frame,
                           const md::mpi::DomainDecomposition& dd, int f,
                           const PackPlan& plan, bool full_rebuild,
                           std::vector<std::byte>& buffer,
                           const std::vector<CommField>& fields) {
    const auto& list = dd.send_lists[f];
    const int n = static_cast<int>(list.size());
    const std::size_t wire_bytes = forward_wire_bytes(plan, n);
    const std::size_t extras =
        checked_mul(static_cast<std::size_t>(n), extra_row_bytes(fields),
                    "halo forward extras");
    buffer.resize(wire_bytes + extras);
    const int dim = face_dim(f);
    const double shift = pbc_shift_for_face(dd, f, frame.simbox.length(dim));
    const double sx = dim == 0 ? shift : 0.0;
    const double sy = dim == 1 ? shift : 0.0;
    const double sz = dim == 2 ? shift : 0.0;
    if (full_rebuild)
        (void)pack_border(frame, plan, list.data(), n, sx, sy, sz,
                          buffer.data(), wire_bytes);
    else
        (void)pack_comm(frame, plan, list.data(), n, sx, sy, sz, buffer.data(),
                        wire_bytes);
    pack_extra_fields(fields, list, buffer, wire_bytes);
}

void prealloc_bufs(FaceBufs& b, int initial_cap, std::size_t border_row_bytes,
                   std::size_t comm_row_bytes, std::size_t reverse_row_bytes) {
    for (int f = 0; f < 6; ++f) {
        const std::size_t forward_row =
            std::max(border_row_bytes, comm_row_bytes);
        b.send_pos[f].reserve(checked_mul(static_cast<std::size_t>(initial_cap),
                                          forward_row,
                                          "halo preallocate forward"));
        b.recv_pos[f].reserve(checked_mul(static_cast<std::size_t>(initial_cap),
                                          forward_row,
                                          "halo preallocate forward"));
        b.send_force[f].reserve(
            checked_mul(static_cast<std::size_t>(initial_cap),
                        reverse_row_bytes, "halo preallocate reverse"));
        b.recv_force[f].reserve(
            checked_mul(static_cast<std::size_t>(initial_cap),
                        reverse_row_bytes, "halo preallocate reverse"));
    }
}

} // namespace

void HaloSend::build(const cpu::BuildContext& ctx) {
    // Compute the bonded skin extension from the rank-local topology.
    // Multi-rank: scan uses `dd->tag_to_local` to map topology atom-tags
    // to the post-partition row indices; tuples whose atoms are not all
    // resolvable locally contribute nothing on this rank, and the global
    // max is reconciled with an `MPI_Allreduce(MAX)` below. Single-rank:
    // skip the allreduce; the local scan alone is the global answer.
    auto* dd = ctx.runtime.md.dd;
    bonded_extension_ = compute_bonded_extension_local(ctx.frame, dd);

    if (cutoff_ <= 0.0)
        return;
    if (!dd || ctx.runtime.mpi_size <= 1)
        return;

#ifdef ATV_HAS_MPI
    if (dd->cart_comm != MPI_COMM_NULL) {
        double local = bonded_extension_;
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, dd->cart_comm);
        bonded_extension_ = global;
    }
#endif

    // The ghost shell cannot be wider than the subdomain it is grown from: the halo
    // exchanges with IMMEDIATE neighbour ranks only, so a shell deeper than one
    // subdomain would need atoms from a rank two hops away, and those simply never
    // arrive. Until now that failed SILENTLY — the ghost region was quietly truncated
    // and the forces came out wrong. It surfaced the first time a component asked for
    // a ghost-of-ghost halo (EAM, halo = 2*cutoff): on a box that was big enough for
    // a normal pair halo but not for a doubled one, the decomposed forces differed
    // from a brute-force reference by 1.6 eV/A, with no diagnostic of any kind.
    //
    // Refuse instead. A user who hits this needs a bigger box, fewer ranks, or a
    // smaller cutoff — and must be told so, not handed wrong physics.
    const double need = halo_cutoff();
    for (int d = 0; d < 3; ++d) {
        if (dd->dims[d] <= 1)
            continue; // not decomposed along this axis
        const double sub = dd->subhi[d] - dd->sublo[d];
        if (need > sub) {
            throw std::runtime_error(
                "HaloSend: the requested ghost shell (" + std::to_string(need)
                + " A) is wider than this rank's subdomain along axis "
                + std::to_string(d) + " (" + std::to_string(sub)
                + " A). The halo only exchanges with immediate neighbour "
                  "ranks, so a "
                  "deeper shell cannot be filled and the ghost region would be "
                  "silently truncated. Use a larger box, fewer ranks along "
                  "that axis, "
                  "or a smaller cutoff. (A many-body style such as EAM asks "
                  "for a "
                  "ghost-of-ghost shell of 2*cutoff.)");
        }
    }

    if (!bufs_)
        bufs_ = std::make_unique<detail::FaceBufs>();
    bufs_->border_plan = make_border_plan(ctx.frame);
    bufs_->comm_plan = make_comm_plan(ctx.frame);
    bufs_->reverse_plan = make_reverse_plan(ctx.frame);
    // Each face ships only the boundary layer (cutoff-deep slab, typically
    // 5-15% of owners; the cascade re-forwarding adds a little). A quarter
    // of capacity is generous; std::vector growth covers outliers.
    const int face_cap = dd->capacity / 4 + 1024;
    prealloc_bufs(*bufs_, face_cap, border_bytes(bufs_->border_plan, 1),
                  comm_bytes(bufs_->comm_plan, 1),
                  plan_row_bytes(bufs_->reverse_plan));
    for (int f = 0; f < 6; ++f)
        dd->send_lists[f].reserve(static_cast<std::size_t>(face_cap));
}

// Per-axis sub-stage helper for the forward path. Both `full_rebuild` and
// the position-only refresh path walk the same three axis stages; what
// differs is whether send lists are rebuilt and whether the tag column is
// included in the packet. Encapsulating one axis as a single function
// keeps the orchestrator readable.
namespace {

/// @brief Drive one axis-`dim` sub-stage of the cascading forward halo.
///
/// `slot_in_out` is the running ghost-slot write cursor: on entry it is
/// the next row to write (`n_local` before axis 0; after every axis it
/// advances by the total ghosts received in that axis). `frame_cap` is
/// the per-atom array capacity used for the overflow guard. When
/// `full_rebuild`, the function rescans `[0, sources_end)` for ghost
/// candidates, packs positions+tag, registers each received tag in
/// `tag_to_local` / `ghost_tags`, and stores per-face `recv_counts`. When
/// `!full_rebuild`, send lists and `recv_counts` are reused from the last
/// rebuild and only positions get refreshed.
///
/// `xfields` is the flattened list of declared extra forward fields; their
/// payload rides both message shapes (border and position-only refresh)
/// after the intrinsic columns.
inline void forward_axis_stage(Frame& frame, md::mpi::DomainDecomposition& dd,
                               FaceBufs& bufs, int dim, int sources_end,
                               double cutoff, bool full_rebuild, int& frame_cap,
                               int my_rank, double bonded_extension_for_log,
                               int& slot_in_out,
                               const std::vector<CommField>& xfields,
                               PackPlan& plan) {
    const int f_lo = 2 * dim;
    const int f_hi = 2 * dim + 1;
    const std::size_t fz_lo = static_cast<std::size_t>(f_lo);
    const std::size_t fz_hi = static_cast<std::size_t>(f_hi);

    if (full_rebuild) {
        rebuild_send_lists_for_axis(frame, dd, dim, sources_end, cutoff);
    }

    const int neighbor_lo = dd.face_neighbors[fz_lo];
    const int neighbor_hi = dd.face_neighbors[fz_hi];

    int send_counts[2] = {static_cast<int>(dd.send_lists[fz_lo].size()),
                          static_cast<int>(dd.send_lists[fz_hi].size())};
    int recv_counts[2];

    if (full_rebuild) {
        recv_counts[0] = 0;
        recv_counts[1] = 0;
        MPI_Request count_reqs[4];
        int n_creqs = 0;
        auto post_count = [&](int s) {
            const int face_s = (s == 0) ? f_lo : f_hi;
            const int neighbor = (s == 0) ? neighbor_lo : neighbor_hi;
            if (neighbor == MPI_PROC_NULL || neighbor == my_rank)
                return;
            MPI_Isend(&send_counts[s], 1, MPI_INT, neighbor, face_s,
                      dd.cart_comm, &count_reqs[n_creqs++]);
            MPI_Irecv(&recv_counts[s], 1, MPI_INT, neighbor,
                      opposite_face(face_s), dd.cart_comm,
                      &count_reqs[n_creqs++]);
        };
        post_count(0);
        post_count(1);
        if (n_creqs > 0)
            MPI_Waitall(n_creqs, count_reqs, MPI_STATUSES_IGNORE);
    } else {
        recv_counts[0] = dd.recv_counts[fz_lo];
        recv_counts[1] = dd.recv_counts[fz_hi];
    }

    for (int s = 0; s < 2; ++s) {
        const int face_s = (s == 0) ? f_lo : f_hi;
        const int neighbor = (s == 0) ? neighbor_lo : neighbor_hi;
        auto& sbuf = bufs.send_pos[static_cast<std::size_t>(face_s)];
        auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(face_s)];
        if (neighbor == MPI_PROC_NULL) {
            sbuf.clear();
        } else {
            pack_forward_for_face(frame, dd, face_s, plan, full_rebuild, sbuf,
                                  xfields);
        }
        const std::size_t row_bytes =
            forward_wire_bytes(plan, 1) + extra_row_bytes(xfields);
        rbuf.assign(checked_mul(static_cast<std::size_t>(recv_counts[s]),
                                row_bytes, "halo receive buffer"),
                    std::byte{0});
    }

    MPI_Request pos_reqs[4];
    int n_preqs = 0;
    for (int s = 0; s < 2; ++s) {
        const int face_s = (s == 0) ? f_lo : f_hi;
        const int neighbor = (s == 0) ? neighbor_lo : neighbor_hi;
        if (neighbor == MPI_PROC_NULL || neighbor == my_rank)
            continue;
        if (send_counts[s] > 0) {
            MPI_Isend(
                bufs.send_pos[static_cast<std::size_t>(face_s)].data(),
                checked_mpi_byte_count(
                    bufs.send_pos[static_cast<std::size_t>(face_s)].size(),
                    "halo forward send"),
                MPI_BYTE, neighbor, 100 + face_s, dd.cart_comm,
                &pos_reqs[n_preqs++]);
        }
        if (recv_counts[s] > 0) {
            MPI_Irecv(
                bufs.recv_pos[static_cast<std::size_t>(face_s)].data(),
                checked_mpi_byte_count(
                    bufs.recv_pos[static_cast<std::size_t>(face_s)].size(),
                    "halo forward receive"),
                MPI_BYTE, neighbor, 100 + opposite_face(face_s), dd.cart_comm,
                &pos_reqs[n_preqs++]);
        }
    }
    if (n_preqs > 0)
        MPI_Waitall(n_preqs, pos_reqs, MPI_STATUSES_IGNORE);

    if (full_rebuild) {
        // Capacity guard for this axis. Ranks no longer hold the global-N
        // backing store (partition_atoms shrinks to owned + headroom to avoid
        // per-rank memory scaling with the total system size), so a thin
        // subdomain can legitimately need more ghost rows than the headroom.
        // Grow on demand (explicit-copy realloc preserving the live rows)
        // instead of aborting; update the shared capacity mirrors and the
        // by-ref `frame_cap` so the remaining axis stages see the new size.
        const int new_ghosts = recv_counts[0] + recv_counts[1];
        if (slot_in_out + new_ghosts > frame_cap) {
            const int needed = slot_in_out + new_ghosts;
            // 50% slack so a cascade of small overflows doesn't realloc each axis.
            const std::int64_t requested =
                static_cast<std::int64_t>(needed) + needed / 2 + 1024;
            if (requested > std::numeric_limits<int>::max())
                throw std::overflow_error("halo: capacity growth overflow");
            const std::uint64_t fingerprint = plan.fingerprint;
            frame.set_capacity(static_cast<int>(requested));
            frame_cap = frame.capacity();
            dd.capacity = frame.capacity();
            plan =
                full_rebuild ? make_border_plan(frame) : make_comm_plan(frame);
            require_schema_fingerprint(plan, fingerprint, plan.version);
            ATV_DEBUG("halo_exchange: rank {} grew ghost capacity to {} "
                      "(needed {}) at axis {}",
                      my_rank, frame_cap, needed, dim);
        }
    }

    // Unpack into ghost slots — for full rebuilds register tags too. The
    // unpack order ALSO defines the slot layout that `Reverse` mirrors
    // via `dd->recv_counts`.
    //
    // Hoisted AFTER the capacity guard above (a `set_capacity` there would
    // invalidate any span resolved before it) and dominating both the
    // full_rebuild and refresh unpack loops below.
    TagInt* __restrict sp_tag = frame.span(kAtom_tag).data();
    int slot = slot_in_out;
    // Ghost tag→slot tracking exists solely for bonded-tuple resolution
    // (ResolveView / select_owned_tuples). Pure pairwise systems never
    // query ghost tags, so skip the per-ghost hash inserts and the
    // companion ghost_tags log entirely — the map stays empty (~50 B per
    // ghost saved, plus the per-rebuild insert/erase churn).
    const bool track_tags = frame.has_bonded();
    for (int s = 0; s < 2; ++s) {
        const int face_s = (s == 0) ? f_lo : f_hi;
        const int n = recv_counts[s];
        const auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(face_s)];
        const std::size_t wire_bytes = forward_wire_bytes(plan, n);
        if (full_rebuild) {
            unpack_border(frame, plan, n, slot, rbuf.data(), wire_bytes);
            unpack_extra_fields(xfields, slot, n, rbuf, wire_bytes);
            for (int k = 0; k < n; ++k) {
                const TagInt tag = sp_tag[slot];
                if (track_tags) {
                    dd.tag_to_local[tag] = slot;
                    dd.ghost_tags.push_back(tag);
                }
                ++slot;
            }
            dd.recv_counts[static_cast<std::size_t>(face_s)] = n;
        } else {
            unpack_comm(frame, plan, n, slot, rbuf.data(), wire_bytes);
            unpack_extra_fields(xfields, slot, n, rbuf, wire_bytes);
            slot += n;
        }
    }
    slot_in_out = slot;
}

std::size_t reverse_payload_bytes(const PackPlan& plan, int count,
                                  const std::vector<CommField>& extras) {
    return checked_mul(static_cast<std::size_t>(count),
                       plan_row_bytes(plan) + extra_row_bytes(extras),
                       "halo reverse payload");
}

void pack_reverse_rows(const PackPlan& plan, int first, int count,
                       const std::vector<CommField>& extras,
                       std::vector<std::byte>& buffer) {
    buffer.resize(reverse_payload_bytes(plan, count, extras));
    std::size_t offset = 0;
    for (const ColumnRef& field : plan.fields) {
        for (int k = 0; k < count; ++k) {
            const std::size_t bytes = field.row_bytes();
            if (bytes != 0)
                std::memcpy(buffer.data() + offset,
                            field.bytes()
                                + static_cast<std::size_t>(first + k) * bytes,
                            bytes);
            offset += bytes;
        }
    }
    for (const CommField& field : extras) {
        const std::size_t bytes = checked_mul(
            static_cast<std::size_t>(count),
            static_cast<std::size_t>(field.width) * sizeof(StorageReal),
            "halo reverse extra");
        if (bytes != 0)
            std::memcpy(buffer.data() + offset,
                        field.data
                            + static_cast<std::size_t>(first) * field.width,
                        bytes);
        offset += bytes;
    }
}

template <typename T>
void accumulate_column(const ColumnRef& field, const std::vector<int>& rows,
                       const std::byte* source) {
    T* destination = static_cast<T*>(field.write_data);
    if (!field.writable)
        throw column_error("halo reverse plan is const");
    for (std::size_t k = 0; k < rows.size(); ++k) {
        for (int c = 0; c < field.cols; ++c) {
            T value{};
            const std::size_t value_index =
                k * static_cast<std::size_t>(field.cols)
                + static_cast<std::size_t>(c);
            std::memcpy(&value, source + value_index * sizeof(T), sizeof(T));
            destination[static_cast<std::size_t>(rows[k]) * field.cols + c] +=
                value;
        }
    }
}

void accumulate_reverse_rows(const PackPlan& plan, const std::vector<int>& rows,
                             const std::vector<CommField>& extras,
                             const std::vector<std::byte>& buffer) {
    if (rows.empty()) {
        if (!buffer.empty())
            throw std::runtime_error("halo reverse payload size mismatch");
        return;
    }
    std::size_t offset = 0;
    for (const ColumnRef& field : plan.fields) {
        const std::size_t bytes =
            checked_mul(rows.size(), field.row_bytes(), "halo reverse column");
        switch (field.dtype) {
        case ColDType::Storage:
            accumulate_column<StorageReal>(field, rows, buffer.data() + offset);
            break;
        case ColDType::Acc:
            accumulate_column<AccReal>(field, rows, buffer.data() + offset);
            break;
        case ColDType::Tag:
            accumulate_column<TagInt>(field, rows, buffer.data() + offset);
            break;
        case ColDType::Int:
            accumulate_column<int>(field, rows, buffer.data() + offset);
            break;
        case ColDType::Int8:
            accumulate_column<std::int8_t>(field, rows, buffer.data() + offset);
            break;
        }
        offset += bytes;
    }
    for (const CommField& field : extras) {
        for (std::size_t k = 0; k < rows.size(); ++k) {
            StorageReal* destination =
                field.data + static_cast<std::size_t>(rows[k]) * field.width;
            for (int c = 0; c < field.width; ++c) {
                StorageReal value{};
                const std::size_t value_index =
                    k * static_cast<std::size_t>(field.width)
                    + static_cast<std::size_t>(c);
                std::memcpy(&value,
                            buffer.data() + offset
                                + value_index * sizeof(StorageReal),
                            sizeof(value));
                destination[c] += value;
            }
        }
        offset += rows.size() * static_cast<std::size_t>(field.width)
                  * sizeof(StorageReal);
    }
    if (offset != buffer.size())
        throw std::runtime_error("halo reverse payload size mismatch");
}

// Zero every schema-declared reverse accumulator on ghost rows. Runtime
// columns opt in with kColReverse and require no endpoint field-list edit.
void zero_ghost_forces(Frame& frame, const md::mpi::DomainDecomposition& dd,
                       PackPlan& reverse_plan,
                       const std::vector<CommField>& reverse_fields) {
    const int lo = dd.n_local;
    const int hi = dd.n_local + dd.n_ghost;
    if (!reverse_plan.valid_for(frame) || !reverse_plan.writable) {
        reverse_plan = make_reverse_plan(frame);
    }
    if (hi <= lo)
        return;
    for (const ColumnRef& field : reverse_plan.fields) {
        std::byte* destination =
            field.mutable_bytes()
            + static_cast<std::size_t>(lo) * field.row_bytes();
        const std::size_t bytes =
            checked_mul(static_cast<std::size_t>(hi - lo), field.row_bytes(),
                        "halo reverse zero");
        if (bytes != 0)
            std::memset(destination, 0, bytes);
    }
    // Reverse extra fields share the force semantics: ghost rows are
    // partial-sum accumulators and must start each step at zero.
    for (const auto& f : reverse_fields) {
        for (int i = lo; i < hi; ++i) {
            const std::size_t base =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(f.width);
            for (int k = 0; k < f.width; ++k)
                f.data[base + k] = StorageReal(0);
        }
    }
}

std::array<ColumnRef, 3> refresh_columns(const PackPlan& plan, bool positions) {
    const std::array<std::uint16_t, 3> ids =
        positions ? std::array<std::uint16_t,
                               3>{static_cast<std::uint16_t>(AtomFieldId::x),
                                  static_cast<std::uint16_t>(AtomFieldId::y),
                                  static_cast<std::uint16_t>(AtomFieldId::z)}
                  : std::array<std::uint16_t, 3>{
                        static_cast<std::uint16_t>(AtomFieldId::vx),
                        static_cast<std::uint16_t>(AtomFieldId::vy),
                        static_cast<std::uint16_t>(AtomFieldId::vz)};
    std::array<ColumnRef, 3> selected{};
    std::array<bool, 3> found{};
    for (const ColumnRef& column : plan.fields) {
        for (std::size_t component = 0; component < ids.size(); ++component) {
            if (column.wire_id != ids[component])
                continue;
            if (column.dtype != ColDType::Storage
                || column.element_size != sizeof(StorageReal)
                || column.cols != 1 || !column.writable)
                throw column_error(
                    "halo refresh column must be writable StorageReal[1]");
            selected[component] = column;
            found[component] = true;
        }
    }
    if (!std::all_of(found.begin(), found.end(),
                     [](bool value) { return value; }))
        throw column_error("halo refresh plan is missing its field triple");
    return selected;
}

std::size_t refresh_payload_bytes(int count) {
    if (count < 0)
        throw std::invalid_argument("negative halo refresh count");
    return checked_mul(checked_mul(static_cast<std::size_t>(count),
                                   std::size_t{3}, "halo refresh payload"),
                       sizeof(StorageReal), "halo refresh payload");
}

void pack_refresh_rows(const std::array<ColumnRef, 3>& columns,
                       const std::vector<int>& rows,
                       const std::array<double, 3>& shift,
                       std::vector<std::byte>& buffer) {
    buffer.resize(refresh_payload_bytes(static_cast<int>(rows.size())));
    std::size_t offset = 0;
    for (int row : rows) {
        for (std::size_t component = 0; component < columns.size();
             ++component) {
            StorageReal value{};
            std::memcpy(&value,
                        columns[component].bytes()
                            + static_cast<std::size_t>(row)
                                  * sizeof(StorageReal),
                        sizeof(value));
            value = static_cast<StorageReal>(static_cast<double>(value)
                                             + shift[component]);
            std::memcpy(buffer.data() + offset, &value, sizeof(value));
            offset += sizeof(value);
        }
    }
}

void unpack_refresh_rows(const std::array<ColumnRef, 3>& columns, int first,
                         int count, const std::vector<std::byte>& buffer) {
    if (buffer.size() != refresh_payload_bytes(count))
        throw std::length_error("halo refresh payload size mismatch");
    std::size_t offset = 0;
    for (int k = 0; k < count; ++k) {
        for (const ColumnRef& column : columns) {
            std::memcpy(column.mutable_bytes()
                            + static_cast<std::size_t>(first + k)
                                  * sizeof(StorageReal),
                        buffer.data() + offset, sizeof(StorageReal));
            offset += sizeof(StorageReal);
        }
    }
}

} // namespace

void HaloSend::launch(const cpu::LaunchContext& ctx) {
    if (ctx.runtime.mpi_size <= 1 || cutoff_ <= 0.0)
        return;
    auto* dd = ctx.runtime.md.dd;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;

    Frame& frame = ctx.frame;
    FaceBufs& bufs = *bufs_;

    // One-time flatten of bound participants' field declarations. Deferred
    // to first launch (not build) because participant arrays are allocated
    // in the participants' own build(), whose plan order may follow ours.
    if (!fields_cached_) {
        append_declared_fields(participants_, /*forward=*/true, fwd_fields_);
        append_declared_fields(participants_, /*forward=*/false, rev_fields_);
        fields_cached_ = true;
    }

    // Decide whether this step needs a full rebuild of per-face send lists
    // and ghost-slot assignments. LAMMPS-style cadence: rebuild every
    // `every_` steps, plus any step where migration has bumped the
    // neighbor generation (owner indices changed → any saved send_list is
    // stale). In between, only ghost positions are refreshed; slot indices
    // and the tag→slot map carry through unchanged, which lets downstream
    // `NeighborList` reuse its CSR from the last reneighbor step.
    ++ago_;
    ++dd->atom_sort_step; // MD-step proxy (HaloSend once per step)
    const bool generation_changed =
        dd->neighbor_generation != last_rebuild_generation_;
    const bool full_rebuild = generation_changed || (ago_ >= every_);

    PackPlan& forward_plan = full_rebuild ? bufs.border_plan : bufs.comm_plan;
    const PackKind expected_kind =
        full_rebuild ? PackKind::Border : PackKind::Comm;
    if (!forward_plan.valid_for(frame) || !forward_plan.writable
        || forward_plan.kind != expected_kind) {
        forward_plan =
            full_rebuild ? make_border_plan(frame) : make_comm_plan(frame);
    }
    require_collective_halo_schema(forward_plan, fwd_fields_, dd->cart_comm,
                                   full_rebuild ? "halo/border" : "halo/comm");

    int my_rank = 0;
    MPI_Comm_rank(dd->cart_comm, &my_rank);

    const double halo_cut =
        (cutoff_ > bonded_extension_) ? cutoff_ : bonded_extension_;
    // Mutable: forward_axis_stage grows the frame (and bumps frame_cap) when a
    // shrunk rank needs more ghost rows than its headroom.
    int frame_cap = frame.capacity();

    if (full_rebuild) {
        // LAMMPS: after exchange, before borders — spatial sort owned atoms
        // so pair/neighbor walks are cache-friendly. Must run *before*
        // send_lists / ghost slots are built (reverse uses those indices).
        if (dd->atom_sort_enabled && dd->atom_sort_nfreq > 0
            && dd->atom_sort_step >= dd->atom_sort_next) {
            double bs = dd->atom_sort_binsize;
            if (!(bs > 0.0))
                bs = 0.5 * (dd->atom_sort_neigh_cut > 0.0
                                ? dd->atom_sort_neigh_cut
                                : cutoff_);
            if (md::sort_owned_atoms(frame, dd->n_local, bs))
                md::notify_sorted(dd, frame, /*clear_ghosts=*/true);
            dd->atom_sort_next =
                dd->atom_sort_step
                + (dd->atom_sort_nfreq > 0 ? dd->atom_sort_nfreq : 1);
        }

        frame.set_nghost(0);
        // Re-check the build-time geometry guards: an NPT/deform run can
        // shrink the subdomain AFTER build() validated it (both the
        // shell-width and the dims==2 self-overlap condition are
        // box-dependent). Same fail-loud contract as build().
        for (int d = 0; d < 3; ++d) {
            if (dd->dims[d] <= 1)
                continue;
            const double sub = dd->subhi[d] - dd->sublo[d];
            if (halo_cut > sub)
                throw std::runtime_error(
                    "HaloSend: the box shrank until the ghost shell ("
                    + std::to_string(halo_cut) + " A) no longer fits axis "
                    + std::to_string(d) + "'s subdomain (" + std::to_string(sub)
                    + " A). See HaloSend::build for "
                      "the geometry contract.");
        }

        // Effective halo = max(pair cutoff, bonded skin): the wider of
        // the two is sufficient for both pair partners and bonded-tuple
        // members near the boundary.
        for (auto t : dd->ghost_tags)
            dd->tag_to_local.erase(t);
        dd->ghost_tags.clear();

        // 3 axis stages; each stage's send list scans [0, slot) so atoms
        // received in earlier axes get *forwarded* to later-axis
        // neighbors. After axis 2 (z), each rank has all 26 cartesian
        // neighbor ghosts via cascade.
        int slot = dd->n_local;
        for (int dim = 0; dim < 3; ++dim) {
            forward_axis_stage(frame, *dd, bufs, dim, slot, halo_cut,
                               /*full_rebuild=*/true, frame_cap, my_rank,
                               bonded_extension_, slot, fwd_fields_,
                               forward_plan);
        }
        dd->n_ghost = slot - dd->n_local;
        frame.set_nghost(dd->n_ghost);

        // Signal to ghost-index-caching consumers (the NeighborList CSR
        // is the load-bearing one) that ghost slot assignments may have
        // changed. Without this bump the NL pair list keeps `j` indices
        // pointing at slots whose atoms have been replaced by the
        // current rebuild, producing a per-rebuild PE / virial
        // discontinuity that drifts the NPT extended Hamiltonian
        // (~1-3 eV per `every_=10` rebuild → catastrophic at step
        // ~1000). See `.claude/notes/msu_stage2_conserved_drift_2026-05-27.md`
        // for why we use a halo-local counter instead of bumping the
        // shared `neighbor_generation`.
        ++dd->halo_generation;

        ago_ = 0;
        last_rebuild_generation_ = dd->neighbor_generation;
        zero_ghost_forces(frame, *dd, bufs.reverse_plan, rev_fields_);
        return;
    }

    // Position-only refresh path (no reneighbor). Same three axis stages
    // play out; per-axis ghost positions must be refreshed in order so
    // a later axis sees the up-to-date forwarded ghosts (a later-axis
    // send list may reference an earlier-axis ghost slot).
    int slot = dd->n_local;
    for (int dim = 0; dim < 3; ++dim) {
        forward_axis_stage(frame, *dd, bufs, dim, /*sources_end=*/0, halo_cut,
                           /*full_rebuild=*/false, frame_cap, my_rank,
                           bonded_extension_, slot, fwd_fields_, forward_plan);
    }
    zero_ghost_forces(frame, *dd, bufs.reverse_plan, rev_fields_);
}

// ── HaloRefresh — a forward field refresh at an arbitrary phase ───────

void HaloRefresh::build(const cpu::BuildContext& /*ctx*/) {}

void HaloRefresh::launch(const cpu::LaunchContext& ctx) {
    auto* dd = ctx.runtime.md.dd;
    if (!dd || ctx.runtime.mpi_size <= 1 || dd->cart_comm == MPI_COMM_NULL)
        return;
    Frame& frame = ctx.frame;
    const bool pos = (field_ == Field::positions);
    PackPlan plan = make_comm_plan(frame);
    constexpr std::uint64_t kPositionRefresh = 0x706f736974696f6eull;
    constexpr std::uint64_t kVelocityRefresh = 0x76656c6f63697479ull;
    require_collective_halo_schema(plan, {}, dd->cart_comm,
                                   pos ? "halo/refresh-position"
                                       : "halo/refresh-velocity",
                                   pos ? kPositionRefresh : kVelocityRefresh);
    const auto columns = refresh_columns(plan, pos);

    // Replay the cascade EXACTLY as the last full rebuild laid it out: axis 0,1,2;
    // within an axis, the -face then the +face; ghost slots appended from n_local.
    // Reusing dd->send_lists and dd->recv_counts is what makes this a refresh rather
    // than a second, conflicting halo.
    // Both faces of an axis go out CONCURRENTLY (Isend/Irecv + Waitall), never as two
    // sequential Sendrecv calls. With two ranks the lo and hi neighbour of an axis are
    // the SAME rank, and a sequential pair crosses its tags — each side blocks sending
    // face 0 while waiting to receive face 1. It deadlocks, and it deadlocked. The
    // rebuild path already does it this way, for exactly this reason.
    std::array<std::vector<std::byte>, 2> sbuf, rbuf;
    int slot = dd->n_local;
    int me = 0;
    MPI_Comm_rank(dd->cart_comm, &me);

    for (int dim = 0; dim < 3; ++dim) {
        const int face[2] = {2 * dim, 2 * dim + 1};
        const double L = frame.simbox.length(dim);

        MPI_Request reqs[4];
        int n_req = 0;

        for (int s = 0; s < 2; ++s) {
            const int f = face[s];
            const int nbr = dd->face_neighbors[static_cast<std::size_t>(f)];
            const auto& list = dd->send_lists[static_cast<std::size_t>(f)];
            const int n_send = static_cast<int>(list.size());
            const int n_recv = dd->recv_counts[static_cast<std::size_t>(f)];

            const double shift = pos ? pbc_shift_for_face(*dd, f, L) : 0.0;
            auto& sb = sbuf[static_cast<std::size_t>(s)];
            auto& rb = rbuf[static_cast<std::size_t>(s)];
            std::array<double, 3> shifts{};
            shifts[static_cast<std::size_t>(dim)] = shift;
            pack_refresh_rows(columns, list, shifts, sb);
            rb.assign(refresh_payload_bytes(n_recv), std::byte{0});

            if (nbr == MPI_PROC_NULL || nbr == me)
                continue;
            if (n_send > 0)
                MPI_Isend(
                    sb.data(),
                    checked_mpi_byte_count(sb.size(), "halo refresh send"),
                    MPI_BYTE, nbr, f, dd->cart_comm, &reqs[n_req++]);
            if (n_recv > 0)
                MPI_Irecv(
                    rb.data(),
                    checked_mpi_byte_count(rb.size(), "halo refresh receive"),
                    MPI_BYTE, nbr, opposite_face(f), dd->cart_comm,
                    &reqs[n_req++]);
        }
        if (n_req > 0)
            MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);

        // Unpack in face order lo, hi — the slot layout the rebuild established.
        for (int s = 0; s < 2; ++s) {
            const int f = face[s];
            const int n_recv = dd->recv_counts[static_cast<std::size_t>(f)];
            const auto& rb = rbuf[static_cast<std::size_t>(s)];
            unpack_refresh_rows(columns, slot, n_recv, rb);
            slot += n_recv;
        }
    }
}

void HaloRecv::build(const cpu::BuildContext& ctx) {
    auto* dd = ctx.runtime.md.dd;
    if (!dd || ctx.runtime.mpi_size <= 1)
        return;
    if (!bufs_)
        bufs_ = std::make_unique<detail::FaceBufs>();
    bufs_->reverse_plan = make_reverse_plan(ctx.frame);
    prealloc_bufs(*bufs_, dd->capacity, 0, 0,
                  plan_row_bytes(bufs_->reverse_plan));
}

void HaloRecv::launch(const cpu::LaunchContext& ctx) {
    if (ctx.runtime.mpi_size <= 1)
        return;
    auto* dd = ctx.runtime.md.dd;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;
    Frame& frame = ctx.frame;
    if (!bufs_)
        throw std::logic_error("HaloRecv::launch called before build");
    FaceBufs& bufs = *bufs_;

    // One-time flatten of bound participants' reverse-field declarations
    // (see HaloSend::launch for why this is deferred to first launch).
    if (!fields_cached_) {
        append_declared_fields(participants_, /*forward=*/false, rev_fields_);
        fields_cached_ = true;
    }

    PackPlan& reverse_plan = bufs.reverse_plan;
    if (!reverse_plan.valid_for(frame) || !reverse_plan.writable
        || reverse_plan.kind != PackKind::Reverse)
        reverse_plan = make_reverse_plan(frame);
    require_collective_halo_schema(reverse_plan, rev_fields_, dd->cart_comm,
                                   "halo/reverse");

    int my_rank = 0;
    MPI_Comm_rank(dd->cart_comm, &my_rank);

    // Per-face ghost slot start positions, fixed by the forward halo's
    // unpack order: slot_start[f] = n_local + Σ_{j<f} recv_counts[j].
    int slot_start[6];
    {
        int s = dd->n_local;
        for (int f = 0; f < 6; ++f) {
            slot_start[f] = s;
            s += dd->recv_counts[static_cast<std::size_t>(f)];
        }
        const int local_layout_valid =
            s == dd->n_local + dd->n_ghost && s <= frame.capacity();
        int all_layouts_valid = 0;
        MPI_Allreduce(&local_layout_valid, &all_layouts_valid, 1, MPI_INT,
                      MPI_MIN, dd->cart_comm);
        if (!all_layouts_valid)
            throw std::runtime_error(
                "halo reverse ghost layout disagrees with Frame capacity");
    }

    // Reverse the axis cascade: z first, then y, then x. Forces accumulated
    // on a swap-K ghost row by a later-axis reverse stage are then sent
    // back further down the cascade in earlier-axis stages. Because the
    // forward axis-K send list referenced rows in [0, n_local + Σ_{j<K}
    // recv_counts[j]), the reverse accumulates correctly into rows that
    // are either owned (axis 0) or earlier-axis ghosts (axes 1/2). By the
    // time axis 0 (x) finishes, all force has flowed back to owners.
    for (int dim = 2; dim >= 0; --dim) {
        const int f_lo = 2 * dim;
        const int f_hi = 2 * dim + 1;
        const std::size_t fz_lo = static_cast<std::size_t>(f_lo);
        const std::size_t fz_hi = static_cast<std::size_t>(f_hi);

        const int neighbor_lo = dd->face_neighbors[fz_lo];
        const int neighbor_hi = dd->face_neighbors[fz_hi];

        // For face K's reverse: we send the K-ghost slot forces back to the
        // rank that owns them (= face_neighbors[K], the same rank we
        // exchanged with in forward), and we receive forces for the atoms
        // we sent forward (those live at indices in dd->send_lists[K] on
        // this rank).
        int send_force_counts[2] = {dd->recv_counts[fz_lo],
                                    dd->recv_counts[fz_hi]};
        int recv_force_counts[2] = {
            static_cast<int>(dd->send_lists[fz_lo].size()),
            static_cast<int>(dd->send_lists[fz_hi].size())};

        for (int s = 0; s < 2; ++s) {
            const int face_s = (s == 0) ? f_lo : f_hi;
            const int n = send_force_counts[s];
            auto& sbuf = bufs.send_force[static_cast<std::size_t>(face_s)];
            pack_reverse_rows(reverse_plan, slot_start[face_s], n, rev_fields_,
                              sbuf);
            bufs.recv_force[static_cast<std::size_t>(face_s)].assign(
                reverse_payload_bytes(reverse_plan, recv_force_counts[s],
                                      rev_fields_),
                std::byte{0});
        }

        MPI_Request reqs[4];
        int n_reqs = 0;
        for (int s = 0; s < 2; ++s) {
            const int face_s = (s == 0) ? f_lo : f_hi;
            const int neighbor = (s == 0) ? neighbor_lo : neighbor_hi;
            if (neighbor == MPI_PROC_NULL || neighbor == my_rank)
                continue;
            if (send_force_counts[s] > 0) {
                MPI_Isend(
                    bufs.send_force[static_cast<std::size_t>(face_s)].data(),
                    checked_mpi_byte_count(
                        bufs.send_force[static_cast<std::size_t>(face_s)]
                            .size(),
                        "halo reverse send"),
                    MPI_BYTE, neighbor, 200 + face_s, dd->cart_comm,
                    &reqs[n_reqs++]);
            }
            if (recv_force_counts[s] > 0) {
                MPI_Irecv(
                    bufs.recv_force[static_cast<std::size_t>(face_s)].data(),
                    checked_mpi_byte_count(
                        bufs.recv_force[static_cast<std::size_t>(face_s)]
                            .size(),
                        "halo reverse receive"),
                    MPI_BYTE, neighbor, 200 + opposite_face(face_s),
                    dd->cart_comm, &reqs[n_reqs++]);
            }
        }
        if (n_reqs > 0)
            MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

        // Accumulate received forces into send_list indices. Critical: for
        // dim > 0 these indices may include ghost-row indices from earlier
        // axes — those forces will be re-packed and sent further along the
        // cascade in the next (lower-dim) iteration of this loop.
        for (int s = 0; s < 2; ++s) {
            const int face_s = (s == 0) ? f_lo : f_hi;
            const auto& list = dd->send_lists[static_cast<std::size_t>(face_s)];
            const auto& rbuf =
                bufs.recv_force[static_cast<std::size_t>(face_s)];
            accumulate_reverse_rows(reverse_plan, list, rev_fields_, rbuf);
        }
    }
}

#else // !ATV_HAS_MPI

void HaloSend::build(const cpu::BuildContext& /*ctx*/) {}
void HaloSend::launch(const cpu::LaunchContext& /*ctx*/) {}
void HaloRefresh::build(const cpu::BuildContext& /*ctx*/) {}
void HaloRefresh::launch(const cpu::LaunchContext& /*ctx*/) {}
void HaloRecv::build(const cpu::BuildContext& /*ctx*/) {}
void HaloRecv::launch(const cpu::LaunchContext& /*ctx*/) {}

#endif // ATV_HAS_MPI

} // namespace comm
} // namespace cpu
} // namespace ATV
