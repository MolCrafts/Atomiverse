/// @file src/cpu/comm/halo_exchange.cpp
/// @brief `HaloSend` / `HaloRecv` real implementation — table driven.
///
/// Orthorhombic, LAMMPS CommBrick forward comm over an ordered swap table
/// (`comm::SwapList`, owned by `HaloSend`): the swaps of one axis form one
/// stage, the stages run in axis order (`x`, then `y`, then `z`), and inside
/// a stage the `(lo, hi)` swap pair of each **forwarding round** runs in hop
/// order. Both swaps of a round are dispatched **concurrently**
/// (`Isend`/`Irecv` + one `Waitall`): at `dims[d] == 2` a rank's low and high
/// neighbour along `d` are the same rank, and two sequential exchanges cross
/// their tags and deadlock. Consecutive rounds are strictly serialized —
/// round `r` ships what round `r - 1` received.
///
/// Two cascades, one per nesting level:
///
///   * **across axes** — round 1 of a stage scans
///     `[0, n_local + n_ghost_accumulated_so_far)`, so ghosts received from
///     an earlier axis are eligible to be forwarded to a later-axis
///     neighbour. After three stages every rank has the owned-atom plus 1-,
///     2- and 3-axis-shifted ghost copies it needs, reaching all 26 cartesian
///     neighbours (6 face + 12 edge + 8 corner). A single 6-face exchange
///     drops every edge/corner ghost, which silently drops bonded tuples
///     whose partner lives diagonally — see
///     `.claude/notes/cpu_mpi_halo_edge_corner_bug_2026-05-27.md`.
///   * **inside an axis** — a ghost shell deeper than one subdomain
///     (`SwapList::max_hops(d) > 1`, the narrow-box / high-rank-count case)
///     is filled by forwarding: round `r > 1` of a face ships exactly the
///     ghosts round `r - 1` delivered on the OPPOSITE face of the same axis,
///     so rank `c - r`'s atoms arrive through the `r - 1` ranks in between.
///     Nothing is ever sent directly to a rank two hops away.
///
/// Ghost slots are appended in swap order — axis, then hop, then face
/// `{lo, hi}` — and recorded as each swap's `firstrecv`/`nrecv` by
/// `SwapList::commit_recv`; `HaloRecv` mirrors that layout and walks it
/// exactly backwards (last axis first, deepest hop first) so both cascades
/// play in reverse.

#include "atomiverse/cpu/communication/halo_exchange.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/communication/atom_pack.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/schema_mpi.h"
#include "atomiverse/cpu/communication/swap_list.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/atom_sort.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace comm {

// ── Per-instance buffer storage (PIMPL) ──────────────────────────────
//
// `FaceBufs` holds only native byte arrays, request scratch and compiled
// plans, so the type is usable regardless of whether MPI is linked in; the
// send-/recv-force arrays simply stay empty on the !MPI path. Keeping the
// full definition here (rather than under `#ifdef ATV_HAS_MPI`) lets the
// non-MPI stubs and the destructors in both classes see a complete type.
//
// Everything is sized by the **swap count**, which is derived
// (`2 * sum_d maxneed_d`) rather than a fixed six.

namespace detail {

struct FaceBufs {
    std::vector<std::vector<std::byte>> send_pos;
    std::vector<std::vector<std::byte>> recv_pos;
    std::vector<std::vector<std::byte>> send_force;
    std::vector<std::vector<std::byte>> recv_force;
    /// Per-swap count-handshake scratch (`MPI_Isend`/`Irecv` need stable
    /// addresses for the duration of the exchange, so these cannot be
    /// stack locals of a helper).
    std::vector<int> send_count;
    std::vector<int> recv_count;
#ifdef ATV_HAS_MPI
    /// Request scratch: at most one send + one receive per swap in flight.
    std::vector<MPI_Request> requests;
#endif
    PackPlan border_plan;
    PackPlan comm_plan;
    PackPlan reverse_plan;

    /// @brief Size every per-swap array to `nswap`, preserving contents.
    void resize_swaps(std::size_t nswap) {
        send_pos.resize(nswap);
        recv_pos.resize(nswap);
        send_force.resize(nswap);
        recv_force.resize(nswap);
        send_count.resize(nswap, 0);
        recv_count.resize(nswap, 0);
#ifdef ATV_HAS_MPI
        requests.resize(2 * nswap);
#endif
    }
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
    return (bufs_ && !bufs_->send_pos.empty()) ? bufs_->send_pos[0].capacity()
                                               : 0;
}

namespace {
// Resident bytes of a FaceBufs (4 arrays × one byte vector per swap).
std::size_t facebufs_bytes(const detail::FaceBufs* b) {
    if (!b)
        return 0;
    // size() (live elements) ≈ resident; reserved slack is lazily mapped.
    std::size_t n = 0;
    for (const auto& buf : b->send_pos)
        n += buf.size();
    for (const auto& buf : b->recv_pos)
        n += buf.size();
    for (const auto& buf : b->send_force)
        n += buf.size();
    for (const auto& buf : b->recv_force)
        n += buf.size();
    return n * sizeof(std::byte);
}
} // namespace

std::size_t HaloSend::memory_bytes() const {
    return facebufs_bytes(bufs_.get());
}

void HaloSend::destroy(const cpu::DestroyContext& ctx) {
    // The table dies with this component; a stale published pointer would
    // turn a consumer's null branch into a use-after-free. MPI-agnostic on
    // purpose: without MPI nothing is ever published, and clearing a null
    // that is not ours is a no-op either way.
    if (ctx.runtime.md.swap_list == &swaps_)
        ctx.runtime.md.swap_list = nullptr;
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
        frame.box.min_image(dx, dy, dz);
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

/// @brief End of the axis stage starting at `begin`: the half-open run of
/// swaps that cross the same axis.
///
/// The swaps of one axis are contiguous in the table (`SwapList::setup`
/// expands axis-major, then round, then face), and a stage is exactly one
/// such run — the unit whose round 1 shares a scan bound (owners plus the
/// ghosts every earlier axis delivered).
/// @param swaps Published swap table.
/// @param begin First swap of the stage.
/// @return One past the stage's last swap.
int stage_end(const SwapList& swaps, int begin) {
    const int nswap = static_cast<int>(swaps.size());
    const int dim = swaps[begin].dim;
    int end = begin;
    while (end < nswap && swaps[end].dim == dim)
        ++end;
    return end;
}

/// @brief End of the forwarding round starting at `begin`: the `(lo, hi)`
/// swap pair of one hop of one axis.
///
/// The round — not the axis stage — is the transport unit. Its two swaps go
/// out **concurrently** (at `dims[d] == 2` the low and high neighbour are the
/// same rank, and two sequential exchanges cross their tags and deadlock),
/// while consecutive rounds are strictly **serialized**: round `r` ships what
/// round `r - 1` received, so it cannot be posted before that arrived.
/// @param swaps Published swap table.
/// @param begin First swap of the round.
/// @return One past the round's last swap.
int round_end(const SwapList& swaps, int begin) {
    const int nswap = static_cast<int>(swaps.size());
    const int dim = swaps[begin].dim;
    const int round = swaps[begin].round;
    int end = begin;
    while (end < nswap && swaps[end].dim == dim && swaps[end].round == round)
        ++end;
    return end;
}

/// @brief First swap of the forwarding round that ends at `end`.
/// The reverse pass walks rounds backwards, deepest hop first.
/// @param swaps Published swap table.
/// @param end One past the round's last swap.
/// @return First swap of that round.
int round_begin(const SwapList& swaps, int end) {
    int begin = end - 1;
    while (begin > 0 && swaps[begin - 1].dim == swaps[begin].dim
           && swaps[begin - 1].round == swaps[begin].round)
        --begin;
    return begin;
}

/// @brief The swap of the same round pointing the other way.
///
/// A message this rank sends on swap `iswap` arrives at the partner as its
/// opposite-facing swap of the same axis and the same round, so the receive
/// tag is derived from the mirrored index — the swap-table spelling of the
/// old `opposite_face(f) == f ^ 1`.
/// @param begin First swap of the round.
/// @param end One past the round's last swap.
/// @param iswap Swap whose partner-side index is wanted.
/// @return Mirrored swap index within `[begin, end)`.
inline int mirror_swap(int begin, int end, int iswap) {
    return begin + (end - 1) - iswap;
}

/// @brief Which exchange a halo message belongs to. One tag stride each, so
/// a table deeper than the old fixed six swaps cannot alias a count
/// handshake onto a payload.
enum class HaloTagKind : int {
    count = 0,
    forward = 1,
    reverse = 2,
    refresh = 3
};

/// Swaps a single stride may address. A multi-hop table holds
/// `2 * sum_d maxneed_d` swaps — up to `2 * (dims - 1)` on a narrow axis —
/// so the stride is sized for a thousands-of-ranks axis while four strides
/// still fit the 32767 tag range MPI guarantees.
constexpr int kHaloTagStride = 4096;

/// @brief MPI tag for one swap's message of one kind.
/// @param kind Which of the four halo exchanges the message belongs to.
/// @param iswap Swap index in the published table.
/// @return Tag, unique across kinds and swaps.
/// @throws std::overflow_error if the table is deeper than one tag stride.
inline int halo_tag(HaloTagKind kind, int iswap) {
    if (iswap < 0 || iswap >= kHaloTagStride)
        throw std::overflow_error(
            "halo: swap table deeper than the MPI tag stride");
    return static_cast<int>(kind) * kHaloTagStride + iswap;
}

/// @brief Periodic image displacement this swap applies at pack time [Å].
///
/// `Swap::pbc` is an integer image flag; the metric displacement is only
/// formed here, against the **current** box, so an NPT / deform rescale
/// cannot ship a stale shift.
/// @param swap Swap being packed.
/// @param box Current simulation box.
/// @param s Output displacement per axis [Angstrom].
inline void pbc_shift_for_swap(const Swap& swap, const Box& box, double s[3]) {
    for (int d = 0; d < 3; ++d)
        s[d] = static_cast<double>(swap.pbc[d]) * box.length(d);
}

/// @brief True when this swap moves no data at all.
///
/// Either the face has no partner (`MPI_PROC_NULL`, non-periodic axis) or
/// the partner is this rank itself (`sendself`, an undecomposed axis). The
/// latter is a **skip, not a LAMMPS local memcpy** — see
/// `Swap::sendself` for why a self-image ghost would double-count against
/// the pair kernel's min-image.
/// @param swap Swap to test.
/// @return True if the transport side must ignore it.
inline bool swap_is_silent(const Swap& swap) {
    return swap.sendself || swap.sendproc == MPI_PROC_NULL;
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

void pack_forward_for_swap(const Frame& frame, const Swap& swap,
                           const PackPlan& plan, bool full_rebuild,
                           std::vector<std::byte>& buffer,
                           const std::vector<CommField>& fields) {
    const auto& list = swap.sendlist;
    const int n = static_cast<int>(list.size());
    const std::size_t wire_bytes = forward_wire_bytes(plan, n);
    const std::size_t extras =
        checked_mul(static_cast<std::size_t>(n), extra_row_bytes(fields),
                    "halo forward extras");
    buffer.resize(wire_bytes + extras);
    double shift[3];
    pbc_shift_for_swap(swap, frame.box, shift);
    if (full_rebuild)
        (void)pack_border(frame, plan, list.data(), n, shift[0], shift[1],
                          shift[2], buffer.data(), wire_bytes);
    else
        (void)pack_comm(frame, plan, list.data(), n, shift[0], shift[1],
                        shift[2], buffer.data(), wire_bytes);
    pack_extra_fields(fields, list, buffer, wire_bytes);
}

void prealloc_bufs(FaceBufs& b, int initial_cap, std::size_t border_row_bytes,
                   std::size_t comm_row_bytes, std::size_t reverse_row_bytes) {
    const std::size_t forward_row = std::max(border_row_bytes, comm_row_bytes);
    const std::size_t forward_bytes =
        checked_mul(static_cast<std::size_t>(initial_cap), forward_row,
                    "halo preallocate forward");
    const std::size_t reverse_bytes =
        checked_mul(static_cast<std::size_t>(initial_cap), reverse_row_bytes,
                    "halo preallocate reverse");
    for (auto& buf : b.send_pos)
        buf.reserve(forward_bytes);
    for (auto& buf : b.recv_pos)
        buf.reserve(forward_bytes);
    for (auto& buf : b.send_force)
        buf.reserve(reverse_bytes);
    for (auto& buf : b.recv_force)
        buf.reserve(reverse_bytes);
}

/// @brief Re-derive the swap table against the current geometry, with a
/// **rank-uniform** round count.
///
/// `SwapList::derive_hops` is pure and per-rank, and its answer is not
/// automatically the same on every rank: a non-uniform (RCB) partition has a
/// different partner-slab chain per rank, and a non-periodic axis truncates
/// its chain at the global boundary. The transport cannot tolerate that —
/// every round is a matched pair of messages, so a rank that stops one round
/// early leaves its partner posting a receive nobody answers. Hence the one
/// `MPI_Allreduce(MAX)` the spec calls for; it also carries the refusal flag,
/// so a minimum-image violation only some ranks can see throws on **all** of
/// them instead of hanging the rest.
///
/// @param swaps Table to rebuild in place.
/// @param dd This rank's decomposition (current `sublo/subhi` [Angstrom]).
/// @param box Current simulation box; edge lengths [Angstrom].
/// @param shell Ghost shell width [Angstrom],
///        `max(pair cutoff, bonded extension)`.
/// @param my_rank This rank's id in the Cartesian communicator.
/// @param comm Cartesian communicator the halo runs on.
/// @throws std::runtime_error if any rank refuses the geometry.
void setup_swaps(SwapList& swaps, const md::mpi::DomainDecomposition& dd,
                 const Box& box, double shell, int my_rank, MPI_Comm comm) {
    int reduced[4] = {0, 0, 0, 0}; // rounds per axis, then the refusal flag
    std::string refusal;
    try {
        const SwapList::HopNeed hops = SwapList::derive_hops(dd, box, shell);
        for (int d = 0; d < 3; ++d)
            reduced[d] = std::max(hops.minus[d], hops.plus[d]);
    } catch (const std::runtime_error& error) {
        refusal = error.what();
        reduced[3] = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, reduced, 4, MPI_INT, MPI_MAX, comm);
    if (reduced[3] != 0)
        throw std::runtime_error(
            refusal.empty()
                ? std::string("halo: another rank refused the ghost-shell "
                              "geometry (minimum-image bound); its message "
                              "names the axis and the bound.")
                : refusal);
    swaps.setup(dd, box, shell, my_rank, reduced);
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
    if (!dd || ctx.runtime.mpi_size <= 1 || dd->cart_comm == MPI_COMM_NULL)
        return;

    {
        double local = bonded_extension_;
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, dd->cart_comm);
        bonded_extension_ = global;
    }

    int my_rank = 0;
    MPI_Comm_rank(dd->cart_comm, &my_rank);

    // Publish how deep a legitimate ghost may sit, BEFORE any consumer sizes
    // itself against it: `CellList` pads every decomposed axis of its bin
    // grid by this width, and a ghost outside that window is silently clamped
    // into the edge cell (wrong cell membership, no error, wrong pairs).
    dd->ghost_shell_width = halo_cutoff();

    // Derive the swap table now, so the geometry contract is checked at
    // build time (as it always was) and the published table is non-null
    // from the first launch of any consumer. A shell wider than this rank's
    // subdomain is NOT a refusal: `SwapList` derives the extra hops and the
    // forward pass fills the shell by forwarding each round's ghosts one
    // rank further (spec `comm-topology-02`). The one geometry that still
    // throws is the minimum-image bound `shell <= L_d - w_self` — past it an
    // atom would interact with its own periodic image, which no number of
    // hops can fix.
    setup_swaps(swaps_, *dd, ctx.frame.box, halo_cutoff(), my_rank,
                dd->cart_comm);
    ctx.runtime.md.swap_list = &swaps_;

    // `setup` keeps the swap geometry and drops every stored row index, so
    // the table this build publishes has no send lists and no ghost slots
    // yet. Force the next launch to take the full-rebuild path: replaying
    // an empty table would silently leave the ghost rows unrefreshed while
    // `dd->n_ghost` still counts them (a re-`build()` on an already-bound
    // decomposition does not reset that count).
    ago_ = every_;
    if (!bufs_)
        bufs_ = std::make_unique<detail::FaceBufs>();
    bufs_->resize_swaps(swaps_.size());
    bufs_->border_plan = make_border_plan(ctx.frame);
    bufs_->comm_plan = make_comm_plan(ctx.frame);
    bufs_->reverse_plan = make_reverse_plan(ctx.frame);
    // Each swap ships only the boundary layer (cutoff-deep slab, typically
    // 5-15% of owners; the cascade re-forwarding adds a little). A quarter
    // of capacity is generous; std::vector growth covers outliers.
    const int swap_cap = dd->capacity / 4 + 1024;
    prealloc_bufs(*bufs_, swap_cap, border_bytes(bufs_->border_plan, 1),
                  comm_bytes(bufs_->comm_plan, 1),
                  plan_row_bytes(bufs_->reverse_plan));
}

// Per-round helper for the forward path. Both the full rebuild and the
// position-only refresh walk the same rounds of the same table; what
// differs is whether send lists are rebuilt and whether the tag column is
// included in the packet. Encapsulating one round as a single function
// keeps the orchestrator readable.
namespace {

/// @brief Drive one forwarding round — the `(lo, hi)` swap pair
/// `[begin, end)` of one hop of one axis — of the cascading forward halo.
///
/// Both swaps of the round go out concurrently (`Isend`/`Irecv` +
/// `Waitall`), never as two sequential exchanges: at `dims[d] == 2` the low
/// and high neighbour of the axis are the SAME rank, and a sequential pair
/// crosses its tags — each side blocks sending its first swap while waiting
/// to receive the second. It deadlocks, and it deadlocked.
///
/// `slot_in_out` is the running ghost-slot write cursor: on entry it is
/// the next row to write (`n_local` before the first round; after every
/// round it advances by the ghosts that round received). `frame_cap` is
/// the per-atom array capacity used for the overflow guard. When
/// `full_rebuild`, the function rescans each swap's own window
/// `[win_first[k], win_last[k])` for ghost candidates, packs positions+tag,
/// registers each received tag in `tag_to_local` / `ghost_tags`, and commits
/// each swap's `nrecv` / `firstrecv`. When `!full_rebuild`, send lists and
/// receive counts are replayed from the table and only positions get
/// refreshed.
///
/// `xfields` is the flattened list of declared extra forward fields; their
/// payload rides both message shapes (border and position-only refresh)
/// after the intrinsic columns.
///
/// @param win_first Per-swap inclusive lower scan bound, indexed by
///        `s - begin`; unused when `!full_rebuild`.
/// @param win_last Per-swap exclusive upper scan bound, same indexing.
inline void forward_round(Frame& frame, md::mpi::DomainDecomposition& dd,
                          SwapList& swaps, FaceBufs& bufs, int begin, int end,
                          const int* win_first, const int* win_last,
                          bool full_rebuild, int& frame_cap, int my_rank,
                          int& slot_in_out,
                          const std::vector<CommField>& xfields,
                          PackPlan& plan) {
    // A silent swap (no partner, or the partner is this rank) ships and
    // receives nothing, so its send list stays as `setup` left it: empty.
    if (full_rebuild) {
        for (int s = begin; s < end; ++s)
            if (!swap_is_silent(swaps[s]))
                swaps.rebuild_sendlist(s, frame, win_first[s - begin],
                                       win_last[s - begin]);
    }

    for (int s = begin; s < end; ++s) {
        bufs.send_count[static_cast<std::size_t>(s)] =
            static_cast<int>(swaps[s].sendlist.size());
        bufs.recv_count[static_cast<std::size_t>(s)] =
            full_rebuild ? 0 : swaps[s].nrecv;
    }

    if (full_rebuild) {
        int n_creqs = 0;
        for (int s = begin; s < end; ++s) {
            if (swap_is_silent(swaps[s]))
                continue;
            MPI_Isend(&bufs.send_count[static_cast<std::size_t>(s)], 1, MPI_INT,
                      swaps[s].sendproc, halo_tag(HaloTagKind::count, s),
                      dd.cart_comm,
                      &bufs.requests[static_cast<std::size_t>(n_creqs++)]);
            MPI_Irecv(&bufs.recv_count[static_cast<std::size_t>(s)], 1, MPI_INT,
                      swaps[s].recvproc,
                      halo_tag(HaloTagKind::count, mirror_swap(begin, end, s)),
                      dd.cart_comm,
                      &bufs.requests[static_cast<std::size_t>(n_creqs++)]);
        }
        if (n_creqs > 0)
            MPI_Waitall(n_creqs, bufs.requests.data(), MPI_STATUSES_IGNORE);
    }

    const std::size_t row_bytes =
        forward_wire_bytes(plan, 1) + extra_row_bytes(xfields);
    for (int s = begin; s < end; ++s) {
        auto& sbuf = bufs.send_pos[static_cast<std::size_t>(s)];
        auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(s)];
        if (swap_is_silent(swaps[s]))
            sbuf.clear();
        else
            pack_forward_for_swap(frame, swaps[s], plan, full_rebuild, sbuf,
                                  xfields);
        rbuf.assign(
            checked_mul(static_cast<std::size_t>(
                            bufs.recv_count[static_cast<std::size_t>(s)]),
                        row_bytes, "halo receive buffer"),
            std::byte{0});
    }

    int n_preqs = 0;
    for (int s = begin; s < end; ++s) {
        if (swap_is_silent(swaps[s]))
            continue;
        auto& sbuf = bufs.send_pos[static_cast<std::size_t>(s)];
        auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(s)];
        if (bufs.send_count[static_cast<std::size_t>(s)] > 0) {
            MPI_Isend(sbuf.data(),
                      checked_mpi_byte_count(sbuf.size(), "halo forward send"),
                      MPI_BYTE, swaps[s].sendproc,
                      halo_tag(HaloTagKind::forward, s), dd.cart_comm,
                      &bufs.requests[static_cast<std::size_t>(n_preqs++)]);
        }
        if (bufs.recv_count[static_cast<std::size_t>(s)] > 0) {
            MPI_Irecv(
                rbuf.data(),
                checked_mpi_byte_count(rbuf.size(), "halo forward receive"),
                MPI_BYTE, swaps[s].recvproc,
                halo_tag(HaloTagKind::forward, mirror_swap(begin, end, s)),
                dd.cart_comm,
                &bufs.requests[static_cast<std::size_t>(n_preqs++)]);
        }
    }
    if (n_preqs > 0)
        MPI_Waitall(n_preqs, bufs.requests.data(), MPI_STATUSES_IGNORE);

    if (full_rebuild) {
        // Capacity guard for this round. Ranks no longer hold the global-N
        // backing store (partition_atoms shrinks to owned + headroom to avoid
        // per-rank memory scaling with the total system size), so a thin
        // subdomain can legitimately need more ghost rows than the headroom —
        // and a multi-hop shell fills several rounds of them. Grow on demand
        // (explicit-copy realloc preserving the live rows) instead of
        // aborting; update the shared capacity mirrors and the by-ref
        // `frame_cap` so the remaining rounds see the new size.
        int new_ghosts = 0;
        for (int s = begin; s < end; ++s)
            new_ghosts += bufs.recv_count[static_cast<std::size_t>(s)];
        if (slot_in_out + new_ghosts > frame_cap) {
            const int needed = slot_in_out + new_ghosts;
            // 50% slack so a cascade of small overflows doesn't realloc
            // each round.
            const std::int64_t requested =
                static_cast<std::int64_t>(needed) + needed / 2 + 1024;
            if (requested > std::numeric_limits<int>::max())
                throw std::overflow_error("halo: capacity growth overflow");
            const std::uint64_t fingerprint = plan.fingerprint;
            frame.set_capacity(static_cast<int>(requested));
            frame_cap = frame.capacity();
            dd.capacity = frame.capacity();
            plan = make_border_plan(frame);
            require_schema_fingerprint(plan, fingerprint, plan.version);
            ATV_DEBUG("halo_exchange: rank {} grew ghost capacity to {} "
                      "(needed {}) at axis {} round {}",
                      my_rank, frame_cap, needed, swaps[begin].dim,
                      swaps[begin].round);
        }
    }

    // Unpack into ghost slots — for full rebuilds register tags too. The
    // unpack order ALSO defines the slot layout `HaloRecv` mirrors, which
    // is why the rebuild commits it to the table (`commit_recv`) instead of
    // leaving each consumer to re-derive the prefix sum.
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
    for (int s = begin; s < end; ++s) {
        const int n = bufs.recv_count[static_cast<std::size_t>(s)];
        const auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(s)];
        const std::size_t wire_bytes = forward_wire_bytes(plan, n);
        if (full_rebuild) {
            unpack_border(frame, plan, n, slot, rbuf.data(), wire_bytes);
            unpack_extra_fields(xfields, slot, n, rbuf, wire_bytes);
            for (int k = 0; k < n; ++k) {
                const TagInt tag = sp_tag[slot + k];
                if (track_tags) {
                    dd.tag_to_local[tag] = slot + k;
                    dd.ghost_tags.push_back(tag);
                }
            }
            slot = swaps.commit_recv(s, n, slot);
        } else {
            // Replay: the table already knows where this swap's ghosts
            // live, and `firstrecv` is that same running prefix sum.
            const int first = swaps[s].firstrecv;
            unpack_comm(frame, plan, n, first, rbuf.data(), wire_bytes);
            unpack_extra_fields(xfields, first, n, rbuf, wire_bytes);
            slot = first + n;
        }
    }
    slot_in_out = slot;
}

/// @brief Run the whole forward cascade over the published table.
///
/// Two nested walks, both in table order:
///
///   * **axis stages** — axis 0, then 1, then 2. Round 1 of an axis scans
///     `[0, slot)`, i.e. owners plus every ghost the earlier axes already
///     delivered, so a ghost gets *forwarded* to a later-axis neighbour and
///     the 12 edge / 8 corner neighbours are reached without ever addressing
///     them directly.
///   * **forwarding rounds inside an axis** — hop 1, then 2, … Round `r > 1`
///     of a face ships exactly what round `r - 1` delivered on the
///     **opposite** face of the same axis: the ghosts my high neighbour gave
///     me are what my low neighbour is missing at depth `r`, and vice versa.
///     That is the whole multi-hop shell — the same per-round shape
///     `Migrate::exchange_along_dim` uses for owners
///     (`src/cpu/communication/atom_migration.cpp:154`). Feeding a round the
///     owners again instead would re-ship them to the same rank at the same
///     periodic image and double-count every boundary pair.
///
/// Rounds are serialized (round `r` cannot be packed before round `r - 1`
/// arrived); the two faces of one round go out concurrently.
///
/// @param slot_in_out Ghost-slot write cursor; `n_local` on entry, one past
///        the last ghost row on return.
inline void forward_cascade(Frame& frame, md::mpi::DomainDecomposition& dd,
                            SwapList& swaps, FaceBufs& bufs, bool full_rebuild,
                            int& frame_cap, int my_rank, int& slot_in_out,
                            const std::vector<CommField>& xfields,
                            PackPlan& plan) {
    const int nswap = static_cast<int>(swaps.size());
    for (int begin = 0; begin < nswap;) {
        const int stop = stage_end(swaps, begin);
        const int sources_end = slot_in_out;
        // Slot range each face of the PREVIOUS round of this axis received.
        int prev_first[2] = {0, 0};
        int prev_last[2] = {0, 0};

        for (int rbegin = begin; rbegin < stop;) {
            const int rend = round_end(swaps, rbegin);
            const int width = rend - rbegin;
            if (width > 2)
                throw std::logic_error(
                    "halo: a swap-table round holds more than its (lo, hi) "
                    "pair");

            int win_first[2] = {0, 0};
            int win_last[2] = {0, 0};
            for (int k = 0; k < width; ++k) {
                if (swaps[rbegin].round <= 1) {
                    win_first[k] = 0;
                    win_last[k] = sources_end;
                } else {
                    const int opposite = width - 1 - k;
                    win_first[k] = prev_first[opposite];
                    win_last[k] = prev_last[opposite];
                }
            }

            forward_round(frame, dd, swaps, bufs, rbegin, rend, win_first,
                          win_last, full_rebuild, frame_cap, my_rank,
                          slot_in_out, xfields, plan);

            for (int k = 0; k < width; ++k) {
                prev_first[k] = swaps[rbegin + k].firstrecv;
                prev_last[k] =
                    swaps[rbegin + k].firstrecv + swaps[rbegin + k].nrecv;
            }
            rbegin = rend;
        }
        begin = stop;
    }
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

std::vector<ColumnRef> refresh_columns(const PackPlan& plan,
                                       HaloRefresh::Field field) {
    std::vector<std::uint16_t> ids;
    switch (field) {
    case HaloRefresh::Field::positions:
        ids = {static_cast<std::uint16_t>(AtomFieldId::x),
               static_cast<std::uint16_t>(AtomFieldId::y),
               static_cast<std::uint16_t>(AtomFieldId::z)};
        break;
    case HaloRefresh::Field::velocities:
        ids = {static_cast<std::uint16_t>(AtomFieldId::vx),
               static_cast<std::uint16_t>(AtomFieldId::vy),
               static_cast<std::uint16_t>(AtomFieldId::vz)};
        break;
    case HaloRefresh::Field::charges:
        ids = {static_cast<std::uint16_t>(AtomFieldId::charge)};
        break;
    }
    std::vector<ColumnRef> selected(ids.size());
    std::vector<bool> found(ids.size(), false);
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
        throw column_error("halo refresh plan is missing its field set");
    return selected;
}

std::size_t refresh_payload_bytes(int count, std::size_t n_columns) {
    if (count < 0)
        throw std::invalid_argument("negative halo refresh count");
    return checked_mul(checked_mul(static_cast<std::size_t>(count), n_columns,
                                   "halo refresh payload"),
                       sizeof(StorageReal), "halo refresh payload");
}

void pack_refresh_rows(const std::vector<ColumnRef>& columns,
                       const std::vector<int>& rows,
                       const std::array<double, 3>& shift,
                       std::vector<std::byte>& buffer) {
    buffer.resize(
        refresh_payload_bytes(static_cast<int>(rows.size()), columns.size()));
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
            // Only the position set carries a component shift; scalar sets
            // (charges) pass shift = {0,0,0} and stay bit-exact.
            value = static_cast<StorageReal>(static_cast<double>(value)
                                             + shift[component % 3]);
            std::memcpy(buffer.data() + offset, &value, sizeof(value));
            offset += sizeof(value);
        }
    }
}

void unpack_refresh_rows(const std::vector<ColumnRef>& columns, int first,
                         int count, const std::vector<std::byte>& buffer) {
    if (buffer.size() != refresh_payload_bytes(count, columns.size()))
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

    // Re-publish defensively: the table's address is stable for this
    // component's lifetime, but a subsequent `build_components()` cycle
    // (e.g. after a frame resize) would reset the runtime pointer.
    ctx.runtime.md.swap_list = &swaps_;

    // One-time flatten of bound participants' field declarations. Deferred
    // to first launch (not build) because participant arrays are allocated
    // in the participants' own build(), whose plan order may follow ours.
    if (!fields_cached_) {
        append_declared_fields(participants_, /*forward=*/true, fwd_fields_);
        append_declared_fields(participants_, /*forward=*/false, rev_fields_);
        fields_cached_ = true;
    }

    // Decide whether this step needs a full rebuild of the swap table's
    // send lists and ghost-slot assignments. LAMMPS-style cadence: rebuild
    // every `every_` steps, plus any step where migration has bumped the
    // neighbor generation (owner indices changed → every stored send list
    // is stale). In between, only ghost positions are refreshed; slot indices
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
    // Mutable: forward_round grows the frame (and bumps frame_cap) when a
    // shrunk rank needs more ghost rows than its headroom.
    int frame_cap = frame.capacity();

    if (full_rebuild) {
        // LAMMPS: after exchange, before borders — spatial sort owned atoms
        // so pair/neighbor walks are cache-friendly. Must run *before* the
        // send lists / ghost slots are built (reverse uses those indices),
        // and it invalidates the swap table's stored rows.
        if (dd->atom_sort_enabled && dd->atom_sort_nfreq > 0
            && dd->atom_sort_step >= dd->atom_sort_next) {
            double bs = dd->atom_sort_binsize;
            if (!(bs > 0.0))
                bs = 0.5
                     * (dd->atom_sort_neigh_cut > 0.0 ? dd->atom_sort_neigh_cut
                                                      : cutoff_);
            if (md::sort_owned_atoms(frame, dd->n_local, bs))
                md::notify_sorted(dd, frame, &swaps_, /*clear_ghosts=*/true);
            dd->atom_sort_next =
                dd->atom_sort_step
                + (dd->atom_sort_nfreq > 0 ? dd->atom_sort_nfreq : 1);
        }

        frame.set_nghost(0);

        // Re-derive the swap table against the CURRENT subdomain and box.
        // This IS the geometry re-check that used to be a second rejection
        // point: an NPT / deform run can shrink the subdomain after build()
        // sized the table, and the slab bands are box-dependent. A shrink
        // that pushes the shell past one subdomain now simply raises the
        // round count (`SwapList::max_hops`) and the shell is filled by
        // forwarding — never a refusal, never a silently truncated shell.
        // The only refusal left is the minimum-image bound, which no number
        // of rounds can satisfy.
        //
        // Effective halo = max(pair cutoff, bonded skin): the wider of
        // the two is sufficient for both pair partners and bonded-tuple
        // members near the boundary. Republished on `dd` so the neighbor
        // grid pads its decomposed axes by the shell that is actually about
        // to be built, before it consumes the ghost rows.
        dd->ghost_shell_width = halo_cut;
        setup_swaps(swaps_, *dd, frame.box, halo_cut, my_rank, dd->cart_comm);
        bufs.resize_swaps(swaps_.size());

        for (auto t : dd->ghost_tags)
            dd->tag_to_local.erase(t);
        dd->ghost_tags.clear();

        // One stage per axis, and inside an axis one forwarding round per
        // hop; see `forward_cascade` for both walks.
        int slot = dd->n_local;
        forward_cascade(frame, *dd, swaps_, bufs, /*full_rebuild=*/true,
                        frame_cap, my_rank, slot, fwd_fields_, forward_plan);
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

    // Position-only refresh path (no reneighbor). The same axes and the same
    // rounds of the same table play out, in the same order: a send list may
    // reference a ghost slot filled by an earlier axis OR by the previous
    // round of this axis, and both must already hold this step's positions
    // when it is packed.
    int slot = dd->n_local;
    forward_cascade(frame, *dd, swaps_, bufs, /*full_rebuild=*/false, frame_cap,
                    my_rank, slot, fwd_fields_, forward_plan);
    zero_ghost_forces(frame, *dd, bufs.reverse_plan, rev_fields_);
}

// ── HaloRefresh — a forward field refresh at an arbitrary phase ───────

void HaloRefresh::build(const cpu::BuildContext& /*ctx*/) {}

void HaloRefresh::launch(const cpu::LaunchContext& ctx) {
    auto* dd = ctx.runtime.md.dd;
    if (!dd || ctx.runtime.mpi_size <= 1 || dd->cart_comm == MPI_COMM_NULL)
        return;
    // No swap table published (no halo in this run, or step 0 before the
    // first `HaloSend` launch — `HaloRefresh{positions}` binds at
    // `post_integrate`, which precedes `pre_neighbor`). Every rank sees the
    // same nullness, so returning here is a collective no-op, not a
    // half-posted exchange.
    const SwapList* swaps = ctx.runtime.md.swap_list;
    if (!swaps)
        return;
    Frame& frame = ctx.frame;
    const bool pos = (field_ == Field::positions);
    PackPlan plan = make_comm_plan(frame);
    constexpr std::uint64_t kPositionRefresh = 0x706f736974696f6eull;
    constexpr std::uint64_t kVelocityRefresh = 0x76656c6f63697479ull;
    constexpr std::uint64_t kChargeRefresh = 0x63686172676573ull; // "charges"
    const char* refresh_name =
        pos ? "halo/refresh-position"
            : (field_ == Field::velocities ? "halo/refresh-velocity"
                                           : "halo/refresh-charge");
    const std::uint64_t refresh_token =
        pos ? kPositionRefresh
            : (field_ == Field::velocities ? kVelocityRefresh : kChargeRefresh);
    require_collective_halo_schema(plan, {}, dd->cart_comm, refresh_name,
                                   refresh_token);
    const auto columns = refresh_columns(plan, field_);

    // Replay the swap table EXACTLY as the last full rebuild laid it out:
    // round by round in table order (axis 0's hop 1, hop 2, …, then axis 1's,
    // …), and within a round the lo swap then the hi swap, each unpacking at
    // its own `firstrecv`. Replaying the table instead of re-deriving anything
    // is what makes this a refresh rather than a second, conflicting halo, and
    // walking it FORWARD is what keeps a forwarding round's send list — which
    // names ghost rows the previous round filled — reading this step's values
    // rather than the last step's.
    // Both swaps of a round go out CONCURRENTLY (Isend/Irecv + Waitall), never as
    // two sequential Sendrecv calls. With two ranks the lo and hi neighbour of an
    // axis are the SAME rank, and a sequential pair crosses its tags — each side
    // blocks sending the first swap while waiting to receive the second. It
    // deadlocks, and it deadlocked. The rebuild path already does it this way, for
    // exactly this reason.
    const int nswap = static_cast<int>(swaps->size());
    std::vector<std::vector<std::byte>> sbuf(static_cast<std::size_t>(nswap)),
        rbuf(static_cast<std::size_t>(nswap));
    std::vector<MPI_Request> reqs(2 * static_cast<std::size_t>(nswap));

    for (int begin = 0; begin < nswap;) {
        const int end = round_end(*swaps, begin);
        int n_req = 0;

        for (int s = begin; s < end; ++s) {
            const Swap& swap = (*swaps)[s];
            const int n_send = static_cast<int>(swap.sendlist.size());
            auto& sb = sbuf[static_cast<std::size_t>(s)];
            auto& rb = rbuf[static_cast<std::size_t>(s)];

            // Positions carry the periodic image displacement; velocities
            // and charges are translation invariants and must not.
            std::array<double, 3> shifts{};
            if (pos) {
                double shift[3];
                pbc_shift_for_swap(swap, frame.box, shift);
                shifts = {shift[0], shift[1], shift[2]};
            }
            pack_refresh_rows(columns, swap.sendlist, shifts, sb);
            rb.assign(refresh_payload_bytes(swap.nrecv, columns.size()),
                      std::byte{0});

            if (swap_is_silent(swap))
                continue;
            if (n_send > 0)
                MPI_Isend(
                    sb.data(),
                    checked_mpi_byte_count(sb.size(), "halo refresh send"),
                    MPI_BYTE, swap.sendproc, halo_tag(HaloTagKind::refresh, s),
                    dd->cart_comm, &reqs[static_cast<std::size_t>(n_req++)]);
            if (swap.nrecv > 0)
                MPI_Irecv(
                    rb.data(),
                    checked_mpi_byte_count(rb.size(), "halo refresh receive"),
                    MPI_BYTE, swap.recvproc,
                    halo_tag(HaloTagKind::refresh, mirror_swap(begin, end, s)),
                    dd->cart_comm, &reqs[static_cast<std::size_t>(n_req++)]);
        }
        if (n_req > 0)
            MPI_Waitall(n_req, reqs.data(), MPI_STATUSES_IGNORE);

        // Unpack in table order — the slot layout the rebuild committed.
        for (int s = begin; s < end; ++s)
            unpack_refresh_rows(columns, (*swaps)[s].firstrecv,
                                (*swaps)[s].nrecv,
                                rbuf[static_cast<std::size_t>(s)]);
        begin = end;
    }
}

void HaloRecv::build(const cpu::BuildContext& ctx) {
    auto* dd = ctx.runtime.md.dd;
    if (!dd || ctx.runtime.mpi_size <= 1)
        return;
    if (!bufs_)
        bufs_ = std::make_unique<detail::FaceBufs>();
    bufs_->reverse_plan = make_reverse_plan(ctx.frame);
    // The swap count is the producer's to decide, and component `build()`
    // runs in Plan order — the halo's send half is bound at `pre_neighbor`
    // and normally builds first, so the table is usually already published
    // here. When it is not, `launch()` sizes the buffers instead.
    if (const SwapList* swaps = ctx.runtime.md.swap_list)
        bufs_->resize_swaps(swaps->size());
    prealloc_bufs(*bufs_, dd->capacity, 0, 0,
                  plan_row_bytes(bufs_->reverse_plan));
}

void HaloRecv::launch(const cpu::LaunchContext& ctx) {
    if (ctx.runtime.mpi_size <= 1)
        return;
    auto* dd = ctx.runtime.md.dd;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;
    // No swap table published: no forward halo ran, so there is no ghost
    // force to give back. Rank-uniform, hence a safe collective no-op.
    const SwapList* swaps = ctx.runtime.md.swap_list;
    if (!swaps)
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

    const int nswap = static_cast<int>(swaps->size());
    if (bufs.send_force.size() != static_cast<std::size_t>(nswap)) {
        bufs.resize_swaps(static_cast<std::size_t>(nswap));
        prealloc_bufs(bufs, dd->capacity, 0, 0, plan_row_bytes(reverse_plan));
    }

    // Ghost slot layout guard. The prefix sum is the swap table's invariant
    // (`firstrecv = n_local + Σ_{j<i} nrecv[j]`, maintained by
    // `commit_recv`), so all that is left to check is that it describes the
    // frame this rank is about to write into — collectively, because a
    // disagreement is a wire-level mismatch, not a local one.
    {
        const int local_layout_valid =
            swaps->total_recv() == dd->n_ghost
            && dd->n_local + dd->n_ghost <= frame.capacity();
        int all_layouts_valid = 0;
        MPI_Allreduce(&local_layout_valid, &all_layouts_valid, 1, MPI_INT,
                      MPI_MIN, dd->cart_comm);
        if (!all_layouts_valid)
            throw std::runtime_error(
                "halo reverse ghost layout disagrees with Frame capacity");
    }

    // Reverse the cascade EXACTLY backwards: the last round of the last axis
    // first, then its earlier rounds, then the previous axis. Both orders
    // matter and for the same reason — a swap's send list names rows that an
    // EARLIER step of the forward pass filled:
    //
    //   * across axes, a later axis' send list may name an earlier axis'
    //     ghost rows, so z must give back before y and y before x;
    //   * inside an axis, a forwarding round's send list names the ghost rows
    //     the PREVIOUS round received, so the deepest hop must give back
    //     first: its ghost-row forces land on the intermediate rank's own
    //     ghost rows, which the next (shallower) round then carries one rank
    //     further, until round 1 hands everything to the owners.
    //
    // Run a round too early and its contribution is packed before it exists —
    // silently dropped force, not an error.
    for (int end = nswap; end > 0;) {
        const int begin = round_begin(*swaps, end);

        // For a swap's reverse: we send its ghost rows' forces back to the
        // rank that owns them (the same rank we exchanged with going
        // forward), and we receive forces for the atoms we sent out, which
        // live at that swap's own send-list indices on this rank.
        for (int s = begin; s < end; ++s) {
            const Swap& swap = (*swaps)[s];
            pack_reverse_rows(reverse_plan, swap.firstrecv, swap.nrecv,
                              rev_fields_,
                              bufs.send_force[static_cast<std::size_t>(s)]);
            bufs.recv_force[static_cast<std::size_t>(s)].assign(
                reverse_payload_bytes(reverse_plan,
                                      static_cast<int>(swap.sendlist.size()),
                                      rev_fields_),
                std::byte{0});
        }

        int n_reqs = 0;
        for (int s = begin; s < end; ++s) {
            const Swap& swap = (*swaps)[s];
            if (swap_is_silent(swap))
                continue;
            auto& sbuf = bufs.send_force[static_cast<std::size_t>(s)];
            auto& rbuf = bufs.recv_force[static_cast<std::size_t>(s)];
            if (swap.nrecv > 0) {
                MPI_Isend(
                    sbuf.data(),
                    checked_mpi_byte_count(sbuf.size(), "halo reverse send"),
                    MPI_BYTE, swap.recvproc, halo_tag(HaloTagKind::reverse, s),
                    dd->cart_comm,
                    &bufs.requests[static_cast<std::size_t>(n_reqs++)]);
            }
            if (!swap.sendlist.empty()) {
                MPI_Irecv(
                    rbuf.data(),
                    checked_mpi_byte_count(rbuf.size(), "halo reverse receive"),
                    MPI_BYTE, swap.sendproc,
                    halo_tag(HaloTagKind::reverse, mirror_swap(begin, end, s)),
                    dd->cart_comm,
                    &bufs.requests[static_cast<std::size_t>(n_reqs++)]);
            }
        }
        if (n_reqs > 0)
            MPI_Waitall(n_reqs, bufs.requests.data(), MPI_STATUSES_IGNORE);

        // Accumulate received forces into the send-list indices, in table
        // order within the round: the two swaps of a round can name the
        // same owner row (a thin subdomain has atoms in both bands), and
        // floating-point addition is not associative.  Critical: past the
        // first round of the first axis these indices may include ghost-row
        // indices — from an earlier axis, or from the previous round of this
        // axis — and those forces are re-packed and sent further along the
        // cascade in the next iteration of this loop.
        for (int s = begin; s < end; ++s) {
            if (swap_is_silent((*swaps)[s]))
                continue;
            accumulate_reverse_rows(
                reverse_plan, (*swaps)[s].sendlist, rev_fields_,
                bufs.recv_force[static_cast<std::size_t>(s)]);
        }
        end = begin;
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
