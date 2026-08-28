/// @file cpu/communication/atom_pack.h
/// @brief Contract-compiled, native-dtype Frame row packing.
///
/// Payloads are bytes described by an exact schema fingerprint.  Values are
/// copied in their native configured dtype; TagInt is never widened through
/// double. A PackPlan owns a dynamically-sized descriptor vector and is
/// invalidated by Frame assignment, typed-meta schema mutation, or any atoms
/// Block structural generation change.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace comm {

inline constexpr std::uint32_t kAtomWireVersion = 2;

enum class PackKind : std::uint8_t { Comm, Border, Exchange, Reverse };

/// @brief Human name of a wire, for diagnostics.
/// @param kind Wire kind.
/// @return Static string naming the wire.
constexpr const char* pack_kind_name(PackKind kind) noexcept {
    switch (kind) {
    case PackKind::Comm:
        return "refresh";
    case PackKind::Border:
        return "border";
    case PackKind::Exchange:
        return "exchange";
    case PackKind::Reverse:
        return "reverse";
    }
    return "?";
}

/// @brief A wire plan compiled from a Frame's schema, and the verbs that
///        drive it.
///
/// This is a **compiled artifact**, the sibling of `PairList` — not a
/// parameter bag. It is produced once per schema change by `make_*_plan` and
/// replayed for the rest of the interval.
///
/// It carries its own `PackKind`, so the caller never selects a kind-specific
/// entry point and no method has to re-check that the caller selected the
/// right one. That check used to be the first line of six near-identical free
/// functions.
struct PackPlan {
    PackKind kind = PackKind::Exchange;
    std::uint32_t version = kAtomWireVersion;
    std::uint64_t fingerprint = 0;
    std::uint64_t frame_generation = 0;
    std::uint64_t block_generation = 0;
    const Frame* frame_owner = nullptr;
    const Block* owner = nullptr;
    bool writable = false;
    std::vector<ColumnRef> fields;

    /// True only while this plan still belongs to the exact Frame instance
    /// and atoms-Block generation it was compiled from.  This check performs
    /// no mapping lookup and is therefore safe in per-row pack loops.
    bool valid_for(const Frame& frame) const noexcept;
    void validate(const Frame& frame) const;

    // ── sizing ────────────────────────────────────────────────────────

    /// @brief Bytes one row occupies on this wire.
    /// @return Sum of every field's row size [bytes].
    std::size_t row_bytes() const;

    /// @brief Bytes `n` rows occupy on this wire, whichever wire it is.
    /// @param n Row count.
    /// @return Payload size [bytes].
    std::size_t bytes(int n) const;

    /// @brief Widest single row this plan can produce, over every group.
    /// Exchange plans only — the variable-length topology row.
    /// @return Upper bound on one packed row [bytes].
    std::size_t max_row_bytes() const;

    /// @brief Wire-shape digest, for the collective agreement check.
    /// @param discriminator Extra value folded in, distinguishing two uses
    ///        of the same plan shape.
    /// @return 64-bit digest, identical on every rank with the same schema.
    std::uint64_t wire_fingerprint(std::uint64_t discriminator = 0) const;

    // ── fixed-stride wires: comm and border ───────────────────────────

    /// @brief Gather `n` rows onto the wire.
    /// @param frame Frame this plan was compiled from.
    /// @param rows Row indices to gather, `n` entries.
    /// @param n Row count.
    /// @param shift Periodic image displacement per axis [Angstrom],
    ///        applied to the coordinate columns only.
    /// @param out Destination buffer.
    /// @param capacity Bytes available at `out`.
    /// @return Bytes written.
    std::size_t pack(const Frame& frame, const int* rows, int n,
                     const double shift[3], std::byte* out,
                     std::size_t capacity) const;

    /// @brief Scatter `n` wire rows into the contiguous run at `first`.
    /// @param frame Frame this plan was compiled from; must be writable.
    /// @param n Row count.
    /// @param first First destination row.
    /// @param data Payload from the matching `pack`.
    /// @param size Payload size [bytes].
    void unpack(Frame& frame, int n, int first, const std::byte* data,
                std::size_t size) const;

    // ── reverse wire: contiguous source, scattered destination ────────

    /// @brief Pack `count` contiguous ghost rows' accumulators.
    /// @param first First ghost row.
    /// @param count Rows to pack.
    /// @param out Output, resized to `bytes(count)`.
    void pack_rows(int first, int count, std::vector<std::byte>& out) const;

    /// @brief Fold a reverse payload onto the owner rows in `rows`, `+=`.
    /// @param rows Owner row indices, one per packed row.
    /// @param in Payload from the matching `pack_rows`.
    void accumulate(const std::vector<int>& rows,
                    const std::vector<std::byte>& in) const;

    // ── exchange wire: one variable-length row at a time ──────────────

    /// @brief Pack one whole atom, topology included.
    /// @param frame Frame this plan was compiled from.
    /// @param row Row to pack.
    /// @param out Destination buffer.
    /// @param capacity Bytes available at `out`.
    /// @return Bytes written.
    std::size_t pack_row(const Frame& frame, int row, std::byte* out,
                         std::size_t capacity) const;

    /// @brief Unpack one whole atom. Non-const: an incoming topology count
    ///        can force a re-stride, which refreshes this plan in place.
    /// @param frame Destination frame.
    /// @param row Destination row.
    /// @param data Payload from the matching `pack_row`.
    /// @param size Payload size [bytes].
    /// @return Bytes consumed.
    std::size_t unpack_row(Frame& frame, int row, const std::byte* data,
                           std::size_t size);
};

/// @brief Compile a wire plan from the folded ghost contract.
///
/// **This is the compile step**, and the only place that maps a
/// data-lifetime contract onto a communication pass:
///
/// | `PackKind` | carries |
/// | --- | --- |
/// | `Border` | every read — presence is implied by reading it |
/// | `Comm` | reads at `Freshness::step` |
/// | `Reverse` | every reduction |
///
/// Nothing above this function names a wire, and nothing below it names a
/// freshness. `make_exchange_plan` stays flag-driven — migration carries
/// what an atom IS, a property of the Frame rather than of whoever happens
/// to be mounted, which is a different question and not a second answer to
/// this one.
/// @param frame Frame to resolve against.
/// @param kind Which wire.
/// @param schema Fields the wire carries, in wire order.
/// @return Compiled plan.
PackPlan make_wire_plan(Frame& frame, PackKind kind,
                        const GhostContract& contract);
PackPlan make_wire_plan(const Frame& frame, PackKind kind,
                        const GhostContract& contract);

PackPlan make_exchange_plan(const Frame& frame);
PackPlan make_exchange_plan(Frame& frame);

/// Fail before an MPI payload operation when ranks disagree.  The MPI owner
/// must compare/all-reduce this value collectively before point-to-point I/O.
void require_schema_fingerprint(const PackPlan& local,
                                std::uint64_t remote_fingerprint,
                                std::uint32_t remote_version);

/// @brief Multiply two sizes, throwing rather than wrapping.
///
/// The single overflow-checked multiply of the communication layer; the
/// gather, broadcast, pack and halo paths all use it so a size overflow
/// reports the same way wherever it happens.
/// @param a First factor.
/// @param b Second factor.
/// @param operation Name used in the failure message.
/// @return `a * b`.
/// @throws std::overflow_error if the product would wrap.
std::size_t checked_mul(std::size_t a, std::size_t b, const char* operation);

/// @brief Bytes of an extra-field packer payload: `n_atoms * nsize` doubles.
///
/// `ForwardPacker::comm_forward()` is doubles per atom; the MPI post must
/// use this product, not `n_atoms` alone. LAMMPS `CommBrick::forward_comm(Fix*)`
/// posts `nsize * recvnum` MPI_DOUBLE.
/// @param n_atoms Atom count on this swap.
/// @param nsize Doubles per atom (`Packer::comm_forward()`).
/// @return Payload size [bytes].
std::size_t packer_bytes(int n_atoms, int nsize);

} // namespace comm
} // namespace cpu
} // namespace ATV
