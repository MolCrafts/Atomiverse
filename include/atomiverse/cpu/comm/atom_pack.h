/// @file cpu/comm/atom_pack.h
/// @brief Schema-compiled, native-dtype Frame row packing.
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
};

PackPlan make_comm_plan(const Frame& frame);
PackPlan make_comm_plan(Frame& frame);
PackPlan make_border_plan(const Frame& frame);
PackPlan make_border_plan(Frame& frame);
PackPlan make_exchange_plan(const Frame& frame);
PackPlan make_exchange_plan(Frame& frame);
PackPlan make_reverse_plan(const Frame& frame);
PackPlan make_reverse_plan(Frame& frame);

/// Fail before an MPI payload operation when ranks disagree.  The MPI owner
/// must compare/all-reduce this value collectively before point-to-point I/O.
void require_schema_fingerprint(const PackPlan& local,
                                std::uint64_t remote_fingerprint,
                                std::uint32_t remote_version);

std::size_t comm_bytes(const PackPlan& plan, int n);
std::size_t border_bytes(const PackPlan& plan, int n);
std::size_t max_exchange_bytes(const Frame& frame, const PackPlan& plan);

std::size_t pack_comm(const Frame& frame, const PackPlan& plan,
                      const int* sendlist, int n, double sx, double sy,
                      double sz, std::byte* out, std::size_t capacity);
void unpack_comm(Frame& frame, const PackPlan& plan, int n, int first,
                 const std::byte* data, std::size_t size);

std::size_t pack_border(const Frame& frame, const PackPlan& plan,
                        const int* sendlist, int n, double sx, double sy,
                        double sz, std::byte* out, std::size_t capacity);
void unpack_border(Frame& frame, const PackPlan& plan, int n, int first,
                   const std::byte* data, std::size_t size);

std::size_t pack_exchange(const Frame& frame, const PackPlan& plan, int row,
                          std::byte* out, std::size_t capacity);

/// Unpack one variable row.  The plan is refreshed in-place if an incoming
/// topology count forces a re-stride before payload materialization.
std::size_t unpack_exchange(Frame& frame, PackPlan& plan, int row,
                            const std::byte* data, std::size_t size);

} // namespace comm
} // namespace cpu
} // namespace ATV
