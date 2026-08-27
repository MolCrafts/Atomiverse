/// @file cpu/communication/domain_runtime.h
/// @brief cpu::communication::DomainRuntime — owns the spatial DomainDecomposition
///        and its per-run lifecycle (allocate, bind-to-frame, adopt/release).
///
/// backend-mirror-05-dissolve pulled this out of the former CPU execution
/// engine: domain decomposition is a communication concern, so it is owned and
/// driven here, not by a god-class. The driver-level Runtime composes one of
/// these and calls into it; components still reach the DD through
/// `Context::md.dd`, which this type publishes.
#pragma once

#include <cstddef>
#include <memory>

namespace ATV {
namespace cpu {

class Context;
class Frame;

namespace md {
namespace mpi {
struct DomainDecomposition;
} // namespace mpi
} // namespace md

class Comm;

namespace comm {
class CommBrick;
}

namespace communication {

/// @brief Owns the DomainDecomposition, the brick communicator, and their
///        bind/partition lifecycle.
class DomainRuntime {
  public:
    DomainRuntime();
    ~DomainRuntime();
    DomainRuntime(const DomainRuntime&) = delete;
    DomainRuntime& operator=(const DomainRuntime&) = delete;
    DomainRuntime(DomainRuntime&&) noexcept;
    DomainRuntime& operator=(DomainRuntime&&) noexcept;

    /// @brief Allocate the DD and the sole `CommBrick`, publish
    ///        `ctx.md.dd` / `ctx.md.comm`, and build MPI topology unless
    ///        running serial-on-replica.
    void allocate(Context& ctx);

    /// @brief Bind the owned DD to a frame (partition / slab bounds / tag maps).
    ///
    /// @param frame System state to bind.
    /// @param ctx   Runtime context (rank / size).
    /// @param tag_index_required True when some component in the driver tree
    ///        declared `Component::requires_tag_index()`. The tag index is then
    ///        kept — and, under decomposition, **rebuilt** for the owned rows
    ///        after the initial partition, because the pre-partition map names
    ///        different atoms than the rows it points at.
    void bind_frame(cpu::Frame& frame, Context& ctx,
                    bool tag_index_required = false);

    /// @brief Free the frame tag index + DD tag map when nothing resolves
    ///        global tags at run time.
    ///
    /// @param frame System state.
    /// @param tag_index_required Keep both maps when true. Bonded topology
    ///        implies the need; a component may also declare it without
    ///        carrying any topology at all (`FixSettle` names waters by tag).
    void release_tag_maps_if_unused(cpu::Frame& frame,
                                    bool tag_index_required = false);

    /// @brief Adopt an externally-owned DD (stage hand-off); publishes it.
    void adopt(std::unique_ptr<md::mpi::DomainDecomposition> dd, Context& ctx);

    /// @brief Release the owned DD for reuse; clears `ctx.md.dd`.
    std::unique_ptr<md::mpi::DomainDecomposition> release(Context& ctx);

    /// @brief Run serially on a full replica even under multi-rank MPI.
    void set_serial_on_replica() noexcept { serial_on_replica_ = true; }

    /// @brief Release the MPI communicator held by the DD (shutdown).
    void release_comm();

    /// @brief Non-owning DD accessor (may be null before allocate()).
    md::mpi::DomainDecomposition* dd() const { return dd_.get(); }

    /// @brief Non-owning communicator (null before allocate()).
    Comm* comm() const;

    /// @brief Heap bytes of the DD tag→local map + ghost-tag list (Comm bucket).
    std::size_t memory_bytes() const;

  private:
    std::unique_ptr<md::mpi::DomainDecomposition> dd_;
    std::unique_ptr<comm::CommBrick> comm_;
    bool serial_on_replica_ = false;
};

} // namespace communication
} // namespace cpu
} // namespace ATV
