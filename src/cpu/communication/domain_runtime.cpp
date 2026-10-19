/// @file cpu/communication/domain_runtime.cpp
/// @brief cpu::communication::DomainRuntime implementation (moved out of the
///        former CPU execution engine by backend-mirror-05-dissolve).

#include "atomiverse/cpu/communication/domain_runtime.h"

#include "atomiverse/cpu/communication/comm_brick.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"

#include <cstdint>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace communication {

DomainRuntime::DomainRuntime() = default;
DomainRuntime::~DomainRuntime() = default;
DomainRuntime::DomainRuntime(DomainRuntime&&) noexcept = default;
DomainRuntime& DomainRuntime::operator=(DomainRuntime&&) noexcept = default;

Comm* DomainRuntime::comm() const { return comm_.get(); }

void DomainRuntime::allocate(Context& ctx) {
    if (!dd_)
        dd_ = std::make_unique<md::mpi::DomainDecomposition>();
    ctx.md.dd = dd_.get();
    if (!comm_)
        comm_ = std::unique_ptr<comm::CommBrick>(new comm::CommBrick(*dd_));
    ctx.md.comm = comm_.get();

#ifdef ATV_HAS_MPI
    // serial_on_replica: molecular whole-system evaluators opt out of spatial
    // decomposition — leaving cart_comm null means bind_frame's partition
    // condition never fires and every rank keeps its replica.
    if (ctx.mpi_size > 1 && !ctx.exec.serial_on_replica
        && dd_->cart_comm == MPI_COMM_NULL) {
        dd_->init_topology(MPI_COMM_WORLD);
    }
#else
    (void)ctx;
#endif
}

void DomainRuntime::adopt(std::unique_ptr<md::mpi::DomainDecomposition> dd,
                          Context& ctx) {
    comm_.reset();
    ctx.md.comm = nullptr;
    dd_ = std::move(dd);
    bound_frame_ = nullptr;
    ctx.md.dd = dd_.get();
}

std::unique_ptr<md::mpi::DomainDecomposition>
DomainRuntime::release(Context& ctx) {
    comm_.reset();
    ctx.md.comm = nullptr;
    ctx.md.dd = nullptr;
    bound_frame_ = nullptr;
    return std::move(dd_);
}

void DomainRuntime::bind_frame(cpu::Frame& frame, Context& ctx,
                               bool tag_index_required) {
    if (!dd_)
        return;
#ifdef ATV_HAS_MPI
    if (ctx.exec.serial_on_replica && dd_->cart_comm != MPI_COMM_NULL) {
        // A retained shard cannot become a complete replica by changing mode.
        // Check identity before releasing any partition or transport state.
        if (dd_->partitioned && (!bound_frame_ || bound_frame_ == &frame))
            throw std::runtime_error(
                "DomainRuntime: changing spatial decomposition to replica mode "
                "requires binding a new complete Frame; the retained Frame "
                "contains only this rank's owned shard");
        comm_.reset();
        dd_ = std::make_unique<md::mpi::DomainDecomposition>();
        comm_.reset(new comm::CommBrick(*dd_));
        ctx.md.dd = dd_.get();
        ctx.md.comm = comm_.get();
        bound_frame_ = nullptr;
    }
#endif
    if (bound_frame_ && bound_frame_ != &frame) {
        comm_.reset(new comm::CommBrick(*dd_));
        ctx.md.comm = comm_.get();
    }
    // Binding does not touch the Frame's row extents: the Frame owns
    // `nlocal` / `nghost`, `Comm::exchange` and `Comm::borders` reset and
    // rebuild the ghost rows at the first rebuild boundary, and an electronic
    // method binding the shared Frame to its private backend mid-pass must
    // not discard the halo rows the hosting Runtime's reverse pass still owns.

#ifdef ATV_HAS_MPI
    if (ctx.mpi_size > 1 && dd_->cart_comm != MPI_COMM_NULL) {
        if (bound_frame_ && bound_frame_ != &frame)
            dd_->partitioned = false;
        // First call: dd has no prior partition → do the full
        // initial scan that classifies the global frame into per-rank slabs.
        //
        // Subsequent calls (e.g. `adopt` across per-stage driver rebuilds):
        // dd already holds the previous stage's partition state. Re-running
        // `partition_atoms` here would iterate only `[0, n_local)` and DROP any
        // atom that has drifted across a slab boundary since the last
        // `Comm::exchange`. Skip the re-partition; refresh slab bounds against
        // the (possibly box-changed) current box and let the next exchange
        // absorb any drift.
        if (!dd_->partitioned) {
            dd_->init_topology(MPI_COMM_WORLD, frame.box.face_height(0),
                               frame.box.face_height(1),
                               frame.box.face_height(2));
            dd_->partition_atoms(frame);
            // Drop the Frame-local tag index: under decomposition the
            // authoritative tag->row map is `dd->tag_to_local`; the local index
            // still holds the PRE-partition identity mapping, so every entry now
            // points at a different atom than the one it names.
            //
            // A component that declared `requires_tag_index()` still needs one,
            // and needs it CORRECT — so rebuild it over this rank's owned rows
            // rather than leaving it absent. Dropping it and letting the
            // consumer notice is what forced `FixSettle` to carry its own O(N)
            // re-establish pass.
            frame.release_tag_index();
            if (tag_index_required)
                frame.rebuild_tag_index(0, frame.nlocal());
        } else {
            dd_->compute_sublo_subhi(frame.box);
        }
        release_tag_maps_if_unused(frame, tag_index_required);
        bound_frame_ = &frame;
        return;
    }
#else
    (void)ctx;
#endif
    bound_frame_ = &frame;
    dd_->bind(frame);
    dd_->compute_sublo_subhi(frame.box);
    release_tag_maps_if_unused(frame, tag_index_required);
}

void DomainRuntime::release_tag_maps_if_unused(cpu::Frame& frame,
                                               bool tag_index_required) {
    // `has_bonded()` alone was the predicate, and it is not the question. A
    // component can name atoms by global tag and carry no topology whatsoever
    // — `FixSettle` is handed water triples as tags — so "no bonds" does not
    // imply "nothing resolves tags". Ask the driver tree instead; bonded
    // topology remains a sufficient reason on its own.
    if (tag_index_required || frame.has_bonded())
        return;
    // Nothing resolves global tags at run time. The loaders may have populated
    // the local index before topology turned out empty; drop it and the DD
    // mirror (~50 B/atom each).
    frame.release_tag_index();
    if (dd_) {
        std::unordered_map<TagInt, int>{}.swap(dd_->tag_to_local);
    }
}

void DomainRuntime::release_comm(Context& ctx) {
    comm_.reset();
    ctx.md.comm = nullptr;
    ctx.md.dd = nullptr;
    // Keep the partition topology and frame identity for continuation.
    // CommBrick owns session-specific wires and row-indexed swap lists; the
    // borrowed domain communicator has no transport storage to release.
}

std::size_t DomainRuntime::memory_bytes() const {
    std::size_t n = 0;
    if (dd_)
        n += dd_->tag_to_local.size() * 32
             + dd_->tag_to_local.bucket_count() * sizeof(void*)
             + dd_->ghost_tags.capacity() * sizeof(std::int64_t);
    if (comm_)
        n += comm_->memory_bytes();
    return n;
}

} // namespace communication
} // namespace cpu
} // namespace ATV
