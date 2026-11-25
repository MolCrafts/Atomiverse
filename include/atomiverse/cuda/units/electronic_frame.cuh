#pragma once
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/frame/frame.cuh"
#include <memory>

namespace ATV::cuda::unit_boundary {
/// Persistent adapter at a fixed-convention electronic solver boundary.
/// Compatible units borrow the parent's Frame without allocations or kernels.
/// Other styles use two fused transformations per evaluation, never per SCF
/// iteration, and publish into the parent's existing allocations.
class ElectronicFrame {
    struct Storage;
    std::unique_ptr<Storage> storage_;

  public:
    ElectronicFrame();
    ~ElectronicFrame();
    void configure(Context&, Frame&, bool electronic);
    class Scope {
        ElectronicFrame& adapter_;
        Context& context_;
        Frame& parent_;
        std::optional<Frame> parent_snapshot_;
        std::optional<ATV::units::UnitSystem> saved_units_;
        std::optional<Box> saved_box_;
        const ATV::units::UnitSystem* inherited_ = nullptr;

      public:
        Scope(ElectronicFrame&, Context&, Frame&, bool refresh = true);
        ~Scope();
        Scope(const Scope&) = delete;
        Frame& frame();
        void publish();
    };
};
} // namespace ATV::cuda::unit_boundary
