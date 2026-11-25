/// @file cpu/units/frame.h
/// Unit operations at data boundaries. Frame owns data, not unit behavior.
#pragma once
#include "atomiverse/core/physical_units.h"
#include "atomiverse/core/unit_system.h"
namespace ATV::cpu {
class Frame;
}
namespace ATV::units {
bool has_frame_units(const cpu::Frame& frame);
std::string_view frame_profile(const cpu::Frame& frame);
UnitSystem for_frame(UnitSystem run, const cpu::Frame& frame);
Style frame_style(const cpu::Frame& frame);
void require_frame_units(const cpu::Frame& frame, Style expected);
/// Annotate existing values without rescaling them.
void annotate(cpu::Frame& frame, std::string_view profile);
void annotate_reduced_reference(cpu::Frame& frame,
                                const ReducedReference& reference);
std::optional<ReducedReference> reduced_reference(const cpu::Frame& frame);
/// Convert a private copy; source buffers are unchanged on success or failure.
cpu::Frame
convert(cpu::Frame frame, std::string_view target,
        const std::optional<ReducedReference>& target_reference = {});
} // namespace ATV::units

#include "atomiverse/cpu/frame/frame.h"
#include <functional>
#include <type_traits>
namespace ATV::units {
/// Publish solver outputs without replacing parent buffers or invalidating
/// column handles held by other components of the same execution plan.
void publish_electronic_results(const cpu::Frame& source,
                                cpu::Frame& destination);
/// Electronic numerical kernels keep one fixed convention. Convert only at
/// their public evaluation boundary, never within electronic iterations.
inline bool needs_conversion(const cpu::Frame& frame, std::string_view target) {
    if (!has_frame_units(frame) || frame_profile(frame) == target)
        return false;
    const auto source = PhysicalUnits::from_style(frame_profile(frame),
                                                  reduced_reference(frame));
    const auto destination = PhysicalUnits::from_style(target);
    // Electronic kernels consume geometry, masses, charges, energy, force
    // and extensive stress. Their nuclear clock is adapted independently.
    if (target == "electronic")
        return source.length().scale != destination.length().scale
               || source.mass().scale != destination.mass().scale
               || source.charge().scale != destination.charge().scale
               || source.energy().scale != destination.energy().scale;
    return true;
}
template <class Function>
decltype(auto) evaluate_in(cpu::Frame& frame, std::string_view target,
                           Function&& function) {
    using Result = std::invoke_result_t<Function, cpu::Frame&>;
    const auto source = frame_profile(frame);
    const auto reference = source == "lj" ? reduced_reference(frame)
                                          : std::optional<ReducedReference>{};
    auto work = convert(frame, target);
    if constexpr (std::is_void_v<Result>) {
        std::invoke(function, work);
        auto restored = convert(std::move(work), source, reference);
        publish_electronic_results(restored, frame);
    } else {
        decltype(auto) result = std::invoke(function, work);
        auto restored = convert(std::move(work), source, reference);
        publish_electronic_results(restored, frame);
        return result;
    }
}
} // namespace ATV::units
