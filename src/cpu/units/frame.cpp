#include "atomiverse/cpu/units/frame.h"
#include "atomiverse/core/field_units.h"
#include "atomiverse/cpu/frame/frame.h"
#include <algorithm>
#include <type_traits>
#include <utility>
namespace ATV::units {
UnitSystem for_frame(UnitSystem run, const cpu::Frame& frame) {
    if (run.style == Style::lj) {
        const auto reference = reduced_reference(frame);
        if (reference)
            run.bind_reference(*reference);
    }
    return run;
}
bool has_frame_units(const cpu::Frame& frame) {
    return frame.meta.contains("units_style");
}
std::string_view frame_profile(const cpu::Frame& frame) {
    if (!has_frame_units(frame))
        throw std::invalid_argument(
            "Frame has no units: declare units_style before exchanging data");
    const auto& name =
        frame.meta.get(MetaKey<std::string>{"units_style", true});
    if (name != "electronic")
        (void)parse_style(name);
    return name;
}
Style frame_style(const cpu::Frame& frame) {
    return parse_style(frame_profile(frame));
}
void require_frame_units(const cpu::Frame& frame, Style expected) {
    if (frame_profile(frame) != style_name(expected))
        throw std::invalid_argument("Frame units disagree with run units; use "
                                    "units::convert at the boundary");
}
void annotate(cpu::Frame& frame, std::string_view profile) {
    if (profile != "electronic")
        (void)parse_style(profile);
    const MetaKey<std::string> key{"units_style", true};
    const std::string name(profile);
    if (has_frame_units(frame))
        frame.meta.set(key, name);
    else
        frame.meta.insert(key, name);
}
using namespace ATV::cpu;

void annotate_reduced_reference(cpu::Frame& frame,
                                const ReducedReference& reference) {
    reference.validate();
    if (frame_profile(frame) != "lj")
        throw std::invalid_argument(
            "Frame: reduced reference belongs to an LJ profile");
    auto put = [&frame]<class T>(const char* name, T value) {
        const MetaKey<T> key{name, true};
        if (frame.meta.contains(name))
            frame.meta.set(key, std::move(value));
        else
            frame.meta.insert(key, std::move(value));
    };
    put("units_lj_identity", reference.identity);
    put("units_lj_mass_kg", reference.mass_kg);
    put("units_lj_length_m", reference.length_m);
    put("units_lj_energy_j", reference.energy_j);
}

std::optional<ReducedReference> reduced_reference(const cpu::Frame& frame) {
    const bool identity = frame.meta.contains("units_lj_identity");
    const bool mass = frame.meta.contains("units_lj_mass_kg");
    const bool length = frame.meta.contains("units_lj_length_m");
    const bool energy = frame.meta.contains("units_lj_energy_j");
    if (!(identity || mass || length || energy))
        return {};
    if (!(identity && mass && length && energy))
        throw std::invalid_argument("Frame: incomplete LJ reference metadata");
    units::ReducedReference reference{
        frame.meta.get(MetaKey<std::string>{"units_lj_identity", true}),
        frame.meta.get(MetaKey<double>{"units_lj_mass_kg", true}),
        frame.meta.get(MetaKey<double>{"units_lj_length_m", true}),
        frame.meta.get(MetaKey<double>{"units_lj_energy_j", true})};
    reference.validate();
    return reference;
}

cpu::Frame convert(cpu::Frame frame, std::string_view target,
                   const std::optional<ReducedReference>& target_reference) {
    const auto source = frame_profile(frame);
    if (target != "electronic")
        (void)units::parse_style(target);
    if (target_reference) {
        target_reference->validate();
        if (target != "lj")
            throw std::invalid_argument(
                "Frame: target reference requires LJ units");
    }
    const auto source_reference = reduced_reference(frame);
    if (source == target
        && (target != "lj" || !target_reference
            || source_reference == target_reference))
        return frame;
    const auto from =
        units::PhysicalUnits::from_style(source, source_reference);
    const auto to = units::PhysicalUnits::from_style(target, target_reference);
    cpu::Frame out(std::move(
        frame)); // All validation/conversion happens on a private copy.
    for (auto& [block_name, block] : out.blocks) {
        for (const auto& col : block.resolve_columns()) {
            if (col.dtype != ColDType::Storage && col.dtype != ColDType::Acc)
                continue;
            if (block_name != "atoms" || col.wire_id == kDynamicFieldId)
                throw std::invalid_argument(
                    "Frame unit conversion: no units for floating column '"
                    + block_name + "." + std::string(col.key) + "'");
            const auto field = static_cast<AtomFieldId>(col.wire_id);
            const double ratio =
                units::convert(1., units::field_unit(field, from),
                               units::field_unit(field, to));
            const std::size_t count =
                static_cast<std::size_t>(std::min(block.rows(), out.n_total()))
                * col.cols;
            auto scale = [count, ratio]<class T>(T* values) {
                for (std::size_t i = 0; i < count; ++i)
                    values[i] = static_cast<T>(values[i] * ratio);
            };
            if (col.dtype == ColDType::Storage)
                scale(static_cast<StorageReal*>(col.write_data));
            else
                scale(static_cast<AccReal*>(col.write_data));
        }
    }
    const double length_ratio = units::convert(1., from.length(), to.length());
    if (out.box.volume() > 0.) {
        Mat3 cell = out.box.cell();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell(i, j) *= length_ratio;
        out.box.set_cell(cell);
    }
    for (const auto& [key, entry] : out.meta.entries()) {
        const auto from_unit = units::metadata_unit(key, from);
        const auto to_unit = units::metadata_unit(key, to);
        std::visit(
            [&]<class T>(const T& value) {
                constexpr bool scalar = std::is_same_v<T, double>;
                constexpr bool array = std::is_same_v<T, MetaF64x3>
                                       || std::is_same_v<T, MetaF64x6>
                                       || std::is_same_v<T, MetaF64x9>;
                if constexpr (scalar || array) {
                    if (!from_unit || !to_unit)
                        throw std::invalid_argument(
                            "Frame unit conversion: no units for metadata '"
                            + key + "'");
                    const double ratio =
                        units::convert(1., *from_unit, *to_unit);
                    T scaled = value;
                    if constexpr (scalar)
                        scaled *= ratio;
                    else
                        for (auto& element : scaled)
                            element *= ratio;
                    out.meta.set(MetaKey<T>{key.c_str(), entry.serializable},
                                 scaled);
                }
            },
            entry.value);
    }
    annotate(out, target);
    if (target_reference)
        annotate_reduced_reference(out, *target_reference);
    return out;
}

void publish_electronic_results(const cpu::Frame& source,
                                cpu::Frame& destination) {
    if (source.nlocal() != destination.nlocal())
        throw std::invalid_argument(
            "electronic evaluation changed the atom count");
    for (const auto& key : {kAtom_fx, kAtom_fy, kAtom_fz}) {
        const auto input = source.span(key);
        auto output = destination.span(key);
        std::copy_n(input.data(), source.nlocal(), output.data());
    }
    if (source.has("atoms", "charge") && destination.has("atoms", "charge")) {
        const auto input = source.span(kAtom_charge);
        auto output = destination.span(kAtom_charge);
        std::copy_n(input.data(), source.nlocal(), output.data());
    }
    for (const auto& [key, entry] : source.meta.entries()) {
        if (!entry.serializable || key == "units_style"
            || key.starts_with("units_lj_"))
            continue;
        std::visit(
            [&]<class T>(const T& value) {
                if constexpr (!std::is_same_v<T, cpu::TagIndex>) {
                    const MetaKey<T> typed{key.c_str(), true};
                    if (destination.meta.contains(key))
                        destination.meta.set(typed, value);
                    else
                        destination.meta.insert(typed, value);
                }
            },
            entry.value);
    }
}

} // namespace ATV::units
