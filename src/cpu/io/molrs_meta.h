/// @file src/cpu/io/molrs_meta.h
/// @brief The single exact typed-meta mapping at the molrs CXX boundary.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/cpu/frame/meta.h"
#include "atomiverse/molpy.hpp"

namespace ATV::cpu::internal {

#define ATV_ASSERT_META_ID(name)                                               \
    static_assert(static_cast<std::uint8_t>(MetaDType::name)                   \
                  == static_cast<std::uint8_t>(molrs::MetaType::name))
ATV_ASSERT_META_ID(Bool);
ATV_ASSERT_META_ID(I32);
ATV_ASSERT_META_ID(I64);
ATV_ASSERT_META_ID(U32);
ATV_ASSERT_META_ID(U64);
ATV_ASSERT_META_ID(F64);
ATV_ASSERT_META_ID(String);
ATV_ASSERT_META_ID(Bool3);
ATV_ASSERT_META_ID(I32x3);
ATV_ASSERT_META_ID(I64x3);
ATV_ASSERT_META_ID(U32x3);
ATV_ASSERT_META_ID(U64x3);
ATV_ASSERT_META_ID(F64x3);
ATV_ASSERT_META_ID(F64x6);
ATV_ASSERT_META_ID(F64x9);
#undef ATV_ASSERT_META_ID

template <typename T, std::size_t N>
void append_values(rust::Vec<T>& destination, const std::array<T, N>& source) {
    for (const T value : source)
        destination.push_back(value);
}

// A std::vector, not a rust::Vec: the bridge passes metadata one value at a
// time (`frame_set_meta`), so cxx emits no `rust::Vec<KeyedMetaValue>`.
inline std::vector<molrs::KeyedMetaValue>
encode_molrs_meta(const MetaMap& meta) {
    std::vector<molrs::KeyedMetaValue> out;
    for (const SerializableMetaRef& source : meta.serializable_entries()) {
        molrs::KeyedMetaValue entry{};
        entry.key = rust::String(std::string(source.key));
        switch (source.dtype) {
        case MetaDType::Bool:
            entry.dtype = molrs::MetaType::Bool;
            entry.bool_value = std::get<bool>(*source.value);
            break;
        case MetaDType::I32:
            entry.dtype = molrs::MetaType::I32;
            entry.i32_value = std::get<std::int32_t>(*source.value);
            break;
        case MetaDType::I64:
            entry.dtype = molrs::MetaType::I64;
            entry.i64_value = std::get<std::int64_t>(*source.value);
            break;
        case MetaDType::U32:
            entry.dtype = molrs::MetaType::U32;
            entry.u32_value = std::get<std::uint32_t>(*source.value);
            break;
        case MetaDType::U64:
            entry.dtype = molrs::MetaType::U64;
            entry.u64_value = std::get<std::uint64_t>(*source.value);
            break;
        case MetaDType::F64:
            entry.dtype = molrs::MetaType::F64;
            entry.f64_value = std::get<double>(*source.value);
            break;
        case MetaDType::String:
            entry.dtype = molrs::MetaType::String;
            entry.string_value =
                rust::String(std::get<std::string>(*source.value));
            break;
        case MetaDType::Bool3: {
            entry.dtype = molrs::MetaType::Bool3;
            for (bool value : std::get<MetaBool3>(*source.value))
                entry.bool_values.push_back(value ? 1u : 0u);
            break;
        }
        case MetaDType::I32x3:
            entry.dtype = molrs::MetaType::I32x3;
            append_values(entry.i32_values, std::get<MetaI32x3>(*source.value));
            break;
        case MetaDType::I64x3:
            entry.dtype = molrs::MetaType::I64x3;
            append_values(entry.i64_values, std::get<MetaI64x3>(*source.value));
            break;
        case MetaDType::U32x3:
            entry.dtype = molrs::MetaType::U32x3;
            append_values(entry.u32_values, std::get<MetaU32x3>(*source.value));
            break;
        case MetaDType::U64x3:
            entry.dtype = molrs::MetaType::U64x3;
            append_values(entry.u64_values, std::get<MetaU64x3>(*source.value));
            break;
        case MetaDType::F64x3:
            entry.dtype = molrs::MetaType::F64x3;
            append_values(entry.f64_values, std::get<MetaF64x3>(*source.value));
            break;
        case MetaDType::F64x6:
            entry.dtype = molrs::MetaType::F64x6;
            append_values(entry.f64_values, std::get<MetaF64x6>(*source.value));
            break;
        case MetaDType::F64x9:
            entry.dtype = molrs::MetaType::F64x9;
            append_values(entry.f64_values, std::get<MetaF64x9>(*source.value));
            break;
        case MetaDType::TagIndexLocal:
            throw meta_error(
                "backend-local tag index reached the molrs meta encoder");
        }
        out.push_back(std::move(entry));
    }
    return out;
}

template <typename Array, typename RustVec>
Array exact_array(const RustVec& values, const std::string& key) {
    Array out{};
    if (values.size() != out.size())
        throw meta_error("molrs meta '" + key + "' has invalid vector width");
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<typename Array::value_type>(values[i]);
    return out;
}

/// Copy every metadata value of a molrs frame into `destination`, in the
/// frame's insertion order (molrs `frame_meta_keys` + `frame_get_meta`).
inline void decode_molrs_meta(const molrs::FrameRef& source,
                              MetaMap& destination) {
    for (const rust::String& raw_key : molrs::frame_meta_keys(source)) {
        const molrs::KeyedMetaValue entry =
            molrs::frame_get_meta(source, {raw_key.data(), raw_key.size()});
        const std::string key(entry.key.data(), entry.key.size());
        switch (entry.dtype) {
        case molrs::MetaType::Bool:
            destination.upsert_serializable(key, entry.bool_value);
            break;
        case molrs::MetaType::I32:
            destination.upsert_serializable(key, entry.i32_value);
            break;
        case molrs::MetaType::I64:
            destination.upsert_serializable(key, entry.i64_value);
            break;
        case molrs::MetaType::U32:
            destination.upsert_serializable(key, entry.u32_value);
            break;
        case molrs::MetaType::U64:
            destination.upsert_serializable(key, entry.u64_value);
            break;
        case molrs::MetaType::F64:
            destination.upsert_serializable(key, entry.f64_value);
            break;
        case molrs::MetaType::String:
            destination.upsert_serializable(
                key, std::string(entry.string_value.data(),
                                 entry.string_value.size()));
            break;
        case molrs::MetaType::Bool3:
            destination.upsert_serializable(
                key, exact_array<MetaBool3>(entry.bool_values, key));
            break;
        case molrs::MetaType::I32x3:
            destination.upsert_serializable(
                key, exact_array<MetaI32x3>(entry.i32_values, key));
            break;
        case molrs::MetaType::I64x3:
            destination.upsert_serializable(
                key, exact_array<MetaI64x3>(entry.i64_values, key));
            break;
        case molrs::MetaType::U32x3:
            destination.upsert_serializable(
                key, exact_array<MetaU32x3>(entry.u32_values, key));
            break;
        case molrs::MetaType::U64x3:
            destination.upsert_serializable(
                key, exact_array<MetaU64x3>(entry.u64_values, key));
            break;
        case molrs::MetaType::F64x3:
            destination.upsert_serializable(
                key, exact_array<MetaF64x3>(entry.f64_values, key));
            break;
        case molrs::MetaType::F64x6:
            destination.upsert_serializable(
                key, exact_array<MetaF64x6>(entry.f64_values, key));
            break;
        case molrs::MetaType::F64x9:
            destination.upsert_serializable(
                key, exact_array<MetaF64x9>(entry.f64_values, key));
            break;
        default:
            throw meta_error("molrs returned an unknown typed-meta ID");
        }
    }
}

} // namespace ATV::cpu::internal
