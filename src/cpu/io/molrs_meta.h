/// @file src/cpu/io/molrs_meta.h
/// @brief The single exact typed-meta mapping at the molrs CXX boundary.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "atomiverse/cpu/frame/meta.h"
#include "atomiverse/molrs.hpp"

namespace ATV::cpu::internal {

inline constexpr std::uint32_t kMolrsFrameSchemaVersion = 2;

inline void require_molrs_frame_schema_v2() {
    const std::uint32_t actual = molrs::frame_schema_version();
    if (actual != kMolrsFrameSchemaVersion)
        throw meta_error("Atomiverse requires molrs Frame schema v2; got v"
                         + std::to_string(actual));
}

#define ATV_ASSERT_META_ID(name)                                               \
    static_assert(static_cast<std::uint8_t>(MetaDType::name)                   \
                  == static_cast<std::uint8_t>(molrs::MetaType::name))
ATV_ASSERT_META_ID(Bool);
ATV_ASSERT_META_ID(I32);
ATV_ASSERT_META_ID(I64);
ATV_ASSERT_META_ID(U32);
ATV_ASSERT_META_ID(U64);
ATV_ASSERT_META_ID(F32);
ATV_ASSERT_META_ID(F64);
ATV_ASSERT_META_ID(String);
ATV_ASSERT_META_ID(Bool3);
ATV_ASSERT_META_ID(I32x3);
ATV_ASSERT_META_ID(I64x3);
ATV_ASSERT_META_ID(U32x3);
ATV_ASSERT_META_ID(U64x3);
ATV_ASSERT_META_ID(F32x3);
ATV_ASSERT_META_ID(F64x3);
ATV_ASSERT_META_ID(F32x6);
ATV_ASSERT_META_ID(F64x6);
ATV_ASSERT_META_ID(F32x9);
ATV_ASSERT_META_ID(F64x9);
#undef ATV_ASSERT_META_ID

template <typename T, std::size_t N>
void append_values(rust::Vec<T>& destination, const std::array<T, N>& source) {
    for (const T value : source)
        destination.push_back(value);
}

inline rust::Vec<molrs::MetaEntry> encode_molrs_meta(const MetaMap& meta) {
    require_molrs_frame_schema_v2();
    rust::Vec<molrs::MetaEntry> out;
    for (const SerializableMetaRef& source : meta.serializable_entries()) {
        molrs::MetaEntry entry{};
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
        case MetaDType::F32:
            entry.dtype = molrs::MetaType::F32;
            entry.f32_value = std::get<float>(*source.value);
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
        case MetaDType::F32x3:
            entry.dtype = molrs::MetaType::F32x3;
            append_values(entry.f32_values, std::get<MetaF32x3>(*source.value));
            break;
        case MetaDType::F64x3:
            entry.dtype = molrs::MetaType::F64x3;
            append_values(entry.f64_values, std::get<MetaF64x3>(*source.value));
            break;
        case MetaDType::F32x6:
            entry.dtype = molrs::MetaType::F32x6;
            append_values(entry.f32_values, std::get<MetaF32x6>(*source.value));
            break;
        case MetaDType::F64x6:
            entry.dtype = molrs::MetaType::F64x6;
            append_values(entry.f64_values, std::get<MetaF64x6>(*source.value));
            break;
        case MetaDType::F32x9:
            entry.dtype = molrs::MetaType::F32x9;
            append_values(entry.f32_values, std::get<MetaF32x9>(*source.value));
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

inline void decode_molrs_meta(const rust::Vec<molrs::MetaEntry>& source,
                              MetaMap& destination) {
    require_molrs_frame_schema_v2();
    for (const molrs::MetaEntry& entry : source) {
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
        case molrs::MetaType::F32:
            destination.upsert_serializable(key, entry.f32_value);
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
        case molrs::MetaType::F32x3:
            destination.upsert_serializable(
                key, exact_array<MetaF32x3>(entry.f32_values, key));
            break;
        case molrs::MetaType::F64x3:
            destination.upsert_serializable(
                key, exact_array<MetaF64x3>(entry.f64_values, key));
            break;
        case molrs::MetaType::F32x6:
            destination.upsert_serializable(
                key, exact_array<MetaF32x6>(entry.f32_values, key));
            break;
        case molrs::MetaType::F64x6:
            destination.upsert_serializable(
                key, exact_array<MetaF64x6>(entry.f64_values, key));
            break;
        case molrs::MetaType::F32x9:
            destination.upsert_serializable(
                key, exact_array<MetaF32x9>(entry.f32_values, key));
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
