/// @file cpu/frame/meta.h
/// @brief Owning typed CPU Frame metadata with stable typed references.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/column.h"

namespace ATV {
namespace cpu {

using TagIndex = std::unordered_map<TagInt, int>;
using MetaValue =
    std::variant<bool, std::int32_t, std::int64_t, std::uint32_t, std::uint64_t,
                 float, double, std::string, MetaBool3, MetaI32x3, MetaI64x3,
                 MetaU32x3, MetaU64x3, MetaF32x3, MetaF64x3, MetaF32x6,
                 MetaF64x6, MetaF32x9, MetaF64x9, TagIndex>;

class meta_error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

template <typename T>
class MetaRef {
  public:
    using value_type = T;

    MetaRef() = default;
    MetaRef(T* ptr, const std::uint64_t* generation, std::uint64_t captured)
        : ptr_(ptr), generation_(generation), captured_(captured) {}

    void validate() const {
        if (generation_ != nullptr && *generation_ != captured_)
            throw stale_handle_error("resolved Frame meta handle is stale "
                                     "after a structural mutation");
    }

    bool stale() const noexcept {
        return generation_ != nullptr && *generation_ != captured_;
    }

    T& get() const {
        validate();
        return *ptr_;
    }

    T* data() const {
        validate();
        return ptr_;
    }

    operator T&() const { return get(); }
    T* operator->() const { return data(); }

  private:
    T* ptr_ = nullptr;
    const std::uint64_t* generation_ = nullptr;
    std::uint64_t captured_ = 0;
};

struct MetaEntry {
    MetaValue value;
    bool serializable = true;
};

enum class MetaDType : std::uint8_t {
    Bool = 0,
    I32 = 1,
    I64 = 2,
    U32 = 3,
    U64 = 4,
    F32 = 5,
    F64 = 6,
    String = 7,
    Bool3 = 8,
    I32x3 = 9,
    I64x3 = 10,
    U32x3 = 11,
    U64x3 = 12,
    F32x3 = 13,
    F64x3 = 14,
    F32x6 = 15,
    F64x6 = 16,
    F32x9 = 17,
    F64x9 = 18,
    TagIndexLocal = 255,
};

MetaDType meta_dtype(const MetaValue& value) noexcept;

struct SerializableMetaRef {
    std::string_view key;
    MetaDType dtype;
    const MetaValue* value;
};

class MetaMap {
  public:
    using Storage = std::map<std::string, MetaEntry, std::less<>>;

    MetaMap() = default;
    MetaMap(const MetaMap& other);
    MetaMap& operator=(const MetaMap& other);
    MetaMap(MetaMap&& other) noexcept;
    MetaMap& operator=(MetaMap&& other) noexcept;

    template <typename T>
    void insert(MetaKey<T> key, std::remove_const_t<T> value) {
        insert_impl_(key.key, MetaValue{std::move(value)}, key.serializable);
    }

    template <typename T>
    MetaRef<T> ref(MetaKey<T> key) {
        static_assert(!std::is_const_v<T>);
        using U = std::remove_const_t<T>;
        auto it = entries_.find(key.key);
        if (it == entries_.end())
            throw meta_error("Frame meta has no key '" + std::string(key.key)
                             + "'");
        U* value = std::get_if<U>(&it->second.value);
        if (value == nullptr)
            throw meta_error("Frame meta dtype mismatch for key '"
                             + std::string(key.key) + "'");
        return MetaRef<T>{value, &generation_, generation_};
    }

    template <typename T>
    MetaRef<const T> ref(MetaKey<T> key) const {
        auto it = entries_.find(key.key);
        if (it == entries_.end())
            throw meta_error("Frame meta has no key '" + std::string(key.key)
                             + "'");
        const T* value = std::get_if<T>(&it->second.value);
        if (value == nullptr)
            throw meta_error("Frame meta dtype mismatch for key '"
                             + std::string(key.key) + "'");
        return MetaRef<const T>{value, &generation_, generation_};
    }

    template <typename T>
    T& get(MetaKey<T> key) {
        return ref<T>(key).get();
    }

    template <typename T>
    const T& get(MetaKey<T> key) const {
        return ref(key).get();
    }

    template <typename T>
    void set(MetaKey<T> key, T value) {
        ref<T>(key).get() = std::move(value);
    }

    bool contains(std::string_view key) const noexcept {
        return entries_.find(key) != entries_.end();
    }

    const Storage& entries() const noexcept { return entries_; }
    std::uint64_t generation() const noexcept { return generation_; }
    const std::shared_ptr<ResolutionState>& resolution_state() const noexcept {
        return resolution_state_;
    }

    /// Deterministic key-ordered serializable plan. Backend-local counts,
    /// strides, capacity and tag index are excluded by construction.
    std::vector<SerializableMetaRef> serializable_entries() const;
    std::uint64_t serializable_fingerprint() const;

    /// Copy the exact serializable schema/value set while retaining every
    /// destination-local entry. Existing serializable destination keys must
    /// be present with the same explicit dtype in source; source-only
    /// serializable keys are inserted canonically.
    void copy_serializable_from(const MetaMap& source);

    /// Restore one runtime-keyed boundary value without weakening the typed
    /// metadata contract. Existing keys must already be serializable and keep
    /// their exact dtype; missing keys become serializable schema entries.
    /// Backend-local values (notably TagIndexLocal) never cross a boundary.
    void upsert_serializable(std::string key, MetaValue value);

  private:
    void insert_impl_(std::string key, MetaValue value, bool serializable) {
        if (serializable && meta_dtype(value) == MetaDType::TagIndexLocal)
            throw meta_error(
                "backend-local tag index cannot be a serializable meta key");
        auto [it, inserted] = entries_.emplace(
            std::move(key), MetaEntry{std::move(value), serializable});
        if (!inserted)
            throw meta_error("duplicate Frame meta key '" + it->first + "'");
        ++generation_;
    }

    Storage entries_;
    std::uint64_t generation_ = 1;
    std::shared_ptr<ResolutionState> resolution_state_ =
        std::make_shared<ResolutionState>();
};

} // namespace cpu
} // namespace ATV
