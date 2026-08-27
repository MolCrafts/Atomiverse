/// @file core/schema.h
/// @brief `Schema` — what fields this system carries, and whether a given
///        Frame satisfies that.
///
/// A **system-level** declaration owned by the Driver, not a communication
/// detail. It answers "what is this chemical model made of": positions
/// always, charges if something is charged, velocities if something reads
/// them. It is **derived, not hand-written** — every mounted component
/// declares what it needs (`Component::declare_fields`) and the Driver folds
/// the declarations at `build()`, the same shape as the existing
/// `Component::default_bindings()` collection.
///
/// **API shape follows pydantic**, and the three borrowed ideas are the
/// reason it is worth borrowing:
///
///   * a field is declared **with its type**, not as a bare name, so a
///     mismatch is caught at declaration rather than at the first memcpy;
///   * validation is `model_validate`-shaped: it checks a Frame against the
///     model and reports **every** problem, not the first one, because
///     fixing errors one build at a time is what makes a schema hateful;
///   * every error carries its **location** — here, which component asked
///     for the field — so the message says who to talk to.
///
/// The relationship to `Frame` is **validation, not construction**. A Frame
/// is a mapping: it arrives with whatever its source gave it (a data file,
/// `Block::insert`, another rank). Nothing builds a Frame from a Schema.
#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "atomiverse/core/frame_schema.h"

namespace ATV {

/// @brief One declared field — a pydantic model field.
struct Field {
    /// @brief Column key, as it appears in the atoms Block.
    std::string_view name;
    /// @brief Element type the declarer expects.
    ColDType dtype = ColDType::Storage;
    /// @brief Components per item; 1 for a scalar, 3 for a fixed triple.
    int cols = 1;
    /// @brief Who asked for it. The `loc` of a pydantic error: it turns
    ///        "field missing" into "PairDpd needs this and it is not there".
    std::string_view required_by;
};

/// @brief One validation failure — pydantic's `{loc, msg}`.
struct SchemaError {
    /// @brief Field the problem is about.
    std::string_view field;
    /// @brief Component that declared it.
    std::string_view required_by;
    /// @brief What is wrong, phrased so it can be acted on.
    std::string reason;
};

/// @brief The set of fields the system must carry, in declaration order.
///
/// Declaration order is Plan order, which is identical on every rank, so two
/// ranks folding the same component set produce the same Schema without
/// sorting and without a collective.
class Schema {
  public:
    /// @brief Declare a typed field, the way a pydantic model declares one.
    ///
    /// Idempotent for an identical redeclaration — two components needing
    /// `charge` is normal. A redeclaration with a **different** type or
    /// width is a contradiction between two components and is reported by
    /// `conflicts()`; it is not silently merged.
    ///
    /// @param key Typed column key; carries both the name and the dtype.
    /// @param required_by Declaring component's name, for error messages.
    template <typename T>
    void require(const ColKey<T>& key, std::string_view required_by) {
        require(key.key, key.dtype, 1, required_by);
    }

    /// @brief Declare a field by name, type and width.
    /// @param name Column key.
    /// @param dtype Element type.
    /// @param cols Components per item.
    /// @param required_by Declaring component's name.
    void require(std::string_view name, ColDType dtype, int cols,
                 std::string_view required_by) {
        if (const Field* seen = find(name)) {
            if (seen->dtype != dtype || seen->cols != cols)
                conflicts_.push_back(SchemaError{
                    name, required_by,
                    "declared as " + std::string(dtype_name(dtype)) + "["
                        + std::to_string(cols) + "] here but as "
                        + std::string(dtype_name(seen->dtype)) + "["
                        + std::to_string(seen->cols) + "] by "
                        + std::string(seen->required_by)
                        + "; two components disagree about what this field "
                          "is"});
            return;
        }
        fields_.push_back(Field{name, dtype, cols, required_by});
    }

    /// @brief Declare several same-typed scalar fields at once.
    /// @param names Column keys.
    /// @param dtype Element type shared by all of them.
    /// @param required_by Declaring component's name.
    void require_all(std::initializer_list<std::string_view> names,
                     ColDType dtype, std::string_view required_by) {
        for (std::string_view n : names)
            require(n, dtype, 1, required_by);
    }

    /// @brief The declared field, or null.
    /// @param name Column key.
    /// @return Pointer into the field list; invalidated by further requires.
    const Field* find(std::string_view name) const {
        auto it = std::find_if(fields_.begin(), fields_.end(),
                               [&](const Field& f) { return f.name == name; });
        return it == fields_.end() ? nullptr : &*it;
    }

    /// @brief Whether the system carries `name`.
    bool contains(std::string_view name) const { return find(name) != nullptr; }

    /// @brief Declared fields, in declaration order — pydantic's
    ///        `model_fields`.
    const std::vector<Field>& fields() const noexcept { return fields_; }

    /// @brief Contradictions found while declaring: two components asking
    ///        for the same field with different types.
    const std::vector<SchemaError>& conflicts() const noexcept {
        return conflicts_;
    }

    std::size_t size() const noexcept { return fields_.size(); }
    bool empty() const noexcept { return fields_.empty(); }
    auto begin() const noexcept { return fields_.begin(); }
    auto end() const noexcept { return fields_.end(); }

    /// @brief Drop everything, keeping heap capacity.
    void clear() noexcept {
        fields_.clear();
        conflicts_.clear();
    }

    /// @brief Order-sensitive equality — two ranks must agree on the ORDER,
    ///        not merely the membership, or one packs what the other cannot
    ///        read.
    bool operator==(const Schema& other) const {
        if (fields_.size() != other.fields_.size())
            return false;
        for (std::size_t i = 0; i < fields_.size(); ++i) {
            const Field& a = fields_[i];
            const Field& b = other.fields_[i];
            if (a.name != b.name || a.dtype != b.dtype || a.cols != b.cols)
                return false;
        }
        return true;
    }
    bool operator!=(const Schema& other) const { return !(*this == other); }

    /// @brief Human name of an element type, for messages.
    static std::string_view dtype_name(ColDType d) {
        switch (d) {
        case ColDType::Storage:
            return "StorageReal";
        case ColDType::Acc:
            return "AccReal";
        case ColDType::Tag:
            return "TagInt";
        case ColDType::Int:
            return "int";
        case ColDType::Int8:
            return "int8";
        }
        return "?";
    }

  private:
    std::vector<Field> fields_;
    std::vector<SchemaError> conflicts_;
};

/// @brief Fields the engine carries whatever the components declare.
///
/// The same idea as LAMMPS' `AtomVec::default_comm` / `default_border` /
/// `default_reverse` (`src/atom_vec.cpp:36-40`, checkout `78a7ddf`): a few
/// fields are the engine's own product rather than any component's request.
///
/// Copyright (2003) Sandia Corporation. GPL-2.0-only.
namespace base_fields {

/// @brief Ghost positions — producing them is what the halo is for.
inline void forward(Schema& s) {
    s.require_all({"x", "y", "z"}, ColDType::Storage, "halo");
}

/// @brief Ghost force accumulators — what the reverse pass folds back.
inline void reverse(Schema& s) {
    s.require_all({"fx", "fy", "fz"}, ColDType::Acc, "halo");
}

/// @brief What a ghost row needs before anything can use it: the tag (bonded
/// resolution) and the type (every pair style indexes by it), plus positions.
inline void border(Schema& s) {
    forward(s);
    s.require("tag", ColDType::Tag, 1, "halo");
    s.require("type", ColDType::Int, 1, "halo");
}

} // namespace base_fields
} // namespace ATV
