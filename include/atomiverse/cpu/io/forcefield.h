/// @file cpu/io/forcefield.h
/// @brief ForceField — classical style tables (molrs/molpy-aligned API).
///
/// **Data only.** Not a Driver, not a Component. Mirrors molrs
/// `ForceField::def_pairstyle` / `def_bondstyle` / … + `def_type` /
/// `set_special_bonds`. Does not install into a Plan.
///
/// Engine-side: read styles/types, then construct `Potential` components and
/// register them with `Driver::set_evaluator` / `on(phase).add`.
///
/// Units here are Atomiverse engineering (eV, Å, rad) unless noted.
#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ATV {
namespace cpu {

/// @brief Key/value parameter bag for one style or type (molrs Params-shaped).
struct StyleParams {
    std::map<std::string, double> num;
    std::map<std::string, std::string> str;

    StyleParams() = default;
    StyleParams(std::initializer_list<std::pair<const char*, double>> pairs) {
        for (auto& p : pairs)
            num[p.first] = p.second;
    }

    double get(std::string_view key, double fallback) const {
        auto it = num.find(std::string(key));
        return it == num.end() ? fallback : it->second;
    }
    double require(std::string_view key) const {
        auto it = num.find(std::string(key));
        if (it == num.end())
            throw std::invalid_argument("StyleParams: missing key '"
                                        + std::string(key) + "'");
        return it->second;
    }
};

/// @brief One named style under a category (pair / bond / angle / …).
struct Style {
    std::string name;
    StyleParams params; ///< style-level params (e.g. pair cutoff)
    /// type name → params (e.g. "O-H" → {k, r0})
    std::map<std::string, StyleParams> types;
};

/// @brief Classical force-field data (molrs-aligned).
class ForceField {
  public:
    explicit ForceField(std::string name = "ff") : name_(std::move(name)) {}

    const std::string& name() const noexcept { return name_; }

    // ── def_*style (molrs ForceField) ─────────────────────────────────

    /// @brief Register or retrieve pair style (e.g. `"lj/cut"`, `"coul/cut"`).
    Style& def_pairstyle(std::string name, StyleParams params = {}) {
        return def_style_("pair", std::move(name), std::move(params));
    }

    /// @brief Register or retrieve bond style (e.g. `"harmonic"`).
    Style& def_bondstyle(std::string name, StyleParams params = {}) {
        return def_style_("bond", std::move(name), std::move(params));
    }

    Style& def_anglestyle(std::string name, StyleParams params = {}) {
        return def_style_("angle", std::move(name), std::move(params));
    }

    Style& def_dihedralstyle(std::string name, StyleParams params = {}) {
        return def_style_("dihedral", std::move(name), std::move(params));
    }

    Style& def_kspacestyle(std::string name, StyleParams params = {}) {
        return def_style_("kspace", std::move(name), std::move(params));
    }

    /// @brief Add a type under an existing style (molrs `def_type` / style.def_*type).
    /// @param category `"pair"` / `"bond"` / `"angle"` / …
    /// @param style    style name already registered
    /// @param type_name dash-form type id (e.g. `"O"`, `"O-H"`, `"H-O-H"`)
    void def_type(std::string_view category, std::string_view style,
                  std::string type_name, StyleParams params) {
        Style& s = require_style_(category, style);
        s.types[std::move(type_name)] = std::move(params);
    }

    /// @brief LAMMPS-style 1-2 / 1-3 / 1-4 scales (lj and coul shared in v1).
    void set_special_bonds(double s12, double s13, double s14) {
        special_[0] = s12;
        special_[1] = s13;
        special_[2] = s14;
    }

    double special_lj(int i) const { return special_[i]; }
    double special_coul(int i) const { return special_[i]; }

    const std::vector<Style>& styles(std::string_view category) const {
        static const std::vector<Style> empty;
        auto it = styles_.find(std::string(category));
        return it == styles_.end() ? empty : it->second;
    }

    Style* find_style(std::string_view category, std::string_view name) {
        auto it = styles_.find(std::string(category));
        if (it == styles_.end())
            return nullptr;
        for (auto& s : it->second)
            if (s.name == name)
                return &s;
        return nullptr;
    }

    const Style* find_style(std::string_view category,
                            std::string_view name) const {
        return const_cast<ForceField*>(this)->find_style(category, name);
    }

    /// @brief Neighbor cutoff hint: max pair/coul style cutoff param [Å].
    double neighbor_cutoff() const {
        double c = 0.0;
        for (const auto& s : styles("pair")) {
            double cut = s.params.get("cutoff", 0.0);
            if (cut > c)
                c = cut;
        }
        return c;
    }

  private:
    Style& def_style_(std::string category, std::string name,
                      StyleParams params) {
        auto& vec = styles_[category];
        for (auto& s : vec) {
            if (s.name == name) {
                if (!params.num.empty() || !params.str.empty())
                    s.params = std::move(params);
                return s;
            }
        }
        vec.push_back(Style{std::move(name), std::move(params), {}});
        return vec.back();
    }

    Style& require_style_(std::string_view category, std::string_view name) {
        Style* s = find_style(category, name);
        if (!s)
            throw std::invalid_argument(
                "ForceField: unknown style '" + std::string(category) + "/"
                + std::string(name) + "' — def_*style first");
        return *s;
    }

    std::string name_;
    std::map<std::string, std::vector<Style>> styles_;
    double special_[3] = {0.0, 0.0, 0.0};
};

} // namespace cpu
} // namespace ATV
