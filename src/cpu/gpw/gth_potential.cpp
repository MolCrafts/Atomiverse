/// @file src/cpu/gpw/gth_potential.cpp
/// @brief GthPotential lookup constructor over the in-tree static tables.

#include "atomiverse/cpu/gpw/gth_potential.h"

#include <stdexcept>
#include <string>

#include "atomiverse/cpu/gpw/basis_data/gpw_tables.h"

namespace ATV {
namespace cpu {
namespace gpw {

GthPotential::GthPotential(molrs::Element element, std::string_view name)
    : element(element), name(name) {
    for (const auto* row : basis_data::kGthPotentials) {
        if (row->z != static_cast<int>(element) || name != row->name)
            continue;
        this->zion = row->zion;
        this->rloc = row->rloc;
        this->cloc.assign(row->cloc, row->cloc + row->n_cloc);
        this->channels.reserve(row->n_channels);
        for (unsigned long c = 0; c < row->n_channels; ++c) {
            const auto& ch = row->channels[c];
            GthNonlocalChannel out;
            out.l = ch.l;
            out.r = ch.r;
            out.nproj = ch.nproj;
            out.h.assign(ch.h, ch.h + ch.h_len);
            channels.push_back(std::move(out));
        }
        return;
    }
    std::string supported;
    for (const auto* row : basis_data::kGthPotentials)
        supported += std::string(" ") + row->symbol + ":" + row->name;
    throw std::runtime_error("GthPotential: no in-tree entry for Z="
                             + std::to_string(static_cast<int>(element))
                             + " name='" + std::string(name)
                             + "'. Supported:" + supported);
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
