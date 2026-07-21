/// @file cpu/observer/rdf_curve.cpp
/// @brief rdf_write_curve — the single emitter of the g(r) file format.

#include "atomiverse/cpu/observer/rdf_curve.h"

#include <cstddef>
#include <fstream>

#include <spdlog/fmt/fmt.h>

namespace ATV {
namespace cpu {

void rdf_write_curve(const std::string& filename, const std::vector<double>& gr,
                     double r_min, double bin_width, int n_bins) {
    std::ofstream out(filename);
    out << "# r [Angstrom]   g(r)\n";
    for (int i = 0; i < n_bins; ++i) {
        const double r = r_min + (static_cast<double>(i) + 0.5) * bin_width;
        const double g = (static_cast<std::size_t>(i) < gr.size())
                             ? gr[static_cast<std::size_t>(i)]
                             : 0.0;
        out << fmt::format("{:.6f}  {:.6f}\n", r, g);
    }
}

} // namespace cpu
} // namespace ATV
