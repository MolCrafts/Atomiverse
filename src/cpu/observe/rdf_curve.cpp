/// @file cpu/observe/rdf_curve.cpp
/// @brief rdf_write_curve — the single emitter of the g(r) file format.

#include "atomiverse/cpu/observe/rdf_curve.h"

#include <cstddef>
#include <fstream>
#include <stdexcept>

#include <spdlog/fmt/fmt.h>

namespace ATV {
namespace cpu {

void rdf_write_curve(const std::string& filename, const std::vector<double>& r,
                     const std::vector<double>& gr) {
    if (r.size() != gr.size())
        throw std::invalid_argument(
            fmt::format("rdf_write_curve({}): {} radii for {} g(r) values",
                        filename, r.size(), gr.size()));
    std::ofstream out(filename);
    out << "# r [Angstrom]   g(r)\n";
    for (std::size_t i = 0; i < gr.size(); ++i)
        out << fmt::format("{:.6f}  {:.6f}\n", r[i], gr[i]);
}

} // namespace cpu
} // namespace ATV
