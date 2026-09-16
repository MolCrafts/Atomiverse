/// @file cpu/vibration/frequency_report.cpp
/// @brief cpu::vib::FrequencyReport — TSV writer implementation.

#include "atomiverse/cpu/vibration/frequency_report.h"

#include <fstream>
#include <iomanip>
#include <limits>

namespace ATV {
namespace cpu {
namespace vib {

void FrequencyReport::launch(const cpu::LaunchContext& ctx) {
    if (ws_ == nullptr)
        return; // test-only role-check ctor.
    // An empty path DISABLES the report. Stated here rather than left to
    // `std::ofstream("")` failing its open: a caller who wants no TSV must be
    // able to say so, and the Python binding defaults to exactly this so that
    // importing a driver never litters the interpreter's working directory.
    if (path_.empty())
        return;
    const int dim = 3 * ws_->n_atoms;
    if (ctx.step < 2 * dim)
        return; // not the finalize pass yet.

    std::ofstream os(path_);
    if (!os)
        return;

    // Full round-trip precision so the trailer ZPE / Gibbs values parse
    // back exactly (default ostream precision drops to 6 digits).
    os << std::setprecision(std::numeric_limits<double>::max_digits10);

    // Header row — tab-separated column names.
    os << "index\tfrequency_cm-1\ttype\n";

    // One data row per normal-mode frequency. Negative wavenumbers are
    // the universal `imaginary_freq=False` convention for imaginary
    // modes (force constant lambda < 0).
    const int n_modes = static_cast<int>(ws_->frequencies.size());
    for (int i = 0; i < n_modes; ++i) {
        const double f = ws_->frequencies(i);
        os << i << '\t' << f << '\t' << (f < 0.0 ? "imaginary" : "real")
           << '\n';
    }

    // Trailer — summary quantities from the Thermochem pass. Prefixed
    // with '#' so a TSV parser skips them as comments.
    os << "# n_modes\t" << n_modes << '\n';
    os << "# n_imaginary\t" << ws_->n_imaginary << '\n';
    os << "# zpe_hartree\t" << ws_->zpe << '\n';
    os << "# g_thermal_hartree\t" << ws_->g_thermal << '\n';
    os << "# g_tot_hartree\t" << ws_->g_tot << '\n';
    os << "# s_tot_hartree_per_K\t" << ws_->s_tot << '\n';
}

} // namespace vib
} // namespace cpu
} // namespace ATV
