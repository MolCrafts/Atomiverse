/// @file examples/08_pw.cpp
/// @brief Plane-wave DFT (LDA) single-point energy for bulk silicon.
///
/// The first four SCF tutorials used atom-centered Gaussian basis sets for
/// isolated molecules. This example switches to the other major basis family
/// — plane waves — which is the natural choice for periodic crystals. The
/// system is silicon in the diamond structure (the 2-atom fcc primitive
/// cell).
///
/// One concept: how a plane-wave single-point energy is assembled from the
/// same Driver/Component vocabulary as every other method. The plane-wave
/// SCF loop lives inside the inner solver `cpu::pw::PW`; the
/// user-facing component is `cpu::pw::PWEvaluator` (role Evaluator,
/// capability Energy), run through the one-pass `cpu::Evaluate` driver via
/// the `cpu::single_point` helper — the same shape as the Gaussian methods
/// (`cpu::scf::RHF{basis}.run(frame)`).
///
/// Build:
///   cmake -B build -DATV_BUILD_PW=ON -DATV_BUILD_EXAMPLES=ON   # needs FFTW3
///   cmake --build build --target 08_pw
///
/// Run (analytic GTH pseudopotential — runnable with zero arguments):
///   ./build/examples/08_pw
///
/// Run against a norm-conserving UPF v1 file instead. A silicon LDA
/// pseudopotential ships with the repository:
///   ./build/examples/08_pw tests/data/pw/Si_lda_nc.upf
///
/// With no argument the solver uses the published Goedecker-Teter-Hutter
/// parameterization "GTH-PBE-q4" for silicon, evaluated in-tree from
/// `cpu/gpw/basis_data/gpw_tables.h`. There is no pseudopotential-free
/// mode: a bare valence-charge Coulomb potential is not representable in a
/// plane-wave basis at any finite cutoff, so a run with neither
/// `upf_path` nor `gth_name` set throws.

#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

/// @brief Silicon diamond structure: 2-atom fcc primitive cell.
/// @param a Conventional cubic lattice constant [Å] (5.43 Å for Si).
///
/// The two basis atoms sit at (0,0,0) and (a/4,a/4,a/4); the cell vectors
/// are the fcc primitive vectors (a/2)(0,1,1), (a/2)(1,0,1), (a/2)(1,1,0).
cpu::Frame make_silicon(double a) {
    cpu::Frame f(2);
    f.span(ATV::kAtom_type)[0] = 14; // Si, Z = 14
    f.span(ATV::kAtom_type)[1] = 14;
    f.span(ATV::kAtom_mass)[0] = 28.0855;
    f.span(ATV::kAtom_mass)[1] = 28.0855;
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = a / 4.0;
    f.span(ATV::kAtom_y)[1] = a / 4.0;
    f.span(ATV::kAtom_z)[1] = a / 4.0;

    const double h = a / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    f.box.set_cell(cell);
    return f;
}

} // namespace

int main(int argc, char** argv) {
    // A plane-wave run has no default pseudopotential. With no argument this
    // uses the in-tree analytic GTH parameterization for silicon; pass a UPF
    // file path to use that instead.
    const std::string upf_path = (argc > 1) ? argv[1] : "";
    const std::string gth_name = upf_path.empty() ? "GTH-PBE-q4" : "";

    std::cout << "=== Silicon (diamond) / plane-wave LDA ===\n"
              << "Cell: 2-atom fcc primitive, a = 5.43 Angstrom\n"
              << "Pseudopotential: "
              << (upf_path.empty() ? gth_name + " (analytic, in-tree)"
                                   : upf_path)
              << "\n\n";

    auto frame = make_silicon(5.43);

    // ── Configure the plane-wave SCF ──────────────────────────────────
    //
    // ecut_ry  : plane-wave kinetic-energy cutoff [Ry]. Larger = more plane
    //            waves = more complete basis. 15 Ry / k_mesh=2 keeps this
    //            tutorial fast (seconds); production silicon uses ~30 Ry and
    //            a denser k-mesh — see benchmarks/pw/si for converged runs.
    // k_mesh   : Monkhorst-Pack Brillouin-zone sampling (k_mesh^3 points).
    // tol      : SCF convergence tolerance on the density change.
    cpu::pw::PW cfg;
    cfg.upf_path = upf_path;
    cfg.gth_name = gth_name;
    cfg.ecut_ry = 15.0;
    cfg.k_mesh = 2;
    cfg.tol = 1e-6;
    cfg.max_iter = 50;

    // ── Run it as an Evaluator component ──────────────────────────────
    //
    // PWEvaluator wraps the inner PW. cpu::single_point binds it at
    // the evaluate phase, builds the one-pass Evaluate driver, and runs it.
    // The component writes frame.meta.get(ATV::kEnergy) in eV; the Ry / Hartree
    // breakdown stays on last_result().
    cpu::pw::PWEvaluator pwe{std::move(cfg)};
    cpu::single_point(frame, pwe);

    const auto& result = pwe.last_result();

    std::cout << std::boolalpha << "Converged:    " << result.converged << " ("
              << result.iterations << " iterations)\n"
              << std::fixed << std::setprecision(6)
              << "E_total:      " << result.energy << " Ry\n"
              << "E_total:      " << result.energy_ha << " Hartree\n"
              << "E_total:      " << frame.meta.get(ATV::kEnergy) << " eV\n";

    if (upf_path.empty())
        std::cout << "\nNote: re-run against a UPF file to compare "
                     "pseudopotentials, e.g.\n"
                     "  ./08_pw tests/data/pw/Si_lda_nc.upf\n";

    std::cout << "\n=== Done ===\n";
    return 0;
}
