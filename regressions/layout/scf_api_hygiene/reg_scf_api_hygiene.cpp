/// @file reg_scf_api_hygiene.cpp
/// @brief scf-review-05-api-hygiene layout/architecture witness — living
///        documentation of two SCF API-hygiene invariants, verified by an
///        OUT-OF-LIBRARY consumer (public headers + link atomiverse, nothing
///        else). This mirrors the reg_architecture_gates precedent: there is
///        no numeric golden and no third-party runtime; compilation + a clean
///        exit 0 is the witness that the locked-down API shape resolves.
///
///          - item 10: MolecularGrid is CONSTRUCTOR-built. The former factory
///            free function `build_molecular_grid(...)` is gone; the only
///            construction path is the member constructor
///            `MolecularGrid(atoms, natm, GridLevel, n_radial, n_angular)`
///            (following the GthPotential(Element, name) lookup-ctor precedent).
///            This file constructs one via that constructor — so it does NOT
///            compile until the constructor lands, which is its RED state.
///          - item 14: an unsupported `nroots` is a CATCHABLE exception from
///            outside the library, not the uncatchable std::abort() it used to
///            be. This file catches the std::invalid_argument thrown by
///            native::rys_roots(8, 1.0, ...).
///
/// The tests/architecture + tests/cpu scrubs are the enforcement; this is the
/// consumer-side witness that the shape actually links and the error path is
/// catchable across the library boundary.
#include "atomiverse/cpu/scf/grid.h" // MolecularGrid ctor, GridLevel, native::Atom
#include "atomiverse/cpu/scf/rys_roots.h" // native::rys_roots

#include <cstdio>
#include <stdexcept>

int main() {
    using ATV::cpu::scf::GridLevel;
    using ATV::cpu::scf::MolecularGrid;
    using ATV::cpu::scf::native::Atom;

    // item 10 — MolecularGrid is constructor-built (no build_molecular_grid
    // factory). Single H atom at the origin; SG0 honours n_radial / n_angular
    // verbatim (50, 110 = the documented SG0 defaults, guaranteed valid).
    const Atom atoms[1] = {{/*Z=*/1, {0.0, 0.0, 0.0}}};
    const MolecularGrid grid(atoms, /*natm=*/1, GridLevel::SG0,
                             /*n_radial=*/50, /*n_angular=*/110);
    if (grid.points.empty()) {
        std::fprintf(stderr,
                     "reg_scf_api_hygiene: MolecularGrid constructor produced "
                     "no grid points\n");
        return 1;
    }

    // item 14 — an unsupported nroots (> 7) is a catchable std::invalid_argument
    // from outside the library, not an uncatchable std::abort(). x = 1.0 sits
    // in the intermediate regime that reaches the domain switch/default branch.
    double u[8] = {};
    double w[8] = {};
    bool caught = false;
    try {
        ATV::cpu::scf::native::rys_roots(8, 1.0, u, w);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    if (!caught) {
        std::fprintf(
            stderr, "reg_scf_api_hygiene: rys_roots(8, 1.0, ...) did not throw "
                    "a catchable std::invalid_argument\n");
        return 2;
    }

    std::puts("scf_api_hygiene: MolecularGrid constructor links; "
              "rys_roots(8, ...) throws a catchable std::invalid_argument");
    return 0;
}
