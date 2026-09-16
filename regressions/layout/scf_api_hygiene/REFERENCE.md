# scf_api_hygiene — SCF API-hygiene layout/architecture witness

Architecture witness for spec `scf-review-05-api-hygiene`. This is a **layout
witness**, not a numerical regression: it has **no literature value and no
golden number**, exactly like the `regressions/layout/architecture_gates`
precedent. Its correctness is that it *compiles*, *links* against the public
library, and *exits 0* — the out-of-library proof that two API-hygiene
invariants hold at the library boundary.

## What it witnesses

1. **item 10 — MolecularGrid is constructor-built.** The former factory free
   function `build_molecular_grid(...)` (a "factory as the primary constructor
   story", forbidden by the OOP rules) is gone. The only construction path is
   the member constructor
   `MolecularGrid(const native::Atom* atoms, int natm, GridLevel level,
   int n_radial = 50, int n_angular = 110)`, following the
   `GthPotential(molrs::Element, name)` lookup-constructor precedent. The
   witness constructs one via that constructor for a single H atom at
   `GridLevel::SG0`. Because it names a constructor that does not exist on
   `dev`, this file **does not compile** pre-fix — that is its RED state.

2. **item 14 — unsupported nroots is catchable, not abort.** `native::rys_roots`
   used to `std::fprintf` + `std::abort()` on an out-of-domain `nroots`,
   terminating the process uncatchably (and reachable through g/h shells). The
   post-fix contract is a `std::invalid_argument`. The witness calls
   `native::rys_roots(8, 1.0, u, w)` inside a `try`/`catch` and asserts the
   exception is caught **across the library boundary**.

`x = 1.0` is chosen so the call lands in the intermediate regime
(`SMALLX_LIMIT < x < 35 + 5*nroots`), i.e. the domain switch/default branch
that must now throw, rather than the small-x or large-x fast paths.

## Enforcement vs witness

The scrubs `tests/architecture/no_factory_scrub.sh` (no `build_molecular_grid`)
and `tests/cpu/compatibility_surface_scrub.sh` (constructor shape, explicit
`GridLevel`) are the enforcement. The unit test
`tests/cpu/scf/test_rys_roots.cpp` pins the throw type/message. This regression
is the consumer-side witness that both shapes resolve and link from outside the
library.

## References

None. A layout/architecture witness cites no literature value; the invariant is
the API shape, not a physical number. Precedent: `regressions/layout/architecture_gates`.
