---
name: atomiverse-docs
description: Writes or updates the Atomiverse Zensical manual, roadmap pages and Doxygen comments and builds them locally. Use when a page under docs/, a backend capability claim or a public C++ symbol's documentation changes.
---

# atomiverse-docs

The manual is Zensical with the shared MolCrafts theme (`zensical.toml`);
the C++ reference is Doxygen (`Doxyfile`). AGENTS.md `## Documentation` owns
the rules; mechanism detail lives in `docs/developer/`.

## Steps

1. **Add a page with both edits.** Write the markdown under `docs/` and add
   its entry to `nav` in `zensical.toml`.
2. **Use the theme only.** Add no CSS, HTML, `extra_css`,
   `extra_javascript`, `custom_dir` or `molcrafts-*` class names; the only
   exception is `[project.extra.molcrafts].accent`.
3. **Claim capabilities only in the roadmap.** State what runs on which
   backend only in `docs/roadmap/`, and only with evidence.
4. **Document public C++ symbols.** Give each `@brief`, `@param` with units
   and `@return` in its header under `include/atomiverse/`.
5. **Preview the manual.**
   `uv run --with zensical --with molcrafts-zensical-theme zensical serve`.
6. **Build the API reference.** `doxygen Doxyfile`, or without the C++
   toolchain `cmake -B build/docs -DATV_DOCS_ONLY=ON` then
   `cmake --build build/docs --target docs`.
7. **Check strictly.** Run `molci verify`; its push checks build the docs
   strictly.

## Done when

- every new page is in `nav` and renders in the preview
- Doxygen builds and new public symbols carry `@brief`/`@param`/`@return`
- `molci verify` passes
