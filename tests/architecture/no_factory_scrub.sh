#!/usr/bin/env bash
# Grep gate: no free-function factories; no intermediate-type construction
# in user-facing trees (examples/app).
#
# Public API: concrete methods in methods.h (RHF / UHF / RKS / UKS) + Frame.
# Kernel unit tests may use scf/internal/ — not gated here.
#
# Usage: no_factory_scrub.sh <repo-root>
set -u

ROOT="${1:?usage: no_factory_scrub.sh <repo-root>}"

FACTORY_PATS=(
    'build_integrals[[:space:]]*\('
    'make_atom_list[[:space:]]*\('
    'build_shell_list[[:space:]]*\('
    'build_integral_derivatives[[:space:]]*\('
    'build_am1_system[[:space:]]*\('
    'build_nddo_system[[:space:]]*\('
    'rhf_config[[:space:]]*\('
    'uhf_config[[:space:]]*\('
    'uks_config[[:space:]]*\('
    'default_config[[:space:]]*\('
    'b3lyp_config[[:space:]]*\('
    'pbe0_config[[:space:]]*\('
    'baseline_config[[:space:]]*\('
    # scf-slater-determinant-04-methods: the 13 memberized build_* free
    # functions stay blacklisted tree-wide so the old spellings cannot creep
    # back (new members dropped the build_ prefix and cannot collide).
    'build_g_rhf_direct[[:space:]]*\('
    'build_g_rhf_direct_block[[:space:]]*\('
    'build_jk_uhf_direct[[:space:]]*\('
    'build_j_direct[[:space:]]*\('
    'build_k_direct_block[[:space:]]*\('
    'build_coulomb_exchange_fused[[:space:]]*\('
    'build_shell_pair_list[[:space:]]*\('
    'build_shell_block_pair_list[[:space:]]*\('
    'build_overlap[[:space:]]*\('
    'build_kinetic[[:space:]]*\('
    'build_nuclear_attraction[[:space:]]*\('
    'build_eri_dense[[:space:]]*\('
    'build_libcint_context[[:space:]]*\('
)

INTERMEDIATE_PATS=(
    'IntegralPack[[:space:]]*\{'
    'AtomList[[:space:]]*\{'
    'ShellList[[:space:]]*\{'
    'IntegralDerivPack[[:space:]]*\{'
    'SCFConfig[[:space:]]*\{'
    'SCFSolver[[:space:]]+[A-Za-z_]'
    'NDDOSystem[[:space:]]+[A-Za-z_]'
)

status=0

for dir in "include/atomiverse" "src" "examples" "tests" "regressions" "app"; do
    target="${ROOT}/${dir}"
    [ -d "${target}" ] || continue
    for pat in "${FACTORY_PATS[@]}"; do
        matches="$(grep -rnE "${pat}" "${target}" 2>/dev/null \
            | grep -v 'no_factory_scrub.sh' \
            | grep -v 'build_integrals_libcint' \
            | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(///|//|\*|/\*)' \
            | grep -vE 'Wrong:|banned|Never `' \
            || true)"
        if [ -n "${matches}" ]; then
            echo "FAIL: banned free-function factory /${pat}/ in ${dir}:"
            echo "${matches}"
            echo "  -> use concrete methods (RHF/UHF/RKS/UKS) with Frame."
            status=1
        fi
    done
done

# Intermediate construction: examples + app only (no internal-include gate).
for dir in "examples" "app"; do
    target="${ROOT}/${dir}"
    [ -d "${target}" ] || continue
    for pat in "${INTERMEDIATE_PATS[@]}"; do
        matches="$(grep -rnE "${pat}" "${target}" 2>/dev/null \
            | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(///|//|\*|/\*)' \
            || true)"
        if [ -n "${matches}" ]; then
            echo "FAIL: intermediate type in user-facing ${dir}/ /${pat}/:"
            echo "${matches}"
            echo "  -> Frame + RHF/UHF/RKS/UKS only."
            status=1
        fi
    done
done

# ── cpu/scf-scoped build_* wildcard (scf-slater-determinant-04-methods) ──────
# The cpu/scf tree admits NO new build_* free function. Scoped to cpu/scf
# only: tree-wide there are ~40 legitimate build_* MEMBER declarations
# (Frame::build_special_list, CellList::build_half, CUDA build_halo, ...)
# that grep cannot tell from free functions. linalg/density.h's
# build_density_matrix (shared with six GPW call sites, to be memberized in
# the GPW round) is naturally outside this scope — do NOT extend the
# wildcard tree-wide without solving both problems.
#
# Exemption whitelist (each with its owning round; delete the line when the
# owning round memberizes it):
#   build_molecular_grid          — grid.h public grid builder; IntegralPack-deletion round
#   build_minao_density           — MINAO initial-guess machinery; IntegralPack-deletion round
#   build_minao_ao_list           — MINAO initial-guess machinery; IntegralPack-deletion round
#   build_shell_list_from_table   — basis_set.cpp file-local helper; IntegralPack-deletion round
#   build_shell_to_atom_from_integrals — integral_derivatives.cpp internal; gradient round
for dir in "include/atomiverse/cpu/scf" "src/cpu/scf"; do
    target="${ROOT}/${dir}"
    [ -d "${target}" ] || continue
    matches="$(grep -rnE 'build_[a-z_0-9]*[[:space:]]*\(' "${target}" 2>/dev/null \
        | grep -vE '(\.|->|::)build_' \
        | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(///|//|\*|/\*|#)' \
        | grep -vE 'build_on[[:space:]]*\(|build_components|build_plan' \
        | grep -vE 'build_molecular_grid|build_minao_density|build_minao_ao_list|build_shell_list_from_table|build_shell_to_atom_from_integrals' \
        || true)"
    if [ -n "${matches}" ]; then
        echo "FAIL: new build_* free function in ${dir} (cpu/scf admits none):"
        echo "${matches}"
        echo "  -> make it a member of the class that owns the data."
        status=1
    fi
done

if [ "${status}" -eq 0 ]; then
    echo "PASS: no banned factories; examples/app free of intermediate types."
fi
exit "${status}"
