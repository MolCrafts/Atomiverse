#!/usr/bin/env bash
# Grep gate: compatibility-only CPU surfaces must not return.  CPU Frame is a
# single hard-cut schema (`blocks + meta + simbox`); only CUDA/Torch are
# deferred.  There are intentionally no aliases, forwarding accessors, or
# legacy wire decoders.
#
# Usage: compatibility_surface_scrub.sh <repo-root>
set -u

ROOT="${1:?usage: compatibility_surface_scrub.sh <repo-root>}"
status=0

fail_matches() { # label, matches
    if [ -n "$2" ]; then
        echo "FAIL: $1"
        echo "$2"
        status=1
    fi
}

CPU_DIRS=(
    "${ROOT}/CMakeLists.txt"
    "${ROOT}/include/atomiverse/core"
    "${ROOT}/include/atomiverse/cpu"
    "${ROOT}/include/atomiverse/facade"
    "${ROOT}/src/cpu"
    "${ROOT}/tests/cpu"
    "${ROOT}/tests/test_xc_reference.cpp"
    "${ROOT}/examples"
    "${ROOT}/regressions"
    "${ROOT}/benchmarks"
    "${ROOT}/app"
    "${ROOT}/tools"
    "${ROOT}/docs"
)

common_grep() { # extended-regex
    grep -RnE --include='*.h' --include='*.hpp' --include='*.cpp' \
        --include='*.md' --include='*.txt' --include='CMakeLists.txt' \
        --exclude-dir=cuda \
        --exclude-dir=torch "$1" "${CPU_DIRS[@]}" 2>/dev/null \
        | grep -vE '/docs/user-guide/libtorch-backend[.]md:|/benchmarks/md/pe/performance/polyethylene_tension[.]cpp:' \
        | grep -vE 'compatibility_(surface_scrub[.]sh|inventory[.]txt)'
}

code_grep() { # extended-regex
    grep -RnE --include='*.h' --include='*.hpp' --include='*.cpp' \
        --exclude-dir=cuda --exclude-dir=torch "$1" "${CPU_DIRS[@]}" \
        2>/dev/null \
        | grep -vE 'compatibility_(surface_scrub[.]sh|inventory[.]txt)'
}

python_grep() { # extended-regex
    grep -RnE --include='*.py' --include='*.md' "$1" \
        "${ROOT}/python/src" "${ROOT}/python/docs" \
        "${ROOT}/python/examples" "${ROOT}/python/benchmarks" 2>/dev/null \
        | grep -vE 'compatibility_(surface_scrub[.]sh|inventory[.]txt)'
}

all_python_grep() { # extended-regex
    grep -RnE --include='*.py' --exclude-dir=cuda --exclude-dir=torch \
        --exclude-dir=build --exclude-dir=.venv "$1" \
        "${ROOT}/python/src" "${ROOT}/python/docs" \
        "${ROOT}/python/examples" "${ROOT}/python/benchmarks" \
        "${ROOT}/examples" "${ROOT}/regressions" \
        "${ROOT}/benchmarks" "${ROOT}/app" "${ROOT}/tools" 2>/dev/null \
        | grep -vE 'compatibility_(surface_scrub[.]sh|inventory[.]txt)'
}

INVENTORY="${ROOT}/tests/cpu/compatibility_inventory.txt"
if [ ! -f "${INVENTORY}" ]; then
    echo 'FAIL: checked CPU compatibility inventory is missing'
    status=1
else
    while IFS='|' read -r scope label pattern; do
        case "${scope}" in
        ''|'#'*) continue ;;
        code) matches="$(code_grep "${pattern}")" ;;
        all) matches="$(common_grep "${pattern}")" ;;
        python) matches="$(python_grep "${pattern}")" ;;
        *)
            echo "FAIL: unknown compatibility inventory scope '${scope}'"
            status=1
            continue
            ;;
        esac
        fail_matches "${label} remains" "${matches}"
    done < "${INVENTORY}"
fi

# The old aggregate spelling is checked only where the receiver is visibly a
# Frame.  Scientific helper types such as IntegralPack::atoms remain valid.
matches="$(common_grep '(ctx[.]frame|(^|[^[:alnum:]_])(frame|f))[.](atoms|scalars|box)([^[:alnum:]_]|$)|->[[:space:]]*(atoms|scalars|box)([^[:alnum:]_]|$)')"
fail_matches 'legacy CPU Frame atoms/scalars/box surface remains' "${matches}"

matches="$(all_python_grep '(^|[^[:alnum:]_])(frame|f)[.](box|metadata)([^[:alnum:]_]|$)')"
fail_matches 'legacy Python Frame box/metadata surface remains' "${matches}"

matches="$(common_grep 'ATV_FRAME_META|ATV_FIELD_BIND|(^|[^[:alnum:]_])ExtraColumn([^[:alnum:]_]|$)')"
fail_matches 'legacy Frame macro/ExtraColumn surface remains' "${matches}"

matches="$(common_grep '#[[:space:]]*include[[:space:]]*[<\"]atomiverse/cpu/frame/atom[.]h[>\"]')"
fail_matches 'deleted cpu/frame/atom.h is still included' "${matches}"

if [ -e "${ROOT}/include/atomiverse/cpu/frame/atom.h" ]; then
    echo 'FAIL: deleted include/atomiverse/cpu/frame/atom.h exists'
    status=1
fi
if [ -e "${ROOT}/src/cpu/frame/atom.cpp" ]; then
    echo 'FAIL: deleted src/cpu/frame/atom.cpp exists'
    status=1
fi

for removed in \
    include/atomiverse/core/constants.h \
    include/atomiverse/core/real.h \
    include/atomiverse/core/int.h \
    include/atomiverse/cpu/scf/scf_solve.h \
    include/atomiverse/cpu/scf/rhf_solve.h \
    include/atomiverse/cpu/scf/uhf_solve.h \
    include/atomiverse/cpu/scf/uks_solve.h \
    include/atomiverse/cpu/pw/pw_scf.h \
    src/cpu/scf/scf_solve.cpp \
    src/cpu/scf/rhf_solve.cpp \
    src/cpu/scf/uhf_solve.cpp \
    src/cpu/scf/uks_solve.cpp; do
    if [ -e "${ROOT}/${removed}" ]; then
        echo "FAIL: deleted compatibility file ${removed} exists"
        status=1
    fi
done

FRAME_H="${ROOT}/include/atomiverse/cpu/frame/frame.h"
matches="$(grep -nE '(^|[^[:alnum:]_])insert[[:space:]]*\(' "${FRAME_H}" 2>/dev/null)"
fail_matches 'Frame compatibility insert forwarder remains' "${matches}"

BOX_H="${ROOT}/include/atomiverse/cpu/frame/box.h"
matches="$(grep -nE 'operator[[:space:]]*\(|operator[[:space:]]*=[[:space:]]*\([^)]*Mat3' "${BOX_H}" 2>/dev/null)"
fail_matches 'legacy Box matrix operators remain' "${matches}"

matches="$(common_grep '(^|[^[:alnum:]_])Mat33([^[:alnum:]_]|$)')"
fail_matches 'legacy Mat33 alias/reference remains' "${matches}"

# Element identity is owned and exported by molrs. Atomiverse callers name the
# owner explicitly; an ATV::Element spelling, namespace re-export, or implicit
# adapter would let the two repositories drift again.
matches="$(grep -RnE --include='*.h' --include='*.hpp' --include='*.cpp' \
    --include='*.md' \
    'ATV::Element|using[[:space:]]+(::)?molrs::Element|using[[:space:]]+Element[[:space:]]*=[[:space:]]*(::)?molrs::Element|typedef[[:space:]]+(::)?molrs::Element[[:space:]]+Element|operator[[:space:]]+(::)?molrs::Element|enum[[:space:]]+(class[[:space:]]+)?Element[[:space:]]*[{:]|struct[[:space:]]+Element[[:space:]]*[{:]' \
    "${ROOT}/include/atomiverse" "${ROOT}/src" "${ROOT}/tests" \
    "${ROOT}/examples" "${ROOT}/regressions" "${ROOT}/benchmarks" \
    "${ROOT}/docs" 2>/dev/null \
    | grep -vE '/(cuda|torch)/' \
    | grep -v 'compatibility_surface_scrub.sh')"
fail_matches 'Element must be used as molrs::Element without an ATV alias or adapter' "${matches}"

matches="$(grep -nE \
    'molrs::Element[[:space:]]+element[[:space:]]*\{[[:space:]]*\}|(GaussianBasis|GthPotential|Kind)[[:space:]]*\([[:space:]]*\)[[:space:]]*=[[:space:]]*default' \
    "${ROOT}/include/atomiverse/cpu/gpw/gaussian_basis.h" \
    "${ROOT}/include/atomiverse/cpu/gpw/gth_potential.h" \
    "${ROOT}/include/atomiverse/cpu/gpw/gpw_evaluator.h" 2>/dev/null)"
fail_matches 'GPW Element-bearing value types must not have a Z=0 default state' "${matches}"

matches="$(common_grep 'cpu::md::Zero(Forces|Energy|Stress|Kinetic)([^[:alnum:]_]|$)|(^|[^[:alnum:]_])ZeroLinearMomentum([^[:alnum:]_]|$)')"
fail_matches 'legacy MD Zero alias/name remains' "${matches}"

# The deleted overload had `int n_radial` immediately after `int natm`;
# the sole canonical declaration requires an explicit GridLevel there.
GRID_H="${ROOT}/include/atomiverse/cpu/scf/grid.h"
matches="$(awk '
    /MolecularGrid build_molecular_grid\(/ {
        first = NR ":" $0
        if (getline > 0 && $0 ~ /^[[:space:]]*int n_radial/) \
            print first "\n" NR ":" $0
    }
' "${GRID_H}")"
fail_matches 'implicit-SG0 build_molecular_grid overload remains' "${matches}"

UPF_H="${ROOT}/include/atomiverse/cpu/pw/internal/upf_reader.h"
UPF_CPP="${ROOT}/src/cpu/pw/internal/upf_reader.cpp"
matches="$(grep -nE '(^|[^[:alnum:]_])kb(\[[^]]+\])?\.d([^[:alnum:]_]|$)|Real[[:space:]]+d[[:space:]]*;' \
    "${UPF_H}" "${UPF_CPP}" 2>/dev/null)"
fail_matches 'deprecated UPF KBProjector diagonal cache remains' "${matches}"

TASK_H="${ROOT}/include/atomiverse/cpu/task/task_spec.h"
TASK_CODEC="${ROOT}/src/cpu/task/task_codec.cpp"
TASK_TEST="${ROOT}/tests/cpu/test_task_codec.cpp"
matches="$(grep -nE '\.index\(\)|kWireVersion[[:space:]]*=[[:space:]]*(1|2)' \
    "${TASK_H}" "${TASK_CODEC}" "${TASK_TEST}" 2>/dev/null)"
fail_matches 'TaskSpec wire identity still depends on variant index or a pre-v3 format' "${matches}"

if ! grep -q 'enum class ComponentId : std::uint16_t' "${TASK_H}"; then
    echo 'FAIL: explicit ComponentId enum missing from TaskSpec'
    status=1
fi
if ! grep -q 'kWireVersion = 3;' "${TASK_H}"; then
    echo 'FAIL: TaskSpec wire version is not exactly v3'
    status=1
fi
if ! grep -q 'if (version != TaskSpec::kWireVersion)' "${TASK_CODEC}"; then
    echo 'FAIL: TaskSpec decoder no longer rejects every non-v3 version'
    status=1
fi
if ! grep -q 'VersionTwoAndUnknownVersionsAreRejected' "${TASK_TEST}"; then
    echo 'FAIL: no unit test pins rejection of the removed v2 wire format'
    status=1
fi

if [ "${status}" -eq 0 ]; then
    echo 'PASS: CPU compatibility shims remain deleted and TaskSpec is strict v3.'
fi
exit "${status}"
