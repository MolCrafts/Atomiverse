#!/usr/bin/env bash
# Grep gate: no hand-written `static_cast<int>` on a Phase enum, anywhere.
#
# A Plan is a dense array of slots, so somewhere the typed phase has to become an
# index. That "somewhere" is `ATV::slot_of` (facade/stage.h) and nowhere else.
#
# The reason this is a gate and not a style preference: the six workflow Phase enums
# (md, minimizer, evaluate, vib, sampling, path) all start at 0. An `int` phase cannot
# tell them apart, so a cast from the wrong workflow lands on a syntactically valid but
# semantically wrong slot and runs silently in the wrong lifecycle position —
# `evaluate::evaluate` == 1 == `md::pre_force`. Before the driver-handler refactor the
# hole was live: `Backend::run_phase(int)` took a raw int and both of its callers reached
# it by writing `run_phase(static_cast<int>(some::Phase::force), s)`.
#
# The typed entry points that replaced it:
#   ATV::slot_of(phase)              — the ONE conversion
#   ATV::slot_count<PhaseEnum>()     — the COUNT sentinel as a slot count
#   Backend::run_stage(phase, step)  — replay one phase
#   LaunchContext::at(phase)         — "is this my stage?"
#   ATV::make_binding(phase, order)  — declare a default binding
#
# `DefaultBinding` still carries an `int phase` + a `PhaseKind`. That pair IS the
# erasure, not a leak: `ComponentBase` is not a template, so `default_bindings()` must
# return one concrete type for every workflow. `make_binding` is the only way to build
# the pair, so no caller ever types the int.
#
# Usage: stage_typing_scrub.sh <repo-root>
set -u

ROOT="${1:?usage: stage_typing_scrub.sh <repo-root>}"

status=0
for dir in "include/atomiverse" "src" "examples" "tests" "regressions" "benchmarks"; do
    target="${ROOT}/${dir}"
    [ -d "${target}" ] || continue

    # A cast whose operand mentions a Phase enum. Doc comments (`///`, `//`, ` *`) are
    # excluded: several of them quote the banned pattern in order to explain the rule,
    # including this gate's own rationale in backend.h and stage.h.
    matches="$(grep -rn 'static_cast<int>([^)]*Phase' "${target}" 2>/dev/null \
        | grep -v 'stage_typing_scrub.sh' \
        | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(///|//|\*)')"

    if [ -n "${matches}" ]; then
        echo "FAIL: hand-written static_cast<int> on a Phase enum in ${dir}:"
        echo "${matches}"
        echo "  -> use ATV::slot_of(phase) / slot_count<P>() / ctx.at(phase) /"
        echo "     backend.run_stage(phase, step) / ATV::make_binding(phase, order)."
        status=1
    fi
done

# The one sanctioned conversion must still exist, or the gate above is vacuous.
if ! grep -q 'return static_cast<int>(p);' "${ROOT}/include/atomiverse/facade/stage.h"; then
    echo "FAIL: ATV::slot_of's cast is gone from facade/stage.h — the gate is vacuous."
    status=1
fi

if [ "${status}" -eq 0 ]; then
    echo "PASS: the only Phase->int conversion is ATV::slot_of."
fi
exit "${status}"
