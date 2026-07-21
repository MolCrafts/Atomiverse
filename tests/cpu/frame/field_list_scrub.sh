#!/usr/bin/env bash
# Grep gate (frame-block-01, ac-001): the registry is THE single field
# enumeration. Structural storage and native-byte exchange must traverse the
# Block/Column plan and must not hand-copy a list of former Atom members.
#
# Usage: field_list_scrub.sh <repo-root>
set -u

ROOT="${1:?usage: field_list_scrub.sh <repo-root>}"

# Distinctive member tokens: a hand-written field list can't avoid these.
TOKENS='vy|vz|fy|fz|charge|mass|bond_atom|angle_atom|dihedral_atom|nspecial|special'

status=0

# Extract one function body: from the signature line to the first column-0 '}'.
body_of() { # file, signature-regex
    awk -v sig="$2" '
        $0 ~ sig { inside = 1 }
        inside { print }
        inside && /^}/ { exit }
    ' "$1"
}

check() { # file, signature-regex, label
    local hits
    hits=$(body_of "$1" "$2" | sed 's://.*::' | grep -nE "(^|[^a-zA-Z0-9_])(${TOKENS})([^a-zA-Z0-9_]|$)" | grep -v for_each_column || true)
    if [ -n "$hits" ]; then
        echo "FAIL: hand-written field names inside $3 (must derive from the registry):"
        echo "$hits"
        status=1
    fi
}

FRAME="$ROOT/src/cpu/frame/frame.cpp"
PACK="$ROOT/src/cpu/comm/atom_pack.cpp"

check "$FRAME" '^void Frame::set_capacity'       'Frame::set_capacity'
check "$FRAME" '^void Block::copy_row'           'Block::copy_row'
check "$PACK" '^std::size_t pack_exchange'       'comm::pack_exchange'
check "$PACK" '^std::size_t unpack_exchange'     'comm::unpack_exchange'
check "$PACK" '^std::size_t max_exchange_bytes'  'comm::max_exchange_bytes'

if [ "$status" -eq 0 ]; then
    echo "PASS: registry functions carry no hand-written field lists"
fi
exit "$status"
