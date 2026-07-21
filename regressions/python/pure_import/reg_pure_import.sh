#!/usr/bin/env bash
# Pure-Python import regression (python-on-demand-install-02):
# isolated sources only — no C++ extension on the import path.
#
# Arguments: $1 = python executable (optional; defaults to python3).
set -euo pipefail

PYTHON="${1:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SRC="$REPO/python/src"

if [[ ! -d "$SRC/atomiverse" ]]; then
    echo "FAIL: pure sources not found at $SRC/atomiverse"
    exit 1
fi

echo "== pure_import: isolated pure-Python path =="
echo "  python: $PYTHON"
echo "  src:    $SRC"

# Replace sys.path with pure sources only (no site-packages extension,
# no skbuild .so). Verifies soft import, install-message ImportError,
# and ``python -m atomiverse`` guidance.
"$PYTHON" - <<PY
import runpy
import sys

src = r"$SRC"
# Prepend pure sources; drop site/dist-packages so an installed _atomiverse.so
# cannot satisfy the extension import. Keep stdlib paths (never sys.path = [src]).
sys.path = [src] + [
    p for p in sys.path
    if p and "site-packages" not in p and "dist-packages" not in p
]
for key in list(sys.modules):
    if key == "atomiverse" or key.startswith("atomiverse.") or key == "_atomiverse":
        del sys.modules[key]

# 1) soft import + version
import atomiverse as av
assert av.__version__, "expected non-empty __version__"
print("ok: import atomiverse; __version__ =", av.__version__)

from atomiverse._binding import extension_available, require_cpp

assert extension_available() is False, "extension must be absent on pure path"
print("ok: extension_available() is False")

# 2) compute path ImportError carries install guidance
raised = False
try:
    _ = av.MD
except ImportError as e:
    raised = True
    msg = str(e)
    assert "python -m atomiverse install" in msg, msg
    print("ok: av.MD raises install-oriented ImportError")
assert raised, "av.MD did not raise"

raised = False
try:
    require_cpp()
except ImportError as e:
    raised = True
    assert "python -m atomiverse install" in str(e)
    print("ok: require_cpp() raises install-oriented ImportError")
assert raised, "require_cpp() did not raise"

# 3) ``python -m atomiverse`` guidance (in-process, same isolated path)
sys.argv = ["atomiverse"]
rc = 0
try:
    runpy.run_module("atomiverse", run_name="__main__")
except SystemExit as e:
    rc = int(e.code or 0)
assert rc == 0, f"python -m atomiverse exited {rc}"
print("ok: python -m atomiverse (in-process) exit 0")
PY

# Separate process for -m entry: capture output and assert install wording.
OUT="$("$PYTHON" -c "
import runpy, sys
src = r'$SRC'
sys.path = [src] + [p for p in sys.path if p and 'site-packages' not in p and 'dist-packages' not in p]
sys.argv = ['atomiverse']
runpy.run_module('atomiverse', run_name='__main__')
" 2>&1)" || {
    echo "FAIL: python -m atomiverse process failed"
    echo "$OUT"
    exit 1
}
echo "$OUT" | grep -qi install || {
    echo "FAIL: -m output missing install guidance"
    echo "$OUT"
    exit 1
}
echo "$OUT" | grep -qi traceback && {
    echo "FAIL: -m output contains traceback"
    echo "$OUT"
    exit 1
}

echo "REGRESSION PASSED: python/pure_import"
