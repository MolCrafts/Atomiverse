"""Reject incompatible native Frame handles before reading their pointers."""

import subprocess
import sys
import textwrap

import pytest

from atomiverse._binding import _cpp
from _domains import skip_module_unless

skip_module_unless("ATV_BUILD_HF")


@pytest.mark.parametrize(
    ("token_change", "expected"),
    [
        ("token[:2]", "fewer than 3 fields"),
        ("('999.0', *token[1:])", "ABI-line mismatch"),
        ("(*token[:2], token[2] + '-stale', *token[3:])", "native build mismatch"),
    ],
)
def test_capsule_token_checked_before_pointer_access(token_change, expected):
    # The handshake is cached per interpreter; each rejection needs a fresh
    # process. Load the exact extension under test, including in a dev build.
    script = textwrap.dedent(
        f"""
        import importlib.util
        import molpy._core as core
        import sys
        spec = importlib.util.spec_from_file_location('_atomiverse', {_cpp.__file__!r})
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        token = core._ffi_abi_token()
        core._ffi_abi_token = lambda: {token_change}

        class Candidate:
            def _ffi_frameref_capsule(self):
                raise AssertionError('must reject the token before extracting a pointer')

        try:
            native.RHF('STO-3G').run(Candidate())
        except RuntimeError as error:
            assert {expected!r} in str(error), str(error)
        else:
            raise AssertionError('incompatible token was accepted')
        """
    )
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
