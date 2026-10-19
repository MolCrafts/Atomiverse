"""Unsupported DFTB2 launcher entries must reject before touching the frame.

The sentinel frame makes the pre-fix failure finite: reaching frame conversion
raises the wrong exception instead of entering unmatched MPI collectives.
"""
import atomiverse as av

method = av.DFTB2()
for entry in (method.run, method.forces, lambda frame: av.single_point(frame, method)):
    try:
        entry(object())
    except RuntimeError as error:
        assert "DFTB2 has no MPI single-point task payload" in str(error), str(error)
    else:
        raise AssertionError("DFTB2 launcher entry did not reject")
print("DFTB2 launcher rejection PASS", flush=True)
