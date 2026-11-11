"""I/O recorders (mirrors ``cpu/record/`` and ``cuda/record/``).

Canonical deep paths: ``atomiverse.recorder.Thermo``,
``atomiverse.recorder.XYZ`` and ``atomiverse.recorder.MrecRecorder``; the
top-level ``atomiverse.Thermo`` / ``atomiverse.XYZ`` /
``atomiverse.MrecRecorder`` are the same classes.

Prefer ``MD.set_thermo(N)`` for the common LAMMPS-style cadence; construct
``Thermo(freq)`` only when passing through ``set_record`` or the phase
escape hatch. ``XYZ(filename, freq)`` always goes through ``set_record``.

All recorders follow the LAMMPS ``dump``/``thermo`` step convention: a
record's ``step`` is the number of completed MD steps and holds the state
after exactly that many steps (``time = step · dt``). A run of ``n`` steps
at cadence ``k`` records the initial state (step 0) and the states after
steps ``k, 2k, …`` up to and including ``n``; ``Thermo`` also prints the
last step of every run. A step is never recorded twice, so continued CUDA
``run()`` calls on one frame extend one step axis (``run(f, 4); run(f, 4)``
at ``every=2`` records steps 0, 2, 4, 6, 8).

``MrecRecorder(path, every)`` writes a molrec (``*.mrec``) trajectory —
positions, velocities, forces, charges, int32 image flags, tags, mass, cell,
energies (total and per term), ``step`` and ``time`` [fs], plus the identity
columns the frame carries: force-field type as ``type_id`` and species as
``atomic_number`` (a coarse-grained frame has the first only; an absent
column stays absent, never zero-filled) — and is accepted by
``MD.set_record`` on **both** devices (``MD(device="cpu")`` and
``MD(device="cuda")``) and under the MPI launcher (rank 0 gathers and
writes)::

    sim = av.MD(device="cuda")
    ...
    sim.set_record(av.MrecRecorder("traj.mrec", every=100))
    sim.run(frame, 10_000)

    from molpy.io.mrec import MrecReader
    with MrecReader("traj.mrec") as traj:
        steps, times = traj.step, traj.time  # [0, 100, …, 10_000]
        first = traj[0]                      # a molpy.Frame

After ``run()`` returns the store holds every frame of the run. On CUDA
recording is asynchronous but lossless: the device freezes each recorded
step once into the driver's ONE shared capture buffer — the same buffer
every asynchronous consumer of that driver reads (thermo, the pressure
series, other recorders), so a step needed by several of them is captured
once — and a worker thread writes it to the store, so the MD loop does not
wait for the disk. The buffer's byte budget belongs to the driver, not to a
recorder: ``MD(device="cuda", capture_buffer_bytes=N)`` (default the build's
``ATV_CAPTURE_BUFFER_BYTES``, 256 MiB, allocated once when the run starts).
When it is full the run waits until a consumer frees space — no frame is
ever dropped. Waits are logged at the end of the run, returned by
``MD.capture_stats()`` and recorded in the store's ``status`` section
(``buffer``: ``budget_bytes``, ``frame_bytes``, ``peak_bytes``,
``blocked_episodes``, ``blocked_seconds``). A budget smaller than one
captured frame raises ``ValueError`` naming both sizes when the run starts.
If writing fails (disk full, I/O error) — or any other consumer of the
driver fails — the run stops with ``RuntimeError`` naming the failed
consumer and the cause, and the status records ``state = "failed"`` with
the error.

All recorders are bound with the MD driver, so they exist on an extension
built with the ``md`` module (``ATV_BUILD_MD=ON``); on one built without it,
importing this module raises ``MissingDomainError`` naming the flag.
"""

from atomiverse._binding import _cpp, require_domain

require_domain(__name__, "md")

#: Thermodynamic output at ``end_of_step`` — ``cpu::Thermo(freq [steps])``.
Thermo = _cpp.Thermo

#: XYZ trajectory output at ``end_of_step`` —
#: ``cpu::XyzRecorder(filename, freq [steps])``.
XYZ = _cpp.XYZ

#: molrec trajectory output at ``end_of_step`` on CPU and CUDA MD —
#: ``cpu::MrecRecorder`` / ``cuda::MrecRecorder(path, every [steps],
#: flush_every [frames] = 0, durable = True)``; the frame buffer is the
#: driver's (``MD(capture_buffer_bytes=...)``).
MrecRecorder = _cpp.MrecRecorder

__all__ = ["Thermo", "XYZ", "MrecRecorder"]
