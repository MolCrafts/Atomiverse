# Initial-force CUDA Velocity Verlet trajectory

oracle: literature

W. C. Swope, H. C. Andersen, P. H. Berens, and K. R. Wilson,
“A computer simulation method for the calculation of equilibrium constants
for the formation of physical clusters of molecules: Application to small
water clusters,” *Journal of Chemical Physics* **76**, 637–649 (1982),
[doi:10.1063/1.442716](https://doi.org/10.1063/1.442716).

The velocity Verlet update uses the force at the original coordinates for
its first half-kick:

    x(t+dt) = x(t) + dt*v(t) + dt²*a(x(t))/2
    v(t+dt) = v(t) + dt*[a(x(t)) + a(x(t+dt))]/2.

Two atoms of masses 2 and 3 amu start at x = 4 and 6.4 Å with zero full-step
velocities. One harmonic bond has K = 7 eV/Å² and equilibrium length 1.2 Å,
with V = K(r−r0)². The public unit conversion supplies a = F/m in Å/fs².
The regression compares positions and velocities after one 0.5 fs step and
a second continued step with these analytic updates. It independently runs
two steps in one call and verifies that chunking does not restart the initial
force pass or lose the current force. Transverse coordinates and velocities
remain unchanged. No force component lifecycle is driven by the consumer.

Maximum absolute position/velocity error must be below 1e−11 in double or
3e−6 with float storage or kernel work. Four total integration steps on two atoms should
finish within ten seconds on Arrhenius, including CUDA initialization.
