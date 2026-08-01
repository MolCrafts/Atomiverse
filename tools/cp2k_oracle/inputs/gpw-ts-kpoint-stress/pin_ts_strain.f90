! Offline libMBD pin driver for gpw-ts-kpoint-stress (never launched from ctest).
!
! Free-atom Tkatchenko-Scheffler dispersion energy of the gpw-ts-ice /
! water dimer in a periodic 12 A cube, plus E_disp at orthorhombic
! diagonally strained cells (one axis scaled by 1 +/- h at fixed fractional
! coordinates, h = 2e-3) and the per-axis strain-FD stress terms
! sigma_TS_aa = -dE/deps_aa / V. Geometry-only: the electronic k-mesh does
! not enter; pins are bit-identical to gpw-ts-stress.
!
! This is a *driver*, not a reimplementation: every number is produced by
! libMBD's own get_ts_energy (src/mbd_ts.F90). The driver only selects
! libMBD's pure real-space lattice-sum branch (param%ewald_on = .false.) with
! ATV's pair-cutoff convention (dipole_cutoff = 50 A, ATV Angstrom->Bohr
! factor 1/0.52917720859) so the pinned convention matches the regression
! (`ts_pbe` rcut). Free-atom alpha_0 / C6 / R_vdw come from libMBD's
! ts_vdw_params; Fermi damping s_R = 0.94, d = 20 (PBE defaults).
!
! Build+run (libmbd git 88069c6 built with the buildenv-gcc/2026.03-mpich
! module): see accompanying water_dimer_ts_kpoint_stress.note.
program pin_ts_strain
    use mbd_constants, only: dp
    use mbd_geom, only: geom_t
    use mbd_ts, only: get_ts_energy
    use mbd_damping, only: damping_t
    use mbd_vdw_param, only: ts_vdw_params, species_index
    use mbd_gradients, only: grad_request_t
    use mbd_utils, only: result_t

    implicit none

    integer, parameter :: n_atoms = 6
    real(dp), parameter :: a2b = 1d0 / 0.52917720859d0   ! ATV Angstrom->Bohr
    real(dp), parameter :: rcut = 50d0 * a2b             ! ATV kRcutTs [Bohr]
    real(dp), parameter :: l_ang = 12d0                  ! cube edge [A]
    real(dp), parameter :: h = 2d-3                      ! strain step
    character(len=2), parameter :: species(n_atoms) = &
        ['O ', 'H ', 'H ', 'O ', 'H ', 'H ']
    character(len=1), parameter :: axis_name(3) = ['x', 'y', 'z']

    real(dp) :: coords_ang(3, n_atoms), free_params(3, n_atoms)
    real(dp) :: alpha_0(n_atoms), c6(n_atoms), r_vdw(n_atoms)
    real(dp) :: scale(3), e0, ep, em, vol
    integer :: a

    coords_ang(:, 1) = [0.000000d0, 0d0, 0.000000d0]
    coords_ang(:, 2) = [0.758602d0, 0d0, 0.504284d0]
    coords_ang(:, 3) = [0.758602d0, 0d0, -0.504284d0]
    coords_ang(:, 4) = [3.000000d0, 0d0, 0.000000d0]
    coords_ang(:, 5) = [3.758602d0, 0d0, 0.504284d0]
    coords_ang(:, 6) = [3.758602d0, 0d0, -0.504284d0]

    free_params = ts_vdw_params(:, species_index(species))
    alpha_0 = free_params(1, :)
    c6 = free_params(2, :)
    r_vdw = free_params(3, :)

    e0 = ts_energy_rs([1d0, 1d0, 1d0], .false.)
    print '(a, es24.16, a)', 'E_disp isolated             = ', e0, ' Ha'
    e0 = ts_energy_rs([1d0, 1d0, 1d0], .true.)
    print '(a, es24.16, a)', 'E_disp unstrained           = ', e0, ' Ha'

    vol = (l_ang * a2b)**3
    do a = 1, 3
        scale = 1d0
        scale(a) = 1d0 + h
        ep = ts_energy_rs(scale, .true.)
        scale(a) = 1d0 - h
        em = ts_energy_rs(scale, .true.)
        print '(a, a, a, es24.16, a)', 'E_disp ', axis_name(a), ' +h            = ', ep, ' Ha'
        print '(a, a, a, es24.16, a)', 'E_disp ', axis_name(a), ' -h            = ', em, ' Ha'
        print '(a, a, a, a, a, es18.10, a)', 'sigma_TS_', axis_name(a), &
            axis_name(a), ' (FD, libMBD)', '  = ', -(ep - em) / (2d0 * h) / vol, ' Ha/Bohr3'
    end do

contains

    !> libMBD TS energy [Ha]; pure real-space branch (no Ewald), rcut 50 A.
    function ts_energy_rs(axis_scale, periodic) result(energy)
        real(dp), intent(in) :: axis_scale(3)
        logical, intent(in) :: periodic
        real(dp) :: energy

        type(geom_t) :: geom
        type(damping_t) :: damp
        type(result_t) :: res
        integer :: i, k

        allocate (geom%coords(3, n_atoms))
        do i = 1, n_atoms
            geom%coords(:, i) = coords_ang(:, i) * a2b * axis_scale
        end do
        if (periodic) then
            allocate (geom%lattice(3, 3), source=0d0)
            do k = 1, 3
                geom%lattice(k, k) = l_ang * a2b * axis_scale(k)
            end do
        end if
        geom%param%ewald_on = .false.        ! pure real-space lattice sum
        geom%param%dipole_cutoff = rcut      ! ATV ts_pbe pair cutoff
        call geom%init()
        damp%version = 'fermi'
        damp%ts_sr = 0.94d0                  ! PBE
        damp%ts_d = 20d0
        damp%r_vdw = r_vdw
        res = get_ts_energy(geom, alpha_0, c6, damp, grad_request_t())
        energy = res%energy
        call geom%destroy()
    end function

end program
