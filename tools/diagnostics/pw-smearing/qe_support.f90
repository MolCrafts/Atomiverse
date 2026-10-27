module kinds
 implicit none
 integer, parameter :: DP=selected_real_kind(15,307)
end module
module constants
 use kinds
 implicit none
 real(DP), parameter :: pi=acos(-1._DP),sqrtpm1=1._DP/sqrt(pi),sqrt2=sqrt(2._DP),rytoev=13.605693122994_DP
end module
module io_global
 integer,parameter :: stdout=6
end module
module mp_pools
 integer,parameter :: inter_pool_comm=0
end module
module mp
 use kinds
 implicit none
contains
 subroutine mp_max(value,comm)
 real(DP),intent(inout)::value
 integer,intent(in)::comm
 end subroutine
 subroutine mp_min(value,comm)
 real(DP),intent(inout)::value
 integer,intent(in)::comm
 end subroutine
 subroutine mp_sum(value,comm)
 real(DP),intent(inout)::value
 integer,intent(in)::comm
 end subroutine
end module
subroutine errore(where,why,n)
 implicit none
 character(len=*),intent(in)::where,why
 integer,intent(in)::n
 print *,trim(where),trim(why),n
 stop 1
end subroutine
