! Read spectra captured by the diagnostic-only PW solver and call actual QE.
program replay_qe
 use kinds
 implicit none
 real(DP),external :: efermig,wgauss,w1gauss,w0gauss
 real(DP),allocatable :: evals(:,:),weights(:)
 integer,allocatable :: isk(:)
 integer :: count,i,j,n,id,order
 real(DP) :: target,width,degen,native_mu,qe_mu,gaussian_mu,number,entropy,slope,x
 read(*,*) count
 do i=1,count
   read(*,*) id,n,target,width,order,degen,native_mu
   allocate(evals(1,n),weights(n),isk(n))
   read(*,*) evals(1,:)
   read(*,*) weights(:)
   weights=weights*degen
   isk=1
   gaussian_mu=efermig(evals,1,n,target,weights,width,0,0,isk)
   qe_mu=efermig(evals,1,n,target,weights,width,order,0,isk)
   number=0;entropy=0;slope=0
   do j=1,n
     x=(native_mu-evals(1,j))/width
     number=number+weights(j)*wgauss(x,order)
     entropy=entropy-weights(j)*w1gauss(x,order)
     slope=slope+weights(j)*w0gauss(x,order)/width
   end do
   write(*,'(a,1x,i8,6(1x,es26.17))') 'REPLAY',id,qe_mu,gaussian_mu,number,entropy,slope,native_mu
   deallocate(evals,weights,isk)
 end do
end program
