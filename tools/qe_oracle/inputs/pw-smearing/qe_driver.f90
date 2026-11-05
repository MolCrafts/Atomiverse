program qe_smearing_oracle
 use kinds
 implicit none
 real(DP), external :: wgauss,w1gauss,efermig
 real(DP) :: x,mu,f,s,number,entropy,width,target,degen
 real(DP) :: evals(5,1), wk(1)
 real(DP), parameter :: xs(13)=[-40._DP,-8._DP,-2._DP,-1._DP,-.25_DP,0._DP,.25_DP,1._DP,2._DP,8._DP,40._DP,-36._DP,36._DP]
 integer, parameter :: orders(5)=[-99,0,1,2,3]
 integer :: n,i,j,k,c,isk(1)
 do k=1,5
 n=orders(k)
 do i=1,size(xs)
 x=xs(i)
 f=wgauss(-x,n)
 s=-w1gauss(-x,n)
 write(*,'(a,1x,i4,3(1x,es26.17))')'POINT',n,x,f,s
 end do
 end do
 isk(1)=1
 do c=1,4
 if(c==1)then
 evals(:,1)=[-.5_DP,-.3_DP,.2_DP,.4_DP,.8_DP];target=4._DP;width=.01_DP;degen=2._DP
 else if(c==2)then
 evals(:,1)=[-.2_DP,-.08_DP,.01_DP,.11_DP,.24_DP];target=3.1_DP;width=.05_DP;degen=2._DP
 else if(c==3)then
 evals(:,1)=[-.02_DP,.02_DP,.8_DP,1._DP,2._DP];target=2._DP;width=.01_DP;degen=2._DP
 else
 evals(:,1)=[-.2_DP,-.08_DP,.01_DP,.11_DP,.24_DP];target=2.7_DP;width=.07_DP;degen=1._DP
 end if
 wk(1)=degen
 do k=1,5
 n=orders(k)
 mu=efermig(evals,5,1,target,wk,width,n,0,isk)
 number=0;entropy=0
 do j=1,5
 number=number+degen*wgauss((mu-evals(j,1))/width,n)
 entropy=entropy-degen*w1gauss((mu-evals(j,1))/width,n)
 end do
 write(*,'(a,2(1x,i4),4(1x,es26.17))')'SPECTRUM',c,n,mu,number,entropy,-width*entropy
 end do
 end do
end program
