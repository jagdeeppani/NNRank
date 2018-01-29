% Ellipsoid Rounding implementation
% Input A,k,rho,    A is document x words matrix
function Ar_I=ER_nmf_lowrank(A,noise_level,k)
rho=k;
%noise_level=0.1;
I=[];
while (length(I)<k)
    [I,Ar_I]= ER_routine(A,rho);
    rho=rho+1;
end
%disp(I);
if length(I)>k
    anchor_index=I(spa(Ar_I,noise_level,k));
else
    anchor_index=I;
end

    
Ar_I=Ar_I(anchor_index);
end



