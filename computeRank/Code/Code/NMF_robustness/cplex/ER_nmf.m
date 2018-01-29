% Ellipsoid Rounding NMF implemented by Jagdeep
% Input A,k,rho,    A is document x words matrix
function anchor_index=ER_nmf(A,noise_level,k)
rho=k;
[m,n]=size(A);
A_un=A;
A = A*spdiags(1./sum(A,1)',0,n,n);

%noise_level=0.1;
I=[];

disp('svd started');
tic
if issparse(A)
    [~,D,V]= svds(A);
else
    [~,D,V]= svd(A);
end
disp('svd done');
toc
t=min(m,n);


increment_counter=2;
while (length(I)<k)
    D_temp=D;
    for i=rho+1:t
        D_temp(i,i)=0;
    end
    P=D_temp*V';
    P=P(1:rho,:); % P is a matrix of size rxn
    disp('SOlveQ_Cutting_Plane started');
    L=solveQ_Cutting_Plane(P); 
    disp('SolveQ_Cutting plane done')
    count=1;
    I=[];
    for i=1:n
        temp = P(:,i)'*L*P(:,i);
        if temp>=0.98 && temp<=1.02
            I(count)=i;
            count=count+1;
        end
    end
        rho=rho+increment_counter;
        increment_counter=increment_counter+1;
end
%disp(I);
if length(I)>k
    anchor_index=I(spa(A_un(:,I),noise_level,k));
else
    anchor_index=I;
end

end



