function [I,Ar_I]= ER_routine_lowrank(A,r)
% This function implements the Algorithm 1 of Mizutani etal.
[m,n]=size(A);
disp('svd started');
if issparse(A)
    [U,D,V]= svds(A);
else
    [U,D,V]= svd(A);
end
disp('svd done');
t=min(m,n);

for i=r+1:t
    D(i,i)=0;
end
P=D*V';
P=P(1:r,:); % P is a matrix of size rxn
disp('SOlveQ_Cutting_Plane started');
L=solveQ_Cutting_Plane(P); 
disp('SolveQ_Cutting plane done')
count=1;
I=[];
for i=1:n
    temp = P(:,i)'*L*P(:,i);
    if temp>=0.99 && temp<=1.01
        I(count)=i;
        count=count+1;
    end
end

Ar_I=U*D*V';
end