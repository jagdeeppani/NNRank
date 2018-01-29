% Ellipsoid Rounding implementation
% Input A,k,rho,    A is document x words matrix
function anchor_index=ER_spa(A,noise_level,k)

% normalize columns of A
[m,n]=size(A);
A = A*spdiags(1./sum(A,1)',0,n,n);

A=full(A); % As the input A sparse

rho=k;
%noise_level=0.1;
I=[];

disp('svd started');
tic
if issparse(A)
    disp('matrix is sparse');
    try
        [~,D,V]= svds(A);
    catch err
        anchor_index=0;
        return
    end
        
    
else
    disp('matrix is dense');
        try
        [~,D,V]= svd(A);
    catch err
        anchor_index=0;
        return
    end
end
disp('svd done');
toc
t=min(m,n);


increment_counter=2;
while (length(I)<k && increment_counter<100)
    D_temp=D;
    for i=rho+1:t
        D_temp(i,i)=0;
    end
    P=D_temp*V';
    P=P(1:rho,:); % P is a matrix of size rxn
    disp('SOlveQ_Cutting_Plane started');
    L=solveQ_Cutting_Plane(P); 
    disp('SolveQ_Cutting plane done')
    
    if (L==0)
        disp('cycle detected in solveQ_Cutting_Plane');
        %anchor_index=0;
        rho=rho+increment_counter;
        increment_counter=increment_counter+1;
        continue;
    end
    
    count=1;
    I=[];
    for i=1:n
        temp = P(:,i)'*L*P(:,i);
        if temp>=0.98 && temp<=1.02
            I(count)=i;
            count=count+1;
        end
    end
        rho=rho+1;
        increment_counter=increment_counter+1;
        
end
%disp(I);
if increment_counter>=100
    anchor_index=0;
    return
end

if length(I)>k
    anchor_index=I(spa(A(:,I),noise_level,k));
else
    anchor_index=I;
end

end



