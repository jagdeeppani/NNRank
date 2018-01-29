% Ellipsoid Rounding implementation
% Input A,k,rho,    A is document x words matrix
function anchor_index=ER_spa_svds(A,noise_level,k)
anchor_index=1:1:k;
return;
%A=full(A); % As the input A sparse
% if ~issparse(A)
%     A=sparse(A);
% end
rho=k;
I=[];
[m,n]=size(A);
A_un=A;

% A = A*spdiags(1./sum(A,1)',0,n,n);
A = spdiags(1./sum(A,2),0,m,m) * A ;  % rows of A are normalized

increment_counter=2;

Ar = zeros(size(A,1),size(A,2));
while (length(I)<k && increment_counter<100 && rho<n)
    disp('svd started');
    try
        if (rho==k)
            [U,D,V]= svds(A,rho);
        else
            [Ut,Dt,Vt]= svds(A-Ar,1);
            U = horzcat(U,Ut); V = horzcat(V,Vt);   D(rho,rho) = Dt(1);
        end
    catch err
        disp('svd failed');
        anchor_index=0;
        return
    end

    disp('svd done');
%     Ar = U * D * V';
    P = D * V';
    Ar = U * P; % Ar is the best rank r approximation of matrix A (in terms of frobenius norm)
    %P = P(1:rho,:); % P is a matrix of size rxn
    disp('SolveQ_Cutting_Plane started');
    L=solveQ_Cutting_Plane(P,noise_level); 
    disp('SolveQ_Cutting plane done')
    
    if (L==0)
        disp('MinvolEllipse didnt stop or cycle detected in solveQ_Cutting_Plane');
        %anchor_index=0;
        rho=rho+1;
        increment_counter=increment_counter+1;
        continue;
    end
    
    count=1;
    I=[];
    for i=1:n
        temp = P(:,i)'*L*P(:,i);
        if temp>=0.85   && temp<=1.15         % Parameter to be changed
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
if rho>n
    anchor_index=0;
    disp('rho value exceeded the number of words');
    return
end

% disp(I)
disp('*********************************************************');
fprintf('number of points after preprocessing is %d',length(I));
disp('*********************************************************');
if length(I)>k
    anchor_index=I(spa(A_un(:,I),noise_level,k));
elseif length(I)==k
    anchor_index=I;
else
    anchor_index=0;
end

end

