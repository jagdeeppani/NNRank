% Ellipsoid Rounding implementation
% Input A,k,rho,    A is document x words matrix
function anchor_index=ER_spa_svds_v2(A,noise_level,k,varargin)

if ~isempty(varargin)   % Algo parameters
    [lim,theta,etta] = deal(varargin{:});
else
    lim = 0.05;
    theta = 0.99;
    etta = 5;
end


l_lim1 = 1-lim;
u_lim1 = 1+lim;

rho=k;
I=[];
[m,n]=size(A);
A_un=A;

% A = A*spdiags(1./sum(A,1)',0,n,n);
% A = spdiags(1./sum(A,2),0,m,m) * A ;  % rows of A are normalized

increment_counter=2;

% Ar = sparse(size(A,1),size(A,2));
while (length(I)<k && increment_counter<100 && rho<min(m,n) )
    disp('svd started');
    if (rho==k)
        [~,D,V]= svds(A,k*2);
    end
    
    if rho>2*k
        fprintf('Iterations exceeded, more rank svd required \n');
        return;
    end
    
    disp('svd done');
    %     Ar = U * D * V';
    P = D(1:rho,1:rho) * (V(:,1:rho))';
    %     Ar = U * P; % Ar is the best rank r approximation of matrix A (in terms of frobenius norm)
    %P = P(1:rho,:); % P is a matrix of size rxn
    disp('SolveQ_Cutting_Plane started');
    L=solveQ_Cutting_Plane(P,noise_level,l_lim1,u_lim1,theta,etta);
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
        if temp>=l_lim1 && temp<=u_lim1         % Parameter to be changed
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
if rho>min(m,n)
    anchor_index=0;
    disp('rho value exceeded the number of words or # of docs');
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

