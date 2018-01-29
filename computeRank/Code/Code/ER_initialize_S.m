function S = ER_initialize_S(P)
% Intialize S for solveQ_Cutting_Plane
% Implemented by Jagdeep Pani
% Algorithm 3.1 in "Minimum-Volume Enclosing Ellipsoids and Core Sets"
initialize_time=tic;
[r,n]=size(P);

if n<=2*r
    S=linspace(1,n,n);
    return;
end
S=[];
psi1=zeros(1,r);    % Each row of psi1 contains a point
psi1_idx=2;
iter=1;
while rank(psi1)<r && iter<500              
    Z=null(psi1);   % columns of Z contains the basis of null space of matrix formed by first l+1 columns of psi1.   The function null assumes rows of argument as datapoints
    
    b=Z*ones(size(Z,2),1);  % b is a vector in the orthogonal complement of space spanned by psi1
    val=P'*b;

    [~,alpha] = max(val);
    [~,beta] = min(val);
    S=union(S,[alpha beta]);
    psi1(psi1_idx,:)=(P(:,beta)-P(:,alpha))';
    psi1_idx=psi1_idx+1;
    iter=iter+1;
end

if iter==500 && rank(psi1)<r
    fprintf('# of iterations exceeded');
    t1= randperm(n);
    S=t1(1:2*r);
end    

% size(S)
fprintf('\nTime taken by ER_initialize_S is %f\n',toc(initialize_time));
% S

end



