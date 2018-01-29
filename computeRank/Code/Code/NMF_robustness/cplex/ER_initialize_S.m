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
while rank(psi1)<r              
    %     l=l-1;  % l is the no of nonzero rows of psi1
    Z=null(psi1);   % columns of Z contains the basis of null space of matrix formed by first l+1 columns of psi1.   The function null assumes rows of argument as datapoints
%     b = vertcat(Z(:,1),zeros(r-l-1,1));  % b is a vector in the orthogonal complement of space spanned by psi1
    % for finding a point from null space, idea taken from http://mathoverflow.net/questions/168721/is-finding-a-single-vector-in-the-null-space-as-difficult-as-discovering-the-who?newreg=45e1410c9ecc4f3bb82f60cf154685af
    
    b=Z*ones(size(Z,2),1);  % b is a vector in the orthogonal complement of space spanned by psi1
    
    % b = normrnd(0,1,size(psi1,2),1);
    % fprintf('value of ************** is %f\n',psi1*b);
    
    val=P'*b;
    
    [~,alpha] = max(val);
    [~,beta] = min(val);
    S=union(S,[alpha beta]);
    psi1(psi1_idx,:)=(P(:,beta)-P(:,alpha))';
    psi1_idx=psi1_idx+1;
end

fprintf('\nTime taken by ER_initialize_S is %f\n',toc(initialize_time));

end



