function X = ER_initialize_Sv2(S)
% Intialize S for solveQ_Cutting_Plane
% Implemented by Jagdeep Pani
% Algorithm 3.1 in "Minimum-Volume Enclosing Ellipsoids and Core Sets"
initialize_time=tic;
[d,n]=size(S);

if n<=2*d
    X=linspace(1,n,n);
    return;
end
X=[];
psi1=[];    % Each row of psi1 contains a point
psi1_idx=1;
first_iter=1;
while rank(psi1)<d
    
    if first_iter==1
        b=unifrnd(0,1,d,1);
        first_iter=0;
    else
        l=size(psi1,1);    % l is the no of nonzero rows of psi1
        if l<=d
        Z=null(psi1(:,1:l+1));   % columns of Z contains the basis of null space of matrix formed by first l+1 columns of psi1.   The function null assumes rows of argument as datapoints
        b = vertcat(Z(:,1),zeros(d-l-1,1));  % b is a vector in the orthogonal complement of space spanned by psi1
        else
            
            
        end
        
        
        % for finding a point from null space, idea taken from http://mathoverflow.net/questions/168721/is-finding-a-single-vector-in-the-null-space-as-difficult-as-discovering-the-who?newreg=45e1410c9ecc4f3bb82f60cf154685af
    end
        
        
        %b=Z*ones(size(Z,2),1);  % b is a vector in the orthogonal complement of space spanned by psi1
        
        % b = normrnd(0,1,size(psi1,2),1);
        % fprintf('value of ************** is %f\n',psi1*b);
        
        d
        l
        size(psi1)
        size(Z)
        size(b)
        size(P)
        
        
        val=P'*b;
        
        [~,alpha] = max(val);
        [~,beta] = min(val);
        X=union(X,[alpha beta]);
        psi1(psi1_idx,:)=(P(:,beta)-P(:,alpha))';
        psi1_idx=psi1_idx+1;
    end
    
    fprintf('\nTime taken by ER_initialize_S is %f\n',toc(initialize_time));
    
end



