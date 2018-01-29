% CVX formulation for the separable NMF model proposed in 
%
% Bittorf, Recht, Ré, Tropp, "Factoring nonnegative matrices with linear
% programs", NIPS 2012. 
% 
% min p^T diag(X)
%     such that    X >= 0
%                  X_ij <= Xii <= 1 for all i,j 
%                  tr(X) = r
%
% K = set of indices corresponding to the r largest diagonal entries of X 

function [K,X] = HottTopixx_cvx(M,epsilon,r,p,norma)

[m,n] = size(M); 

if nargin <= 3, p = randn(n,1); end
if nargin <= 4, norma = 1; end

if norma == 1
    % 1. Normalize the columns of M
    D = diag(1./sum(M)); M = M*D; 
end

% Create and solve the model
cvx_begin
    variable X(n,n) 
    minimize( p'*diag(X) )
    subject to
        X(:) >= 0; 
        for i = 1 : n
            norm(M(:,i) - M*X(:,i),1) <= 2*epsilon; 
            X(i,i) <= 1; 
            for j = 1 : n
                X(i,j) <= X(i,i);
            end
        end
        trace(X) == r; 
cvx_end

x = diag(X); 
for i = 1 : r
    [a,b] = max(x); 
    K(i) = b; x(b) = -1; 
end
