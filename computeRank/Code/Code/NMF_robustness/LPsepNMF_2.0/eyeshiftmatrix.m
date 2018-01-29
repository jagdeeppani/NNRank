function S = eyeshiftmatrix(n, colnorms)

% Matrix to encode the constraints of the type
% 
%    colnorms_i * X_ij <= colnorms_j * X_ii
%
% which is needed in all LP models for near-separable NMF.
%
% ****** Input ******
%   n        :  number of columns in M (so X is n-by-n)
%   colnorms :  column norms of M (default: all-ones as in HottTopixx)
%
% ****** Output ******
%  S         : matrix S, such that S*vec(X) <= 0 represents the contraints 
%              above. 

if nargin < 2 || isempty(colnorms)
    colnorms = ones(n,1);
end

if n ~= length(colnorms)
    error('Incompatible dimension of column norm vector');
end

if any(colnorms < 0)
    error('Column norms must not be negative');
end

% Constraint matrix is of shape n*(n-1) X n*n  and each row has exactly two
% nonzero elements.
I = zeros(2*n*(n-1),1);
J = zeros(2*n*(n-1),1);
V = zeros(2*n*(n-1),1);

next_nz_idx = 1;
next_row = 1;
for k=1:n
    % For each diagonal position figure out column indices of vec(X)
    % variables.
    left_idx = k:n:((k-2)*n + k);
    diag_idx = (k-1)*n + k;
    right_idx = (k*n + k):n:(n*(n-1) + k);
    
    first_row = next_row;
    last_row = first_row + n - 2;
    
    first_nz1 = next_nz_idx;
    last_nz1 = first_nz1 + n - 2;
    first_nz2 = last_nz1 + 1;
    last_nz2 = first_nz2 + n - 2;
    
    I(first_nz1:last_nz1) = first_row:last_row;
    J(first_nz1:last_nz1) = diag_idx;
    V(first_nz1:last_nz1) = -[colnorms(1:k-1)' colnorms(k+1:end)'];
    
    I(first_nz2:last_nz2) = first_row:last_row;
    J(first_nz2:last_nz2) = [left_idx right_idx];
    V(first_nz2:last_nz2) = colnorms(k);
    
    % Upkeep
    next_row = last_row + 1;
    next_nz_idx = last_nz2 + 1;
end

S = sparse(I,J,V,n*(n-1), n*n);
end % of function eyeshiftmatrix
