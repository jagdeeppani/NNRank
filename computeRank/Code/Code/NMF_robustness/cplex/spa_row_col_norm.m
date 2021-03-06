% Wrapper for fast recursive algorithm for separable NMF by Gillis & Vavasis
function [K] = spa_row_col_norm(M, ~, k)

[m,n]=size(M);
M = spdiags(1./sum(M,2),0,m,m) * M ;
K = FastSepNMF(M,k,1);
% X = [];
end