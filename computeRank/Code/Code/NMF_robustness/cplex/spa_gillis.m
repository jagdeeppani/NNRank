% Wrapper for fast recursive algorithm for separable NMF by Gillis & Vavasis
function [K] = spa_gillis(M, ~, k)
K = FastSepNMF(M,k,1);
% X = [];
end