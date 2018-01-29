% Wrapper for fast recursive algorithm for separable NMF by Gillis & Vavasis
function [K] = spa_un(M, ~, k)
K = FastSepNMF(M,k,0);
% X = [];
end