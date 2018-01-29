% Wrapper for fast recursive algorithm for separable NMF by Gillis & Vavasis
function [K] = spa(M, ~, k)
K = FastSepNMF(M,k,0);
% X = [];
end