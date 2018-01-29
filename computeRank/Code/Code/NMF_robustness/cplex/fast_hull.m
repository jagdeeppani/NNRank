% Wrapper for the fast conical hull algorithm by Kumar, Sindhwani & Kambadur
function [K,X] = fast_hull(M, ~, k)
K = FastConicalHull(M,k);
X = [];
end