
% This function is a wrapper around the original LP model (Hottopixx) by 
% Bittorf, Recht, R� & Tropp
function [K, X] = hottopixx(M, epsilon, k)
[~, X] = hottopixx_v2(M, epsilon, k, [], 1, false);
K = PostProHybrid(M,diag(X),epsilon,k); 
end