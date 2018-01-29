% Wrapper for our new LP model with rho=2
function [K, X] = LPsepNMF_otwo(M, epsilon, r)
[~, X] = LPsepNMF_cplex_v2(M, epsilon, r, [], 'absolute', false,...
    'rhs_rho', 2.0);
K = PostProHybrid(M,diag(X),epsilon,r); 
end
