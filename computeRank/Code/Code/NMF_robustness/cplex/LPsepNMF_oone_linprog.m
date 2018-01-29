% Wrapper for our new LP model with rho=1
function [K, X] = LPsepNMF_oone_linprog(M, epsilon, r)
[~, X] = LPsepNMF_linprog_routine(M, epsilon, r, [], 'absolute', false,...
    'rhs_rho', 1.0);
K = PostProHybrid(M,diag(X),epsilon,r); 
end
