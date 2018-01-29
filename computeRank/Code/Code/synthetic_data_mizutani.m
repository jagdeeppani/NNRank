function [A_orig,A,W,M,true_index,pi1] = synthetic_data_mizutani(m,n,k,sigma)
% for mizutani paper implementation input m=5000, n=250, k= 10, sigma
% between 0:0.1:0.5

alpha=unifrnd(0,1,k,1);
W=unifrnd(0,1,k,n);

N=normrnd(0,sigma,m,n);

M_dash=sample_dirichlet(alpha,m-k);
M=vertcat(eye(k),M_dash);
pi1=randperm(m);
M = M(pi1,:);

A_orig = M*W;
A = A_orig + N ;

true_index = find(pi1<=k);

end
